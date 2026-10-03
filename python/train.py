import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

import connect4_az
import plot
from network import AlphaZeroNet, evaluate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/connect4")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    # Self-play
    p.add_argument("--games-per-iteration", type=int, default=128)
    p.add_argument("--parallel-games", type=int, default=128)
    p.add_argument("--simulations", type=int, default=100)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--dirichlet-alpha", type=float, default=1.0)
    p.add_argument("--noise-fraction", type=float, default=0.25)
    p.add_argument("--temperature-moves", type=int, default=10)
    # Network and training
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--buffer-positions", type=int, default=200_000)
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # Evaluation
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-games", type=int, default=40)
    p.add_argument("--eval-simulations", type=int, default=100)
    p.add_argument("--opponent-simulations", type=int, default=1000)
    return p.parse_args()


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity, obs_size, num_actions, num_players):
        self.observations = np.zeros((capacity, obs_size), np.float32)
        self.policies = np.zeros((capacity, num_actions), np.float32)
        self.values = np.zeros((capacity, num_players), np.float32)
        self.capacity, self.size, self.next = capacity, 0, 0

    def add(self, observations, policies, values):
        for o, p, v in zip(observations, policies, values):
            self.observations[self.next], self.policies[self.next], self.values[self.next] = o, p, v
            self.next = (self.next + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        i = rng.integers(0, self.size, batch_size)
        return self.observations[i], self.policies[i], self.values[i]

# flip a position horizontally for augmentation
def mirror_half(observations, policies, shape, rng):
    obs = observations.reshape(-1, *shape).copy()
    pol = policies.copy()
    flip = rng.random(len(obs)) < 0.5
    obs[flip] = obs[flip][..., ::-1]
    pol[flip] = pol[flip][:, ::-1]
    return obs, pol


def self_play(net, driver, num_games, device):
    parts = []
    finished = 0
    while finished < num_games:
        observations, masks = driver.pending()
        driver.submit(*evaluate(net, observations, masks, device))
        examples = driver.take_examples()
        if len(examples["game_lengths"]):
            parts.append(examples)
            finished += len(examples["game_lengths"])
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def train(net, optimizer, buffer, steps, batch_size, device, rng):
    net.train()
    shape = net.in_shape
    policy_total = value_total = 0.0
    for _ in range(steps):
        observations, policies, values = buffer.sample(batch_size, rng)
        observations, policies = mirror_half(observations, policies, shape, rng)
        x = torch.from_numpy(observations).to(device)
        target_policy = torch.from_numpy(policies).to(device)
        target_value = torch.from_numpy(values).to(device)

        logits, predicted = net(x)
        policy_loss = -(target_policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(predicted, target_value)
        optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        optimizer.step()

        policy_total += policy_loss.item()
        value_total += value_loss.item()
    return policy_total / steps, value_total / steps


def arena(net, args, device, seed):
    driver = connect4_az.ArenaDriver(
        args.eval_games, args.eval_simulations, args.opponent_simulations, args.c_puct, seed
    )
    while not driver.is_finished():
        observations, masks = driver.pending()
        driver.submit(*evaluate(net, observations, masks, device))
    returns = np.array(driver.network_returns())
    return (returns > 0).mean(), (returns == 0).mean(), (returns < 0).mean()


def main():
    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    shape, actions, players = connect4_az.OBSERVATION_SHAPE, connect4_az.NUM_ACTIONS, connect4_az.NUM_PLAYERS
    net = AlphaZeroNet(shape, actions, players, args.channels, args.blocks).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buffer = ReplayBuffer(args.buffer_positions, int(np.prod(shape)), actions, players)

    start_iteration, games, positions = 0, 0, 0
    checkpoint_path = out / "checkpoint.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration, games, positions = checkpoint["iteration"], checkpoint["games"], checkpoint["positions"]
        print(f"resumed at iteration {start_iteration} (the replay buffer starts empty)")
    elif (out / "metrics.jsonl").exists():
        (out / "metrics.jsonl").unlink()  

    driver = connect4_az.SelfPlayDriver(
        args.parallel_games, args.simulations, args.c_puct, args.dirichlet_alpha,
        args.noise_fraction, args.temperature_moves, args.seed + start_iteration,
    )
    print(f"training on {device}; writing to {out}")

    for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
        started = time.time()
        examples = self_play(net, driver, args.games_per_iteration, device)
        buffer.add(examples["observations"], examples["policies"], examples["values"])
        policy_loss, value_loss = train(net, optimizer, buffer, args.train_steps, args.batch_size, device, rng)

        games += len(examples["game_lengths"])
        positions += len(examples["values"])
        row = {
            "iteration": iteration,
            "games": games,
            "positions": positions,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "mean_game_length": float(examples["game_lengths"].mean()),
            "first_player_score": float((examples["returns"][:, 0].mean() + 1) / 2),
        }
        if iteration % args.eval_every == 0:
            wins, draws, losses = arena(net, args, device, seed=iteration)
            row.update(eval_wins=wins, eval_draws=draws, eval_losses=losses, eval_score=wins + 0.5 * draws)
        row["seconds"] = time.time() - started

        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps({k: float(v) if isinstance(v, np.floating) else v for k, v in row.items()}) + "\n")
        torch.save(
            {"net": net.state_dict(), "optimizer": optimizer.state_dict(),
             "iteration": iteration, "games": games, "positions": positions, "args": vars(args)},
            checkpoint_path,
        )
        plot.plot(out)

        summary = f"iter {iteration:4d}  games {games:6d}  policy {policy_loss:.3f}  value {value_loss:.3f}"
        if "eval_score" in row:
            summary += f"  vs rollouts {row['eval_score']:.2f}"
        print(f"{summary}  ({row['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
