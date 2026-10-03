import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt


def load(run_dir):
    path = pathlib.Path(run_dir) / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot(run_dir):
    rows = load(run_dir)
    if not rows:
        return
    games = [r["games"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(games, [r["policy_loss"] for r in rows], label="policy loss")
    ax.plot(games, [r["value_loss"] for r in rows], label="value loss")
    ax.set_title("Training loss")
    ax.set_xlabel("self-play games")
    ax.legend()

    ax = axes[0, 1]
    evaluated = [r for r in rows if "eval_score" in r]
    if evaluated:
        x = [r["games"] for r in evaluated]
        ax.plot(x, [r["eval_score"] for r in evaluated], marker="o", label="score")
        ax.plot(x, [r["eval_wins"] for r in evaluated], linestyle="--", alpha=0.6, label="wins")
        ax.plot(x, [r["eval_losses"] for r in evaluated], linestyle="--", alpha=0.6, label="losses")
        ax.legend()
    ax.axhline(0.5, color="gray", linewidth=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Against rollout MCTS (score: win 1, draw 0.5)")
    ax.set_xlabel("self-play games")

    ax = axes[1, 0]
    ax.plot(games, [r["mean_game_length"] for r in rows])
    ax.set_title("Self-play game length (moves)")
    ax.set_xlabel("self-play games")

    ax = axes[1, 1]
    ax.plot(games, [r["first_player_score"] for r in rows])
    ax.axhline(0.5, color="gray", linewidth=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Self-play: first player's score")
    ax.set_xlabel("self-play games")

    fig.tight_layout()
    fig.savefig(pathlib.Path(run_dir) / "progress.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    plot(sys.argv[1] if len(sys.argv) > 1 else "runs/connect4")
