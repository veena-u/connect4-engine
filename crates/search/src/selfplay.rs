use crate::{Config, Mcts, RandomRollout};
use game::rng::Rng;
use game::{Action, Game, Turn};

/// this contains both the selfplay driver (for collecting train data) 
/// and the arena driver (for evaluating a new network against an old one i.e. randomrollout)

#[derive(Default)]
pub struct Examples {
    pub observations: Vec<f32>,
    pub policies: Vec<f32>,
    pub values: Vec<f32>,
    pub game_lengths: Vec<u32>,
    pub returns: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct SelfPlayConfig {
    pub search: Config,
    pub temperature_moves: u32,
}

pub struct SelfPlay<G: Game> {
    game: G,
    config: SelfPlayConfig,
    games: Vec<SelfPlayGame<G>>,
    rng: Rng,
    finished: Examples,
    scratch: Scratch,
}

struct SelfPlayGame<G: Game> {
    state: G::State,
    mcts: Mcts<G>,
    moves: u32,
    records: Vec<Record>,
    leaf: Option<G::State>,
}

struct Record {
    observation: Vec<f32>,
    policy: Vec<f32>,
    mover: usize,
}

impl<G: Game> SelfPlay<G> {
    pub fn new(game: G, config: SelfPlayConfig, parallel_games: usize, seed: u64) -> Self {
        assert!(config.search.simulations >= 2, "visit counts need at least 2 simulations");
        let mut rng = Rng::new(seed);
        let scratch = Scratch::new(G::NUM_PLAYERS);
        let games = (0..parallel_games)
            .map(|_| {
                let state = game.initial_state();
                let mut mcts = Mcts::new(config.search, rng.next_u64());
                mcts.start(&state);
                SelfPlayGame { state, mcts, moves: 0, records: Vec::new(), leaf: None }
            })
            .collect();
        let mut driver = Self { game, config, games, rng, finished: Examples::default(), scratch };
        for i in 0..parallel_games {
            driver.advance(i);
        }
        driver
    }

    pub fn num_pending(&self) -> usize {
        self.games.iter().filter(|g| g.leaf.is_some()).count()
    }

    pub fn write_pending(&mut self, observations: &mut [f32], masks: &mut [bool]) {
        let leaves = self.games.iter().filter_map(|g| g.leaf.as_ref());
        write_leaves(&self.game, leaves, observations, masks, &mut self.scratch.actions);
    }

    pub fn submit(&mut self, priors: &[f32], values: &[f32]) {
        let pending: Vec<usize> = (0..self.games.len()).filter(|&i| self.games[i].leaf.is_some()).collect();
        check_batch::<G>(pending.len(), priors, values);
        let (a, p) = (G::NUM_ACTIONS, G::NUM_PLAYERS);
        for (row, &i) in pending.iter().enumerate() {
            let g = &mut self.games[i];
            let leaf = g.leaf.take().expect("pending games have a leaf");
            let (prior_row, value_row) = (&priors[row * a..][..a], &values[row * p..][..p]);
            submit_leaf(&self.game, &mut g.mcts, &leaf, prior_row, value_row, &mut self.scratch);
            self.advance(i);
        }
    }

    pub fn take_examples(&mut self) -> Examples {
        std::mem::take(&mut self.finished)
    }

    fn advance(&mut self, i: usize) {
        loop {
            if self.games[i].mcts.is_done() {
                self.play_move(i);
            } else if let Some(leaf) = self.games[i].mcts.select(&self.game) {
                self.games[i].leaf = Some(leaf);
                return;
            }
        }
    }

    fn play_move(&mut self, i: usize) {
        let game = &self.game;
        let g = &mut self.games[i];
        let Turn::Player(mover) = game.turn(&g.state) else { unreachable!("searches start on a player's turn") };

        let result = g.mcts.result();
        let total: u32 = result.visits.iter().map(|&(_, n)| n).sum();
        let mut policy = vec![0.0; G::NUM_ACTIONS];
        for &(action, n) in &result.visits {
            policy[action] = n as f32 / total as f32;
        }
        let mut observation = vec![0.0; G::OBSERVATION_SIZE];
        game.observe(&g.state, mover, &mut observation);
        g.records.push(Record { observation, policy, mover });

        let action = if g.moves < self.config.temperature_moves {
            let weighted: Vec<(Action, f64)> =
                result.visits.iter().filter(|&&(_, n)| n > 0).map(|&(a, n)| (a, n as f64)).collect();
            self.rng.weighted(&weighted)
        } else {
            result.best
        };
        game.apply(&mut g.state, action);
        g.moves += 1;

        if game.turn(&g.state) == Turn::Terminal {
            let returns = &mut self.scratch.values;
            game.returns(&g.state, returns);
            let out = &mut self.finished;
            let mut relative = vec![0.0; G::NUM_PLAYERS];
            for record in g.records.drain(..) {
                to_relative(returns, record.mover, &mut relative);
                out.observations.extend(record.observation);
                out.policies.extend(record.policy);
                out.values.extend_from_slice(&relative);
            }
            out.game_lengths.push(g.moves);
            out.returns.extend_from_slice(returns);

            g.state = game.initial_state();
            g.moves = 0;
        }
        g.mcts.start(&g.state);
    }
}

pub struct Arena<G: Game> {
    game: G,
    games: Vec<ArenaGame<G>>,
    opponent: Mcts<G>,
    rollout: RandomRollout,
    scratch: Scratch,
}

struct ArenaGame<G: Game> {
    state: G::State,
    mcts: Mcts<G>,
    network_seat: usize,
    searching: bool,
    leaf: Option<G::State>,
    result: Option<f32>,
}

impl<G: Game> Arena<G> {
    pub fn new(game: G, network: Config, opponent: Config, num_games: usize, seed: u64) -> Self {
        let network = Config { root_noise: None, ..network };
        let opponent = Config { root_noise: None, ..opponent };
        let mut rng = Rng::new(seed);
        let games = (0..num_games)
            .map(|i| ArenaGame {
                state: game.initial_state(),
                mcts: Mcts::new(network, rng.next_u64()),
                network_seat: i % G::NUM_PLAYERS,
                searching: false,
                leaf: None,
                result: None,
            })
            .collect();
        let (opponent, rollout) = (Mcts::new(opponent, rng.next_u64()), RandomRollout::new(rng.next_u64()));
        let mut arena = Self { game, games, opponent, rollout, scratch: Scratch::new(G::NUM_PLAYERS) };
        for i in 0..num_games {
            arena.advance(i);
        }
        arena
    }

    pub fn is_finished(&self) -> bool {
        self.games.iter().all(|g| g.result.is_some())
    }

    pub fn num_pending(&self) -> usize {
        self.games.iter().filter(|g| g.leaf.is_some()).count()
    }

    pub fn write_pending(&mut self, observations: &mut [f32], masks: &mut [bool]) {
        let leaves = self.games.iter().filter_map(|g| g.leaf.as_ref());
        write_leaves(&self.game, leaves, observations, masks, &mut self.scratch.actions);
    }

    pub fn submit(&mut self, priors: &[f32], values: &[f32]) {
        let pending: Vec<usize> = (0..self.games.len()).filter(|&i| self.games[i].leaf.is_some()).collect();
        check_batch::<G>(pending.len(), priors, values);
        let (a, p) = (G::NUM_ACTIONS, G::NUM_PLAYERS);
        for (row, &i) in pending.iter().enumerate() {
            let g = &mut self.games[i];
            let leaf = g.leaf.take().expect("pending games have a leaf");
            let (prior_row, value_row) = (&priors[row * a..][..a], &values[row * p..][..p]);
            submit_leaf(&self.game, &mut g.mcts, &leaf, prior_row, value_row, &mut self.scratch);
            self.advance(i);
        }
    }

    pub fn network_returns(&self) -> Vec<f32> {
        self.games.iter().map(|g| g.result.expect("call once is_finished() is true")).collect()
    }

    fn advance(&mut self, i: usize) {
        let game = &self.game;
        let g = &mut self.games[i];
        loop {
            match game.turn(&g.state) {
                Turn::Terminal => {
                    game.returns(&g.state, &mut self.scratch.values);
                    g.result = Some(self.scratch.values[g.network_seat]);
                    return;
                }
                Turn::Player(p) if p != g.network_seat => {
                    let best = self.opponent.search(game, &g.state, &mut self.rollout).best;
                    game.apply(&mut g.state, best);
                }
                Turn::Player(_) => {
                    if !g.searching {
                        g.mcts.start(&g.state);
                        g.searching = true;
                    }
                    if g.mcts.is_done() {
                        game.apply(&mut g.state, g.mcts.result().best);
                        g.searching = false;
                    } else if let Some(leaf) = g.mcts.select(game) {
                        g.leaf = Some(leaf);
                        return;
                    }
                }
            }
        }
    }
}

struct Scratch {
    actions: Vec<Action>,
    priors: Vec<f32>,
    values: Vec<f32>,
}

impl Scratch {
    fn new(num_players: usize) -> Self {
        Self { actions: Vec::new(), priors: Vec::new(), values: vec![0.0; num_players] }
    }
}

pub fn to_absolute(relative: &[f32], mover: usize, absolute: &mut [f32]) {
    let n = absolute.len();
    for (k, &value) in relative.iter().enumerate() {
        absolute[(mover + k) % n] = value;
    }
}

pub fn to_relative(absolute: &[f32], mover: usize, relative: &mut [f32]) {
    let n = absolute.len();
    for (k, value) in relative.iter_mut().enumerate() {
        *value = absolute[(mover + k) % n];
    }
}

fn check_batch<G: Game>(rows: usize, priors: &[f32], values: &[f32]) {
    assert_eq!(priors.len(), rows * G::NUM_ACTIONS, "priors must be pending x NUM_ACTIONS");
    assert_eq!(values.len(), rows * G::NUM_PLAYERS, "values must be pending x NUM_PLAYERS");
}

fn write_leaves<'a, G: Game + 'a>(
    game: &G,
    leaves: impl Iterator<Item = &'a G::State>,
    observations: &mut [f32],
    masks: &mut [bool],
    actions: &mut Vec<Action>,
) {
    let (o, a) = (G::OBSERVATION_SIZE, G::NUM_ACTIONS);
    for (row, leaf) in leaves.enumerate() {
        let Turn::Player(mover) = game.turn(leaf) else { unreachable!("leaves are player turns") };
        game.observe(leaf, mover, &mut observations[row * o..][..o]);
        game.legal_actions(leaf, actions);
        let mask = &mut masks[row * a..][..a];
        mask.fill(false);
        actions.iter().for_each(|&action| mask[action] = true);
    }
}

fn submit_leaf<G: Game>(
    game: &G,
    mcts: &mut Mcts<G>,
    leaf: &G::State,
    prior_row: &[f32],
    value_row: &[f32],
    scratch: &mut Scratch,
) {
    let Turn::Player(mover) = game.turn(leaf) else { unreachable!("leaves are player turns") };
    game.legal_actions(leaf, &mut scratch.actions);
    scratch.priors.clear();
    scratch.priors.extend(scratch.actions.iter().map(|&action| prior_row[action]));
    let total: f32 = scratch.priors.iter().sum();
    let fallback = 1.0 / scratch.priors.len() as f32;
    scratch.priors.iter_mut().for_each(|p| *p = if total > 0.0 { *p / total } else { fallback });
    to_absolute(value_row, mover, &mut scratch.values);
    mcts.expand_and_backup(game, &scratch.priors, &scratch.values);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Evaluator;
    use connect4::Connect4;

    const OBS: usize = 84;
    const ACTIONS: usize = 7;

    #[test]
    fn relative_values_round_trip() {
        let absolute = [0.1, 0.2, 0.3, 0.4];
        let (mut relative, mut back) = ([0.0; 4], [0.0; 4]);
        to_relative(&absolute, 2, &mut relative);
        assert_eq!(relative, [0.3, 0.4, 0.1, 0.2]);
        to_absolute(&relative, 2, &mut back);
        assert_eq!(back, absolute);
    }

    fn uniform_batch(masks: &[bool], rows: usize) -> (Vec<f32>, Vec<f32>) {
        let priors = masks.iter().map(|&legal| if legal { 1.0 } else { 0.0 }).collect();
        (priors, vec![0.0; rows * 2])
    }

    #[test]
    fn self_play_records_consistent_examples() {
        let config = SelfPlayConfig { search: Config { simulations: 16, ..Config::default() }, temperature_moves: 4 };
        let mut driver = SelfPlay::new(Connect4, config, 8, 0);
        let (mut obs, mut masks) = (vec![0.0; 8 * OBS], vec![false; 8 * ACTIONS]);
        let mut examples = Examples::default();
        while examples.game_lengths.len() < 30 {
            let rows = driver.num_pending();
            assert_eq!(rows, 8, "every self-play game always has a leaf waiting");
            driver.write_pending(&mut obs, &mut masks);
            let (priors, values) = uniform_batch(&masks, rows);
            driver.submit(&priors, &values);
            let new = driver.take_examples();
            examples.observations.extend(new.observations);
            examples.policies.extend(new.policies);
            examples.values.extend(new.values);
            examples.game_lengths.extend(new.game_lengths);
            examples.returns.extend(new.returns);
        }

        let positions: u32 = examples.game_lengths.iter().sum();
        assert_eq!(examples.observations.len(), positions as usize * OBS);
        assert_eq!(examples.policies.len(), positions as usize * ACTIONS);
        assert_eq!(examples.values.len(), positions as usize * 2);

        let mut row = 0;
        for (game, &length) in examples.game_lengths.iter().enumerate() {
            let seat0 = examples.returns[game * 2];
            for k in 0..length as usize {
                let policy = &examples.policies[row * ACTIONS..][..ACTIONS];
                assert!((policy.iter().sum::<f32>() - 1.0).abs() < 1e-5);
                let o = &examples.observations[row * OBS..][..OBS];
                let (mine, theirs) = (o[..42].iter().sum::<f32>(), o[42..].iter().sum::<f32>());
                assert!(theirs - mine == 0.0 || theirs - mine == 1.0);
                let expected = if k % 2 == 0 { seat0 } else { -seat0 };
                assert_eq!(examples.values[row * 2], expected, "game {game}, move {k}");
                row += 1;
            }
        }
    }

    #[test]
    fn batched_path_matches_direct_search() {
        for (moves, seed) in [(vec![], 0), (vec![0, 1, 0, 1, 0], 1), (vec![3], 2)] {
            let mut state = Connect4.initial_state();
            for col in moves {
                Connect4.apply(&mut state, col);
            }
            let config = Config { simulations: 200, ..Config::default() };
            let direct = Mcts::new(config, seed).search(&Connect4, &state, &mut RandomRollout::new(seed));

            let (mut mcts, mut rollout) = (Mcts::new(config, seed), RandomRollout::new(seed));
            let mut scratch = Scratch::new(2);
            let (mut actions, mut priors, mut absolute) = (Vec::new(), Vec::new(), [0.0; 2]);
            mcts.start(&state);
            while !mcts.is_done() {
                if let Some(leaf) = mcts.select(&Connect4) {
                    Connect4.legal_actions(&leaf, &mut actions);
                    priors.resize(actions.len(), 0.0);
                    rollout.evaluate(&Connect4, &leaf, &actions, &mut priors, &mut absolute);
                    // Reshape into what a network returns: dense priors and relative values.
                    let mut prior_row = [0.0; ACTIONS];
                    actions.iter().zip(&priors).for_each(|(&a, &p)| prior_row[a] = p);
                    let Turn::Player(mover) = Connect4.turn(&leaf) else { unreachable!() };
                    let mut relative = [0.0; 2];
                    to_relative(&absolute, mover, &mut relative);
                    submit_leaf(&Connect4, &mut mcts, &leaf, &prior_row, &relative, &mut scratch);
                }
            }
            assert_eq!(mcts.result().visits, direct.visits, "seed {seed}");
        }
    }

    #[test]
    fn arena_passes_values_to_the_right_seat() {
        let network = Config { simulations: 300, ..Config::default() };
        let opponent = Config { simulations: 20, ..Config::default() };
        let mut arena = Arena::new(Connect4, network, opponent, 10, 0);
        let mut rollout = RandomRollout::new(9);
        let (mut obs, mut masks) = (vec![0.0; 10 * OBS], vec![false; 10 * ACTIONS]);
        let mut leaves = Vec::new();

        while !arena.is_finished() {
            let rows = arena.num_pending();
            arena.write_pending(&mut obs, &mut masks);
            leaves.clear();
            leaves.extend(arena.games.iter().filter_map(|g| g.leaf));
            let (mut priors, mut values) = (vec![0.0; rows * ACTIONS], vec![0.0; rows * 2]);
            let (mut actions, mut absolute, mut row_priors) = (Vec::new(), [0.0; 2], Vec::new());
            for (row, leaf) in leaves.iter().enumerate() {
                Connect4.legal_actions(leaf, &mut actions);
                row_priors.resize(actions.len(), 0.0);
                rollout.evaluate(&Connect4, leaf, &actions, &mut row_priors, &mut absolute);
                actions.iter().zip(&row_priors).for_each(|(&a, &p)| priors[row * ACTIONS + a] = p);
                let Turn::Player(mover) = Connect4.turn(leaf) else { unreachable!() };
                to_relative(&absolute, mover, &mut values[row * 2..][..2]);
            }
            arena.submit(&priors, &values);
        }
        let wins = arena.network_returns().iter().filter(|&&r| r == 1.0).count();
        assert!(wins >= 8, "network side won only {wins} of 10");
    }
}
