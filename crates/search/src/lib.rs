pub mod selfplay;

use game::rng::Rng;
use game::{Action, Game, Turn};
use std::ops::Range;

pub trait Evaluator<G: Game> {
    fn evaluate(
        &mut self,
        game: &G,
        state: &G::State,
        actions: &[Action],
        priors: &mut [f32],
        values: &mut [f32],
    );
}

pub struct RandomRollout {
    rng: Rng,
    actions: Vec<Action>,
}

impl RandomRollout {
    pub fn new(seed: u64) -> Self {
        Self { rng: Rng::new(seed), actions: Vec::new() }
    }
}

impl<G: Game> Evaluator<G> for RandomRollout {
    fn evaluate(
        &mut self,
        game: &G,
        state: &G::State,
        actions: &[Action],
        priors: &mut [f32],
        values: &mut [f32],
    ) {
        priors.fill(1.0 / actions.len() as f32);
        let mut state = *state;
        loop {
            let step = match game.turn(&state) {
                Turn::Terminal => break,
                Turn::Player(_) => {
                    game.legal_actions(&state, &mut self.actions);
                    self.actions[self.rng.below(self.actions.len())]
                }
            };
            game.apply(&mut state, step);
        }
        game.returns(&state, values);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub simulations: u32,
    pub c_puct: f32,
    pub root_noise: Option<RootNoise>,
}

impl Default for Config {
    fn default() -> Self {
        Self { simulations: 800, c_puct: 1.5, root_noise: None }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RootNoise {
    pub alpha: f64,
    pub fraction: f32,
}

pub struct SearchResult {
    pub visits: Vec<(Action, u32)>,
    pub best: Action,
}

struct Node<S> {
    state: Option<S>,
    action: Action,
    prior: f32,
    mover: Option<usize>,
    visits: u32,
    value_sum: f32,
    children: Range<usize>,
    expanded: bool,
}

pub struct Mcts<G: Game> {
    config: Config,
    rng: Rng,
    nodes: Vec<Node<G::State>>,
    pending: Option<usize>,
    path: Vec<usize>,
    actions: Vec<Action>,
    priors: Vec<f32>,
    values: Vec<f32>,
    noise: Vec<f32>,
    eval_priors: Vec<f32>,
    eval_values: Vec<f32>,
}

impl<G: Game> Mcts<G> {
    pub fn new(config: Config, seed: u64) -> Self {
        Self {
            config,
            rng: Rng::new(seed),
            nodes: Vec::new(),
            pending: None,
            path: Vec::new(),
            actions: Vec::new(),
            priors: Vec::new(),
            values: vec![0.0; G::NUM_PLAYERS],
            noise: Vec::new(),
            eval_priors: Vec::new(),
            eval_values: vec![0.0; G::NUM_PLAYERS],
        }
    }

    pub fn search(
        &mut self,
        game: &G,
        root: &G::State,
        evaluator: &mut impl Evaluator<G>,
    ) -> SearchResult {
        self.start(root);
        while !self.is_done() {
            if let Some(leaf) = self.select(game) {
                game.legal_actions(&leaf, &mut self.actions);
                let (mut priors, mut values) =
                    (std::mem::take(&mut self.eval_priors), std::mem::take(&mut self.eval_values));
                priors.resize(self.actions.len(), 0.0);
                values.resize(G::NUM_PLAYERS, 0.0);
                evaluator.evaluate(game, &leaf, &self.actions, &mut priors, &mut values);
                self.expand_and_backup(game, &priors, &values);
                (self.eval_priors, self.eval_values) = (priors, values);
            }
        }
        self.result()
    }

    pub fn start(&mut self, root: &G::State) {
        self.nodes.clear();
        self.pending = None;
        self.nodes.push(Node {
            state: Some(*root),
            action: 0,
            prior: 1.0,
            mover: None,
            visits: 0,
            value_sum: 0.0,
            children: 0..0,
            expanded: false,
        });
    }

    pub fn is_done(&self) -> bool {
        self.nodes[0].visits >= self.config.simulations
    }

    pub fn select(&mut self, game: &G) -> Option<G::State> {
        assert!(self.pending.is_none(), "the previous leaf has not been evaluated");
        self.path.clear();
        let mut node = 0;
        loop {
            self.path.push(node);
            let state = self.nodes[node].state.expect("state is set before a node is visited");

            if game.turn(&state) == Turn::Terminal {
                game.returns(&state, &mut self.values);
                self.backup();
                return None;
            }
            if !self.nodes[node].expanded {
                self.pending = Some(node);
                return Some(state);
            }

            let child = self.puct_child(node);
            if self.nodes[child].state.is_none() {
                let mut next = state;
                game.apply(&mut next, self.nodes[child].action);
                self.nodes[child].state = Some(next);
            }
            node = child;
        }
    }

    pub fn expand_and_backup(&mut self, game: &G, priors: &[f32], values: &[f32]) {
        let node = self.pending.take().expect("no leaf is awaiting evaluation");
        let state = self.nodes[node].state.expect("a pending leaf has a state");
        let Turn::Player(player) = game.turn(&state) else {
            unreachable!("only player nodes wait for evaluation")
        };
        game.legal_actions(&state, &mut self.actions);
        assert_eq!(priors.len(), self.actions.len(), "one prior per legal action");
        assert_eq!(values.len(), G::NUM_PLAYERS, "one value per player");

        self.priors.clear();
        self.priors.extend_from_slice(priors);
        if let (0, Some(noise)) = (node, self.config.root_noise) {
            self.noise.resize(self.priors.len(), 0.0);
            self.rng.dirichlet(noise.alpha, &mut self.noise);
            for (p, n) in self.priors.iter_mut().zip(&self.noise) {
                *p = (1.0 - noise.fraction) * *p + noise.fraction * n;
            }
        }
        let children = self.actions.iter().copied().zip(self.priors.iter().copied());
        expand(&mut self.nodes, node, player, children);

        self.values.copy_from_slice(values);
        self.backup();
    }

    pub fn result(&self) -> SearchResult {
        let children = &self.nodes[self.nodes[0].children.clone()];
        let visits: Vec<(Action, u32)> = children.iter().map(|c| (c.action, c.visits)).collect();
        let best = visits.iter().fold(visits[0], |best, &v| if v.1 > best.1 { v } else { best }).0;
        SearchResult { visits, best }
    }

    fn backup(&mut self) {
        for &n in &self.path {
            let node = &mut self.nodes[n];
            node.visits += 1;
            if let Some(mover) = node.mover {
                node.value_sum += self.values[mover];
            }
        }
    }

    fn puct_child(&self, parent: usize) -> usize {
        let sqrt_parent = (self.nodes[parent].visits.max(1) as f32).sqrt();
        let score = |c: usize| {
            let child = &self.nodes[c];
            let q = if child.visits > 0 { child.value_sum / child.visits as f32 } else { 0.0 };
            q + self.config.c_puct * child.prior * sqrt_parent / (1.0 + child.visits as f32)
        };
        self.nodes[parent]
            .children
            .clone()
            .fold(None, |best: Option<(usize, f32)>, c| {
                let s = score(c);
                match best {
                    Some((_, b)) if b >= s => best,
                    _ => Some((c, s)),
                }
            })
            .expect("an expanded player node has children")
            .0
    }
}

fn expand<S>(
    nodes: &mut Vec<Node<S>>,
    parent: usize,
    mover: usize,
    children: impl Iterator<Item = (Action, f32)>,
) {
    let start = nodes.len();
    nodes.extend(children.map(|(action, prior)| Node {
        state: None,
        action,
        prior,
        mover: Some(mover),
        visits: 0,
        value_sum: 0.0,
        children: 0..0,
        expanded: false,
    }));
    nodes[parent].children = start..nodes.len();
    nodes[parent].expanded = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    use connect4::Connect4;

    fn play(moves: &[usize]) -> connect4::State {
        let mut state = Connect4.initial_state();
        for &col in moves {
            Connect4.apply(&mut state, col);
        }
        state
    }

    fn best_move(state: &connect4::State, seed: u64) -> Action {
        let mut mcts = Mcts::new(Config::default(), seed);
        mcts.search(&Connect4, state, &mut RandomRollout::new(seed)).best
    }

    #[test]
    fn takes_an_immediate_win() {
        let state = play(&[0, 1, 0, 1, 0, 6]);
        for seed in 0..5 {
            assert_eq!(best_move(&state, seed), 0, "seed {seed}");
        }
    }

    #[test]
    fn blocks_an_immediate_loss() {
        let state = play(&[0, 1, 0, 1, 0]);
        for seed in 0..5 {
            assert_eq!(best_move(&state, seed), 0, "seed {seed}");
        }
    }

    #[test]
    fn child_visits_add_up() {
        let mut mcts = Mcts::new(Config { simulations: 300, ..Config::default() }, 0);
        let result = mcts.search(&Connect4, &Connect4.initial_state(), &mut RandomRollout::new(0));
        let total: u32 = result.visits.iter().map(|&(_, n)| n).sum();
        assert_eq!(total, 300 - 1);
        assert_eq!(result.visits.len(), 7);
    }

    #[test]
    fn same_seeds_give_same_search() {
        let run = || {
            let mut mcts = Mcts::new(Config { simulations: 200, ..Config::default() }, 3);
            mcts.search(&Connect4, &Connect4.initial_state(), &mut RandomRollout::new(3)).visits
        };
        assert_eq!(run(), run());
    }
}
