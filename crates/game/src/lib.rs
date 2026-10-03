pub mod rng;

pub type Action = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Turn {
    Player(usize),
    Terminal,
}

pub trait Game {
    type State: Copy;

    const NUM_PLAYERS: usize;
    const NUM_ACTIONS: usize;
    const OBSERVATION_SIZE: usize;

    fn initial_state(&self) -> Self::State;

    fn turn(&self, state: &Self::State) -> Turn;

    fn apply(&self, state: &mut Self::State, action: Action);

    /// These 3 fns do not return a result to avoid allocations on hot paths
    /// They modify a mutable buffer in place
    fn legal_actions(&self, state: &Self::State, out: &mut Vec<Action>);

    fn returns(&self, state: &Self::State, out: &mut [f32]);

    fn observe(&self, state: &Self::State, player: usize, out: &mut [f32]);
}