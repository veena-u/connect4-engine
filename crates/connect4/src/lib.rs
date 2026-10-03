use game::{Action, Game, Turn};
use std::fmt;

pub const WIDTH: usize = 7;
pub const HEIGHT: usize = 6;

/// store the board as u64, one bit per cell (ordered by cols)
const STRIDE: usize = HEIGHT + 1;

const fn bottom(col: usize) -> u64 {
    1 << (col * STRIDE)
}

const fn top(col: usize) -> u64 {
    1 << (col * STRIDE + HEIGHT - 1)
}

const fn column(col: usize) -> u64 {
    ((1 << HEIGHT) - 1) << (col * STRIDE)
}

/// win condition
fn has_four(stones: u64) -> bool {
    [1, STRIDE, STRIDE - 1, STRIDE + 1].into_iter().any(|step| {
        let pairs = stones & (stones >> step);
        pairs & (pairs >> (2 * step)) != 0
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct State {
    stones: [u64; 2],
    moves: u8,
    winner: Option<u8>,
}

impl State {
    fn occupied(&self) -> u64 {
        self.stones[0] | self.stones[1]
    }

    fn has_stone(&self, player: usize, col: usize, row: usize) -> bool {
        self.stones[player] >> (col * STRIDE + row) & 1 == 1
    }

    fn to_move(&self) -> usize {
        self.moves as usize % 2
    }

    fn is_terminal(&self) -> bool {
        self.winner.is_some() || self.moves as usize == WIDTH * HEIGHT
    }

    fn is_legal(&self, col: usize) -> bool {
        !self.is_terminal() && col < WIDTH && self.occupied() & top(col) == 0
    }
}

pub struct Connect4;

impl Game for Connect4 {
    type State = State;
    const NUM_PLAYERS: usize = 2;
    const NUM_ACTIONS: usize = WIDTH;
    const OBSERVATION_SIZE: usize = 2 * WIDTH * HEIGHT;

    fn initial_state(&self) -> State {
        State::default()
    }

    fn turn(&self, state: &State) -> Turn {
        if state.is_terminal() {
            Turn::Terminal
        } else {
            Turn::Player(state.to_move())
        }
    }

    fn legal_actions(&self, state: &State, out: &mut Vec<Action>) {
        out.clear();
        out.extend((0..WIDTH).filter(|&col| state.is_legal(col)));
    }

    fn apply(&self, state: &mut State, col: Action) {
        debug_assert!(state.is_legal(col), "illegal move: column {col}");
        let player = state.to_move();
        let cell = (state.occupied() + bottom(col)) & column(col);
        state.stones[player] |= cell;
        state.moves += 1;
        if has_four(state.stones[player]) {
            state.winner = Some(player as u8);
        }
    }

    fn returns(&self, state: &State, out: &mut [f32]) {
        out.fill(0.0);
        if let Some(winner) = state.winner {
            out[winner as usize] = 1.0;
            out[1 - winner as usize] = -1.0;
        }
    }

    fn observe(&self, state: &State, player: usize, out: &mut [f32]) {
        for (plane, owner) in [player, 1 - player].into_iter().enumerate() {
            for col in 0..WIDTH {
                for row in 0..HEIGHT {
                    let set = state.has_stone(owner, col, row);
                    out[plane * WIDTH * HEIGHT + row * WIDTH + col] = if set { 1.0 } else { 0.0 };
                }
            }
        }
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for row in (0..HEIGHT).rev() {
            for col in 0..WIDTH {
                let cell = match (self.has_stone(0, col, row), self.has_stone(1, col, row)) {
                    (true, _) => 'x',
                    (_, true) => 'o',
                    _ => '.',
                };
                write!(f, "{cell}")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perft(state: &State, depth: usize) -> u64 {
        if depth == 0 {
            return 1;
        }
        let mut actions = Vec::new();
        Connect4.legal_actions(state, &mut actions);
        actions
            .into_iter()
            .map(|col| {
                let mut next = *state;
                Connect4.apply(&mut next, col);
                perft(&next, depth - 1)
            })
            .sum()
    }

    fn cell(col: usize, row: usize) -> u64 {
        1 << (col * STRIDE + row)
    }

    #[test]
    fn detects_four_in_every_direction() {
        let vertical = (0..4).map(|row| cell(3, row)).sum();
        let horizontal = (0..4).map(|col| cell(col, 0)).sum();
        let rising = (0..4).map(|i| cell(i, i)).sum();
        let falling = (0..4).map(|i| cell(i, 3 - i)).sum();
        for stones in [vertical, horizontal, rising, falling] {
            assert!(has_four(stones), "missed a four");
        }
    }

    #[test]
    fn lines_do_not_wrap_between_columns() {
        let stones = cell(0, 3) | cell(0, 4) | cell(0, 5) | cell(1, 0);
        assert!(!has_four(stones));
    }

    #[test]
    fn stones_stack_and_fill_a_column() {
        let mut state = Connect4.initial_state();
        for row in 0..HEIGHT {
            assert!(state.is_legal(2));
            let player = state.to_move();
            Connect4.apply(&mut state, 2);
            assert!(state.has_stone(player, 2, row));
        }
        assert!(!state.is_legal(2));
    }

    #[test]
    fn perft_counts() {
        let expected = [7, 49, 343, 2_401, 16_807, 117_649, 823_536];
        for (depth, &count) in (1..).zip(&expected) {
            assert_eq!(perft(&Connect4.initial_state(), depth), count, "depth {depth}");
        }
    }
}
