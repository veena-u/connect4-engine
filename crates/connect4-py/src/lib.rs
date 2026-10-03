use connect4::{Connect4, State, HEIGHT, WIDTH};
use game::{Game, Turn};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use search::selfplay::{Arena, SelfPlay, SelfPlayConfig};
use search::{Config, Mcts, RootNoise};

const OBS: usize = <Connect4 as Game>::OBSERVATION_SIZE;
const ACTIONS: usize = <Connect4 as Game>::NUM_ACTIONS;
const PLAYERS: usize = <Connect4 as Game>::NUM_PLAYERS;

fn array2<T: numpy::Element>(py: Python<'_>, data: Vec<T>, cols: usize) -> Bound<'_, PyArray2<T>> {
    let rows = data.len() / cols;
    Array2::from_shape_vec((rows, cols), data).expect("data fills whole rows").into_pyarray_bound(py)
}

type Pending<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>);

fn pending_arrays<'py>(
    py: Python<'py>,
    rows: usize,
    write: impl FnOnce(&mut [f32], &mut [bool]),
) -> Pending<'py> {
    let (mut observations, mut masks) = (vec![0.0; rows * OBS], vec![false; rows * ACTIONS]);
    write(&mut observations, &mut masks);
    (array2(py, observations, OBS), array2(py, masks, ACTIONS))
}

fn batch<'a>(
    rows: usize,
    priors: &'a PyReadonlyArray2<f32>,
    values: &'a PyReadonlyArray2<f32>,
) -> PyResult<(&'a [f32], &'a [f32])> {
    if priors.shape() != [rows, ACTIONS] || values.shape() != [rows, PLAYERS] {
        return Err(PyValueError::new_err(format!(
            "expected priors of shape ({rows}, {ACTIONS}) and values of shape ({rows}, {PLAYERS}), \
             got {:?} and {:?}",
            priors.shape(),
            values.shape()
        )));
    }
    Ok((priors.as_slice()?, values.as_slice()?))
}

#[pyclass]
struct SelfPlayDriver {
    inner: SelfPlay<Connect4>,
}

#[pymethods]
impl SelfPlayDriver {
    #[new]
    #[pyo3(signature = (
        parallel_games, simulations, c_puct = 1.5, dirichlet_alpha = 1.0,
        noise_fraction = 0.25, temperature_moves = 10, seed = 0
    ))]
    fn new(
        parallel_games: usize,
        simulations: u32,
        c_puct: f32,
        dirichlet_alpha: f64,
        noise_fraction: f32,
        temperature_moves: u32,
        seed: u64,
    ) -> Self {
        let noise = RootNoise { alpha: dirichlet_alpha, fraction: noise_fraction };
        let search = Config { simulations, c_puct, root_noise: Some(noise) };
        let config = SelfPlayConfig { search, temperature_moves };
        Self { inner: SelfPlay::new(Connect4, config, parallel_games, seed) }
    }

    fn pending<'py>(&mut self, py: Python<'py>) -> Pending<'py> {
        let rows = self.inner.num_pending();
        pending_arrays(py, rows, |obs, masks| self.inner.write_pending(obs, masks))
    }

    fn submit(&mut self, priors: PyReadonlyArray2<f32>, values: PyReadonlyArray2<f32>) -> PyResult<()> {
        let (priors, values) = batch(self.inner.num_pending(), &priors, &values)?;
        self.inner.submit(priors, values);
        Ok(())
    }

    fn take_examples<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let examples = self.inner.take_examples();
        let dict = PyDict::new_bound(py);
        dict.set_item("observations", array2(py, examples.observations, OBS))?;
        dict.set_item("policies", array2(py, examples.policies, ACTIONS))?;
        dict.set_item("values", array2(py, examples.values, PLAYERS))?;
        dict.set_item("game_lengths", PyArray1::from_vec_bound(py, examples.game_lengths))?;
        dict.set_item("returns", array2(py, examples.returns, PLAYERS))?;
        Ok(dict)
    }
}

#[pyclass]
struct ArenaDriver {
    inner: Arena<Connect4>,
}

#[pymethods]
impl ArenaDriver {
    #[new]
    #[pyo3(signature = (games, network_simulations, opponent_simulations, c_puct = 1.5, seed = 0))]
    fn new(games: usize, network_simulations: u32, opponent_simulations: u32, c_puct: f32, seed: u64) -> Self {
        let network = Config { simulations: network_simulations, c_puct, root_noise: None };
        let opponent = Config { simulations: opponent_simulations, c_puct, root_noise: None };
        Self { inner: Arena::new(Connect4, network, opponent, games, seed) }
    }

    fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    fn pending<'py>(&mut self, py: Python<'py>) -> Pending<'py> {
        let rows = self.inner.num_pending();
        pending_arrays(py, rows, |obs, masks| self.inner.write_pending(obs, masks))
    }

    fn submit(&mut self, priors: PyReadonlyArray2<f32>, values: PyReadonlyArray2<f32>) -> PyResult<()> {
        let (priors, values) = batch(self.inner.num_pending(), &priors, &values)?;
        self.inner.submit(priors, values);
        Ok(())
    }

    fn network_returns(&self) -> Vec<f32> {
        self.inner.network_returns()
    }
}

#[pyclass]
struct PlayDriver {
    state: State,
    mcts: Mcts<Connect4>,
    leaf: Option<State>,
}

impl PlayDriver {
    fn advance(&mut self) {
        while !self.mcts.is_done() {
            if let Some(leaf) = self.mcts.select(&Connect4) {
                self.leaf = Some(leaf);
                return;
            }
        }
    }
}

#[pymethods]
impl PlayDriver {
    #[new]
    #[pyo3(signature = (simulations, c_puct = 1.5, seed = 0))]
    fn new(simulations: u32, c_puct: f32, seed: u64) -> Self {
        let config = Config { simulations, c_puct, root_noise: None };
        let mut mcts = Mcts::new(config, seed);
        let state = Connect4.initial_state();
        mcts.start(&state);
        let mut driver = Self { state, mcts, leaf: None };
        driver.advance();
        driver
    }

    fn turn(&self) -> i64 {
        match Connect4.turn(&self.state) {
            Turn::Player(player) => player as i64,
            Turn::Terminal => -1,
        }
    }

    fn winner(&self) -> Option<usize> {
        let mut returns = vec![0.0; PLAYERS];
        Connect4.returns(&self.state, &mut returns);
        returns.iter().position(|&r| r == 1.0)
    }

    fn legal_moves(&self) -> Vec<usize> {
        let mut actions = Vec::new();
        Connect4.legal_actions(&self.state, &mut actions);
        actions
    }

    fn board(&self) -> String {
        format!("{}", self.state)
    }

    fn is_thinking_done(&self) -> bool {
        self.mcts.is_done()
    }

    fn pending<'py>(&mut self, py: Python<'py>) -> Pending<'py> {
        let rows = usize::from(self.leaf.is_some());
        pending_arrays(py, rows, |observations, masks| {
            let Some(leaf) = &self.leaf else { return };
            let Turn::Player(mover) = Connect4.turn(leaf) else { unreachable!("leaves are player turns") };
            Connect4.observe(leaf, mover, observations);
            let mut actions = Vec::new();
            Connect4.legal_actions(leaf, &mut actions);
            actions.iter().for_each(|&action| masks[action] = true);
        })
    }

    fn submit(&mut self, priors: PyReadonlyArray2<f32>, values: PyReadonlyArray2<f32>) -> PyResult<()> {
        let rows = usize::from(self.leaf.is_some());
        let (prior_row, value_row) = batch(rows, &priors, &values)?;
        let Some(leaf) = self.leaf.take() else { return Ok(()) };

        let mut actions = Vec::new();
        Connect4.legal_actions(&leaf, &mut actions);
        let mut sparse: Vec<f32> = actions.iter().map(|&action| prior_row[action]).collect();
        let total: f32 = sparse.iter().sum();
        let fallback = 1.0 / sparse.len() as f32;
        sparse.iter_mut().for_each(|p| *p = if total > 0.0 { *p / total } else { fallback });

        let Turn::Player(mover) = Connect4.turn(&leaf) else { unreachable!("leaves are player turns") };
        let mut absolute = vec![0.0; PLAYERS];
        search::selfplay::to_absolute(value_row, mover, &mut absolute);

        self.mcts.expand_and_backup(&Connect4, &sparse, &absolute);
        self.advance();
        Ok(())
    }

    fn apply_network_move(&mut self) -> PyResult<usize> {
        if !self.mcts.is_done() {
            return Err(PyValueError::new_err("the search isn't finished; keep calling pending()/submit()"));
        }
        let best = self.mcts.result().best;
        Connect4.apply(&mut self.state, best);
        self.mcts.start(&self.state);
        self.leaf = None;
        self.advance();
        Ok(best)
    }

    fn apply_human_move(&mut self, col: usize) -> PyResult<()> {
        if !self.legal_moves().contains(&col) {
            return Err(PyValueError::new_err(format!("column {col} is not a legal move")));
        }
        Connect4.apply(&mut self.state, col);
        self.mcts.start(&self.state);
        self.leaf = None;
        self.advance();
        Ok(())
    }
}

#[pymodule]
fn connect4_az(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SelfPlayDriver>()?;
    m.add_class::<ArenaDriver>()?;
    m.add_class::<PlayDriver>()?;
    m.add("NUM_ACTIONS", ACTIONS)?;
    m.add("NUM_PLAYERS", PLAYERS)?;
    m.add("OBSERVATION_SHAPE", (2, HEIGHT, WIDTH))?;
    Ok(())
}
