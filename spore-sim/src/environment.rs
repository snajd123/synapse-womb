//! The Environment: Teacher that provides inputs and rewards.
//!
//! The Environment:
//! - Generates random input patterns
//! - Holds inputs for a configurable number of ticks
//! - Judges Spore output against the correct (pipeline-delayed) input
//! - Schedules and delivers rewards with configurable latency

use std::collections::VecDeque;
use rand::Rng;
use crate::constants::*;

/// The training environment for the Spore.
#[derive(Debug, Clone)]
pub struct Environment {
    /// Reward delivery latency in ticks
    pub reward_latency: u64,

    /// Pending rewards: (tick_to_deliver, correct_bits)
    pending_rewards: VecDeque<(u64, u8)>,

    /// Input history for pipeline-aware judging
    /// Front = oldest (what spore is responding to)
    /// Back = newest (current input)
    input_history: VecDeque<u8>,

    /// Current input pattern
    current_input: u8,

    /// How long to hold each input pattern
    pub input_hold_ticks: u64,

    /// Ticks spent on current input
    ticks_on_current: u64,
}

impl Environment {
    /// Create a new Environment with default settings.
    pub fn new(reward_latency: u64) -> Self {
        Self::with_params(reward_latency, DEFAULT_INPUT_HOLD_TICKS as u64)
    }

    /// Create a new Environment with custom settings.
    pub fn with_params(reward_latency: u64, input_hold_ticks: u64) -> Self {
        let mut rng = rand::thread_rng();
        let current_input = rng.gen::<u8>();

        // Pre-fill input history for pipeline-aware judging
        // Need exactly PIPELINE_LATENCY entries (NOT +1!)
        // At tick T, history[0] = input from tick T-PIPELINE_LATENCY
        // After tick T: pop history[0], push input_T
        // History length stays constant at PIPELINE_LATENCY
        let mut input_history = VecDeque::with_capacity(PIPELINE_LATENCY);
        for _ in 0..PIPELINE_LATENCY {
            input_history.push_back(current_input);
        }

        Self {
            reward_latency,
            pending_rewards: VecDeque::new(),
            input_history,
            current_input,
            input_hold_ticks,
            ticks_on_current: 0,
        }
    }

    /// Get the current input pattern.
    pub fn get_input(&self) -> u8 {
        self.current_input
    }

    /// Get the length of input history (for testing).
    pub fn input_history_len(&self) -> usize {
        self.input_history.len()
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new(DEFAULT_REWARD_LATENCY as u64)
    }
}
