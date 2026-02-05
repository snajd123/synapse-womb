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

    /// Process one simulation tick.
    ///
    /// # Arguments
    /// * `tick` - Current simulation tick
    /// * `spore_output` - The Spore's output byte
    ///
    /// # Returns
    /// * `Some(correct_bits)` - If a reward is being delivered this tick
    /// * `None` - If no reward is due
    ///
    /// # Pipeline-Aware Judging
    /// The Spore's output at tick T reflects the input from tick T-PIPELINE_LATENCY.
    /// We judge against `input_history[0]` (oldest entry).
    pub fn tick(&mut self, tick: u64, spore_output: u8) -> Option<u8> {
        let mut rng = rand::thread_rng();

        // ====================================================================
        // JUDGE OUTPUT (against pipeline-delayed input)
        // ====================================================================
        let judge_input = self.input_history[0];  // Oldest entry
        let error_bits = (spore_output ^ judge_input).count_ones() as u8;
        let correct_bits = 8 - error_bits;

        // ====================================================================
        // SCHEDULE REWARD
        // ====================================================================
        let deliver_at = tick + self.reward_latency;
        self.pending_rewards.push_back((deliver_at, correct_bits));

        // ====================================================================
        // DELIVER PENDING REWARDS
        // ====================================================================
        let mut reward = None;
        while let Some(&(t, bits)) = self.pending_rewards.front() {
            if t <= tick {
                self.pending_rewards.pop_front();
                reward = Some(bits);  // Take most recent if multiple
            } else {
                break;
            }
        }

        // ====================================================================
        // UPDATE INPUT HISTORY (sliding window)
        // ====================================================================
        self.input_history.pop_front();
        self.input_history.push_back(self.current_input);

        // ====================================================================
        // ADVANCE INPUT PATTERN
        // ====================================================================
        self.ticks_on_current += 1;
        if self.ticks_on_current >= self.input_hold_ticks {
            self.ticks_on_current = 0;
            self.current_input = rng.gen::<u8>();
        }

        reward
    }

    // ========================================================================
    // TEST HELPERS
    // ========================================================================

    /// Set input history for testing (front = oldest, back = newest).
    pub fn set_input_history_for_test(&mut self, history: &[u8]) {
        self.input_history.clear();
        for &h in history {
            self.input_history.push_back(h);
        }
    }

    /// Get ticks on current input for testing.
    pub fn ticks_on_current_for_test(&self) -> u64 {
        self.ticks_on_current
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new(DEFAULT_REWARD_LATENCY as u64)
    }
}
