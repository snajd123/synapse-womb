//! The Environment: Generates input patterns for the Swarm.
//!
//! Simplified for V2: no pipeline delay, no reward scheduling.
//! Just generates random 8-bit patterns and holds them for input_hold_ticks.

use rand::Rng;
use crate::constants::*;

/// The training environment.
#[derive(Debug, Clone)]
pub struct Environment {
    /// Current input pattern.
    current_input: u8,

    /// How long to hold each input pattern.
    pub input_hold_ticks: u64,

    /// Ticks spent on current input.
    ticks_on_current: u64,
}

impl Environment {
    /// Create a new Environment.
    pub fn new(input_hold_ticks: u64) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            current_input: rng.gen::<u8>(),
            input_hold_ticks,
            ticks_on_current: 0,
        }
    }

    /// Get the current input pattern as a u8.
    pub fn get_input(&self) -> u8 {
        self.current_input
    }

    /// Get the current input as an f32 array (one element per bit).
    /// Bit 0 = index 0, Bit 7 = index 7.
    pub fn get_input_f32(&self) -> [f32; INPUT_SIZE] {
        let mut inputs = [0.0_f32; INPUT_SIZE];
        for i in 0..INPUT_SIZE {
            inputs[i] = ((self.current_input >> i) & 1) as f32;
        }
        inputs
    }

    /// Get the target bits for the mirror task (target = input).
    pub fn get_target_bits(&self) -> Vec<bool> {
        (0..INPUT_SIZE)
            .map(|i| (self.current_input >> i) & 1 == 1)
            .collect()
    }

    /// Advance the environment: increment hold counter, switch pattern if needed.
    pub fn advance(&mut self) {
        self.ticks_on_current += 1;
        if self.ticks_on_current >= self.input_hold_ticks {
            self.ticks_on_current = 0;
            self.current_input = rand::thread_rng().gen::<u8>();
        }
    }

    /// Get ticks on current input (for testing).
    pub fn ticks_on_current_for_test(&self) -> u64 {
        self.ticks_on_current
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new(DEFAULT_INPUT_HOLD_TICKS)
    }
}
