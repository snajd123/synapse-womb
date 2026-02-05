//! The Simulation: Orchestrates the Spore and Environment.

use crate::spore::Spore;
use crate::environment::Environment;
use crate::constants::*;

/// The main simulation runner.
#[derive(Debug)]
pub struct Simulation {
    /// The neural network being trained
    spore: Spore,

    /// The training environment
    env: Environment,

    /// Current simulation tick
    pub tick: u64,

    /// Rolling average of accuracy
    pub recent_accuracy: f32,

    /// History of accuracy for plotting
    pub accuracy_history: Vec<f32>,
}

impl Simulation {
    /// Create a new Simulation with default parameters.
    pub fn new() -> Self {
        Self::with_params(
            DEFAULT_REWARD_LATENCY as u64,
            DEFAULT_TRACE_DECAY as f32,
            DEFAULT_INPUT_HOLD_TICKS as u64,
        )
    }

    /// Create a new Simulation with custom parameters.
    pub fn with_params(
        reward_latency: u64,
        trace_decay: f32,
        input_hold_ticks: u64,
    ) -> Self {
        let spore = Spore::with_params(
            DEFAULT_LEARNING_RATE as f32,
            trace_decay,
            DEFAULT_BASE_NOISE as f32,
            DEFAULT_MAX_NOISE_BOOST as f32,
        );
        let env = Environment::with_params(reward_latency, input_hold_ticks);

        Self {
            spore,
            env,
            tick: 0,
            recent_accuracy: 0.0,
            accuracy_history: Vec::new(),
        }
    }

    /// Get a reference to the Environment (for testing).
    pub fn env(&self) -> &Environment {
        &self.env
    }

    /// Get a reference to the Spore (for inspection).
    pub fn spore(&self) -> &Spore {
        &self.spore
    }

    /// Execute one simulation step.
    ///
    /// The heartbeat - CRITICAL: tick_end() BEFORE propagate()!
    /// 1. TICK_END FIRST: Advance pipeline, decay traces
    /// 2. SENSE: Get input from environment
    /// 3. PROPAGATE: Signal flows through network
    /// 4. OUTPUT: Read output byte
    /// 5. ENVIRONMENT: Judge output, schedule/deliver reward
    /// 6. REWARD: Inject dopamine if reward delivered
    /// 7. LEARN: Apply Hebbian update
    /// 8. MAINTAIN: Weight decay, normalization, threshold drift
    pub fn step(&mut self) {
        // 1. TICK_END FIRST (advance pipeline, decay traces)
        // This MUST happen before propagate for correct 2-tick latency!
        // - Swaps hidden_next -> hidden, output_next -> output
        // - Decays traces from previous tick
        self.spore.tick_end();

        // 2. SENSE
        let input = self.env.get_input();

        // 3. PROPAGATE (compute new states, set fresh traces)
        self.spore.propagate(input);

        // 4. OUTPUT (reads from buffer swapped in step 1)
        // This output reflects input from 2 ticks ago (PIPELINE_LATENCY = 2)
        let output = self.spore.output_as_byte();

        // 5. ENVIRONMENT STEP
        if let Some(correct_bits) = self.env.tick(self.tick, output) {
            // 6. REWARD
            self.spore.receive_reward(correct_bits);

            // Track accuracy
            let accuracy = correct_bits as f32 / 8.0;
            self.recent_accuracy = 0.95 * self.recent_accuracy + 0.05 * accuracy;
        }

        // 7. LEARN
        self.spore.learn();

        // 8. MAINTAIN
        self.spore.maintain(self.tick);

        self.tick += 1;
    }

    /// Run simulation for a given number of ticks.
    pub fn run(&mut self, max_ticks: u64, log_interval: u64) -> f32 {
        while self.tick < max_ticks {
            self.step();

            // Logging
            if log_interval > 0 && self.tick % log_interval == 0 {
                println!(
                    "Tick {}: accuracy={:.2}% frustration={:.3}",
                    self.tick,
                    self.recent_accuracy * 100.0,
                    self.spore.frustration
                );
                self.accuracy_history.push(self.recent_accuracy);
            }
        }

        self.recent_accuracy
    }

    /// Check if simulation has converged.
    pub fn has_converged(&self, threshold: f32) -> bool {
        self.recent_accuracy >= threshold
    }
}

impl Default for Simulation {
    fn default() -> Self {
        Self::new()
    }
}
