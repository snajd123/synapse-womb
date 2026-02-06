//! The Simulation: Orchestrates the Swarm and Environment.

use crate::swarm::Swarm;
use crate::environment::Environment;
use crate::constants::*;

/// The main simulation runner.
#[derive(Debug)]
pub struct Simulation {
    /// The colony of specialist Spores
    swarm: Swarm,

    /// The training environment
    env: Environment,

    /// Current simulation tick
    pub tick: u64,

    /// Rolling average of accuracy (Swarm mean)
    pub recent_accuracy: f32,

    /// History of accuracy for plotting
    pub accuracy_history: Vec<f32>,
}

impl Simulation {
    /// Create a new Simulation with default parameters.
    pub fn new() -> Self {
        Self::with_full_params(
            DEFAULT_SWARM_SIZE,
            DEFAULT_INPUT_HOLD_TICKS,
            DEFAULT_LEARNING_RATE,
            DEFAULT_TRACE_DECAY,
            DEFAULT_MAX_NOISE_BOOST,
            DEFAULT_WEIGHT_DECAY_INTERVAL,
            DEFAULT_FRUSTRATION_ALPHA,
            DEFAULT_CORTISOL_STRENGTH,
        )
    }

    /// Create a new Simulation with all tunable parameters.
    ///
    /// Used by the genetic tuner and --params CLI flag.
    pub fn with_full_params(
        n_outputs: usize,
        input_hold_ticks: u64,
        learning_rate: f32,
        trace_decay: f32,
        max_noise_boost: f32,
        weight_decay_interval: u64,
        frustration_alpha: f32,
        cortisol_strength: f32,
    ) -> Self {
        let swarm = Swarm::new(
            n_outputs,
            learning_rate,
            trace_decay,
            DEFAULT_BASE_NOISE,
            max_noise_boost,
            frustration_alpha,
            weight_decay_interval,
            cortisol_strength,
        );
        let env = Environment::new(input_hold_ticks);

        Self {
            swarm,
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

    /// Get a reference to the Swarm (for inspection).
    pub fn swarm(&self) -> &Swarm {
        &self.swarm
    }

    /// Execute one simulation step.
    ///
    /// Returns the per-tick mean accuracy across all Spores.
    pub fn step(&mut self) -> Option<f32> {
        // 1. Get input
        let inputs = self.env.get_input_f32();
        let targets = self.env.get_target_bits();

        // 2. Swarm tick: fire, compare, reward, learn, maintain
        let accuracy = self.swarm.tick(&inputs, &targets, self.tick);

        // 3. Update EMA
        self.recent_accuracy = 0.95 * self.recent_accuracy + 0.05 * accuracy;

        // 4. Advance environment
        self.env.advance();

        self.tick += 1;

        Some(accuracy)
    }

    /// Run simulation for a given number of ticks.
    pub fn run(&mut self, max_ticks: u64, log_interval: u64) -> f32 {
        while self.tick < max_ticks {
            let _ = self.step();

            // Logging
            if log_interval > 0 && self.tick % log_interval == 0 {
                // Report per-Spore accuracies
                let spore_accs: Vec<String> = self.swarm.spores.iter()
                    .map(|s| format!("{:.0}", s.recent_accuracy * 100.0))
                    .collect();
                eprintln!(
                    "Tick {}: swarm_acc={:.2}% bits=[{}]",
                    self.tick,
                    self.recent_accuracy * 100.0,
                    spore_accs.join(","),
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
