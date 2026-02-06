//! Genetic hyperparameter tuner for the Spore neural network.
//!
//! Evolves optimal hyperparameters using a genetic algorithm:
//! - Population of 50 Genomes
//! - 20 generations of evolution
//! - Crossover + mutation + diversity injection
//! - Parallel evaluation with rayon
//! - Final Exam on top candidates (full marathon, 3 runs)

use rand::Rng;
use serde::{Serialize, Deserialize};

/// A genome encoding 6 tunable hyperparameters for the Spore network.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    /// Learning rate for Hebbian weight updates. Range: 0.05 - 0.5
    pub learning_rate: f32,

    /// Eligibility trace decay per tick. Range: 0.85 - 0.999
    /// Exponentially sensitive - small changes have large effects.
    pub trace_decay: f32,

    /// Maximum noise boost at full frustration. Range: 0.01 - 0.05
    pub max_noise_boost: f32,

    /// Ticks between weight decay applications. Range: 50 - 200
    pub weight_decay_interval: u64,

    /// EMA alpha for frustration updates (>50% accuracy). Range: 0.05 - 0.5
    pub frustration_alpha: f32,

    /// Ticks to hold each input pattern. Range: 20 - 200
    pub input_hold_ticks: u64,
}

impl Genome {
    /// Create a random genome with all genes in valid ranges.
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            learning_rate: rng.gen_range(0.05..=0.5),
            trace_decay: rng.gen_range(0.85..=0.999),
            max_noise_boost: rng.gen_range(0.01..=0.05),
            weight_decay_interval: rng.gen_range(50..=200),
            frustration_alpha: rng.gen_range(0.05..=0.5),
            input_hold_ticks: rng.gen_range(20..=200),
        }
    }

    /// Mutate this genome in-place.
    ///
    /// Each gene has a 10% independent chance of mutation.
    /// `magnitude_mult` scales the STEP SIZE (not probability).
    /// Use magnitude_mult=2.0 to escape local minima.
    pub fn mutate(&mut self, magnitude_mult: f32) {
        let mut rng = rand::thread_rng();

        if rng.gen::<f32>() < 0.1 {
            self.learning_rate += rng.gen_range(-0.02..=0.02) * magnitude_mult;
            self.learning_rate = self.learning_rate.clamp(0.05, 0.5);
        }
        if rng.gen::<f32>() < 0.1 {
            // Fine-grained: trace_decay is exponentially sensitive
            self.trace_decay += rng.gen_range(-0.005..=0.005) * magnitude_mult;
            self.trace_decay = self.trace_decay.clamp(0.85, 0.999);
        }
        if rng.gen::<f32>() < 0.1 {
            self.max_noise_boost += rng.gen_range(-0.005..=0.005) * magnitude_mult;
            self.max_noise_boost = self.max_noise_boost.clamp(0.01, 0.05);
        }
        if rng.gen::<f32>() < 0.1 {
            let delta = (rng.gen_range(-10..=10) as f32 * magnitude_mult) as i64;
            self.weight_decay_interval = (self.weight_decay_interval as i64 + delta)
                .clamp(50, 200) as u64;
        }
        if rng.gen::<f32>() < 0.1 {
            self.frustration_alpha += rng.gen_range(-0.02..=0.02) * magnitude_mult;
            self.frustration_alpha = self.frustration_alpha.clamp(0.05, 0.5);
        }
        if rng.gen::<f32>() < 0.1 {
            let delta = (rng.gen_range(-10..=10) as f32 * magnitude_mult) as i64;
            self.input_hold_ticks = (self.input_hold_ticks as i64 + delta)
                .clamp(20, 200) as u64;
        }
    }

    /// Sexual reproduction: combine genes from two parents.
    ///
    /// Each gene is independently selected from parent_a or parent_b
    /// with equal probability (uniform crossover).
    pub fn crossover(parent_a: &Genome, parent_b: &Genome) -> Genome {
        let mut rng = rand::thread_rng();
        Genome {
            learning_rate: if rng.gen() { parent_a.learning_rate } else { parent_b.learning_rate },
            trace_decay: if rng.gen() { parent_a.trace_decay } else { parent_b.trace_decay },
            max_noise_boost: if rng.gen() { parent_a.max_noise_boost } else { parent_b.max_noise_boost },
            weight_decay_interval: if rng.gen() { parent_a.weight_decay_interval } else { parent_b.weight_decay_interval },
            frustration_alpha: if rng.gen() { parent_a.frustration_alpha } else { parent_b.frustration_alpha },
            input_hold_ticks: if rng.gen() { parent_a.input_hold_ticks } else { parent_b.input_hold_ticks },
        }
    }
}
