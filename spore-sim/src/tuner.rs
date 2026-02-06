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

use crate::simulation::Simulation;

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

/// Result of evaluating a single Genome.
#[derive(Clone, Debug)]
pub struct EvalResult {
    /// Fitness score (higher = better)
    pub score: f32,

    /// Mean accuracy since convergence (or final accuracy if never converged)
    pub final_accuracy: f32,

    /// Tick at which stable convergence was achieved (None = never)
    pub convergence_tick: Option<u64>,

    /// Whether the genome achieved stable convergence
    pub stable: bool,
}

/// Fast evaluation for use during evolution loop.
///
/// Runs the genome 3 times (different random sequences) and returns the WORST result.
/// Includes early exit: if stable for 2000+ ticks at 98%+, stop.
pub fn evaluate_fast(genome: &Genome, ticks: u64) -> EvalResult {
    let results: Vec<EvalResult> = (0..3)
        .map(|_| evaluate_single(genome, ticks, true))
        .collect();

    results.into_iter()
        .min_by(|a, b| a.score.total_cmp(&b.score))
        .unwrap()
}

/// Full marathon evaluation for Final Exam.
///
/// Runs the genome 3 times with NO early exit. Must survive all ticks.
/// Returns the WORST result (robustness check).
pub fn evaluate_full(genome: &Genome, ticks: u64) -> EvalResult {
    let results: Vec<EvalResult> = (0..3)
        .map(|_| evaluate_single(genome, ticks, false))
        .collect();

    results.into_iter()
        .min_by(|a, b| a.score.total_cmp(&b.score))
        .unwrap()
}

/// Evaluate a single genome for a given number of ticks.
///
/// IMPORTANT: Uses INSTANTANEOUS per-tick accuracy from sim.step() return value,
/// NOT the EMA (sim.recent_accuracy). The EMA lags by ~30-45 ticks and would make
/// convergence revocation sluggish. Per-tick accuracy gives honest, real-time signal.
///
/// Stability detection (Option B):
/// - Track when accuracy first crosses 90%
/// - If accuracy stays >= 85% for 1000 consecutive ticks -> stable convergence
/// - Dip below 85% -> reset timer AND revoke convergence
///
/// Score:
/// - Stable: mean_accuracy_since_converge * 2000 - (convergence_tick / 100)
/// - Unstable: recent_accuracy * 500
fn evaluate_single(genome: &Genome, ticks: u64, allow_early_exit: bool) -> EvalResult {
    let mut sim = Simulation::with_full_params(
        0,  // reward_latency (fixed for tuner)
        genome.trace_decay,
        genome.input_hold_ticks,
        genome.learning_rate,
        genome.max_noise_boost,
        genome.weight_decay_interval,
        genome.frustration_alpha,
    );

    let mut stability_window_start: Option<u64> = None;
    let mut convergence_tick: Option<u64> = None;
    let mut accuracy_sum_since_converge: f32 = 0.0;
    let mut ticks_since_converge: u64 = 0;
    let mut last_accuracy: f32 = 0.0;

    for tick in 0..ticks {
        // step() returns Some(accuracy) when reward delivered, None otherwise
        // With reward_latency=0, this is always Some
        let tick_accuracy = sim.step();

        // Use per-tick accuracy for stability detection (NOT the EMA)
        let acc = match tick_accuracy {
            Some(a) => { last_accuracy = a; a }
            None => last_accuracy,  // Hold last known accuracy between rewards
        };

        // Stability detection
        if acc >= 0.90 {
            if stability_window_start.is_none() {
                stability_window_start = Some(tick);
            }
            if tick - stability_window_start.unwrap() >= 1000 && convergence_tick.is_none() {
                convergence_tick = Some(stability_window_start.unwrap());
            }
        } else if acc < 0.85 {
            stability_window_start = None;
            convergence_tick = None;  // REVOKE on crash
            accuracy_sum_since_converge = 0.0;
            ticks_since_converge = 0;
        }
        // 0.85 <= acc < 0.90: timer continues, no action

        // Track integral accuracy after convergence
        if convergence_tick.is_some() {
            accuracy_sum_since_converge += acc;
            ticks_since_converge += 1;

            // Early exit (fast eval only): stable for 2000+ ticks at 98%+
            if allow_early_exit && ticks_since_converge >= 2000 {
                let mean_acc = accuracy_sum_since_converge / ticks_since_converge as f32;
                if mean_acc >= 0.98 {
                    return EvalResult {
                        score: mean_acc * 2000.0 - (convergence_tick.unwrap() as f32 / 100.0),
                        final_accuracy: mean_acc,
                        convergence_tick,
                        stable: true,
                    };
                }
            }
        }
    }

    // End of run scoring
    let stable = convergence_tick.is_some() && ticks_since_converge > 0;
    let mean_acc = if ticks_since_converge > 0 {
        accuracy_sum_since_converge / ticks_since_converge as f32
    } else {
        sim.recent_accuracy  // Fall back to EMA only for unstable genomes
    };

    let score = if stable {
        mean_acc * 2000.0 - (convergence_tick.unwrap() as f32 / 100.0)
    } else {
        sim.recent_accuracy * 500.0
    };

    EvalResult {
        score,
        final_accuracy: mean_acc,
        convergence_tick,
        stable,
    }
}
