//! Genetic hyperparameter tuner for the Swarm V2 neural network.
//!
//! Evolves optimal hyperparameters using a genetic algorithm:
//! - Population of 50 Genomes (7 genes including cortisol_strength)
//! - 20 generations of evolution
//! - Crossover + mutation + diversity injection
//! - Parallel evaluation with rayon
//! - Final Exam on top candidates (full marathon, 3 runs)

use rand::Rng;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

use crate::simulation::Simulation;
use crate::constants::*;

/// A genome encoding 7 tunable hyperparameters for the Swarm.
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

    /// Cortisol strength (anti-Hebbian punishment ratio). Range: 0.1 - 0.8
    /// Default 0.3 (asymmetric — gentle carving).
    pub cortisol_strength: f32,
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
            cortisol_strength: rng.gen_range(0.1..=0.8),
        }
    }

    /// Mutate this genome in-place.
    ///
    /// Each gene has a 10% independent chance of mutation.
    /// `magnitude_mult` scales the STEP SIZE (not probability).
    pub fn mutate(&mut self, magnitude_mult: f32) {
        let mut rng = rand::thread_rng();

        if rng.gen::<f32>() < 0.1 {
            self.learning_rate += rng.gen_range(-0.02..=0.02) * magnitude_mult;
            self.learning_rate = self.learning_rate.clamp(0.05, 0.5);
        }
        if rng.gen::<f32>() < 0.1 {
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
        if rng.gen::<f32>() < 0.1 {
            self.cortisol_strength += rng.gen_range(-0.05..=0.05) * magnitude_mult;
            self.cortisol_strength = self.cortisol_strength.clamp(0.1, 0.8);
        }
    }

    /// Sexual reproduction: combine genes from two parents.
    pub fn crossover(parent_a: &Genome, parent_b: &Genome) -> Genome {
        let mut rng = rand::thread_rng();
        Genome {
            learning_rate: if rng.gen() { parent_a.learning_rate } else { parent_b.learning_rate },
            trace_decay: if rng.gen() { parent_a.trace_decay } else { parent_b.trace_decay },
            max_noise_boost: if rng.gen() { parent_a.max_noise_boost } else { parent_b.max_noise_boost },
            weight_decay_interval: if rng.gen() { parent_a.weight_decay_interval } else { parent_b.weight_decay_interval },
            frustration_alpha: if rng.gen() { parent_a.frustration_alpha } else { parent_b.frustration_alpha },
            input_hold_ticks: if rng.gen() { parent_a.input_hold_ticks } else { parent_b.input_hold_ticks },
            cortisol_strength: if rng.gen() { parent_a.cortisol_strength } else { parent_b.cortisol_strength },
        }
    }
}

/// Result of evaluating a single Genome.
#[derive(Clone, Debug)]
pub struct EvalResult {
    pub score: f32,
    pub final_accuracy: f32,
    pub convergence_tick: Option<u64>,
    pub stable: bool,
}

/// Configuration for the genetic tuner.
pub struct TunerConfig {
    pub population_size: usize,
    pub generations: usize,
    pub elite_count: usize,
    pub ticks_per_eval: u64,
    pub finalist_count: usize,
}

impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            generations: 20,
            elite_count: 10,
            ticks_per_eval: 20_000,
            finalist_count: 15,
        }
    }
}

/// Run the genetic hyperparameter tuner.
///
/// Returns the best genome and its evaluation result.
pub fn tune(config: &TunerConfig) -> (Genome, EvalResult) {
    let mut rng = rand::thread_rng();

    let mut population: Vec<Genome> = (0..config.population_size)
        .map(|_| Genome::random())
        .collect();

    let mut elite_cache: Vec<(Genome, EvalResult)> = Vec::new();
    let mut best_score: f32 = f32::NEG_INFINITY;
    let mut gens_without_improvement: usize = 0;

    for gen in 0..config.generations {
        let new_scored: Vec<(Genome, EvalResult)> = population
            .par_iter()
            .map(|g| (g.clone(), evaluate_fast(g, config.ticks_per_eval)))
            .collect();

        let mut all_scored: Vec<(Genome, EvalResult)> = elite_cache.clone();
        all_scored.extend(new_scored);
        all_scored.sort_by(|a, b| b.1.score.total_cmp(&a.1.score));
        all_scored.truncate(config.population_size + config.elite_count);

        let gen_best = all_scored[0].1.score;

        if gen_best > best_score {
            best_score = gen_best;
            gens_without_improvement = 0;
        } else {
            gens_without_improvement += 1;
        }
        let magnitude_mult = if gens_without_improvement >= 3 { 2.0 } else { 1.0 };

        let stable_count = all_scored.iter().filter(|s| s.1.stable).count();
        let eval_count = all_scored.len().min(config.population_size);
        let avg_score: f32 = all_scored.iter().take(eval_count)
            .map(|s| s.1.score).sum::<f32>() / eval_count as f32;
        eprintln!("Gen {:2}: best={:.1} avg={:.1} stable={}/{}{}",
            gen, gen_best, avg_score, stable_count, eval_count,
            if magnitude_mult > 1.0 { " [BOOST]" } else { "" });

        elite_cache = all_scored[0..config.elite_count.min(all_scored.len())].to_vec();

        let elites: Vec<&Genome> = elite_cache.iter().map(|(g, _)| g).collect();
        let diversity_count = config.population_size / 5;
        let children_count = config.population_size - diversity_count;

        population = Vec::with_capacity(config.population_size);

        for _ in 0..children_count {
            let parent_a = &elites[rng.gen_range(0..elites.len())];
            let parent_b = &elites[rng.gen_range(0..elites.len())];
            let mut child = Genome::crossover(parent_a, parent_b);
            child.mutate(magnitude_mult);
            population.push(child);
        }

        for _ in 0..diversity_count {
            population.push(Genome::random());
        }
    }

    // FINAL EXAM
    let finalist_count = config.finalist_count.min(elite_cache.len());
    eprintln!("\n=== FINAL EXAM (Top {}, Full Marathon, 3 Runs) ===", finalist_count);

    let finalist_genomes: Vec<Genome> = elite_cache[0..finalist_count]
        .iter()
        .map(|(g, _)| g.clone())
        .collect();

    let finalists: Vec<(Genome, EvalResult)> = finalist_genomes
        .par_iter()
        .map(|g| (g.clone(), evaluate_full(g, config.ticks_per_eval)))
        .collect();

    for (i, (genome, result)) in finalists.iter().enumerate() {
        eprintln!("  #{:2}: score={:.1} acc={:.2}% stable={} conv@{:?} lr={:.3} td={:.4} cs={:.2}",
            i + 1, result.score, result.final_accuracy * 100.0,
            result.stable, result.convergence_tick,
            genome.learning_rate, genome.trace_decay, genome.cortisol_strength);
    }

    finalists
        .into_iter()
        .max_by(|a, b| a.1.score.total_cmp(&b.1.score))
        .unwrap()
}

/// Fast evaluation: 3 runs, worst score, with early exit.
pub fn evaluate_fast(genome: &Genome, ticks: u64) -> EvalResult {
    let results: Vec<EvalResult> = (0..3)
        .map(|_| evaluate_single(genome, ticks, true))
        .collect();

    results.into_iter()
        .min_by(|a, b| a.score.total_cmp(&b.score))
        .unwrap()
}

/// Full marathon evaluation: 3 runs, worst score, NO early exit.
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
/// Uses per-tick accuracy from sim.step() for stability detection.
/// Stability: 1000 consecutive ticks >= 90%, revoked if < 85%.
fn evaluate_single(genome: &Genome, ticks: u64, allow_early_exit: bool) -> EvalResult {
    let mut sim = Simulation::with_full_params(
        DEFAULT_SWARM_SIZE,  // n_outputs (mirror task)
        genome.input_hold_ticks,
        genome.learning_rate,
        genome.trace_decay,
        genome.max_noise_boost,
        genome.weight_decay_interval,
        genome.frustration_alpha,
        genome.cortisol_strength,
    );

    let mut stability_window_start: Option<u64> = None;
    let mut convergence_tick: Option<u64> = None;
    let mut accuracy_sum_since_converge: f32 = 0.0;
    let mut ticks_since_converge: u64 = 0;
    let mut last_accuracy: f32 = 0.0;

    for tick in 0..ticks {
        let tick_accuracy = sim.step();

        let acc = match tick_accuracy {
            Some(a) => { last_accuracy = a; a }
            None => last_accuracy,
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
            convergence_tick = None;
            accuracy_sum_since_converge = 0.0;
            ticks_since_converge = 0;
        }

        if convergence_tick.is_some() {
            accuracy_sum_since_converge += acc;
            ticks_since_converge += 1;

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

    let stable = convergence_tick.is_some() && ticks_since_converge > 0;
    let mean_acc = if ticks_since_converge > 0 {
        accuracy_sum_since_converge / ticks_since_converge as f32
    } else {
        sim.recent_accuracy
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
