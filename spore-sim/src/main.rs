//! Spore SWARM V2 — Mirror Experiment
//!
//! Proves that a colony of 8→4→1 Spores with per-bit credit assignment
//! can learn the byte-copy reflex from random noise.
//!
//! Usage:
//!   spore-sim [OPTIONS]
//!
//! Modes:
//!   (default)         Run a single simulation
//!   --tune            Run genetic hyperparameter tuner
//!
//! Simulation Options:
//!   --ticks N             Number of ticks to run (default: 100000)
//!   --swarm-size N        Number of Spores / output bits (default: 8)
//!   --trace-decay F       Trace decay rate (default: 0.9)
//!   --hold N              Input hold ticks (default: 50)
//!   --learning-rate F     Learning rate (default: 0.1)
//!   --noise-boost F       Max noise boost (default: 0.05)
//!   --decay-interval N    Weight decay interval (default: 100)
//!   --frustration-alpha F Frustration EMA alpha (default: 0.2)
//!   --cortisol F          Cortisol strength (default: 0.3)
//!   --log-interval N      Log every N ticks (default: 1000)
//!   --quiet               Suppress logging
//!   --params FILE         Load parameters from JSON file
//!
//! Tuner Options:
//!   --tune                Run genetic hyperparameter tuner
//!   --population N        Tuner population size (default: 50)
//!   --generations N       Tuner generations (default: 20)
//!   --output FILE         Output JSON file (default: best_params.json)

use spore_sim::simulation::Simulation;
use spore_sim::tuner::{self, Genome, TunerConfig};
use spore_sim::constants::*;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.iter().any(|a| a == "--tune") {
        run_tuner(&args);
        return;
    }

    run_simulation(&args);
}

fn run_simulation(args: &[String]) {
    let mut max_ticks: u64 = 100_000;
    let mut swarm_size: usize = DEFAULT_SWARM_SIZE;
    let mut trace_decay: f32 = DEFAULT_TRACE_DECAY;
    let mut input_hold_ticks: u64 = DEFAULT_INPUT_HOLD_TICKS;
    let mut learning_rate: f32 = DEFAULT_LEARNING_RATE;
    let mut max_noise_boost: f32 = DEFAULT_MAX_NOISE_BOOST;
    let mut weight_decay_interval: u64 = DEFAULT_WEIGHT_DECAY_INTERVAL;
    let mut frustration_alpha: f32 = DEFAULT_FRUSTRATION_ALPHA;
    let mut cortisol_strength: f32 = DEFAULT_CORTISOL_STRENGTH;
    let mut log_interval: u64 = 1000;
    let mut quiet = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--ticks" => {
                i += 1;
                max_ticks = args[i].parse().expect("Invalid --ticks value");
            }
            "--swarm-size" => {
                i += 1;
                swarm_size = args[i].parse().expect("Invalid --swarm-size value");
            }
            "--trace-decay" => {
                i += 1;
                trace_decay = args[i].parse().expect("Invalid --trace-decay value");
            }
            "--hold" => {
                i += 1;
                input_hold_ticks = args[i].parse().expect("Invalid --hold value");
            }
            "--learning-rate" => {
                i += 1;
                learning_rate = args[i].parse().expect("Invalid --learning-rate value");
            }
            "--noise-boost" => {
                i += 1;
                max_noise_boost = args[i].parse().expect("Invalid --noise-boost value");
            }
            "--decay-interval" => {
                i += 1;
                weight_decay_interval = args[i].parse().expect("Invalid --decay-interval value");
            }
            "--frustration-alpha" => {
                i += 1;
                frustration_alpha = args[i].parse().expect("Invalid --frustration-alpha value");
            }
            "--cortisol" => {
                i += 1;
                cortisol_strength = args[i].parse().expect("Invalid --cortisol value");
            }
            "--log-interval" => {
                i += 1;
                log_interval = args[i].parse().expect("Invalid --log-interval value");
            }
            "--quiet" => {
                quiet = true;
            }
            "--params" => {
                i += 1;
                let json = fs::read_to_string(&args[i])
                    .unwrap_or_else(|e| panic!("Failed to read {}: {}", args[i], e));
                let genome: Genome = serde_json::from_str(&json)
                    .unwrap_or_else(|e| panic!("Failed to parse {}: {}", args[i], e));
                learning_rate = genome.learning_rate;
                trace_decay = genome.trace_decay;
                max_noise_boost = genome.max_noise_boost;
                weight_decay_interval = genome.weight_decay_interval;
                frustration_alpha = genome.frustration_alpha;
                input_hold_ticks = genome.input_hold_ticks;
                cortisol_strength = genome.cortisol_strength;
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_help();
                return;
            }
        }
        i += 1;
    }

    if !quiet {
        println!("========================================");
        println!("  SWARM V2 — MIRROR EXPERIMENT         ");
        println!("========================================");
        println!(" {} Spores x (8->4->1) | {} per bit | Consensus", swarm_size, swarm_size / 8);
        println!();
        println!("Configuration:");
        println!("  Swarm size:          {}", swarm_size);
        println!("  Max ticks:           {}", max_ticks);
        println!("  Learning rate:       {}", learning_rate);
        println!("  Trace decay:         {}", trace_decay);
        println!("  Max noise boost:     {}", max_noise_boost);
        println!("  Weight decay intv:   {}", weight_decay_interval);
        println!("  Frustration alpha:   {}", frustration_alpha);
        println!("  Cortisol strength:   {}", cortisol_strength);
        println!("  Input hold ticks:    {}", input_hold_ticks);
        println!();
    }

    let mut sim = Simulation::with_full_params(
        swarm_size,
        input_hold_ticks,
        learning_rate,
        trace_decay,
        max_noise_boost,
        weight_decay_interval,
        frustration_alpha,
        cortisol_strength,
    );

    let actual_log_interval = if quiet { 0 } else { log_interval };
    let final_accuracy = sim.run(max_ticks, actual_log_interval);

    println!();
    println!("========================================");
    println!("FINAL RESULTS");
    println!("========================================");
    println!("  Ticks run:         {}", sim.tick);
    println!("  Consensus accuracy: {:.2}%", sim.swarm().accuracy() * 100.0);
    println!("  Per-Bit Consensus (best Spore per bit):");
    for k in 0..8 {
        let mut assigned: Vec<(usize, f32)> = sim.swarm().spores.iter()
            .enumerate()
            .filter(|(i, _)| i % 8 == k)
            .map(|(i, s)| (i, s.recent_accuracy))
            .collect();
        assigned.sort_by(|a, b| b.1.total_cmp(&a.1));
        let best = assigned[0];
        let others: Vec<String> = assigned[1..].iter()
            .map(|(i, a)| format!("S{}={:.0}%", i, a * 100.0))
            .collect();
        println!("    Bit {}: best=Spore {} ({:.1}%)  peers=[{}]",
            k, best.0, best.1 * 100.0, others.join(", "));
    }
    println!();

    if sim.has_converged(0.95) {
        println!("SUCCESS: Swarm learned to mirror! (accuracy > 95%)");
    } else if final_accuracy > 0.8 {
        println!("PARTIAL: Swarm is learning but hasn't converged (accuracy > 80%)");
    } else if final_accuracy > 0.5 {
        println!("SLOW: Swarm is above baseline but learning slowly");
    } else {
        println!("FAILED: Swarm did not converge (accuracy <= 50%)");
    }
}

fn run_tuner(args: &[String]) {
    let mut config = TunerConfig::default();
    let mut output_file = String::from("best_params.json");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--tune" => {}
            "--population" => {
                i += 1;
                config.population_size = args[i].parse().expect("Invalid --population");
            }
            "--generations" => {
                i += 1;
                config.generations = args[i].parse().expect("Invalid --generations");
            }
            "--ticks" => {
                i += 1;
                config.ticks_per_eval = args[i].parse().expect("Invalid --ticks");
            }
            "--output" | "-o" => {
                i += 1;
                output_file = args[i].clone();
            }
            _ => {}
        }
        i += 1;
    }

    println!("========================================");
    println!("  SWARM V2 GENETIC TUNER               ");
    println!("========================================");
    println!();
    println!("Configuration:");
    println!("  Population:    {}", config.population_size);
    println!("  Generations:   {}", config.generations);
    println!("  Ticks/eval:    {}", config.ticks_per_eval);
    println!("  Finalists:     {}", config.finalist_count);
    println!("  Output file:   {}", output_file);
    println!();
    println!("Starting evolution...");
    println!();

    let start = std::time::Instant::now();
    let (best_genome, best_result) = tuner::tune(&config);
    let elapsed = start.elapsed();

    println!();
    println!("========================================");
    println!("  EVOLUTION COMPLETE");
    println!("========================================");
    println!("  Time elapsed:  {:.1}s", elapsed.as_secs_f32());
    println!("  Best score:    {:.1}", best_result.score);
    println!("  Mean accuracy: {:.2}%", best_result.final_accuracy * 100.0);
    println!("  Stable:        {}", best_result.stable);
    println!("  Converged at:  {:?}", best_result.convergence_tick);
    println!();
    println!("OPTIMAL PARAMETERS:");
    println!("  learning_rate:         {:.4}", best_genome.learning_rate);
    println!("  trace_decay:           {:.4}", best_genome.trace_decay);
    println!("  max_noise_boost:       {:.5}", best_genome.max_noise_boost);
    println!("  weight_decay_interval: {}", best_genome.weight_decay_interval);
    println!("  frustration_alpha:     {:.4}", best_genome.frustration_alpha);
    println!("  input_hold_ticks:      {}", best_genome.input_hold_ticks);
    println!("  cortisol_strength:     {:.4}", best_genome.cortisol_strength);
    println!();

    let json = serde_json::to_string_pretty(&best_genome)
        .expect("Failed to serialize genome");
    fs::write(&output_file, &json)
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", output_file, e));

    println!("Saved to: {}", output_file);
    println!();
    println!("To run with these parameters:");
    println!("  cargo run --release -- --params {}", output_file);
}

fn print_help() {
    println!("SWARM V2 — Mirror Experiment");
    println!();
    println!("USAGE:");
    println!("  spore-sim [OPTIONS]");
    println!("  spore-sim --tune [TUNER OPTIONS]");
    println!();
    println!("SIMULATION OPTIONS:");
    println!("  --ticks N             Number of ticks to run (default: 100000)");
    println!("  --swarm-size N        Number of Spores (default: 8)");
    println!("  --trace-decay F       Trace decay rate (default: 0.9)");
    println!("  --hold N              Input hold ticks (default: 50)");
    println!("  --learning-rate F     Learning rate (default: 0.1)");
    println!("  --noise-boost F       Max noise boost (default: 0.05)");
    println!("  --decay-interval N    Weight decay interval (default: 100)");
    println!("  --frustration-alpha F Frustration EMA alpha (default: 0.2)");
    println!("  --cortisol F          Cortisol strength (default: 0.3)");
    println!("  --log-interval N      Log every N ticks (default: 1000)");
    println!("  --quiet               Suppress logging");
    println!("  --params FILE         Load parameters from JSON file");
    println!("  --help, -h            Show this help");
    println!();
    println!("TUNER OPTIONS:");
    println!("  --tune                Run genetic hyperparameter tuner");
    println!("  --population N        Tuner population size (default: 50)");
    println!("  --generations N       Tuner generations (default: 20)");
    println!("  --ticks N             Ticks per evaluation (default: 20000)");
    println!("  --output FILE, -o     Output JSON file (default: best_params.json)");
}
