//! Spore Mirror Experiment - Phase 1
//!
//! Proves that Hebbian learning + Dopamine reinforcement can evolve
//! a byte-copy reflex from random noise.
//!
//! Usage:
//!   spore-sim [OPTIONS]
//!
//! Options:
//!   --ticks N         Number of ticks to run (default: 100000)
//!   --latency N       Reward latency in ticks (default: 0)
//!   --trace-decay F   Trace decay rate (default: 0.9)
//!   --hold N          Input hold ticks (default: 50)
//!   --log-interval N  Log every N ticks (default: 1000)
//!   --quiet           Suppress logging

use spore_sim::simulation::Simulation;
use spore_sim::constants::*;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let mut max_ticks: u64 = 100_000;
    let mut reward_latency: u64 = DEFAULT_REWARD_LATENCY as u64;
    let mut trace_decay: f32 = DEFAULT_TRACE_DECAY as f32;
    let mut input_hold_ticks: u64 = DEFAULT_INPUT_HOLD_TICKS as u64;
    let mut log_interval: u64 = 1000;
    let mut quiet = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--ticks" => {
                i += 1;
                max_ticks = args[i].parse().expect("Invalid --ticks value");
            }
            "--latency" => {
                i += 1;
                reward_latency = args[i].parse().expect("Invalid --latency value");
            }
            "--trace-decay" => {
                i += 1;
                trace_decay = args[i].parse().expect("Invalid --trace-decay value");
            }
            "--hold" => {
                i += 1;
                input_hold_ticks = args[i].parse().expect("Invalid --hold value");
            }
            "--log-interval" => {
                i += 1;
                log_interval = args[i].parse().expect("Invalid --log-interval value");
            }
            "--quiet" => {
                quiet = true;
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
        println!("  SPORE MIRROR EXPERIMENT - PHASE 1    ");
        println!("========================================");
        println!(" Proving: Hebbian + Dopamine = Emergent Byte Copy");
        println!();
        println!("Configuration:");
        println!("  Max ticks:        {}", max_ticks);
        println!("  Reward latency:   {}", reward_latency);
        println!("  Trace decay:      {}", trace_decay);
        println!("  Input hold ticks: {}", input_hold_ticks);
        println!();

        // Validate timing constraint (Fix 4)
        let min_hold = min_input_hold_ticks(reward_latency as usize);
        if (input_hold_ticks as usize) < min_hold {
            println!("WARNING: input_hold_ticks ({}) < minimum ({}) for latency {}",
                input_hold_ticks, min_hold, reward_latency);
            println!("   This may cause superstitious learning (Fix 4).");
            println!();
        }

        // Recommend trace decay
        let rec_decay = recommended_trace_decay(reward_latency as usize);
        if (trace_decay as f64) < rec_decay - 0.02 {
            println!("WARNING: trace_decay ({}) may be too fast for latency {}",
                trace_decay, reward_latency);
            println!("   Recommended: >= {:.2}", rec_decay);
            println!();
        }
    }

    // Create and run simulation
    let mut sim = Simulation::with_params(
        reward_latency,
        trace_decay,
        input_hold_ticks,
    );

    let actual_log_interval = if quiet { 0 } else { log_interval };
    let final_accuracy = sim.run(max_ticks, actual_log_interval);

    // Report results
    println!();
    println!("========================================");
    println!("FINAL RESULTS");
    println!("========================================");
    println!("  Ticks run:      {}", sim.tick);
    println!("  Final accuracy: {:.2}%", final_accuracy * 100.0);
    println!("  Frustration:    {:.3}", sim.spore().frustration);
    println!();

    if sim.has_converged(0.95) {
        println!("SUCCESS: Spore learned to mirror! (accuracy > 95%)");
    } else if final_accuracy > 0.8 {
        println!("PARTIAL: Spore is learning but hasn't converged (accuracy > 80%)");
        println!("   Try running for more ticks or tuning hyperparameters.");
    } else if final_accuracy > 0.5 {
        println!("SLOW: Spore is above baseline but learning slowly");
        println!("   Check: learning_rate, trace_decay, input_hold_ticks");
    } else {
        println!("FAILED: Spore did not converge (accuracy <= 50%)");
        println!("   Check failure modes in docs/plans/");
    }
}

fn print_help() {
    println!("Spore Mirror Experiment - Phase 1");
    println!();
    println!("USAGE:");
    println!("  spore-sim [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --ticks N         Number of ticks to run (default: 100000)");
    println!("  --latency N       Reward latency in ticks (default: 0)");
    println!("  --trace-decay F   Trace decay rate (default: 0.9)");
    println!("  --hold N          Input hold ticks (default: 50)");
    println!("  --log-interval N  Log every N ticks (default: 1000)");
    println!("  --quiet           Suppress logging");
    println!("  --help, -h        Show this help");
}
