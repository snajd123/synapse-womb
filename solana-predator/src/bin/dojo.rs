//! Dojo: Train a Scanning Swarm on raw Solana market data.
//!
//! Each Spore reads a different byte of the 1024-byte AMM account state
//! and learns to predict price direction. Evolutionary feature selection:
//! Spores on price-relevant bytes converge, Spores on junk get rejuvenated.
//!
//! Usage:
//!   dojo --input market_vibrations.bin --epochs 10 --spores 200 --lookahead 100

use anyhow::{bail, Context, Result};
use clap::Parser;
use rand::Rng;
use solana_predator::record::{read_record, Record, AMM_DATA_SIZE};
use spore_sim::constants::*;
use spore_sim::swarm::Swarm;
use std::fs::File;
use std::io::BufReader;

#[derive(Parser)]
#[command(name = "dojo", about = "Train Scanning Swarm on Solana market data")]
struct Args {
    /// Input .bin file path
    #[arg(long, short, default_value = "market_vibrations.bin")]
    input: String,

    /// Number of training epochs (full passes over data)
    #[arg(long, short, default_value = "10")]
    epochs: usize,

    /// Number of Spores in the scanning swarm
    #[arg(long, default_value = "200")]
    spores: usize,

    /// Oracle lookahead in records
    #[arg(long, default_value = "100")]
    lookahead: usize,

    /// Print progress every N ticks (0 = epoch-end only)
    #[arg(long, default_value = "1000")]
    log_interval: usize,

    /// Coin decimals for price computation (SOL = 9)
    #[arg(long, default_value = "9")]
    coin_decimals: u8,

    /// PC decimals for price computation (USDC = 6)
    #[arg(long, default_value = "6")]
    pc_decimals: u8,
}

/// Convert a single byte to 8 f32 inputs (LSB to MSB).
fn byte_to_inputs(byte: u8) -> [f32; INPUT_SIZE] {
    let mut inputs = [0.0f32; INPUT_SIZE];
    for bit in 0..INPUT_SIZE {
        inputs[bit] = if byte & (1 << bit) != 0 { 1.0 } else { 0.0 };
    }
    inputs
}

/// Oracle: will the price be higher `lookahead` records from now?
fn oracle(prices: &[f64], t: usize, lookahead: usize) -> bool {
    prices[t + lookahead] > prices[t]
}

/// Load all records from a .bin file. Returns (records, prices).
fn load_data(
    path: &str,
    coin_dec: u8,
    pc_dec: u8,
) -> Result<(Vec<Record>, Vec<f64>)> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open: {}", path))?;
    let mut reader = BufReader::new(file);
    let mut records = Vec::new();
    let mut prices = Vec::new();

    while let Some(record) = read_record(&mut reader)? {
        let price = record.price(coin_dec, pc_dec);
        if price > 0.0 {
            prices.push(price);
            records.push(record);
        }
    }

    Ok((records, prices))
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("========================================");
    eprintln!("  DOJO -- Scanning Swarm Trainer");
    eprintln!("========================================");
    eprintln!("  Input:      {}", args.input);
    eprintln!("  Spores:     {}", args.spores);
    eprintln!("  Epochs:     {}", args.epochs);
    eprintln!("  Lookahead:  {} records", args.lookahead);
    eprintln!();

    // Load data
    let (records, prices) = load_data(&args.input, args.coin_decimals, args.pc_decimals)?;
    let n_records = records.len();

    if n_records <= args.lookahead {
        bail!(
            "Need more than {} records (have {}). Increase data or decrease --lookahead.",
            args.lookahead,
            n_records
        );
    }

    let n_samples = n_records - args.lookahead;
    let price_min = prices.iter().cloned().fold(f64::INFINITY, f64::min);
    let price_max = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    eprintln!("  Records:    {}", n_records);
    eprintln!("  Usable:     {} (after lookahead)", n_samples);
    eprintln!("  Price range: {:.4} - {:.4}", price_min, price_max);

    // Compute target balance (what % of targets are "up"?)
    let ups: usize = (0..n_samples)
        .filter(|&t| oracle(&prices, t, args.lookahead))
        .count();
    eprintln!("  Target balance: {:.1}% up", 100.0 * ups as f64 / n_samples as f64);
    eprintln!();

    // Assign random byte offsets to each Spore
    let mut rng = rand::thread_rng();
    let offsets: Vec<usize> = (0..args.spores)
        .map(|_| rng.gen_range(0..AMM_DATA_SIZE))
        .collect();

    // Create Swarm with proven defaults
    let mut swarm = Swarm::new(
        args.spores,
        DEFAULT_LEARNING_RATE,
        DEFAULT_TRACE_DECAY,
        DEFAULT_BASE_NOISE,
        DEFAULT_MAX_NOISE_BOOST,
        DEFAULT_FRUSTRATION_ALPHA,
        DEFAULT_WEIGHT_DECAY_INTERVAL,
        DEFAULT_CORTISOL_STRENGTH,
    );

    eprintln!("  Swarm initialized: {} Spores", swarm.size());
    eprintln!("  Byte offsets assigned (random, 0-1023)");
    eprintln!();

    // Training loop
    let mut global_tick: u64 = 0;
    let start = std::time::Instant::now();

    for epoch in 0..args.epochs {
        let mut epoch_correct: u64 = 0;
        let mut epoch_total: u64 = 0;

        for t in 0..n_samples {
            let target = oracle(&prices, t, args.lookahead);

            // Per-Spore training: each Spore sees its own byte
            for (i, spore) in swarm.spores.iter_mut().enumerate() {
                let byte = records[t].amm_data[offsets[i]];
                let inputs = byte_to_inputs(byte);
                spore.fire(&inputs);
                let correct = spore.output == target;
                spore.receive_reward(correct);
                spore.learn();
                spore.maintain(global_tick);

                if correct {
                    epoch_correct += 1;
                }
                epoch_total += 1;
            }

            swarm.rejuvenate();
            global_tick += 1;

            // Intra-epoch logging
            if args.log_interval > 0 && (t + 1) % args.log_interval == 0 {
                let best = swarm.spores.iter()
                    .map(|s| s.recent_accuracy)
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap_or(0.0);
                eprintln!(
                    "  [E{}] tick {}/{} | mean_acc: {:.4} | best_spore: {:.4}",
                    epoch + 1,
                    t + 1,
                    n_samples,
                    epoch_correct as f64 / epoch_total as f64,
                    best,
                );
            }
        }

        // Per-epoch summary
        let elapsed = start.elapsed().as_secs();
        eprintln!();
        eprintln!("========== Epoch {}/{} ({:.0}s) ==========", epoch + 1, args.epochs, elapsed as f64);
        eprintln!("  Mean tick accuracy: {:.4}", epoch_correct as f64 / epoch_total as f64);

        // Top 10 Spores by recent_accuracy
        let mut ranked: Vec<(usize, f32)> = swarm.spores.iter()
            .enumerate()
            .map(|(i, s)| (i, s.recent_accuracy))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

        eprintln!("  Top 10 Spores:");
        for &(idx, acc) in ranked.iter().take(10) {
            let label = if acc >= 0.70 {
                "CONVERGED"
            } else if acc >= 0.55 {
                "learning"
            } else {
                "chance"
            };
            eprintln!(
                "    Spore {:>3} | byte {:>4} | acc: {:.4} | {}",
                idx, offsets[idx], acc, label
            );
        }
        eprintln!();
    }

    // Final summary
    let elapsed = start.elapsed();
    let mut converged: Vec<(usize, usize, f32)> = swarm.spores.iter()
        .enumerate()
        .filter(|(_, s)| s.recent_accuracy >= 0.55)
        .map(|(i, s)| (i, offsets[i], s.recent_accuracy))
        .collect();
    converged.sort_by(|a, b| b.2.total_cmp(&a.2));

    eprintln!("========================================");
    eprintln!("  DOJO TRAINING COMPLETE");
    eprintln!("========================================");
    eprintln!("  Input:       {}", args.input);
    eprintln!("  Records:     {}", n_records);
    eprintln!("  Epochs:      {}", args.epochs);
    eprintln!("  Total ticks: {}", global_tick);
    eprintln!("  Time:        {:.1}s", elapsed.as_secs_f64());
    eprintln!();

    if converged.is_empty() {
        eprintln!("  No Spores converged above 55%.");
        eprintln!("  Data may lack learnable signal (expected for random walk).");
    } else {
        eprintln!("  Converged Spores (>55% accuracy):");
        for (idx, offset, acc) in &converged {
            let label = if *acc >= 0.70 { "CONVERGED" } else { "learning" };
            eprintln!(
                "    Spore {:>3} | byte {:>4} | acc: {:.4} | {}",
                idx, offset, acc, label
            );
        }
        eprintln!();
        eprintln!("  Signal-bearing byte offsets:");
        let mut unique_offsets: Vec<usize> = converged.iter().map(|(_, o, _)| *o).collect();
        unique_offsets.sort();
        unique_offsets.dedup();
        for offset in &unique_offsets {
            let best = converged.iter()
                .filter(|(_, o, _)| o == offset)
                .map(|(_, _, a)| *a)
                .max_by(|a, b| a.total_cmp(b))
                .unwrap_or(0.0);
            eprintln!("    byte {:>4} | best_acc: {:.4}", offset, best);
        }
    }
    eprintln!("========================================");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_to_inputs_zero() {
        let inputs = byte_to_inputs(0x00);
        assert_eq!(inputs, [0.0; 8]);
    }

    #[test]
    fn test_byte_to_inputs_all_ones() {
        let inputs = byte_to_inputs(0xFF);
        assert_eq!(inputs, [1.0; 8]);
    }

    #[test]
    fn test_byte_to_inputs_lsb() {
        let inputs = byte_to_inputs(0x01);
        assert_eq!(inputs[0], 1.0);
        for i in 1..8 {
            assert_eq!(inputs[i], 0.0, "bit {} should be 0", i);
        }
    }

    #[test]
    fn test_byte_to_inputs_msb() {
        let inputs = byte_to_inputs(0x80);
        for i in 0..7 {
            assert_eq!(inputs[i], 0.0, "bit {} should be 0", i);
        }
        assert_eq!(inputs[7], 1.0);
    }

    #[test]
    fn test_byte_to_inputs_pattern() {
        // 0b10100101 = 0xA5 → bits 0,2,5,7 are set
        let inputs = byte_to_inputs(0xA5);
        assert_eq!(inputs[0], 1.0);
        assert_eq!(inputs[1], 0.0);
        assert_eq!(inputs[2], 1.0);
        assert_eq!(inputs[3], 0.0);
        assert_eq!(inputs[4], 0.0);
        assert_eq!(inputs[5], 1.0);
        assert_eq!(inputs[6], 0.0);
        assert_eq!(inputs[7], 1.0);
    }

    #[test]
    fn test_oracle_price_up() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        assert!(oracle(&prices, 0, 2)); // 102 > 100
        assert!(oracle(&prices, 1, 3)); // 104 > 101
    }

    #[test]
    fn test_oracle_price_down() {
        let prices = vec![104.0, 103.0, 102.0, 101.0, 100.0];
        assert!(!oracle(&prices, 0, 2)); // 102 < 104
    }

    #[test]
    fn test_oracle_price_flat() {
        let prices = vec![100.0, 100.0, 100.0];
        assert!(!oracle(&prices, 0, 1)); // 100 is NOT > 100
    }

}
