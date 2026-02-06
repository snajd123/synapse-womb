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

    /// Ticks to hold each record (1 = no hold, best for weight learning)
    #[arg(long, default_value = "1")]
    input_hold: usize,

    /// Print progress every N records (0 = epoch-end only)
    #[arg(long, default_value = "1000")]
    log_interval: usize,

    /// Coin decimals for price computation (SOL = 9)
    #[arg(long, default_value = "9")]
    coin_decimals: u8,

    /// PC decimals for price computation (USDC = 6)
    #[arg(long, default_value = "6")]
    pc_decimals: u8,

    /// Force Spore 0 to this byte offset (for testing known signals). -1 = disabled.
    #[arg(long, default_value = "-1")]
    force_offset: i32,
}

/// Convert a single byte to 16 f32 inputs: 8 positive + 8 complementary.
/// Inputs 0-7: the bits of the byte (LSB to MSB).
/// Inputs 8-15: the inverted bits (1.0 - bit).
/// Complementary encoding ensures Hebbian traces fire for both 0 and 1 states.
fn byte_to_inputs(byte: u8) -> [f32; INPUT_SIZE] {
    let mut inputs = [0.0f32; INPUT_SIZE];
    for i in 0..8 {
        let val = if byte & (1 << i) != 0 { 1.0 } else { 0.0 };
        inputs[i] = val;           // Positive channel
        inputs[i + 8] = 1.0 - val; // Negative channel
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
    eprintln!("  Input hold: {} ticks per record", args.input_hold);
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
    let mut offsets: Vec<usize> = (0..args.spores)
        .map(|_| rng.gen_range(0..AMM_DATA_SIZE))
        .collect();
    if args.force_offset >= 0 {
        offsets[0] = args.force_offset as usize;
    }

    // Create Swarm with dojo-specific params.
    // Cortisol = 1.0 (symmetric reward): zero-mean at chance prevents
    // drift-based false convergence on noise bytes. The mirror task needs
    // 0.5 for bootstrapping, but the dojo's per-Spore independent training
    // + INITIAL_BIAS provides enough asymmetry to break symmetry.
    // Homeostasis stays ON — it anchors bias_o at a neutral point,
    // forcing the Spore to use input-dependent weights rather than
    // bias-tracking the current target during the hold period.
    let mut swarm = Swarm::new(
        args.spores,
        DEFAULT_LEARNING_RATE,
        DEFAULT_TRACE_DECAY,
        DEFAULT_BASE_NOISE,
        DEFAULT_MAX_NOISE_BOOST,
        DEFAULT_FRUSTRATION_ALPHA,
        DEFAULT_WEIGHT_DECAY_INTERVAL,
        1.0, // cortisol: symmetric for streaming data (not DEFAULT_CORTISOL_STRENGTH)
    );

    eprintln!("  Swarm initialized: {} Spores", swarm.size());
    eprintln!("  Byte offsets assigned (random, 0-1023)");
    eprintln!();

    // First-tick accuracy tracking (per-Spore: correct on first tick of each record)
    let mut first_tick_correct: Vec<u64> = vec![0; args.spores];
    let mut first_tick_total: Vec<u64> = vec![0; args.spores];

    // Training loop
    let mut global_tick: u64 = 0;
    let start = std::time::Instant::now();

    for epoch in 0..args.epochs {
        let mut epoch_correct: u64 = 0;
        let mut epoch_total: u64 = 0;

        for t in 0..n_samples {
            let target = oracle(&prices, t, args.lookahead);

            // Hold each record for input_hold ticks (repeated exposure for Hebbian learning).
            // Same input-target pair presented multiple times so traces can accumulate
            // directional weight changes. Analogous to mini-batch SGD.
            for hold in 0..args.input_hold {
                for (i, spore) in swarm.spores.iter_mut().enumerate() {
                    let byte = records[t].amm_data[offsets[i]];
                    let inputs = byte_to_inputs(byte);
                    spore.fire(&inputs);
                    let correct = spore.output == target;
                    spore.receive_reward(correct);
                    // Zero bias traces so learn() only updates weights, not biases.
                    // Prevents bias_o from tracking the current target during hold.
                    // Biases are controlled by homeostasis instead.
                    spore.trace_bias_o = 0.0;
                    spore.trace_bias_h = [0.0; HIDDEN_SIZE];
                    spore.learn();
                    spore.maintain(global_tick);

                    if correct {
                        epoch_correct += 1;
                    }
                    epoch_total += 1;

                    // Track first-tick accuracy (before hold memorization).
                    // This is the true predictive metric.
                    if hold == 0 {
                        first_tick_correct[i] += correct as u64;
                        first_tick_total[i] += 1;
                    }
                }
                global_tick += 1;
            }

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

    // Final summary — ranked by FIRST-TICK accuracy (true predictive metric)
    let elapsed = start.elapsed();
    let mut ranked: Vec<(usize, usize, f64, f32)> = (0..args.spores)
        .map(|i| {
            let ft_acc = if first_tick_total[i] > 0 {
                first_tick_correct[i] as f64 / first_tick_total[i] as f64
            } else {
                0.0
            };
            (i, offsets[i], ft_acc, swarm.spores[i].recent_accuracy)
        })
        .collect();
    ranked.sort_by(|a, b| b.2.total_cmp(&a.2));

    eprintln!("========================================");
    eprintln!("  DOJO TRAINING COMPLETE");
    eprintln!("========================================");
    eprintln!("  Input:       {}", args.input);
    eprintln!("  Records:     {}", n_records);
    eprintln!("  Epochs:      {}", args.epochs);
    eprintln!("  Total ticks: {}", global_tick);
    eprintln!("  Time:        {:.1}s", elapsed.as_secs_f64());
    eprintln!();

    eprintln!("  Top 20 by FIRST-TICK accuracy (true predictive metric):");
    for &(idx, offset, ft_acc, ema) in ranked.iter().take(20) {
        let label = if ft_acc >= 0.60 {
            "SIGNAL"
        } else if ft_acc >= 0.53 {
            "maybe"
        } else {
            "noise"
        };
        eprintln!(
            "    Spore {:>3} | byte {:>4} | 1st-tick: {:.4} | ema: {:.4} | {}",
            idx, offset, ft_acc, ema, label
        );
    }
    eprintln!();

    eprintln!("  Bottom 5:");
    for &(idx, offset, ft_acc, ema) in ranked.iter().rev().take(5) {
        eprintln!(
            "    Spore {:>3} | byte {:>4} | 1st-tick: {:.4} | ema: {:.4}",
            idx, offset, ft_acc, ema
        );
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
        assert_eq!(inputs[0..8], [0.0; 8]);
    }

    #[test]
    fn test_byte_to_inputs_all_ones() {
        let inputs = byte_to_inputs(0xFF);
        assert_eq!(inputs[0..8], [1.0; 8]);
    }

    #[test]
    fn test_byte_to_inputs_lsb() {
        // 0b00000001 → bit 0 = 1, rest = 0
        let inputs = byte_to_inputs(0x01);
        assert_eq!(inputs[0], 1.0);
        for i in 1..8 {
            assert_eq!(inputs[i], 0.0, "bit {} should be 0", i);
        }
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