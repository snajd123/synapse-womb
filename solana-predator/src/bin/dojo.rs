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
use solana_predator::features::{
    compute_features, compute_dual_features,
    NUM_FEATURES, NUM_DUAL_FEATURES,
    FEATURE_NAMES, DUAL_FEATURE_NAMES,
};
use solana_predator::record::{read_record, read_dual_record, Record, DualRecord, AMM_DATA_SIZE};
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

    /// Feature mode: "raw" (scan amm_data bytes) or "features" (computed features)
    #[arg(long, default_value = "features")]
    feature_mode: String,

    /// Training mode: "direction" (predict price up/down) or "arb" (predict arb opportunity)
    #[arg(long, default_value = "direction")]
    mode: String,

    /// Arb spread threshold in USD (for arb mode)
    #[arg(long, default_value = "0.01")]
    arb_threshold: f64,

    /// Dual-pool input file (for arb mode, DualRecord format)
    #[arg(long)]
    dual_input: Option<String>,
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

/// Arb oracle: will there be a cross-pool spread exceeding `threshold` at lookahead?
fn arb_oracle(
    ray_prices: &[f64],
    orca_prices: &[f64],
    t: usize,
    lookahead: usize,
    threshold: f64,
) -> bool {
    let spread = (ray_prices[t + lookahead] - orca_prices[t + lookahead]).abs();
    spread > threshold
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

/// Load dual-pool records from a .bin file. Returns (records, ray_prices, orca_prices).
fn load_dual_data(
    path: &str,
    coin_dec: u8,
    pc_dec: u8,
) -> Result<(Vec<DualRecord>, Vec<f64>, Vec<f64>)> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open: {}", path))?;
    let mut reader = BufReader::new(file);
    let mut records = Vec::new();
    let mut ray_prices = Vec::new();
    let mut orca_prices = Vec::new();

    while let Some(dr) = read_dual_record(&mut reader)? {
        let rp = dr.ray_price(coin_dec, pc_dec);
        let op = dr.orca_price(coin_dec, pc_dec);
        if rp > 0.0 && op > 0.0 {
            ray_prices.push(rp);
            orca_prices.push(op);
            records.push(dr);
        }
    }
    Ok((records, ray_prices, orca_prices))
}

fn main() -> Result<()> {
    let args = Args::parse();

    let is_arb_mode = args.mode == "arb";

    eprintln!("========================================");
    eprintln!("  DOJO -- Scanning Swarm Trainer");
    eprintln!("========================================");
    eprintln!("  Mode:       {}", if is_arb_mode { "ARB (cross-pool spread)" } else { "DIRECTION (price up/down)" });
    eprintln!("  Input:      {}", args.input);
    if is_arb_mode {
        if let Some(ref di) = args.dual_input {
            eprintln!("  Dual input: {}", di);
        }
        eprintln!("  Arb thresh: ${:.4}", args.arb_threshold);
    }
    eprintln!("  Spores:     {}", args.spores);
    eprintln!("  Epochs:     {}", args.epochs);
    eprintln!("  Lookahead:  {} records", args.lookahead);
    eprintln!("  Input hold: {} ticks per record", args.input_hold);
    eprintln!();

    // --- ARB MODE: dual-pool training ---
    if is_arb_mode {
        let dual_path = args.dual_input.as_deref().unwrap_or(&args.input);
        let (dual_records, ray_prices, orca_prices) =
            load_dual_data(dual_path, args.coin_decimals, args.pc_decimals)?;
        let n_records = dual_records.len();

        if n_records <= args.lookahead {
            bail!(
                "Need more than {} dual records (have {}). Increase data or decrease --lookahead.",
                args.lookahead, n_records
            );
        }

        let n_samples = n_records - args.lookahead;
        eprintln!("  DualRecords: {}", n_records);
        eprintln!("  Usable:      {} (after lookahead)", n_samples);
        eprintln!("  Ray price:   {:.4} - {:.4}",
            ray_prices.iter().cloned().fold(f64::INFINITY, f64::min),
            ray_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        eprintln!("  Orca price:  {:.4} - {:.4}",
            orca_prices.iter().cloned().fold(f64::INFINITY, f64::min),
            orca_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

        // Compute arb target balance
        let arb_ups: usize = (0..n_samples)
            .filter(|&t| arb_oracle(&ray_prices, &orca_prices, t, args.lookahead, args.arb_threshold))
            .count();
        eprintln!("  Arb events:  {:.1}% of samples", 100.0 * arb_ups as f64 / n_samples as f64);
        eprintln!();

        // Compute dual feature buffers
        let feature_buffers: Vec<Vec<u8>> = dual_records.iter().enumerate().map(|(i, dr)| {
            let ray = dr.ray_record();
            let orca = dr.orca_record();
            if i > 0 {
                let ray_prev = dual_records[i - 1].ray_record();
                let orca_prev = dual_records[i - 1].orca_record();
                compute_dual_features(&ray, Some(&ray_prev), &orca, Some(&orca_prev), args.coin_decimals, args.pc_decimals)
            } else {
                compute_dual_features(&ray, None, &orca, None, args.coin_decimals, args.pc_decimals)
            }
        }).collect();
        let feature_space = NUM_DUAL_FEATURES;
        let feature_names = DUAL_FEATURE_NAMES;

        eprintln!("  Feature mode: DUAL ({} features per record)", feature_space);

        // Assign random feature offsets
        let mut rng = rand::thread_rng();
        let offsets: Vec<usize> = (0..args.spores)
            .map(|_| rng.gen_range(0..feature_space))
            .collect();

        let mut swarm = Swarm::new(
            args.spores,
            DEFAULT_LEARNING_RATE, DEFAULT_TRACE_DECAY, DEFAULT_BASE_NOISE,
            DEFAULT_MAX_NOISE_BOOST, DEFAULT_FRUSTRATION_ALPHA,
            DEFAULT_WEIGHT_DECAY_INTERVAL, 1.0,
        );

        eprintln!("  Swarm initialized: {} Spores", swarm.size());
        eprintln!();

        let mut first_tick_correct: Vec<u64> = vec![0; args.spores];
        let mut first_tick_total: Vec<u64> = vec![0; args.spores];
        let mut global_tick: u64 = 0;
        let start = std::time::Instant::now();

        for epoch in 0..args.epochs {
            let mut epoch_correct: u64 = 0;
            let mut epoch_total: u64 = 0;

            for t in 0..n_samples {
                let target = arb_oracle(&ray_prices, &orca_prices, t, args.lookahead, args.arb_threshold);

                for hold in 0..args.input_hold {
                    for (i, spore) in swarm.spores.iter_mut().enumerate() {
                        let byte = feature_buffers[t][offsets[i]];
                        let inputs = byte_to_inputs(byte);
                        spore.fire(&inputs);
                        let correct = spore.output == target;
                        spore.receive_reward(correct);
                        spore.trace_bias_o = 0.0;
                        spore.trace_bias_h = [0.0; HIDDEN_SIZE];
                        spore.learn();
                        spore.maintain(global_tick);

                        if correct { epoch_correct += 1; }
                        epoch_total += 1;

                        if hold == 0 {
                            first_tick_correct[i] += correct as u64;
                            first_tick_total[i] += 1;
                        }
                    }
                    global_tick += 1;
                }

                if args.log_interval > 0 && (t + 1) % args.log_interval == 0 {
                    let best = swarm.spores.iter()
                        .map(|s| s.recent_accuracy)
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap_or(0.0);
                    eprintln!(
                        "  [E{}] tick {}/{} | mean_acc: {:.4} | best_spore: {:.4}",
                        epoch + 1, t + 1, n_samples,
                        epoch_correct as f64 / epoch_total as f64, best,
                    );
                }
            }

            let elapsed = start.elapsed().as_secs();
            eprintln!();
            eprintln!("========== Epoch {}/{} ({:.0}s) ==========", epoch + 1, args.epochs, elapsed as f64);
            eprintln!("  Mean tick accuracy: {:.4}", epoch_correct as f64 / epoch_total as f64);

            let mut ranked: Vec<(usize, f32)> = swarm.spores.iter()
                .enumerate()
                .map(|(i, s)| (i, s.recent_accuracy))
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

            eprintln!("  Top 10 Spores:");
            for &(idx, acc) in ranked.iter().take(10) {
                let label = if acc >= 0.70 { "CONVERGED" } else if acc >= 0.55 { "learning" } else { "chance" };
                eprintln!(
                    "    Spore {:>3} | {:>16} | acc: {:.4} | {}",
                    idx,
                    if offsets[idx] < feature_names.len() { feature_names[offsets[idx]] } else { "???" },
                    acc, label
                );
            }
            eprintln!();
        }

        // Final arb summary
        let elapsed = start.elapsed();
        let mut ranked: Vec<(usize, usize, f64, f32)> = (0..args.spores)
            .map(|i| {
                let ft_acc = if first_tick_total[i] > 0 {
                    first_tick_correct[i] as f64 / first_tick_total[i] as f64
                } else { 0.0 };
                (i, offsets[i], ft_acc, swarm.spores[i].recent_accuracy)
            })
            .collect();
        ranked.sort_by(|a, b| b.2.total_cmp(&a.2));

        eprintln!("========================================");
        eprintln!("  DOJO ARB TRAINING COMPLETE");
        eprintln!("========================================");
        eprintln!("  DualRecords: {}", n_records);
        eprintln!("  Epochs:      {}", args.epochs);
        eprintln!("  Arb thresh:  ${:.4}", args.arb_threshold);
        eprintln!("  Total ticks: {}", global_tick);
        eprintln!("  Time:        {:.1}s", elapsed.as_secs_f64());
        eprintln!();

        eprintln!("  Top 20 by FIRST-TICK accuracy:");
        for &(idx, offset, ft_acc, ema) in ranked.iter().take(20) {
            let label = if ft_acc >= 0.60 { "SIGNAL" } else if ft_acc >= 0.53 { "maybe" } else { "noise" };
            eprintln!(
                "    Spore {:>3} | {:>16} | 1st-tick: {:.4} | ema: {:.4} | {}",
                idx,
                if offset < feature_names.len() { feature_names[offset] } else { "???" },
                ft_acc, ema, label
            );
        }
        eprintln!();
        eprintln!("  Bottom 5:");
        for &(idx, offset, ft_acc, ema) in ranked.iter().rev().take(5) {
            eprintln!(
                "    Spore {:>3} | {:>16} | 1st-tick: {:.4} | ema: {:.4}",
                idx,
                if offset < feature_names.len() { feature_names[offset] } else { "???" },
                ft_acc, ema
            );
        }
        eprintln!("========================================");

        return Ok(());
    }

    // --- DIRECTION MODE (original) ---

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

    // Feature mode: precompute feature buffers or use raw amm_data bytes
    let use_features = args.feature_mode == "features";
    let feature_buffers: Vec<Vec<u8>> = if use_features {
        eprintln!("  Feature mode: COMPUTED ({} features per record)", NUM_FEATURES);
        records.iter().enumerate().map(|(i, r)| {
            let prev = if i > 0 { Some(&records[i - 1]) } else { None };
            compute_features(r, prev, args.coin_decimals, args.pc_decimals, None)
        }).collect()
    } else {
        eprintln!("  Feature mode: RAW ({} amm_data bytes)", AMM_DATA_SIZE);
        Vec::new()
    };
    let feature_space = if use_features { NUM_FEATURES } else { AMM_DATA_SIZE };

    // Assign random byte offsets to each Spore
    let mut rng = rand::thread_rng();
    let mut offsets: Vec<usize> = (0..args.spores)
        .map(|_| rng.gen_range(0..feature_space))
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
    if use_features {
        eprintln!("  Feature offsets assigned (random, 0-{})", feature_space - 1);
    } else {
        eprintln!("  Byte offsets assigned (random, 0-1023)");
    }
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
                    let byte = if use_features {
                        feature_buffers[t][offsets[i]]
                    } else {
                        records[t].amm_data[offsets[i]]
                    };
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
                "    Spore {:>3} | {:>12} | acc: {:.4} | {}",
                idx,
                if use_features && offsets[idx] < FEATURE_NAMES.len() {
                    FEATURE_NAMES[offsets[idx]].to_string()
                } else {
                    format!("byte {:>4}", offsets[idx])
                },
                acc, label
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
            "    Spore {:>3} | {:>12} | 1st-tick: {:.4} | ema: {:.4} | {}",
            idx,
            if use_features && offset < FEATURE_NAMES.len() {
                FEATURE_NAMES[offset].to_string()
            } else {
                format!("byte {:>4}", offset)
            },
            ft_acc, ema, label
        );
    }
    eprintln!();

    eprintln!("  Bottom 5:");
    for &(idx, offset, ft_acc, ema) in ranked.iter().rev().take(5) {
        eprintln!(
            "    Spore {:>3} | {:>12} | 1st-tick: {:.4} | ema: {:.4}",
            idx,
            if use_features && offset < FEATURE_NAMES.len() {
                FEATURE_NAMES[offset].to_string()
            } else {
                format!("byte {:>4}", offset)
            },
            ft_acc, ema
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

    // --- arb_oracle tests ---

    #[test]
    fn test_arb_oracle_no_spread() {
        let ray = vec![100.0, 100.0, 100.0];
        let orca = vec![100.0, 100.0, 100.0];
        assert!(!arb_oracle(&ray, &orca, 0, 1, 0.01)); // no spread
    }

    #[test]
    fn test_arb_oracle_spread_above_threshold() {
        let ray = vec![100.0, 101.0];
        let orca = vec![100.0, 100.0];
        assert!(arb_oracle(&ray, &orca, 0, 1, 0.5)); // $1 spread > $0.50
    }

    #[test]
    fn test_arb_oracle_spread_below_threshold() {
        let ray = vec![100.0, 100.005];
        let orca = vec![100.0, 100.0];
        assert!(!arb_oracle(&ray, &orca, 0, 1, 0.01)); // $0.005 < $0.01
    }
}