//! Computed byte features from market data Records.
//!
//! Each feature produces a single u8 that a Spore reads via byte_to_inputs().
//! Features are computed from a current Record and optionally a previous Record.

use crate::record::{Record, whirlpool_price};

/// Feature names (for logging). Index matches position in compute_features() output.
pub const FEATURE_NAMES: &[&str] = &[
    "coin_b0", "coin_b1", "coin_b2", "coin_b3",
    "coin_b4", "coin_b5", "coin_b6", "coin_b7",
    "pc_b0", "pc_b1", "pc_b2", "pc_b3",
    "pc_b4", "pc_b5", "pc_b6", "pc_b7",
    "price",
    "coin_delta", "pc_delta", "price_delta",
    "volume",
    "activity",
];

/// Number of features produced by compute_features().
pub const NUM_FEATURES: usize = 22;

/// Dual-pool feature names (single-pool features + cross-pool features).
pub const DUAL_FEATURE_NAMES: &[&str] = &[
    // [0-21] Raydium single-pool features (same as FEATURE_NAMES)
    "ray_coin_b0", "ray_coin_b1", "ray_coin_b2", "ray_coin_b3",
    "ray_coin_b4", "ray_coin_b5", "ray_coin_b6", "ray_coin_b7",
    "ray_pc_b0", "ray_pc_b1", "ray_pc_b2", "ray_pc_b3",
    "ray_pc_b4", "ray_pc_b5", "ray_pc_b6", "ray_pc_b7",
    "ray_price",
    "ray_coin_delta", "ray_pc_delta", "ray_price_delta",
    "ray_volume", "ray_activity",
    // [22-43] Orca single-pool features
    "orca_coin_b0", "orca_coin_b1", "orca_coin_b2", "orca_coin_b3",
    "orca_coin_b4", "orca_coin_b5", "orca_coin_b6", "orca_coin_b7",
    "orca_pc_b0", "orca_pc_b1", "orca_pc_b2", "orca_pc_b3",
    "orca_pc_b4", "orca_pc_b5", "orca_pc_b6", "orca_pc_b7",
    "orca_price",
    "orca_coin_delta", "orca_pc_delta", "orca_price_delta",
    "orca_volume", "orca_activity",
    // [44-49] Cross-pool features
    "spread",        // |ray_price - orca_price|
    "spread_delta",  // change in spread vs previous
    "price_ratio",   // ray_price / orca_price encoded around 128
    "ray_premium",   // ray > orca → >128, ray < orca → <128
    "liq_imbalance", // relative liquidity difference
    "spread_accel",  // spread_delta change (second derivative)
];

/// Number of features produced by compute_dual_features().
pub const NUM_DUAL_FEATURES: usize = 50;

/// Quantize signed delta to byte: 128 = zero, 0 = max negative, 255 = max positive.
/// `scale` controls sensitivity: delta is divided by scale before clamping.
fn delta_to_byte(delta: i64, scale: i64) -> u8 {
    let scaled = if scale == 0 { 0 } else { delta / scale };
    (scaled.clamp(-128, 127) + 128) as u8
}

/// Quantize price into byte. Maps [low..high] to [0..255].
fn price_to_byte(price: f64, low: f64, high: f64) -> u8 {
    if high <= low { return 128; }
    let normalized = (price - low) / (high - low);
    (normalized * 255.0).clamp(0.0, 255.0) as u8
}

/// Price function type: extracts price from a Record's amm_data.
/// Used to inject Whirlpool sqrt_price for Orca instead of vault ratio.
pub type PriceFn = fn(&[u8; 1024], u8, u8) -> f64;

/// Default price function: vault ratio (correct for Raydium constant-product AMM).
fn vault_ratio_price(r: &Record, coin_dec: u8, pc_dec: u8) -> f64 {
    r.price(coin_dec, pc_dec)
}

/// Compute feature buffer from current record and optional previous record.
/// Returns exactly NUM_FEATURES bytes.
/// `price_fn` allows injecting a price source (e.g. `whirlpool_price` for Orca).
/// When None, uses vault ratio (correct for Raydium constant-product AMM).
pub fn compute_features(
    current: &Record,
    prev: Option<&Record>,
    coin_dec: u8,
    pc_dec: u8,
    price_fn: Option<PriceFn>,
) -> Vec<u8> {
    let mut f = Vec::with_capacity(NUM_FEATURES);

    let get_price = |r: &Record| -> f64 {
        match price_fn {
            Some(pf) => pf(&r.amm_data, coin_dec, pc_dec),
            None => vault_ratio_price(r, coin_dec, pc_dec),
        }
    };

    // [0-7] Vault balance bytes: coin_amount LE
    f.extend_from_slice(&current.coin_amount.to_le_bytes());
    // [8-15] Vault balance bytes: pc_amount LE
    f.extend_from_slice(&current.pc_amount.to_le_bytes());

    // [16] Price byte
    let price = get_price(current);
    f.push(price_to_byte(price, 50.0, 200.0));

    // [17-19] Delta features
    if let Some(prev) = prev {
        let coin_delta = current.coin_amount as i64 - prev.coin_amount as i64;
        let pc_delta = current.pc_amount as i64 - prev.pc_amount as i64;
        let price_prev = get_price(prev);
        let price_delta = ((price - price_prev) * 10000.0) as i64;
        f.push(delta_to_byte(coin_delta, 10_000_000));  // ±128 covers ±1.28 SOL
        f.push(delta_to_byte(pc_delta, 100_000));        // ±128 covers ±$12.80
        f.push(delta_to_byte(price_delta, 1));           // ±128 covers ±$0.0128
    } else {
        f.extend_from_slice(&[128, 128, 128]);
    }

    // [20] Volume proxy: |coin_delta| + |pc_delta|, quantized
    if let Some(prev) = prev {
        let cd = (current.coin_amount as i64 - prev.coin_amount as i64).unsigned_abs();
        let pd = (current.pc_amount as i64 - prev.pc_amount as i64).unsigned_abs();
        let vol = ((cd / 10_000_000) + (pd / 100_000)).min(255) as u8;
        f.push(vol);
    } else {
        f.push(0);
    }

    // [21] Activity: count of changed amm_data bytes vs previous
    if let Some(prev) = prev {
        let changed = current.amm_data.iter()
            .zip(prev.amm_data.iter())
            .filter(|(a, b)| a != b)
            .count()
            .min(255) as u8;
        f.push(changed);
    } else {
        f.push(0);
    }

    debug_assert_eq!(f.len(), NUM_FEATURES);
    f
}

/// Quantize absolute spread (USD) to byte: 0 = zero, 255 = $1+ spread.
/// Scale: $0.004 per unit → 255 covers ~$1.02.
fn spread_to_byte(spread: f64) -> u8 {
    (spread / 0.004).clamp(0.0, 255.0) as u8
}

/// Encode price ratio (ray/orca) as byte: 128 = equal, >128 = ray premium, <128 = orca premium.
/// Scale: ±0.1% per unit → covers ±12.8% divergence.
fn ratio_to_byte(ray_price: f64, orca_price: f64) -> u8 {
    if orca_price <= 0.0 { return 128; }
    let ratio = ray_price / orca_price;
    let offset = (ratio - 1.0) * 1000.0; // 0.1% = 1 unit
    (offset.clamp(-128.0, 127.0) + 128.0) as u8
}

/// Compute dual-pool features: single-pool features for both Raydium and Orca,
/// plus 6 cross-pool arbitrage features. Returns exactly NUM_DUAL_FEATURES bytes.
pub fn compute_dual_features(
    ray: &Record,
    ray_prev: Option<&Record>,
    orca: &Record,
    orca_prev: Option<&Record>,
    coin_dec: u8,
    pc_dec: u8,
) -> Vec<u8> {
    let mut f = Vec::with_capacity(NUM_DUAL_FEATURES);

    // [0-21] Raydium single-pool features (vault ratio price — correct for constant-product AMM)
    f.extend_from_slice(&compute_features(ray, ray_prev, coin_dec, pc_dec, None));
    // [22-43] Orca single-pool features (whirlpool sqrt_price — correct for concentrated liquidity)
    f.extend_from_slice(&compute_features(orca, orca_prev, coin_dec, pc_dec, Some(whirlpool_price)));

    // Cross-pool features
    let ray_price = ray.price(coin_dec, pc_dec);
    let orca_price_val = whirlpool_price(&orca.amm_data, coin_dec, pc_dec);
    let spread = (ray_price - orca_price_val).abs();

    // [44] Spread: absolute price difference
    f.push(spread_to_byte(spread));

    // [45] Spread delta: change in spread vs previous
    if let (Some(rp), Some(op)) = (ray_prev, orca_prev) {
        let prev_ray_price = rp.price(coin_dec, pc_dec);
        let prev_orca_price = whirlpool_price(&op.amm_data, coin_dec, pc_dec);
        let prev_spread = (prev_ray_price - prev_orca_price).abs();
        let spread_change = ((spread - prev_spread) * 10000.0) as i64;
        f.push(delta_to_byte(spread_change, 1));
    } else {
        f.push(128);
    }

    // [46] Price ratio: ray/orca encoded around 128
    f.push(ratio_to_byte(ray_price, orca_price_val));

    // [47] Ray premium: directional (ray > orca → >128)
    let premium = if orca_price_val > 0.0 {
        ((ray_price - orca_price_val) / orca_price_val * 1000.0).clamp(-128.0, 127.0) + 128.0
    } else {
        128.0
    };
    f.push(premium as u8);

    // [48] Liquidity imbalance: ratio of total vault values
    let ray_liq = ray.coin_amount as f64 + ray.pc_amount as f64;
    let orca_liq = orca.coin_amount as f64 + orca.pc_amount as f64;
    let liq_ratio = if orca_liq > 0.0 { ray_liq / (ray_liq + orca_liq) } else { 0.5 };
    f.push((liq_ratio * 255.0).clamp(0.0, 255.0) as u8);

    // [49] Spread acceleration (second derivative)
    // Requires two previous records; without them, neutral
    if let (Some(rp), Some(op)) = (ray_prev, orca_prev) {
        let prev_ray_p = rp.price(coin_dec, pc_dec);
        let prev_orca_p = whirlpool_price(&op.amm_data, coin_dec, pc_dec);
        let prev_spread = (prev_ray_p - prev_orca_p).abs();
        let spread_delta = spread - prev_spread;
        // We don't have prev-prev, so use spread_delta sign as proxy
        let accel = (spread_delta * 100000.0) as i64;
        f.push(delta_to_byte(accel, 1));
    } else {
        f.push(128);
    }

    debug_assert_eq!(f.len(), NUM_DUAL_FEATURES);
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Record;

    fn make_record(coin: u64, pc: u64) -> Record {
        Record { slot: 1, amm_data: [0u8; 1024], coin_amount: coin, pc_amount: pc }
    }

    /// Make an Orca-style Record with sqrt_price embedded at offset 65.
    /// `target_price` is in USD (e.g. 87.0 for SOL/USDC).
    fn make_orca_record(coin: u64, pc: u64, target_price: f64) -> Record {
        // price = (sqrt_price / 2^64)^2 * 10^(coin_dec - pc_dec)
        // For coin_dec=9, pc_dec=6: decimal_adj = 1000
        // sqrt_price = 2^64 * sqrt(target_price / 1000)
        let sqrt_val = (target_price / 1000.0).sqrt();
        let sqrt_price = (sqrt_val * (1u128 << 64) as f64) as u128;
        let mut amm_data = [0u8; 1024];
        amm_data[65..81].copy_from_slice(&sqrt_price.to_le_bytes());
        Record { slot: 1, amm_data, coin_amount: coin, pc_amount: pc }
    }

    // --- delta_to_byte tests ---

    #[test]
    fn test_delta_zero_is_128() {
        assert_eq!(delta_to_byte(0, 1), 128);
    }

    #[test]
    fn test_delta_positive() {
        assert!(delta_to_byte(50, 1) > 128);
        assert_eq!(delta_to_byte(50, 1), 178); // 50 + 128
    }

    #[test]
    fn test_delta_negative() {
        assert!(delta_to_byte(-50, 1) < 128);
        assert_eq!(delta_to_byte(-50, 1), 78); // -50 + 128
    }

    #[test]
    fn test_delta_clamps_high() {
        assert_eq!(delta_to_byte(1000, 1), 255); // clamps at 127 + 128
    }

    #[test]
    fn test_delta_clamps_low() {
        assert_eq!(delta_to_byte(-1000, 1), 0); // clamps at -128 + 128
    }

    #[test]
    fn test_delta_scaling() {
        // 100 / 10 = 10 → 10 + 128 = 138
        assert_eq!(delta_to_byte(100, 10), 138);
    }

    #[test]
    fn test_delta_zero_scale() {
        assert_eq!(delta_to_byte(999, 0), 128); // zero scale → 0 → 128
    }

    // --- price_to_byte tests ---

    #[test]
    fn test_price_midpoint() {
        let b = price_to_byte(125.0, 50.0, 200.0);
        assert!(b >= 127 && b <= 128, "midpoint should be ~128, got {}", b);
    }

    #[test]
    fn test_price_at_low() {
        assert_eq!(price_to_byte(50.0, 50.0, 200.0), 0);
    }

    #[test]
    fn test_price_at_high() {
        assert_eq!(price_to_byte(200.0, 50.0, 200.0), 255);
    }

    #[test]
    fn test_price_below_range() {
        assert_eq!(price_to_byte(10.0, 50.0, 200.0), 0);
    }

    #[test]
    fn test_price_above_range() {
        assert_eq!(price_to_byte(300.0, 50.0, 200.0), 255);
    }

    // --- compute_features tests ---

    #[test]
    fn test_feature_count() {
        let r = make_record(1_000_000_000, 87_000_000);
        assert_eq!(compute_features(&r, None, 9, 6, None).len(), NUM_FEATURES);
    }

    #[test]
    fn test_feature_count_with_prev() {
        let r1 = make_record(1_000_000_000, 87_000_000);
        let r2 = make_record(1_100_000_000, 88_000_000);
        assert_eq!(compute_features(&r2, Some(&r1), 9, 6, None).len(), NUM_FEATURES);
    }

    #[test]
    fn test_vault_bytes_are_le() {
        let r = make_record(0x0102030405060708, 0x1112131415161718);
        let f = compute_features(&r, None, 9, 6, None);
        assert_eq!(f[0], 0x08);
        assert_eq!(f[7], 0x01);
        assert_eq!(f[8], 0x18);
        assert_eq!(f[15], 0x11);
    }

    #[test]
    fn test_delta_neutral_without_prev() {
        let r = make_record(100, 200);
        let f = compute_features(&r, None, 9, 6, None);
        assert_eq!(f[17], 128);
        assert_eq!(f[18], 128);
        assert_eq!(f[19], 128);
    }

    #[test]
    fn test_volume_zero_without_prev() {
        let r = make_record(100, 200);
        let f = compute_features(&r, None, 9, 6, None);
        assert_eq!(f[20], 0);
    }

    #[test]
    fn test_activity_byte_counts_changes() {
        let r1 = make_record(100, 200);
        let mut r2 = make_record(100, 200);
        r2.amm_data[0] = 0xFF;
        r2.amm_data[10] = 0xAA;
        r2.amm_data[999] = 0x01;
        let f = compute_features(&r2, Some(&r1), 9, 6, None);
        assert_eq!(f[21], 3);
    }

    #[test]
    fn test_activity_zero_without_prev() {
        let r = make_record(100, 200);
        let f = compute_features(&r, None, 9, 6, None);
        assert_eq!(f[21], 0);
    }

    // --- spread_to_byte tests ---

    #[test]
    fn test_spread_zero() {
        assert_eq!(spread_to_byte(0.0), 0);
    }

    #[test]
    fn test_spread_small() {
        // $0.01 spread → 0.01 / 0.004 = 2.5 → 2
        let b = spread_to_byte(0.01);
        assert!(b > 0 && b < 50, "small spread should be low, got {}", b);
    }

    #[test]
    fn test_spread_large() {
        // $1.00 spread → 1.0 / 0.004 = 250 → 250
        let b = spread_to_byte(1.0);
        assert!(b >= 250, "$1 spread should be near 255, got {}", b);
    }

    #[test]
    fn test_spread_saturates() {
        assert_eq!(spread_to_byte(10.0), 255);
    }

    // --- ratio_to_byte tests ---

    #[test]
    fn test_ratio_equal() {
        assert_eq!(ratio_to_byte(100.0, 100.0), 128); // 1.0 → midpoint
    }

    #[test]
    fn test_ratio_ray_higher() {
        let b = ratio_to_byte(105.0, 100.0);
        assert!(b > 128, "ray higher should be >128, got {}", b);
    }

    #[test]
    fn test_ratio_orca_higher() {
        let b = ratio_to_byte(95.0, 100.0);
        assert!(b < 128, "orca higher should be <128, got {}", b);
    }

    #[test]
    fn test_ratio_zero_orca() {
        assert_eq!(ratio_to_byte(100.0, 0.0), 128); // fallback
    }

    // --- compute_dual_features tests ---

    #[test]
    fn test_dual_feature_count() {
        let r1 = make_record(100, 200);
        let r2 = make_record(300, 400);
        let f = compute_dual_features(&r1, None, &r2, None, 9, 6);
        assert_eq!(f.len(), NUM_DUAL_FEATURES);
    }

    #[test]
    fn test_dual_features_include_single_pool() {
        let ray = make_record(1_000_000_000, 87_000_000);
        let orca = make_record(500_000_000, 43_500_000);
        let dual = compute_dual_features(&ray, None, &orca, None, 9, 6);
        let ray_only = compute_features(&ray, None, 9, 6, None);
        // First NUM_FEATURES bytes of dual should equal ray-only features
        assert_eq!(&dual[..NUM_FEATURES], &ray_only[..]);
    }

    #[test]
    fn test_dual_features_orca_section() {
        let ray = make_record(1_000_000_000, 87_000_000);
        let orca = make_record(500_000_000, 43_500_000);
        let dual = compute_dual_features(&ray, None, &orca, None, 9, 6);
        let orca_only = compute_features(&orca, None, 9, 6, Some(whirlpool_price));
        // Bytes [22..44] should equal orca-only features
        assert_eq!(&dual[NUM_FEATURES..NUM_FEATURES * 2], &orca_only[..]);
    }

    #[test]
    fn test_dual_spread_zero_same_price() {
        // Raydium: vault ratio gives 87.0. Orca: sqrt_price set to give 87.0.
        let ray = make_record(1_000_000_000, 87_000_000);
        let orca = make_orca_record(1_000_000_000, 87_000_000, 87.0);
        let dual = compute_dual_features(&ray, None, &orca, None, 9, 6);
        assert_eq!(dual[44], 0); // spread = 0
    }

    #[test]
    fn test_dual_ratio_same_price() {
        let ray = make_record(1_000_000_000, 87_000_000);
        let orca = make_orca_record(1_000_000_000, 87_000_000, 87.0);
        let dual = compute_dual_features(&ray, None, &orca, None, 9, 6);
        assert_eq!(dual[46], 128); // ratio = 1.0 → midpoint
    }

    #[test]
    fn test_dual_feature_names_count() {
        assert_eq!(DUAL_FEATURE_NAMES.len(), NUM_DUAL_FEATURES);
    }
}
