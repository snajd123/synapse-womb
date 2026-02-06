//! Computed byte features from market data Records.
//!
//! Each feature produces a single u8 that a Spore reads via byte_to_inputs().
//! Features are computed from a current Record and optionally a previous Record.

use crate::record::Record;

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

/// Compute feature buffer from current record and optional previous record.
/// Returns exactly NUM_FEATURES bytes.
pub fn compute_features(
    current: &Record,
    prev: Option<&Record>,
    coin_dec: u8,
    pc_dec: u8,
) -> Vec<u8> {
    let mut f = Vec::with_capacity(NUM_FEATURES);

    // [0-7] Vault balance bytes: coin_amount LE
    f.extend_from_slice(&current.coin_amount.to_le_bytes());
    // [8-15] Vault balance bytes: pc_amount LE
    f.extend_from_slice(&current.pc_amount.to_le_bytes());

    // [16] Price byte
    let price = current.price(coin_dec, pc_dec);
    f.push(price_to_byte(price, 50.0, 200.0));

    // [17-19] Delta features
    if let Some(prev) = prev {
        let coin_delta = current.coin_amount as i64 - prev.coin_amount as i64;
        let pc_delta = current.pc_amount as i64 - prev.pc_amount as i64;
        let price_prev = prev.price(coin_dec, pc_dec);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Record;

    fn make_record(coin: u64, pc: u64) -> Record {
        Record { slot: 1, amm_data: [0u8; 1024], coin_amount: coin, pc_amount: pc }
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
        assert_eq!(compute_features(&r, None, 9, 6).len(), NUM_FEATURES);
    }

    #[test]
    fn test_feature_count_with_prev() {
        let r1 = make_record(1_000_000_000, 87_000_000);
        let r2 = make_record(1_100_000_000, 88_000_000);
        assert_eq!(compute_features(&r2, Some(&r1), 9, 6).len(), NUM_FEATURES);
    }

    #[test]
    fn test_vault_bytes_are_le() {
        let r = make_record(0x0102030405060708, 0x1112131415161718);
        let f = compute_features(&r, None, 9, 6);
        assert_eq!(f[0], 0x08);
        assert_eq!(f[7], 0x01);
        assert_eq!(f[8], 0x18);
        assert_eq!(f[15], 0x11);
    }

    #[test]
    fn test_delta_neutral_without_prev() {
        let r = make_record(100, 200);
        let f = compute_features(&r, None, 9, 6);
        assert_eq!(f[17], 128);
        assert_eq!(f[18], 128);
        assert_eq!(f[19], 128);
    }

    #[test]
    fn test_volume_zero_without_prev() {
        let r = make_record(100, 200);
        let f = compute_features(&r, None, 9, 6);
        assert_eq!(f[20], 0);
    }

    #[test]
    fn test_activity_byte_counts_changes() {
        let r1 = make_record(100, 200);
        let mut r2 = make_record(100, 200);
        r2.amm_data[0] = 0xFF;
        r2.amm_data[10] = 0xAA;
        r2.amm_data[999] = 0x01;
        let f = compute_features(&r2, Some(&r1), 9, 6);
        assert_eq!(f[21], 3);
    }

    #[test]
    fn test_activity_zero_without_prev() {
        let r = make_record(100, 200);
        let f = compute_features(&r, None, 9, 6);
        assert_eq!(f[21], 0);
    }
}
