//! Tests for utility functions in spore-sim.
//!
//! These tests verify the stochastic rounding behavior which is critical
//! for preventing weight updates from truncating to zero during Hebbian learning.

use spore_sim::utils::{random_weight, stochastic_round};
use spore_sim::constants::{WEIGHT_INIT_MIN, WEIGHT_INIT_MAX};

/// Test that integer values round to themselves (positive case).
/// 5.0 should always return 5.
#[test]
fn test_stochastic_round_positive_integer() {
    for _ in 0..100 {
        assert_eq!(stochastic_round(5.0), 5);
    }
}

/// Test that integer values round to themselves (negative case).
/// -5.0 should always return -5.
#[test]
fn test_stochastic_round_negative_integer() {
    for _ in 0..100 {
        assert_eq!(stochastic_round(-5.0), -5);
    }
}

/// Test that zero always rounds to zero.
#[test]
fn test_stochastic_round_zero() {
    for _ in 0..100 {
        assert_eq!(stochastic_round(0.0), 0);
    }
}

/// Test stochastic distribution for positive fractional values.
/// 0.3 should round to 1 approximately 30% of the time and 0 approximately 70% of the time.
#[test]
fn test_stochastic_round_positive_fractional_distribution() {
    let iterations = 10000;
    let mut ones = 0;
    let mut zeros = 0;

    for _ in 0..iterations {
        let result = stochastic_round(0.3);
        match result {
            0 => zeros += 1,
            1 => ones += 1,
            _ => panic!("Unexpected result {} for stochastic_round(0.3)", result),
        }
    }

    // 0.3 should give ~30% 1s, ~70% 0s
    let ones_ratio = ones as f64 / iterations as f64;
    let zeros_ratio = zeros as f64 / iterations as f64;

    // Allow 5% tolerance for statistical variation
    assert!(
        (ones_ratio - 0.3).abs() < 0.05,
        "Expected ~30% ones, got {:.1}%",
        ones_ratio * 100.0
    );
    assert!(
        (zeros_ratio - 0.7).abs() < 0.05,
        "Expected ~70% zeros, got {:.1}%",
        zeros_ratio * 100.0
    );
}

/// Test stochastic distribution for negative fractional values.
/// -0.7 should round to -1 approximately 70% of the time and 0 approximately 30% of the time.
#[test]
fn test_stochastic_round_negative_fractional_distribution() {
    let iterations = 10000;
    let mut neg_ones = 0;
    let mut zeros = 0;

    for _ in 0..iterations {
        let result = stochastic_round(-0.7);
        match result {
            0 => zeros += 1,
            -1 => neg_ones += 1,
            _ => panic!("Unexpected result {} for stochastic_round(-0.7)", result),
        }
    }

    // -0.7 should give ~70% -1s, ~30% 0s
    let neg_ones_ratio = neg_ones as f64 / iterations as f64;
    let zeros_ratio = zeros as f64 / iterations as f64;

    // Allow 5% tolerance for statistical variation
    assert!(
        (neg_ones_ratio - 0.7).abs() < 0.05,
        "Expected ~70% negative ones, got {:.1}%",
        neg_ones_ratio * 100.0
    );
    assert!(
        (zeros_ratio - 0.3).abs() < 0.05,
        "Expected ~30% zeros, got {:.1}%",
        zeros_ratio * 100.0
    );
}

/// Test stochastic rounding for larger positive values.
/// 100.9 should round to 101 approximately 90% of the time and 100 approximately 10% of the time.
#[test]
fn test_stochastic_round_large_positive() {
    let iterations = 10000;
    let mut count_101 = 0;
    let mut count_100 = 0;

    for _ in 0..iterations {
        let result = stochastic_round(100.9);
        match result {
            100 => count_100 += 1,
            101 => count_101 += 1,
            _ => panic!("Unexpected result {} for stochastic_round(100.9)", result),
        }
    }

    // 100.9 should give ~90% 101s
    let ratio_101 = count_101 as f64 / iterations as f64;

    // Allow 5% tolerance for statistical variation
    assert!(
        (ratio_101 - 0.9).abs() < 0.05,
        "Expected ~90% 101s, got {:.1}%",
        ratio_101 * 100.0
    );
}

/// Test stochastic rounding for larger negative values.
/// -100.1 should round to -101 approximately 10% of the time and -100 approximately 90% of the time.
#[test]
fn test_stochastic_round_large_negative() {
    let iterations = 10000;
    let mut count_neg_101 = 0;
    let mut count_neg_100 = 0;

    for _ in 0..iterations {
        let result = stochastic_round(-100.1);
        match result {
            -100 => count_neg_100 += 1,
            -101 => count_neg_101 += 1,
            _ => panic!("Unexpected result {} for stochastic_round(-100.1)", result),
        }
    }

    // -100.1 should give ~10% -101s
    let ratio_neg_101 = count_neg_101 as f64 / iterations as f64;

    // Allow 5% tolerance for statistical variation
    assert!(
        (ratio_neg_101 - 0.1).abs() < 0.05,
        "Expected ~10% -101s, got {:.1}%",
        ratio_neg_101 * 100.0
    );
}

/// Test that random_weight returns values within the expected range.
#[test]
fn test_random_weight_in_range() {
    for _ in 0..1000 {
        let weight = random_weight();
        assert!(
            weight >= WEIGHT_INIT_MIN as i16 && weight <= WEIGHT_INIT_MAX as i16,
            "Weight {} outside range [{}, {}]",
            weight,
            WEIGHT_INIT_MIN,
            WEIGHT_INIT_MAX
        );
    }
}

/// Test that random_weight produces varied values (not always the same).
#[test]
fn test_random_weight_distribution() {
    let mut seen_values = std::collections::HashSet::new();

    for _ in 0..1000 {
        seen_values.insert(random_weight());
    }

    // We should see many different values over 1000 iterations
    // With range [-50, 50] (101 possible values), we expect good coverage
    assert!(
        seen_values.len() > 50,
        "Expected varied random weights, but only saw {} unique values",
        seen_values.len()
    );
}
