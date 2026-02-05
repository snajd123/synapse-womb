//! Utility functions for the SPORE neural network simulation.
//!
//! This module provides critical utility functions for probabilistic operations
//! in the network, particularly stochastic rounding which is essential for
//! preventing small weight updates from truncating to zero.

use crate::constants::{WEIGHT_INIT_MAX, WEIGHT_INIT_MIN};
use rand::Rng;

/// Stochastically rounds a floating-point value to an integer.
///
/// This function implements probabilistic rounding where the probability of
/// rounding up equals the fractional part of the value. This is CRITICAL for
/// Hebbian learning where small weight updates (e.g., 0.1) would otherwise
/// truncate to 0, effectively freezing the network.
///
/// # Algorithm
///
/// For a value like 2.3:
/// - Floor is 2, ceiling is 3
/// - Fractional part is 0.3
/// - Returns 3 with probability 0.3, returns 2 with probability 0.7
///
/// For negative values like -2.3:
/// - Floor is -3, ceiling is -2
/// - Fractional part (distance from floor) is 0.7
/// - Returns -2 with probability 0.7, returns -3 with probability 0.3
///
/// # Arguments
///
/// * `value` - The floating-point value to round
///
/// # Returns
///
/// An i16 integer that is either the floor or ceiling of the input value,
/// chosen probabilistically based on the fractional part.
///
/// # Examples
///
/// ```
/// use spore_sim::utils::stochastic_round;
///
/// // Integer values always round to themselves
/// assert_eq!(stochastic_round(5.0), 5);
/// assert_eq!(stochastic_round(-3.0), -3);
///
/// // Fractional values round probabilistically
/// // Over many trials, 2.3 should give ~30% 3s and ~70% 2s
/// let result = stochastic_round(2.3);
/// assert!(result == 2 || result == 3);
/// ```
pub fn stochastic_round(value: f32) -> i16 {
    let floor = value.floor();
    let frac = value - floor;

    // Generate random number in [0, 1)
    let mut rng = rand::thread_rng();
    let random: f32 = rng.gen();

    // Round up if random < fractional part
    if random < frac {
        (floor as i16) + 1
    } else {
        floor as i16
    }
}

/// Generates a random weight value within the initialization range.
///
/// Returns a random integer in the range [WEIGHT_INIT_MIN, WEIGHT_INIT_MAX]
/// (inclusive), using the constants defined in constants.rs.
///
/// # Returns
///
/// A random i16 weight value suitable for network initialization.
///
/// # Examples
///
/// ```
/// use spore_sim::utils::random_weight;
/// use spore_sim::constants::{WEIGHT_INIT_MIN, WEIGHT_INIT_MAX};
///
/// let weight = random_weight();
/// assert!(weight >= WEIGHT_INIT_MIN as i16);
/// assert!(weight <= WEIGHT_INIT_MAX as i16);
/// ```
pub fn random_weight() -> i16 {
    let mut rng = rand::thread_rng();
    rng.gen_range(WEIGHT_INIT_MIN as i16..=WEIGHT_INIT_MAX as i16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that random_weight produces values within the valid range.
    #[test]
    fn test_random_weight_range() {
        for _ in 0..100 {
            let weight = random_weight();
            assert!(
                weight >= WEIGHT_INIT_MIN as i16 && weight <= WEIGHT_INIT_MAX as i16,
                "Weight {} out of range [{}, {}]",
                weight,
                WEIGHT_INIT_MIN,
                WEIGHT_INIT_MAX
            );
        }
    }

    /// Test that random_weight produces varied values.
    #[test]
    fn test_random_weight_variability() {
        let mut values = std::collections::HashSet::new();
        for _ in 0..500 {
            values.insert(random_weight());
        }
        // Should see significant variety
        assert!(
            values.len() > 30,
            "Expected varied weights, only got {} unique values",
            values.len()
        );
    }
}
