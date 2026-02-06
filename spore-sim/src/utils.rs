//! Utility functions for the SWARM V2 neural network simulation.

use rand::Rng;
use crate::constants::WEIGHT_INIT_RANGE;

/// Generate a random f32 weight in [-WEIGHT_INIT_RANGE, +WEIGHT_INIT_RANGE].
pub fn random_weight() -> f32 {
    let mut rng = rand::thread_rng();
    rng.gen_range(-WEIGHT_INIT_RANGE..=WEIGHT_INIT_RANGE)
}
