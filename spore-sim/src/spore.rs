//! The Spore: A neural substrate that learns through Hebbian reinforcement.
//!
//! An 8-32-8 network topology with:
//! - Hard threshold activation with noise injection
//! - Eligibility traces for temporal credit assignment
//! - Signed dopamine for reward/punishment
//! - Homeostasis to prevent runaway

use crate::constants::*;
use crate::utils::random_weight;

/// The Spore neural network.
///
/// A pipelined 8-32-8 network that learns to copy input to output
/// using only Hebbian learning and dopamine reinforcement.
#[derive(Debug, Clone)]
pub struct Spore {
    // ========================================================================
    // WEIGHTS (learnable)
    // ========================================================================

    /// Input → Hidden weights. Shape: [HIDDEN_SIZE][INPUT_SIZE] = [32][8]
    pub weights_ih: [[i16; INPUT_SIZE]; HIDDEN_SIZE],

    /// Hidden → Output weights. Shape: [OUTPUT_SIZE][HIDDEN_SIZE] = [8][32]
    pub weights_ho: [[i16; HIDDEN_SIZE]; OUTPUT_SIZE],

    // ========================================================================
    // THRESHOLDS (learnable)
    // ========================================================================

    /// Per-neuron firing thresholds for hidden layer
    pub thresholds_h: [i16; HIDDEN_SIZE],

    /// Per-neuron firing thresholds for output layer
    pub thresholds_o: [i16; OUTPUT_SIZE],

    // ========================================================================
    // ELIGIBILITY TRACES
    // ========================================================================

    /// Traces for Input → Hidden synapses
    pub traces_ih: [[f32; INPUT_SIZE]; HIDDEN_SIZE],

    /// Traces for Hidden → Output synapses
    pub traces_ho: [[f32; HIDDEN_SIZE]; OUTPUT_SIZE],

    /// Traces for hidden thresholds
    pub traces_th: [f32; HIDDEN_SIZE],

    /// Traces for output thresholds
    pub traces_to: [f32; OUTPUT_SIZE],

    // ========================================================================
    // NEURON STATE (double-buffered for pipeline)
    // ========================================================================

    /// Hidden layer activations from LAST tick
    pub hidden: [bool; HIDDEN_SIZE],

    /// Hidden layer activations being computed THIS tick
    pub hidden_next: [bool; HIDDEN_SIZE],

    /// Output layer activations from LAST tick
    pub output: [bool; OUTPUT_SIZE],

    /// Output layer activations being computed THIS tick
    pub output_next: [bool; OUTPUT_SIZE],

    // ========================================================================
    // LEARNING STATE
    // ========================================================================

    /// Current dopamine level (can be negative for punishment)
    pub dopamine: f32,

    /// Frustration level (drives exploration via noise injection)
    /// 0.0 = calm (succeeding), 1.0 = frantic (failing)
    pub frustration: f32,

    // ========================================================================
    // HYPERPARAMETERS
    // ========================================================================

    /// Learning rate (scaled by 100x internally)
    pub learning_rate: f32,

    /// Trace decay per tick
    pub trace_decay: f32,

    /// Base spontaneous firing rate
    pub base_noise: f32,

    /// Maximum noise boost at full frustration
    pub max_noise_boost: f32,
}

impl Spore {
    /// Create a new Spore with random weights and default hyperparameters.
    ///
    /// Thresholds are initialized to 0 (Fix 5: "trigger happy" bootstrap)
    /// to ensure neurons fire at birth and generate traces for learning.
    pub fn new() -> Self {
        Self::with_params(
            DEFAULT_LEARNING_RATE as f32,
            DEFAULT_TRACE_DECAY as f32,
            DEFAULT_BASE_NOISE as f32,
            DEFAULT_MAX_NOISE_BOOST as f32,
        )
    }

    /// Create a new Spore with custom hyperparameters.
    pub fn with_params(
        learning_rate: f32,
        trace_decay: f32,
        base_noise: f32,
        max_noise_boost: f32,
    ) -> Self {
        // Initialize weights randomly
        let mut weights_ih = [[0i16; INPUT_SIZE]; HIDDEN_SIZE];
        let mut weights_ho = [[0i16; HIDDEN_SIZE]; OUTPUT_SIZE];

        for h in 0..HIDDEN_SIZE {
            for i in 0..INPUT_SIZE {
                weights_ih[h][i] = random_weight();
            }
        }

        for o in 0..OUTPUT_SIZE {
            for h in 0..HIDDEN_SIZE {
                weights_ho[o][h] = random_weight();
            }
        }

        Self {
            weights_ih,
            weights_ho,

            // Fix 5: Initialize thresholds to 0, NOT DEFAULT_THRESHOLD
            thresholds_h: [INIT_THRESHOLD as i16; HIDDEN_SIZE],
            thresholds_o: [INIT_THRESHOLD as i16; OUTPUT_SIZE],

            // All traces start at 0
            traces_ih: [[0.0; INPUT_SIZE]; HIDDEN_SIZE],
            traces_ho: [[0.0; HIDDEN_SIZE]; OUTPUT_SIZE],
            traces_th: [0.0; HIDDEN_SIZE],
            traces_to: [0.0; OUTPUT_SIZE],

            // All activations start false
            hidden: [false; HIDDEN_SIZE],
            hidden_next: [false; HIDDEN_SIZE],
            output: [false; OUTPUT_SIZE],
            output_next: [false; OUTPUT_SIZE],

            // Learning state
            dopamine: 0.0,
            frustration: 1.0,  // Start fully frustrated (exploring)

            // Hyperparameters
            learning_rate,
            trace_decay,
            base_noise,
            max_noise_boost,
        }
    }
}

impl Default for Spore {
    fn default() -> Self {
        Self::new()
    }
}
