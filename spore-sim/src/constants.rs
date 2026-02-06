//! Constants for the SWARM V2 neural network simulation.
//!
//! V2 Architecture: N x (8→4→1) Spores with per-bit credit assignment.
//! Each Spore: 8 inputs → 4 hidden (hard threshold) → 1 output (hard threshold)
//! Weights are f32. Activations are binary.

// =============================================================================
// Network Topology (per Spore)
// =============================================================================

/// Number of input neurons per Spore (all Spores see the full input).
pub const INPUT_SIZE: usize = 8;

/// Number of hidden neurons per Spore.
/// Fixed at 4 for the "hydrogen atom" topology. Phase 3 may make configurable.
pub const HIDDEN_SIZE: usize = 4;

// Note: OUTPUT per Spore is always 1 (hardcoded, not a constant needed).
// The Swarm's N determines total output bits.

// =============================================================================
// Weight Initialization
// =============================================================================

/// Range for random weight initialization: uniform in [-WEIGHT_INIT_RANGE, +WEIGHT_INIT_RANGE].
pub const WEIGHT_INIT_RANGE: f32 = 0.5;

/// Initial bias for all neurons. Positive so neurons fire from tick 1,
/// giving traces for learning. Cortisol carves away wrong firings.
/// Without this, neurons with unlucky random weights never fire,
/// never set traces, and learn() does weight += LR * reward * 0.0 forever.
pub const INITIAL_BIAS: f32 = 0.5;

// =============================================================================
// Learning Parameters
// =============================================================================

/// Baseline accuracy for frustration calculation (chance level for 1-bit output).
pub const BASELINE_ACCURACY: f32 = 0.5;

/// Default learning rate for Hebbian weight updates.
pub const DEFAULT_LEARNING_RATE: f32 = 0.1;

/// Default eligibility trace decay per tick.
/// Exponentially sensitive — small changes have large effects.
pub const DEFAULT_TRACE_DECAY: f32 = 0.9;

/// Default base noise level for spontaneous firing.
pub const DEFAULT_BASE_NOISE: f32 = 0.001;

/// Default maximum noise boost at full frustration.
pub const DEFAULT_MAX_NOISE_BOOST: f32 = 0.05;

/// Default cortisol strength (symmetric punishment).
/// Dopamine is always +1.0; cortisol is 1.0 (symmetric at chance).
/// This is the mathematically correct value for zero-mean reward at 50%
/// accuracy: E[d] = 0.5*(+1) + 0.5*(-1) = 0. Only predictive inputs
/// get net positive weight change. Safe because accuracy-gated LR
/// prevents catastrophic forgetting at high accuracy.
pub const DEFAULT_CORTISOL_STRENGTH: f32 = 1.0;

// =============================================================================
// Homeostatic Regulation
// =============================================================================

/// Default interval (in ticks) between weight decay applications.
/// Every N ticks, weights *= 0.99.
pub const DEFAULT_WEIGHT_DECAY_INTERVAL: u64 = 100;

/// Target firing rate for activity homeostasis (10%).
/// If a Spore fires less than this, bias_o is nudged up; more, nudged down.
/// Prevents Dead Neuron problem: no Spore can stay silent forever.
pub const DEFAULT_TARGET_RATE: f32 = 0.1;

/// Rate at which homeostasis adjusts bias_o toward target firing rate.
/// bias_o += homeostasis_rate * (target_rate - firing_rate) each tick.
pub const DEFAULT_HOMEOSTASIS_RATE: f32 = 0.01;

// =============================================================================
// Environment / Trial Parameters
// =============================================================================

/// Default number of ticks to hold each input pattern.
pub const DEFAULT_INPUT_HOLD_TICKS: u64 = 50;

/// Default number of output bits (Spores) for the mirror task.
pub const DEFAULT_SWARM_SIZE: usize = 32;

// =============================================================================
// Frustration
// =============================================================================

/// Default EMA alpha for frustration updates (>50% accuracy).
pub const DEFAULT_FRUSTRATION_ALPHA: f32 = 0.2;

// =============================================================================
// Rejuvenation
// =============================================================================

/// If a Spore's recent_accuracy stays below this for REJUVENATION_GRACE_TICKS, reset it.
pub const REJUVENATION_THRESHOLD: f32 = 0.55;

/// Minimum ticks alive before a Spore can be rejuvenated.
pub const REJUVENATION_GRACE_TICKS: u64 = 10_000;
