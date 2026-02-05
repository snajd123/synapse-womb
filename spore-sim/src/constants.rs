//! Constants for the SPORE neural network simulation.
//!
//! This module defines all constants used throughout the simulation, including:
//! - Network topology (layer sizes)
//! - Neuron thresholds
//! - Pipeline timing
//! - Homeostatic regulation parameters
//! - Learning parameters
//! - Environment/trial parameters
//! - Weight initialization ranges
//!
//! Many constants are designed to address specific failure modes identified in the
//! original design. These are annotated with "Fix N" references:
//!
//! - **Fix 1**: Proper initialization (thresholds, weights)
//! - **Fix 2**: Timing constraints (pipeline latency, input hold duration)
//! - **Fix 3**: Homeostatic regulation (weight sum limits, decay intervals)
//! - **Fix 4**: Exploration noise (base noise, adaptive noise boost)
//! - **Fix 5**: Trace decay tuning (proper decay rate for credit assignment)

// =============================================================================
// Network Topology
// =============================================================================

/// Number of input neurons.
///
/// The input layer receives binary patterns from the environment.
/// With 8 inputs, we can represent 256 distinct input patterns (2^8).
pub const INPUT_SIZE: usize = 8;

/// Number of hidden layer neurons.
///
/// The hidden layer provides the computational capacity for learning.
/// A 4:1 ratio (32 hidden for 8 inputs) provides sufficient representational
/// power while keeping the network tractable for analysis.
pub const HIDDEN_SIZE: usize = 32;

/// Number of output neurons.
///
/// The output layer produces the network's decision/action.
/// Matching the input size (8) allows for tasks like pattern matching,
/// autoencoding, or classification into 8 categories.
pub const OUTPUT_SIZE: usize = 8;

// =============================================================================
// Neuron Thresholds
// =============================================================================

/// Default firing threshold for neurons during normal operation.
///
/// A neuron fires when its membrane potential exceeds this threshold.
/// Value of 50 is chosen to work well with weight range [-50, 50],
/// requiring multiple correlated inputs to cause firing.
///
/// **Fix 1**: This threshold ensures neurons don't fire too easily or too rarely.
pub const DEFAULT_THRESHOLD: i32 = 50;

/// Initial threshold for neurons during weight initialization phase.
///
/// Setting to 0 during initialization allows neurons to fire more easily,
/// enabling initial exploration of the weight space before settling into
/// learned patterns.
///
/// **Fix 1**: Zero threshold during init prevents "dead neuron" problem where
/// neurons never fire because random weights can't overcome high threshold.
pub const INIT_THRESHOLD: i32 = 0;

// =============================================================================
// Pipeline Timing
// =============================================================================

/// Pipeline latency in ticks (input -> hidden -> output).
///
/// This represents the number of simulation ticks required for a signal
/// to propagate from input through hidden to output layers:
/// - Tick 0: Input applied
/// - Tick 1: Hidden layer processes
/// - Tick 2: Output layer produces result
///
/// **Fix 2**: Understanding pipeline latency is critical for proper credit
/// assignment. Rewards must be delayed appropriately so that eligibility
/// traces connect actions to their consequences.
pub const PIPELINE_LATENCY: usize = 2;

// =============================================================================
// Homeostatic Regulation
// =============================================================================

/// Maximum allowed sum of absolute weights for any neuron.
///
/// Homeostatic regulation prevents runaway weight growth by constraining
/// the total "budget" of synaptic strength per neuron. When a neuron's
/// weight sum exceeds this limit, all weights are scaled down proportionally.
///
/// **Fix 3**: Without this constraint, positive feedback loops can cause
/// weights to grow unboundedly, leading to saturated neurons that always
/// or never fire regardless of input.
pub const MAX_WEIGHT_SUM: i32 = 400;

/// Interval (in ticks) between homeostatic weight decay applications.
///
/// Every N ticks, a small decay is applied to all weights, providing a
/// "forgetting" mechanism that prevents indefinite accumulation and allows
/// the network to adapt to changing reward contingencies.
///
/// **Fix 3**: Regular decay ensures that unused or weakly-used synapses
/// gradually return to baseline, freeing up weight budget for active learning.
pub const WEIGHT_DECAY_INTERVAL: usize = 100;

// =============================================================================
// Learning Parameters
// =============================================================================

/// Baseline accuracy for performance comparison.
///
/// This represents chance-level performance (50% for binary classification).
/// The network's performance is measured against this baseline to determine
/// whether it's learning effectively.
pub const BASELINE_ACCURACY: f64 = 0.5;

/// Default learning rate for SPORE weight updates.
///
/// Controls the magnitude of weight changes during learning. A value of 0.5
/// provides a balance between:
/// - Fast enough learning to make progress
/// - Slow enough to avoid oscillation and instability
///
/// Weight update magnitude = learning_rate * eligibility_trace * reward_signal
pub const DEFAULT_LEARNING_RATE: f64 = 0.5;

/// Default decay rate for eligibility traces.
///
/// Eligibility traces track recent pre-post spike coincidences and decay
/// exponentially each tick. A decay of 0.9 means traces retain 90% of their
/// value per tick.
///
/// **Fix 5**: The trace decay rate determines the "credit assignment window".
/// With decay=0.9:
/// - After 2 ticks (pipeline latency): trace = 0.81 (still strong)
/// - After 10 ticks: trace = 0.35 (moderate)
/// - After 20 ticks: trace = 0.12 (weak but present)
///
/// This allows rewards delivered shortly after actions to properly
/// strengthen the synapses that caused those actions.
pub const DEFAULT_TRACE_DECAY: f64 = 0.9;

/// Default base noise level for synaptic transmission.
///
/// Small random perturbations are added to weight calculations to enable
/// exploration. This base level (0.1%) provides minimal exploration when
/// the network is performing well.
///
/// **Fix 4**: Base noise prevents the network from getting stuck in local
/// optima by occasionally causing different neurons to fire.
pub const DEFAULT_BASE_NOISE: f64 = 0.001;

/// Maximum noise boost when performance is poor.
///
/// When accuracy drops below baseline, noise is increased to encourage
/// exploration. This maximum (5%) is added on top of base noise when
/// the network is performing at its worst.
///
/// **Fix 4**: Adaptive noise implements "frustration-driven exploration" -
/// when things aren't working, try something different.
pub const DEFAULT_MAX_NOISE_BOOST: f64 = 0.05;

// =============================================================================
// Environment / Trial Parameters
// =============================================================================

/// Default number of ticks to hold each input pattern.
///
/// Each trial presents an input pattern for this many ticks before
/// switching to the next pattern. This duration must be long enough for:
/// 1. Signal propagation through the pipeline
/// 2. Reward delivery and processing
/// 3. Sufficient learning signal accumulation
///
/// **Fix 2**: The formula `input_hold >= 2 * (pipeline_latency + reward_latency) + 10`
/// ensures enough time for the full learning cycle. With defaults (latency=2, reward=0),
/// minimum is 14 ticks; we use 50 for comfortable margin.
pub const DEFAULT_INPUT_HOLD_TICKS: usize = 50;

/// Default latency (in ticks) before reward is delivered after output.
///
/// In biological systems, rewards are often delayed. This parameter allows
/// simulation of delayed reinforcement. A value of 0 means immediate reward
/// (delivered same tick as output evaluation).
///
/// **Fix 2**: Reward latency must be accounted for in trace decay calculations.
/// Higher latency requires slower trace decay to maintain credit assignment.
pub const DEFAULT_REWARD_LATENCY: usize = 0;

// =============================================================================
// Weight Initialization
// =============================================================================

/// Minimum value for random weight initialization.
///
/// Weights are initialized uniformly in [WEIGHT_INIT_MIN, WEIGHT_INIT_MAX].
/// Symmetric range around 0 prevents initial bias toward excitation or inhibition.
///
/// **Fix 1**: Proper weight initialization range ensures neurons can potentially
/// fire (when weights align) but don't trivially fire on any input.
pub const WEIGHT_INIT_MIN: i32 = -50;

/// Maximum value for random weight initialization.
///
/// Matches the magnitude of DEFAULT_THRESHOLD, so a single strong synapse
/// could potentially trigger firing, while typical random combinations won't.
///
/// **Fix 1**: Symmetric with WEIGHT_INIT_MIN for balanced initial exploration.
pub const WEIGHT_INIT_MAX: i32 = 50;

// =============================================================================
// Helper Functions
// =============================================================================

/// Calculate minimum input hold ticks for a given reward latency.
///
/// This implements the timing constraint from Fix 2:
/// ```text
/// input_hold_ticks >= 2 * (PIPELINE_LATENCY + reward_latency) + 10
/// ```
///
/// The formula ensures:
/// - One full forward pass (pipeline_latency)
/// - Reward delivery delay (reward_latency)
/// - Another full pass to apply learning
/// - 10 tick margin for stability
///
/// # Arguments
/// * `reward_latency` - The delay between output and reward delivery
///
/// # Returns
/// The minimum safe value for input_hold_ticks
///
/// # Examples
/// ```
/// use spore_sim::constants::min_input_hold_ticks;
///
/// assert_eq!(min_input_hold_ticks(0), 14);  // 2*(2+0)+10
/// assert_eq!(min_input_hold_ticks(5), 24);  // 2*(2+5)+10
/// ```
pub fn min_input_hold_ticks(reward_latency: usize) -> usize {
    2 * (PIPELINE_LATENCY + reward_latency) + 10
}

/// Calculate recommended trace decay for a given reward latency.
///
/// This implements the guidance from Fix 5: the trace decay should be chosen
/// so that traces remain significant (> 0.5) when rewards arrive.
///
/// The formula ensures:
/// ```text
/// decay^(PIPELINE_LATENCY + reward_latency) > 0.5
/// ```
///
/// Solving for decay:
/// ```text
/// decay > 0.5^(1/(PIPELINE_LATENCY + reward_latency))
/// ```
///
/// We add a small margin (0.05) to ensure robust credit assignment.
///
/// # Arguments
/// * `reward_latency` - The delay between output and reward delivery
///
/// # Returns
/// Recommended trace decay value (between 0 and 1)
///
/// # Examples
/// ```
/// use spore_sim::constants::recommended_trace_decay;
///
/// let decay = recommended_trace_decay(0);
/// assert!(decay > 0.7 && decay < 0.95);
/// ```
pub fn recommended_trace_decay(reward_latency: usize) -> f64 {
    let total_latency = (PIPELINE_LATENCY + reward_latency) as f64;
    // We want decay^total_latency > 0.5
    // So decay > 0.5^(1/total_latency)
    // Add margin of 0.05 for robustness
    let min_decay = 0.5_f64.powf(1.0 / total_latency);
    (min_decay + 0.05).min(0.99) // Cap at 0.99 to ensure some decay occurs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_dimensions_consistent() {
        // Sanity check: dimensions should be positive
        assert!(INPUT_SIZE > 0);
        assert!(HIDDEN_SIZE > 0);
        assert!(OUTPUT_SIZE > 0);
    }

    #[test]
    fn test_weight_range_symmetric() {
        // Weight range should be symmetric around 0
        assert_eq!(WEIGHT_INIT_MIN, -WEIGHT_INIT_MAX);
    }

    #[test]
    fn test_default_timing_satisfies_constraint() {
        // Default input hold should satisfy minimum constraint
        let min_required = min_input_hold_ticks(DEFAULT_REWARD_LATENCY);
        assert!(DEFAULT_INPUT_HOLD_TICKS >= min_required);
    }

    #[test]
    fn test_default_trace_decay_adequate() {
        // Default trace decay should be at least as good as recommended
        let recommended = recommended_trace_decay(DEFAULT_REWARD_LATENCY);
        assert!(DEFAULT_TRACE_DECAY >= recommended);
    }
}
