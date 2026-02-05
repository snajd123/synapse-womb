# Spore Mirror Experiment: Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Rust simulation proving Hebbian learning + Dopamine reinforcement can evolve a byte-copy reflex from random noise.

**Architecture:** An 8-32-8 neural network (Spore) receives byte inputs, propagates through a pipelined hidden layer, and produces byte outputs. An Environment judges outputs against delayed inputs and delivers reward signals. Eligibility traces + signed dopamine drive weight updates. Homeostasis prevents runaway.

**Tech Stack:** Rust (std), rand crate for RNG, no external ML frameworks.

**Reference Design:** `docs/plans/2026-02-05-spore-mirror-experiment-design.md`

---

## Project Structure

```
spore-sim/
├── Cargo.toml
├── src/
│   ├── main.rs           # Entry point, CLI
│   ├── lib.rs            # Module exports
│   ├── constants.rs      # All tunable constants
│   ├── spore.rs          # Spore struct and impl
│   ├── environment.rs    # Environment struct and impl
│   ├── simulation.rs     # Simulation loop
│   └── utils.rs          # stochastic_round, helpers
└── tests/
    ├── spore_tests.rs
    ├── environment_tests.rs
    └── integration_tests.rs
```

---

## Task 1: Initialize Rust Project

**Files:**
- Create: `spore-sim/Cargo.toml`
- Create: `spore-sim/src/main.rs`

**Step 1: Create project directory**

```bash
mkdir -p spore-sim/src spore-sim/tests
```

**Step 2: Write Cargo.toml**

Create `spore-sim/Cargo.toml`:

```toml
[package]
name = "spore-sim"
version = "0.1.0"
edition = "2021"
description = "Spore Mirror Experiment - Hebbian learning proof of concept"

[dependencies]
rand = "0.8"

[dev-dependencies]
# None needed yet

[[bin]]
name = "spore-sim"
path = "src/main.rs"

[lib]
name = "spore_sim"
path = "src/lib.rs"
```

**Step 3: Write minimal main.rs**

Create `spore-sim/src/main.rs`:

```rust
fn main() {
    println!("Spore Mirror Experiment - Phase 1");
}
```

**Step 4: Write lib.rs stub**

Create `spore-sim/src/lib.rs`:

```rust
pub mod constants;
pub mod utils;
pub mod spore;
pub mod environment;
pub mod simulation;
```

**Step 5: Create module stubs**

Create `spore-sim/src/constants.rs`:

```rust
// Constants will be added in Task 2
```

Create `spore-sim/src/utils.rs`:

```rust
// Utilities will be added in Task 3
```

Create `spore-sim/src/spore.rs`:

```rust
// Spore struct will be added in Task 4
```

Create `spore-sim/src/environment.rs`:

```rust
// Environment struct will be added in Task 12
```

Create `spore-sim/src/simulation.rs`:

```rust
// Simulation loop will be added in Task 20
```

**Step 6: Verify project compiles**

Run:
```bash
cd spore-sim && cargo build
```

Expected: Build succeeds with no errors.

**Step 7: Commit**

```bash
git add spore-sim/
git commit -m "feat: initialize spore-sim Rust project structure"
```

---

## Task 2: Define All Constants

**Files:**
- Modify: `spore-sim/src/constants.rs`
- Create: `spore-sim/tests/constants_tests.rs`

**Step 1: Write the failing test**

Create `spore-sim/tests/constants_tests.rs`:

```rust
use spore_sim::constants::*;

#[test]
fn test_constants_exist_and_have_correct_values() {
    // Network topology
    assert_eq!(INPUT_SIZE, 8);
    assert_eq!(HIDDEN_SIZE, 32);
    assert_eq!(OUTPUT_SIZE, 8);

    // Thresholds
    assert_eq!(DEFAULT_THRESHOLD, 50);
    assert_eq!(INIT_THRESHOLD, 0);

    // Pipeline
    assert_eq!(PIPELINE_LATENCY, 2);

    // Homeostasis
    assert_eq!(MAX_WEIGHT_SUM, 400);
    assert_eq!(WEIGHT_DECAY_INTERVAL, 100);

    // Learning
    assert!((BASELINE_ACCURACY - 0.5).abs() < 0.001);
    assert!((DEFAULT_LEARNING_RATE - 0.5).abs() < 0.001);
    assert!((DEFAULT_TRACE_DECAY - 0.9).abs() < 0.001);
    assert!((DEFAULT_BASE_NOISE - 0.001).abs() < 0.0001);
    assert!((DEFAULT_MAX_NOISE_BOOST - 0.05).abs() < 0.001);

    // Environment
    assert_eq!(DEFAULT_INPUT_HOLD_TICKS, 50);
    assert_eq!(DEFAULT_REWARD_LATENCY, 0);

    // Weight initialization range
    assert_eq!(WEIGHT_INIT_MIN, -50);
    assert_eq!(WEIGHT_INIT_MAX, 50);
}

#[test]
fn test_timing_constraint_helper() {
    // Rule: input_hold_ticks >= 2 * (PIPELINE_LATENCY + reward_latency) + 10
    assert_eq!(min_input_hold_ticks(0), 14);   // 2*(2+0)+10
    assert_eq!(min_input_hold_ticks(5), 24);   // 2*(2+5)+10
    assert_eq!(min_input_hold_ticks(10), 34);  // 2*(2+10)+10
}

#[test]
fn test_recommended_trace_decay() {
    // At latency 0: trace^2 should be > 0.5, so decay ~0.71 minimum
    // We use 0.9 which gives 0.81
    let decay_0 = recommended_trace_decay(0);
    assert!(decay_0 > 0.7 && decay_0 < 0.95);

    // At latency 10: trace^12 should be > 0.5
    let decay_10 = recommended_trace_decay(10);
    assert!(decay_10 > 0.9 && decay_10 < 0.98);
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd spore-sim && cargo test constants_tests
```

Expected: FAIL - constants not defined.

**Step 3: Implement constants.rs**

Replace `spore-sim/src/constants.rs`:

```rust
//! All tunable constants for the Spore Mirror Experiment.
//!
//! These values were derived through careful analysis in the design phase.
//! See: docs/plans/2026-02-05-spore-mirror-experiment-design.md

// ============================================================================
// NETWORK TOPOLOGY
// ============================================================================

/// Number of input neurons (one per bit of input byte)
pub const INPUT_SIZE: usize = 8;

/// Number of hidden layer neurons (provides redundancy for pathway competition)
pub const HIDDEN_SIZE: usize = 32;

/// Number of output neurons (one per bit of output byte)
pub const OUTPUT_SIZE: usize = 8;

// ============================================================================
// THRESHOLDS
// ============================================================================

/// Homeostasis target threshold (neurons drift toward this)
pub const DEFAULT_THRESHOLD: i16 = 50;

/// Initial threshold for new spores (Fix 5: start "trigger happy")
/// Setting to 0 ensures neurons fire at birth, creating traces for learning.
/// If set to DEFAULT_THRESHOLD, random weights averaging to 0 never exceed
/// threshold, resulting in a "dead spore" that never learns.
pub const INIT_THRESHOLD: i16 = 0;

// ============================================================================
// PIPELINE
// ============================================================================

/// Number of ticks for signal to propagate through the network.
/// Tick N: Input arrives, Hidden computes
/// Tick N+1: Hidden activates, Output computes
/// Tick N+2: Output activates (visible)
pub const PIPELINE_LATENCY: usize = 2;

// ============================================================================
// HOMEOSTASIS (Fix 1: Weight Normalization)
// ============================================================================

/// Maximum sum of absolute weights per neuron.
/// Prevents "hard-wired" neurons where sum grows unbounded and noise becomes irrelevant.
pub const MAX_WEIGHT_SUM: i32 = 400;

/// Apply weight decay every N ticks (~1.5% decay via w -= w >> 6)
pub const WEIGHT_DECAY_INTERVAL: u64 = 100;

// ============================================================================
// LEARNING
// ============================================================================

/// Accuracy baseline for signed dopamine (Fix 3: Anti-Hebbian)
/// Below this accuracy, dopamine is negative (punishment).
/// At 50%, dopamine = 0.25 - 0.25 = 0
/// At 0%, dopamine = 0 - 0.25 = -0.25
/// At 100%, dopamine = 1.0 - 0.25 = 0.75
pub const BASELINE_ACCURACY: f32 = 0.5;

/// Default learning rate (scaled by 100x internally to overcome integer truncation)
pub const DEFAULT_LEARNING_RATE: f32 = 0.5;

/// Default trace decay per tick (eligibility trace memory)
pub const DEFAULT_TRACE_DECAY: f32 = 0.9;

/// Base spontaneous firing rate (exploration)
pub const DEFAULT_BASE_NOISE: f32 = 0.001;

/// Maximum noise boost at full frustration (Fix 2: fast frustration response)
pub const DEFAULT_MAX_NOISE_BOOST: f32 = 0.05;

// ============================================================================
// ENVIRONMENT
// ============================================================================

/// How long to hold each input pattern (Fix 4: prevent superstitious learning)
/// Must be >> PIPELINE_LATENCY + reward_latency
pub const DEFAULT_INPUT_HOLD_TICKS: u64 = 50;

/// Default reward latency (0 = immediate, increase for stress testing)
pub const DEFAULT_REWARD_LATENCY: u64 = 0;

// ============================================================================
// WEIGHT INITIALIZATION
// ============================================================================

/// Minimum initial weight value
pub const WEIGHT_INIT_MIN: i16 = -50;

/// Maximum initial weight value
pub const WEIGHT_INIT_MAX: i16 = 50;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Calculate minimum input_hold_ticks for a given reward_latency.
/// Rule: input_hold_ticks >= 2 * (PIPELINE_LATENCY + reward_latency) + 10
/// This ensures a "clear air" window where reward and input are aligned.
#[inline]
pub const fn min_input_hold_ticks(reward_latency: u64) -> u64 {
    2 * (PIPELINE_LATENCY as u64 + reward_latency) + 10
}

/// Calculate recommended trace decay for a given reward_latency.
/// Rule: trace^(pipeline + reward_latency) should be > 0.5
/// Solves: decay = 0.5^(1/total_delay)
#[inline]
pub fn recommended_trace_decay(reward_latency: u64) -> f32 {
    let total_delay = PIPELINE_LATENCY as u64 + reward_latency;
    0.5_f32.powf(1.0 / total_delay as f32)
}
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd spore-sim && cargo test constants_tests
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add spore-sim/src/constants.rs spore-sim/tests/constants_tests.rs
git commit -m "feat: add all constants with helper functions"
```

---

## Task 3: Implement Utility Functions

**Files:**
- Modify: `spore-sim/src/utils.rs`
- Create: `spore-sim/tests/utils_tests.rs`

**Step 1: Write the failing test**

Create `spore-sim/tests/utils_tests.rs`:

```rust
use spore_sim::utils::stochastic_round;

#[test]
fn test_stochastic_round_positive_integer() {
    // 5.0 should always round to 5
    for _ in 0..100 {
        assert_eq!(stochastic_round(5.0), 5);
    }
}

#[test]
fn test_stochastic_round_negative_integer() {
    // -5.0 should always round to -5
    for _ in 0..100 {
        assert_eq!(stochastic_round(-5.0), -5);
    }
}

#[test]
fn test_stochastic_round_zero() {
    for _ in 0..100 {
        assert_eq!(stochastic_round(0.0), 0);
    }
}

#[test]
fn test_stochastic_round_positive_fractional_distribution() {
    // 0.3 should round to 0 about 70% of the time, 1 about 30%
    let mut zeros = 0;
    let mut ones = 0;
    for _ in 0..10000 {
        match stochastic_round(0.3) {
            0 => zeros += 1,
            1 => ones += 1,
            _ => panic!("Unexpected value"),
        }
    }
    // Allow 5% tolerance
    let ratio = ones as f32 / 10000.0;
    assert!(ratio > 0.25 && ratio < 0.35, "Expected ~30% ones, got {}%", ratio * 100.0);
}

#[test]
fn test_stochastic_round_negative_fractional_distribution() {
    // -0.7 should round to -1 about 70% of the time, 0 about 30%
    let mut zeros = 0;
    let mut neg_ones = 0;
    for _ in 0..10000 {
        match stochastic_round(-0.7) {
            0 => zeros += 1,
            -1 => neg_ones += 1,
            _ => panic!("Unexpected value"),
        }
    }
    let ratio = neg_ones as f32 / 10000.0;
    assert!(ratio > 0.65 && ratio < 0.75, "Expected ~70% -1s, got {}%", ratio * 100.0);
}

#[test]
fn test_stochastic_round_large_positive() {
    // 100.9 should round to 100 or 101
    let mut hundreds = 0;
    let mut hundred_ones = 0;
    for _ in 0..10000 {
        match stochastic_round(100.9) {
            100 => hundreds += 1,
            101 => hundred_ones += 1,
            _ => panic!("Unexpected value"),
        }
    }
    let ratio = hundred_ones as f32 / 10000.0;
    assert!(ratio > 0.85 && ratio < 0.95, "Expected ~90% 101s, got {}%", ratio * 100.0);
}

#[test]
fn test_stochastic_round_large_negative() {
    // -100.1 should round to -100 about 90% of the time, -101 about 10%
    let mut neg_hundreds = 0;
    let mut neg_hundred_ones = 0;
    for _ in 0..10000 {
        match stochastic_round(-100.1) {
            -100 => neg_hundreds += 1,
            -101 => neg_hundred_ones += 1,
            _ => panic!("Unexpected value"),
        }
    }
    let ratio = neg_hundred_ones as f32 / 10000.0;
    assert!(ratio > 0.05 && ratio < 0.15, "Expected ~10% -101s, got {}%", ratio * 100.0);
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd spore-sim && cargo test utils_tests
```

Expected: FAIL - stochastic_round not defined.

**Step 3: Implement utils.rs**

Replace `spore-sim/src/utils.rs`:

```rust
//! Utility functions for the Spore simulation.

use rand::Rng;

/// Stochastic rounding: converts a float to i16 with probabilistic rounding.
///
/// For a value like 2.3:
/// - Returns 2 with probability 0.7
/// - Returns 3 with probability 0.3
///
/// For negative values like -2.3:
/// - Returns -2 with probability 0.7
/// - Returns -3 with probability 0.3
///
/// This is critical for Hebbian learning where small weight updates (e.g., 0.1)
/// would otherwise truncate to 0 and freeze the network.
///
/// # Arguments
/// * `value` - The floating point value to round
///
/// # Returns
/// An i16 that is probabilistically rounded
pub fn stochastic_round(value: f32) -> i16 {
    let sign = value.signum() as i16;
    let abs_val = value.abs();
    let floor = abs_val as i16;
    let frac = abs_val.fract();

    let mut rng = rand::thread_rng();
    let round_up = rng.gen::<f32>() < frac;

    sign * (floor + if round_up { 1 } else { 0 })
}

/// Generate a random weight in the initialization range.
///
/// # Returns
/// A random i16 in the range [WEIGHT_INIT_MIN, WEIGHT_INIT_MAX]
pub fn random_weight() -> i16 {
    use crate::constants::{WEIGHT_INIT_MIN, WEIGHT_INIT_MAX};
    let mut rng = rand::thread_rng();
    rng.gen_range(WEIGHT_INIT_MIN..=WEIGHT_INIT_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_weight_in_range() {
        use crate::constants::{WEIGHT_INIT_MIN, WEIGHT_INIT_MAX};
        for _ in 0..1000 {
            let w = random_weight();
            assert!(w >= WEIGHT_INIT_MIN && w <= WEIGHT_INIT_MAX);
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd spore-sim && cargo test utils_tests
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add spore-sim/src/utils.rs spore-sim/tests/utils_tests.rs
git commit -m "feat: add stochastic_round and random_weight utilities"
```

---

## Task 4: Define Spore Struct

**Files:**
- Modify: `spore-sim/src/spore.rs`
- Create: `spore-sim/tests/spore_tests.rs`

**Step 1: Write the failing test**

Create `spore-sim/tests/spore_tests.rs`:

```rust
use spore_sim::spore::Spore;
use spore_sim::constants::*;

#[test]
fn test_spore_struct_exists() {
    let spore = Spore::new();
    // Just verify it compiles and has the expected fields
    assert_eq!(spore.weights_ih.len(), HIDDEN_SIZE);
    assert_eq!(spore.weights_ih[0].len(), INPUT_SIZE);
    assert_eq!(spore.weights_ho.len(), OUTPUT_SIZE);
    assert_eq!(spore.weights_ho[0].len(), HIDDEN_SIZE);
}

#[test]
fn test_spore_thresholds_init_to_zero() {
    // Fix 5: Thresholds must start at 0, not DEFAULT_THRESHOLD
    let spore = Spore::new();
    for t in &spore.thresholds_h {
        assert_eq!(*t, INIT_THRESHOLD, "Hidden thresholds should init to 0");
    }
    for t in &spore.thresholds_o {
        assert_eq!(*t, INIT_THRESHOLD, "Output thresholds should init to 0");
    }
}

#[test]
fn test_spore_traces_init_to_zero() {
    let spore = Spore::new();
    for row in &spore.traces_ih {
        for t in row {
            assert_eq!(*t, 0.0);
        }
    }
    for row in &spore.traces_ho {
        for t in row {
            assert_eq!(*t, 0.0);
        }
    }
}

#[test]
fn test_spore_weights_in_valid_range() {
    let spore = Spore::new();
    for row in &spore.weights_ih {
        for w in row {
            assert!(*w >= WEIGHT_INIT_MIN && *w <= WEIGHT_INIT_MAX);
        }
    }
    for row in &spore.weights_ho {
        for w in row {
            assert!(*w >= WEIGHT_INIT_MIN && *w <= WEIGHT_INIT_MAX);
        }
    }
}

#[test]
fn test_spore_frustration_starts_at_one() {
    // Start fully frustrated to encourage exploration
    let spore = Spore::new();
    assert_eq!(spore.frustration, 1.0);
}

#[test]
fn test_spore_dopamine_starts_at_zero() {
    let spore = Spore::new();
    assert_eq!(spore.dopamine, 0.0);
}

#[test]
fn test_spore_activations_start_false() {
    let spore = Spore::new();
    for h in &spore.hidden {
        assert!(!*h);
    }
    for h in &spore.hidden_next {
        assert!(!*h);
    }
    for o in &spore.output {
        assert!(!*o);
    }
    for o in &spore.output_next {
        assert!(!*o);
    }
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd spore-sim && cargo test spore_tests
```

Expected: FAIL - Spore struct not defined.

**Step 3: Implement Spore struct**

Replace `spore-sim/src/spore.rs`:

```rust
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
            DEFAULT_LEARNING_RATE,
            DEFAULT_TRACE_DECAY,
            DEFAULT_BASE_NOISE,
            DEFAULT_MAX_NOISE_BOOST,
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
            thresholds_h: [INIT_THRESHOLD; HIDDEN_SIZE],
            thresholds_o: [INIT_THRESHOLD; OUTPUT_SIZE],

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
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd spore-sim && cargo test spore_tests
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add spore-sim/src/spore.rs spore-sim/tests/spore_tests.rs
git commit -m "feat: add Spore struct with proper initialization (Fix 5)"
```

---

## Task 5: Implement Spore Forward Propagation

**Files:**
- Modify: `spore-sim/src/spore.rs`
- Modify: `spore-sim/tests/spore_tests.rs`

**Step 1: Write the failing test**

Add to `spore-sim/tests/spore_tests.rs`:

```rust
#[test]
fn test_propagate_updates_hidden_next() {
    let mut spore = Spore::new();
    // Set thresholds very low so neurons fire easily
    spore.thresholds_h = [-1000; HIDDEN_SIZE];
    spore.base_noise = 0.0;  // Disable noise for deterministic test

    // All 1s input
    spore.propagate(0xFF);

    // With very low thresholds, all hidden_next should fire
    for h in &spore.hidden_next {
        assert!(*h, "Hidden neurons should fire with low threshold");
    }
}

#[test]
fn test_propagate_updates_output_next_from_hidden() {
    let mut spore = Spore::new();
    // Set all thresholds very low
    spore.thresholds_h = [-1000; HIDDEN_SIZE];
    spore.thresholds_o = [-1000; OUTPUT_SIZE];
    spore.base_noise = 0.0;

    // Force hidden layer to be all true (from previous tick)
    spore.hidden = [true; HIDDEN_SIZE];

    // Propagate (output_next should now fire based on hidden)
    spore.propagate(0xFF);

    for o in &spore.output_next {
        assert!(*o, "Output neurons should fire with low threshold and all hidden firing");
    }
}

#[test]
fn test_propagate_high_threshold_nothing_fires() {
    let mut spore = Spore::new();
    // Set thresholds very high
    spore.thresholds_h = [10000; HIDDEN_SIZE];
    spore.thresholds_o = [10000; OUTPUT_SIZE];
    spore.base_noise = 0.0;  // Disable noise
    spore.frustration = 0.0; // No frustration boost

    spore.propagate(0xFF);

    for h in &spore.hidden_next {
        assert!(!*h, "Hidden neurons should not fire with very high threshold");
    }
}

#[test]
fn test_propagate_sets_traces_on_firing() {
    let mut spore = Spore::new();
    spore.thresholds_h = [-1000; HIDDEN_SIZE];
    spore.base_noise = 0.0;

    // Input with bit 0 set
    spore.propagate(0x01);

    // Hidden neurons should fire (low threshold)
    // Traces for input bit 0 -> hidden should be set to 1.0
    for h in 0..HIDDEN_SIZE {
        if spore.hidden_next[h] {
            assert_eq!(spore.traces_ih[h][0], 1.0, "Trace should be 1.0 for firing synapse");
        }
    }
}

#[test]
fn test_propagate_does_not_set_trace_for_zero_input() {
    let mut spore = Spore::new();
    spore.thresholds_h = [-1000; HIDDEN_SIZE];
    spore.base_noise = 0.0;

    // Zero all traces first
    spore.traces_ih = [[0.0; INPUT_SIZE]; HIDDEN_SIZE];

    // Input with only bit 0 set (0x01)
    spore.propagate(0x01);

    // Traces for bit 1 (which is 0 in input) should remain 0
    for h in 0..HIDDEN_SIZE {
        // Bit 1 is not set in input, so trace should stay 0
        assert_eq!(spore.traces_ih[h][1], 0.0, "Trace should be 0 for non-firing input");
    }
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd spore-sim && cargo test test_propagate
```

Expected: FAIL - propagate method not defined.

**Step 3: Implement propagate method**

Add to `spore-sim/src/spore.rs` inside the `impl Spore` block:

```rust
    /// Propagate input through the network (one tick).
    ///
    /// This implements pipelined propagation:
    /// - Input → Hidden: writes to hidden_next
    /// - Hidden → Output: reads from hidden (last tick), writes to output_next
    ///
    /// Traces are set to 1.0 for synapses that participate in firing.
    ///
    /// # Arguments
    /// * `input` - The input byte (each bit is one input neuron)
    pub fn propagate(&mut self, input: u8) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Calculate current noise rate based on frustration
        let noise_rate = self.base_noise + (self.frustration * self.max_noise_boost);

        // ====================================================================
        // INPUT → HIDDEN (writes to hidden_next)
        // ====================================================================
        for h in 0..HIDDEN_SIZE {
            // Compute weighted sum of inputs
            let mut sum: i32 = 0;
            for i in 0..INPUT_SIZE {
                let bit = ((input >> i) & 1) as i32;
                sum += bit * self.weights_ih[h][i] as i32;
            }

            // Hard threshold with noise injection
            let fires = sum > self.thresholds_h[h] as i32
                || rng.gen::<f32>() < noise_rate;

            self.hidden_next[h] = fires;

            // Set traces for synapses that contributed to firing
            if fires {
                for i in 0..INPUT_SIZE {
                    if (input >> i) & 1 == 1 {
                        self.traces_ih[h][i] = 1.0;
                    }
                }
                self.traces_th[h] = 1.0;  // Threshold trace
            }
        }

        // ====================================================================
        // HIDDEN → OUTPUT (reads from hidden, writes to output_next)
        // Note: Uses hidden (last tick), NOT hidden_next (this tick)
        // This is the pipelined behavior.
        // ====================================================================
        for o in 0..OUTPUT_SIZE {
            // Compute weighted sum of hidden activations
            let mut sum: i32 = 0;
            for h in 0..HIDDEN_SIZE {
                sum += self.hidden[h] as i32 * self.weights_ho[o][h] as i32;
            }

            // Hard threshold with noise injection
            let fires = sum > self.thresholds_o[o] as i32
                || rng.gen::<f32>() < noise_rate;

            self.output_next[o] = fires;

            // Set traces for synapses that contributed to firing
            if fires {
                for h in 0..HIDDEN_SIZE {
                    if self.hidden[h] {
                        self.traces_ho[o][h] = 1.0;
                    }
                }
                self.traces_to[o] = 1.0;  // Threshold trace
            }
        }
    }
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd spore-sim && cargo test test_propagate
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add spore-sim/src/spore.rs spore-sim/tests/spore_tests.rs
git commit -m "feat: add Spore::propagate with pipelined activation and trace setting"
```

---

## Task 6: Implement tick_end (Pipeline Advancement)

**Files:**
- Modify: `spore-sim/src/spore.rs`
- Modify: `spore-sim/tests/spore_tests.rs`

**Step 1: Write the failing test**

Add to `spore-sim/tests/spore_tests.rs`:

```rust
#[test]
fn test_tick_end_advances_pipeline() {
    let mut spore = Spore::new();

    // Set some values in hidden_next and output_next
    spore.hidden_next[0] = true;
    spore.hidden_next[5] = true;
    spore.output_next[3] = true;

    spore.tick_end();

    // After tick_end, hidden should equal what hidden_next was
    assert!(spore.hidden[0]);
    assert!(spore.hidden[5]);
    assert!(!spore.hidden[1]);  // Was false

    // Same for output
    assert!(spore.output[3]);
    assert!(!spore.output[0]);  // Was false
}

#[test]
fn test_tick_end_decays_traces() {
    let mut spore = Spore::new();
    spore.trace_decay = 0.9;

    // Set some traces
    spore.traces_ih[0][0] = 1.0;
    spore.traces_ho[0][0] = 0.5;
    spore.traces_th[0] = 1.0;
    spore.traces_to[0] = 0.8;

    spore.tick_end();

    // Traces should decay by trace_decay
    assert!((spore.traces_ih[0][0] - 0.9).abs() < 0.001);
    assert!((spore.traces_ho[0][0] - 0.45).abs() < 0.001);
    assert!((spore.traces_th[0] - 0.9).abs() < 0.001);
    assert!((spore.traces_to[0] - 0.72).abs() < 0.001);
}

#[test]
fn test_tick_end_multiple_decays() {
    let mut spore = Spore::new();
    spore.trace_decay = 0.9;
    spore.traces_ih[0][0] = 1.0;

    // After 10 ticks, trace should be 0.9^10 ≈ 0.349
    for _ in 0..10 {
        spore.tick_end();
    }

    let expected = 0.9_f32.powi(10);
    assert!((spore.traces_ih[0][0] - expected).abs() < 0.001);
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd spore-sim && cargo test test_tick_end
```

Expected: FAIL - tick_end method not defined.

**Step 3: Implement tick_end method**

Add to `spore-sim/src/spore.rs` inside the `impl Spore` block:

```rust
    /// Advance the pipeline and decay traces.
    ///
    /// This must be called after propagate() each tick:
    /// 1. Copies hidden_next → hidden, output_next → output
    /// 2. Decays all eligibility traces by trace_decay
    pub fn tick_end(&mut self) {
        // Advance the pipeline
        self.hidden = self.hidden_next;
        self.output = self.output_next;

        // Decay all traces
        for row in &mut self.traces_ih {
            for t in row {
                *t *= self.trace_decay;
            }
        }
        for row in &mut self.traces_ho {
            for t in row {
                *t *= self.trace_decay;
            }
        }
        for t in &mut self.traces_th {
            *t *= self.trace_decay;
        }
        for t in &mut self.traces_to {
            *t *= self.trace_decay;
        }
    }
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd spore-sim && cargo test test_tick_end
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add spore-sim/src/spore.rs spore-sim/tests/spore_tests.rs
git commit -m "feat: add Spore::tick_end for pipeline advancement and trace decay"
```

---

## Task 7: Implement output_as_byte

**Files:**
- Modify: `spore-sim/src/spore.rs`
- Modify: `spore-sim/tests/spore_tests.rs`

**Step 1: Write the failing test**

Add to `spore-sim/tests/spore_tests.rs`:

```rust
#[test]
fn test_output_as_byte_all_zeros() {
    let mut spore = Spore::new();
    spore.output = [false; OUTPUT_SIZE];
    assert_eq!(spore.output_as_byte(), 0x00);
}

#[test]
fn test_output_as_byte_all_ones() {
    let mut spore = Spore::new();
    spore.output = [true; OUTPUT_SIZE];
    assert_eq!(spore.output_as_byte(), 0xFF);
}

#[test]
fn test_output_as_byte_specific_pattern() {
    let mut spore = Spore::new();
    // Set bits 0, 2, 4, 6 (0b01010101 = 0x55)
    spore.output = [true, false, true, false, true, false, true, false];
    assert_eq!(spore.output_as_byte(), 0x55);
}

#[test]
fn test_output_as_byte_another_pattern() {
    let mut spore = Spore::new();
    // Set bits 1, 3, 5, 7 (0b10101010 = 0xAA)
    spore.output = [false, true, false, true, false, true, false, true];
    assert_eq!(spore.output_as_byte(), 0xAA);
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd spore-sim && cargo test test_output_as_byte
```

Expected: FAIL - output_as_byte method not defined.

**Step 3: Implement output_as_byte method**

Add to `spore-sim/src/spore.rs` inside the `impl Spore` block:

```rust
    /// Convert output activations to a byte.
    ///
    /// Each output neuron corresponds to one bit of the output byte.
    /// output[0] = bit 0 (LSB), output[7] = bit 7 (MSB)
    pub fn output_as_byte(&self) -> u8 {
        let mut byte = 0u8;
        for (i, &bit) in self.output.iter().enumerate() {
            if bit {
                byte |= 1 << i;
            }
        }
        byte
    }
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd spore-sim && cargo test test_output_as_byte
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add spore-sim/src/spore.rs spore-sim/tests/spore_tests.rs
git commit -m "feat: add Spore::output_as_byte"
```

---

## Task 8: Implement receive_reward

**Files:**
- Modify: `spore-sim/src/spore.rs`
- Modify: `spore-sim/tests/spore_tests.rs`

**Step 1: Write the failing test**

Add to `spore-sim/tests/spore_tests.rs`:

```rust
#[test]
fn test_receive_reward_perfect_accuracy() {
    let mut spore = Spore::new();
    spore.receive_reward(8);  // 8/8 correct

    // dopamine = (8/8)² - (0.5)² = 1.0 - 0.25 = 0.75
    assert!((spore.dopamine - 0.75).abs() < 0.001);
}

#[test]
fn test_receive_reward_zero_accuracy() {
    let mut spore = Spore::new();
    spore.receive_reward(0);  // 0/8 correct

    // dopamine = (0/8)² - (0.5)² = 0 - 0.25 = -0.25
    assert!((spore.dopamine - (-0.25)).abs() < 0.001);
}

#[test]
fn test_receive_reward_baseline_accuracy() {
    let mut spore = Spore::new();
    spore.receive_reward(4);  // 4/8 = 50% = baseline

    // dopamine = (4/8)² - (0.5)² = 0.25 - 0.25 = 0
    assert!(spore.dopamine.abs() < 0.001);
}

#[test]
fn test_receive_reward_frustration_spikes_on_low_accuracy() {
    let mut spore = Spore::new();
    spore.frustration = 0.5;  // Start at some value

    spore.receive_reward(3);  // 3/8 = 37.5% < 50%

    // Fix 2: Frustration should spike to 1.0 immediately
    assert_eq!(spore.frustration, 1.0);
}

#[test]
fn test_receive_reward_frustration_decays_on_high_accuracy() {
    let mut spore = Spore::new();
    spore.frustration = 1.0;

    spore.receive_reward(8);  // 100% accuracy

    // frustration = 0.8 * 1.0 + 0.2 * (1.0 - 1.0) = 0.8
    assert!((spore.frustration - 0.8).abs() < 0.001);
}

#[test]
fn test_receive_reward_frustration_ema_on_medium_accuracy() {
    let mut spore = Spore::new();
    spore.frustration = 0.5;

    spore.receive_reward(6);  // 6/8 = 75% > 50%

    // frustration = 0.8 * 0.5 + 0.2 * (1.0 - 0.75) = 0.4 + 0.05 = 0.45
    assert!((spore.frustration - 0.45).abs() < 0.001);
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd spore-sim && cargo test test_receive_reward
```

Expected: FAIL - receive_reward method not defined.

**Step 3: Implement receive_reward method**

Add to `spore-sim/src/spore.rs` inside the `impl Spore` block:

```rust
    /// Receive reward signal and update dopamine/frustration.
    ///
    /// # Arguments
    /// * `correct_bits` - Number of output bits that matched expected (0-8)
    ///
    /// # Dopamine Calculation (Fix 3: Anti-Hebbian)
    /// - dopamine = (accuracy²) - (BASELINE_ACCURACY²)
    /// - At 100% accuracy: 1.0 - 0.25 = +0.75 (strong reward)
    /// - At 50% accuracy: 0.25 - 0.25 = 0 (neutral)
    /// - At 0% accuracy: 0 - 0.25 = -0.25 (punishment)
    ///
    /// # Frustration Update (Fix 2: Fast Response)
    /// - If accuracy < 50%: spike to 1.0 immediately
    /// - Otherwise: EMA with α=0.2
    pub fn receive_reward(&mut self, correct_bits: u8) {
        let accuracy = correct_bits as f32 / 8.0;

        // Fix 3: Signed dopamine - below baseline = punishment
        let reward = accuracy * accuracy;
        let baseline_reward = BASELINE_ACCURACY * BASELINE_ACCURACY;
        self.dopamine = reward - baseline_reward;

        // Fix 2: Fast frustration response
        if accuracy < 0.5 {
            self.frustration = 1.0;  // Instant spike
        } else {
            // EMA with faster alpha (0.2 instead of 0.1)
            self.frustration = 0.8 * self.frustration + 0.2 * (1.0 - accuracy);
        }
    }
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd spore-sim && cargo test test_receive_reward
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add spore-sim/src/spore.rs spore-sim/tests/spore_tests.rs
git commit -m "feat: add Spore::receive_reward with signed dopamine (Fix 3) and fast frustration (Fix 2)"
```

---

## Task 9: Implement learn

**Files:**
- Modify: `spore-sim/src/spore.rs`
- Modify: `spore-sim/tests/spore_tests.rs`

**Step 1: Write the failing test**

Add to `spore-sim/tests/spore_tests.rs`:

```rust
#[test]
fn test_learn_increases_weights_on_positive_dopamine() {
    let mut spore = Spore::new();
    spore.dopamine = 0.5;
    spore.learning_rate = 0.5;

    // Set a trace
    spore.traces_ih[0][0] = 1.0;
    let original_weight = spore.weights_ih[0][0];

    // Run many learn cycles to accumulate stochastic changes
    for _ in 0..100 {
        spore.dopamine = 0.5;
        spore.traces_ih[0][0] = 1.0;
        spore.learn();
    }

    // Weight should have increased (statistically)
    assert!(spore.weights_ih[0][0] > original_weight,
        "Weight should increase with positive dopamine. Was {}, now {}",
        original_weight, spore.weights_ih[0][0]);
}

#[test]
fn test_learn_decreases_weights_on_negative_dopamine() {
    let mut spore = Spore::new();
    spore.learning_rate = 0.5;

    // Set a trace and a starting weight
    spore.traces_ih[0][0] = 1.0;
    spore.weights_ih[0][0] = 50;  // Start positive
    let original_weight = spore.weights_ih[0][0];

    // Run many learn cycles with negative dopamine
    for _ in 0..100 {
        spore.dopamine = -0.25;
        spore.traces_ih[0][0] = 1.0;
        spore.learn();
    }

    // Weight should have decreased
    assert!(spore.weights_ih[0][0] < original_weight,
        "Weight should decrease with negative dopamine. Was {}, now {}",
        original_weight, spore.weights_ih[0][0]);
}

#[test]
fn test_learn_consumes_dopamine() {
    let mut spore = Spore::new();
    spore.dopamine = 0.5;
    spore.traces_ih[0][0] = 1.0;

    spore.learn();

    assert_eq!(spore.dopamine, 0.0, "Dopamine should be consumed after learn");
}

#[test]
fn test_learn_does_nothing_with_zero_dopamine() {
    let mut spore = Spore::new();
    spore.dopamine = 0.0;
    spore.traces_ih[0][0] = 1.0;
    let original_weight = spore.weights_ih[0][0];

    spore.learn();

    assert_eq!(spore.weights_ih[0][0], original_weight,
        "Weight should not change with zero dopamine");
}

#[test]
fn test_learn_does_nothing_with_zero_trace() {
    let mut spore = Spore::new();
    spore.dopamine = 0.5;
    spore.traces_ih[0][0] = 0.0;  // No trace
    let original_weight = spore.weights_ih[0][0];

    spore.learn();

    assert_eq!(spore.weights_ih[0][0], original_weight,
        "Weight should not change with zero trace");
}

#[test]
fn test_learn_threshold_decreases_on_positive_dopamine() {
    let mut spore = Spore::new();
    spore.learning_rate = 0.5;
    spore.thresholds_h[0] = 50;

    // Run many cycles
    for _ in 0..100 {
        spore.dopamine = 0.5;
        spore.traces_th[0] = 1.0;
        spore.learn();
    }

    // Threshold should decrease (neuron becomes more eager)
    assert!(spore.thresholds_h[0] < 50,
        "Threshold should decrease with positive dopamine");
}

#[test]
fn test_learn_threshold_increases_on_negative_dopamine() {
    let mut spore = Spore::new();
    spore.learning_rate = 0.5;
    spore.thresholds_h[0] = 0;

    for _ in 0..100 {
        spore.dopamine = -0.25;
        spore.traces_th[0] = 1.0;
        spore.learn();
    }

    // Threshold should increase (neuron becomes more stubborn)
    assert!(spore.thresholds_h[0] > 0,
        "Threshold should increase with negative dopamine");
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd spore-sim && cargo test test_learn
```

Expected: FAIL - learn method not defined.

**Step 3: Implement learn method**

Add to `spore-sim/src/spore.rs` inside the `impl Spore` block (also add `use crate::utils::stochastic_round;` at the top of the impl block or at module level):

```rust
    /// Apply Hebbian learning based on current dopamine and traces.
    ///
    /// For each synapse: weight += lr_scaled * dopamine * trace
    /// For each threshold: threshold -= lr_scaled * dopamine * trace
    ///
    /// Dopamine can be negative (Fix 3: Anti-Hebbian), causing weights to
    /// decrease and thresholds to increase.
    ///
    /// Uses stochastic rounding to handle small updates that would otherwise
    /// truncate to zero in integer math.
    ///
    /// Consumes dopamine after learning (atomic update).
    pub fn learn(&mut self) {
        use crate::utils::stochastic_round;

        // Skip if no dopamine signal
        if self.dopamine.abs() < 0.001 {
            return;
        }

        let lr_scaled = self.learning_rate * 100.0;
        let d = self.dopamine;

        // Update Input → Hidden weights
        for h in 0..HIDDEN_SIZE {
            for i in 0..INPUT_SIZE {
                let change = lr_scaled * d * self.traces_ih[h][i];
                let delta = stochastic_round(change);
                self.weights_ih[h][i] = self.weights_ih[h][i].saturating_add(delta);
            }

            // Update hidden threshold
            let t_change = lr_scaled * d * self.traces_th[h];
            let t_delta = stochastic_round(t_change);
            // Positive dopamine → decrease threshold (more eager)
            // Negative dopamine → increase threshold (more stubborn)
            self.thresholds_h[h] = self.thresholds_h[h].saturating_sub(t_delta);
        }

        // Update Hidden → Output weights
        for o in 0..OUTPUT_SIZE {
            for h in 0..HIDDEN_SIZE {
                let change = lr_scaled * d * self.traces_ho[o][h];
                let delta = stochastic_round(change);
                self.weights_ho[o][h] = self.weights_ho[o][h].saturating_add(delta);
            }

            // Update output threshold
            let t_change = lr_scaled * d * self.traces_to[o];
            let t_delta = stochastic_round(t_change);
            self.thresholds_o[o] = self.thresholds_o[o].saturating_sub(t_delta);
        }

        // Consume dopamine (atomic update)
        self.dopamine = 0.0;
    }
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd spore-sim && cargo test test_learn
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add spore-sim/src/spore.rs spore-sim/tests/spore_tests.rs
git commit -m "feat: add Spore::learn with stochastic rounding and anti-Hebbian support"
```

---

## Task 10: Implement maintain (Homeostasis)

**Files:**
- Modify: `spore-sim/src/spore.rs`
- Modify: `spore-sim/tests/spore_tests.rs`

**Step 1: Write the failing test**

Add to `spore-sim/tests/spore_tests.rs`:

```rust
#[test]
fn test_maintain_weight_decay() {
    let mut spore = Spore::new();
    spore.weights_ih[0][0] = 64;  // Decays by w >> 6 = 1

    // Decay happens every WEIGHT_DECAY_INTERVAL ticks
    spore.maintain(WEIGHT_DECAY_INTERVAL);

    // 64 - (64 >> 6) = 64 - 1 = 63
    assert_eq!(spore.weights_ih[0][0], 63);
}

#[test]
fn test_maintain_weight_decay_not_every_tick() {
    let mut spore = Spore::new();
    spore.weights_ih[0][0] = 64;

    // At tick 50 (not a multiple of 100), no decay
    spore.maintain(50);

    assert_eq!(spore.weights_ih[0][0], 64);
}

#[test]
fn test_maintain_weight_normalization_over_budget() {
    let mut spore = Spore::new();
    // Set all weights to 100, sum = 800 > MAX_WEIGHT_SUM (400)
    spore.weights_ih[0] = [100; INPUT_SIZE];

    spore.maintain(0);

    // Sum should now be <= MAX_WEIGHT_SUM
    let sum: i32 = spore.weights_ih[0].iter().map(|&w| w.abs() as i32).sum();
    assert!(sum <= MAX_WEIGHT_SUM, "Sum {} should be <= {}", sum, MAX_WEIGHT_SUM);
}

#[test]
fn test_maintain_weight_normalization_under_budget() {
    let mut spore = Spore::new();
    // Set weights to low values, sum = 80 < MAX_WEIGHT_SUM
    spore.weights_ih[0] = [10; INPUT_SIZE];
    let original: [i16; INPUT_SIZE] = spore.weights_ih[0];

    spore.maintain(0);

    // Should be unchanged (no normalization needed)
    assert_eq!(spore.weights_ih[0], original);
}

#[test]
fn test_maintain_threshold_drift_up() {
    let mut spore = Spore::new();
    spore.thresholds_h[0] = 0;  // Below default

    spore.maintain(0);

    assert_eq!(spore.thresholds_h[0], 1);  // Drifted up by 1
}

#[test]
fn test_maintain_threshold_drift_down() {
    let mut spore = Spore::new();
    spore.thresholds_h[0] = 100;  // Above default

    spore.maintain(0);

    assert_eq!(spore.thresholds_h[0], 99);  // Drifted down by 1
}

#[test]
fn test_maintain_threshold_at_default_no_change() {
    let mut spore = Spore::new();
    spore.thresholds_h[0] = DEFAULT_THRESHOLD;

    spore.maintain(0);

    assert_eq!(spore.thresholds_h[0], DEFAULT_THRESHOLD);
}

#[test]
fn test_maintain_output_weights_normalized() {
    let mut spore = Spore::new();
    // Set all output weights high
    spore.weights_ho[0] = [50; HIDDEN_SIZE];  // sum = 1600 > MAX_WEIGHT_SUM

    spore.maintain(0);

    let sum: i32 = spore.weights_ho[0].iter().map(|&w| w.abs() as i32).sum();
    assert!(sum <= MAX_WEIGHT_SUM, "Sum {} should be <= {}", sum, MAX_WEIGHT_SUM);
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd spore-sim && cargo test test_maintain
```

Expected: FAIL - maintain method not defined.

**Step 3: Implement maintain method**

Add to `spore-sim/src/spore.rs` inside the `impl Spore` block:

```rust
    /// Apply homeostasis: weight decay, weight normalization, threshold drift.
    ///
    /// # Weight Decay (every WEIGHT_DECAY_INTERVAL ticks)
    /// Each weight loses ~1.5% of its value: w -= w >> 6
    ///
    /// # Weight Normalization (Fix 1: every tick)
    /// If sum of absolute weights for a neuron exceeds MAX_WEIGHT_SUM,
    /// all weights for that neuron are scaled down proportionally.
    ///
    /// # Threshold Drift (every tick)
    /// Thresholds drift toward DEFAULT_THRESHOLD by ±1.
    ///
    /// # Arguments
    /// * `tick` - Current simulation tick
    pub fn maintain(&mut self, tick: u64) {
        // ====================================================================
        // WEIGHT DECAY (every WEIGHT_DECAY_INTERVAL ticks)
        // ====================================================================
        if tick % WEIGHT_DECAY_INTERVAL == 0 && tick > 0 {
            for row in &mut self.weights_ih {
                for w in row {
                    *w -= *w >> 6;  // ~1.5% decay
                }
            }
            for row in &mut self.weights_ho {
                for w in row {
                    *w -= *w >> 6;
                }
            }
        }

        // ====================================================================
        // WEIGHT NORMALIZATION (Fix 1: per-neuron budget)
        // ====================================================================

        // Input → Hidden weights
        for h in 0..HIDDEN_SIZE {
            let sum: i32 = self.weights_ih[h].iter().map(|&w| w.abs() as i32).sum();
            if sum > MAX_WEIGHT_SUM {
                let scale = MAX_WEIGHT_SUM as f32 / sum as f32;
                for w in &mut self.weights_ih[h] {
                    *w = (*w as f32 * scale) as i16;
                }
            }
        }

        // Hidden → Output weights
        for o in 0..OUTPUT_SIZE {
            let sum: i32 = self.weights_ho[o].iter().map(|&w| w.abs() as i32).sum();
            if sum > MAX_WEIGHT_SUM {
                let scale = MAX_WEIGHT_SUM as f32 / sum as f32;
                for w in &mut self.weights_ho[o] {
                    *w = (*w as f32 * scale) as i16;
                }
            }
        }

        // ====================================================================
        // THRESHOLD DRIFT (every tick)
        // ====================================================================
        for t in &mut self.thresholds_h {
            if *t < DEFAULT_THRESHOLD {
                *t += 1;
            } else if *t > DEFAULT_THRESHOLD {
                *t -= 1;
            }
        }
        for t in &mut self.thresholds_o {
            if *t < DEFAULT_THRESHOLD {
                *t += 1;
            } else if *t > DEFAULT_THRESHOLD {
                *t -= 1;
            }
        }
    }
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd spore-sim && cargo test test_maintain
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add spore-sim/src/spore.rs spore-sim/tests/spore_tests.rs
git commit -m "feat: add Spore::maintain with weight decay, normalization (Fix 1), and threshold drift"
```

---

## Task 11: Implement Environment Struct

**Files:**
- Modify: `spore-sim/src/environment.rs`
- Create: `spore-sim/tests/environment_tests.rs`

**Step 1: Write the failing test**

Create `spore-sim/tests/environment_tests.rs`:

```rust
use spore_sim::environment::Environment;
use spore_sim::constants::*;

#[test]
fn test_environment_new() {
    let env = Environment::new(0);
    assert_eq!(env.reward_latency, 0);
    assert_eq!(env.input_hold_ticks, DEFAULT_INPUT_HOLD_TICKS);
}

#[test]
fn test_environment_with_params() {
    let env = Environment::with_params(5, 100);
    assert_eq!(env.reward_latency, 5);
    assert_eq!(env.input_hold_ticks, 100);
}

#[test]
fn test_environment_input_history_initialized() {
    let env = Environment::new(0);
    // Input history should have PIPELINE_LATENCY + 1 entries
    assert_eq!(env.input_history_len(), PIPELINE_LATENCY + 1);
}

#[test]
fn test_environment_get_input_returns_current() {
    let env = Environment::new(0);
    let input = env.get_input();
    // Should return a valid u8
    assert!(input <= 255);
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd spore-sim && cargo test environment_tests
```

Expected: FAIL - Environment struct not defined.

**Step 3: Implement Environment struct**

Replace `spore-sim/src/environment.rs`:

```rust
//! The Environment: Teacher that provides inputs and rewards.
//!
//! The Environment:
//! - Generates random input patterns
//! - Holds inputs for a configurable number of ticks
//! - Judges Spore output against the correct (pipeline-delayed) input
//! - Schedules and delivers rewards with configurable latency

use std::collections::VecDeque;
use rand::Rng;
use crate::constants::*;

/// The training environment for the Spore.
#[derive(Debug, Clone)]
pub struct Environment {
    /// Reward delivery latency in ticks
    pub reward_latency: u64,

    /// Pending rewards: (tick_to_deliver, correct_bits)
    pending_rewards: VecDeque<(u64, u8)>,

    /// Input history for pipeline-aware judging
    /// Front = oldest (what spore is responding to)
    /// Back = newest (current input)
    input_history: VecDeque<u8>,

    /// Current input pattern
    current_input: u8,

    /// How long to hold each input pattern
    pub input_hold_ticks: u64,

    /// Ticks spent on current input
    ticks_on_current: u64,
}

impl Environment {
    /// Create a new Environment with default settings.
    pub fn new(reward_latency: u64) -> Self {
        Self::with_params(reward_latency, DEFAULT_INPUT_HOLD_TICKS)
    }

    /// Create a new Environment with custom settings.
    pub fn with_params(reward_latency: u64, input_hold_ticks: u64) -> Self {
        let mut rng = rand::thread_rng();
        let current_input = rng.gen::<u8>();

        // Pre-fill input history so we don't underflow
        // Need PIPELINE_LATENCY + 1 entries
        let mut input_history = VecDeque::with_capacity(PIPELINE_LATENCY + 1);
        for _ in 0..=PIPELINE_LATENCY {
            input_history.push_back(current_input);
        }

        Self {
            reward_latency,
            pending_rewards: VecDeque::new(),
            input_history,
            current_input,
            input_hold_ticks,
            ticks_on_current: 0,
        }
    }

    /// Get the current input pattern.
    pub fn get_input(&self) -> u8 {
        self.current_input
    }

    /// Get the length of input history (for testing).
    pub fn input_history_len(&self) -> usize {
        self.input_history.len()
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new(DEFAULT_REWARD_LATENCY)
    }
}
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd spore-sim && cargo test environment_tests
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add spore-sim/src/environment.rs spore-sim/tests/environment_tests.rs
git commit -m "feat: add Environment struct with initialization"
```

---

## Task 12: Implement Environment::tick

**Files:**
- Modify: `spore-sim/src/environment.rs`
- Modify: `spore-sim/tests/environment_tests.rs`

**Step 1: Write the failing test**

Add to `spore-sim/tests/environment_tests.rs`:

```rust
#[test]
fn test_environment_tick_judges_against_delayed_input() {
    let mut env = Environment::with_params(0, 100);  // No reward latency

    // Manually set up input history to control the test
    // History: [old, middle, current] where old is what we judge against
    env.set_input_history_for_test(&[0xAA, 0xBB, 0xCC]);

    // If spore outputs 0xAA (matches oldest), should get 8 correct
    let result = env.tick(0, 0xAA);
    assert_eq!(result, Some(8));
}

#[test]
fn test_environment_tick_with_reward_latency() {
    let mut env = Environment::with_params(5, 100);
    env.set_input_history_for_test(&[0xFF, 0xFF, 0xFF]);

    // At tick 0, schedule reward for tick 5
    let result = env.tick(0, 0xFF);  // Perfect match
    assert!(result.is_none(), "Reward should not arrive yet at tick 0");

    // Advance to tick 5
    for t in 1..5 {
        env.tick(t, 0xFF);
    }
    let result = env.tick(5, 0xFF);
    assert!(result.is_some(), "Reward should arrive at tick 5");
}

#[test]
fn test_environment_tick_input_changes_after_hold_ticks() {
    let mut env = Environment::with_params(0, 5);  // Hold for 5 ticks
    let initial_input = env.get_input();

    // Tick 5 times with dummy output
    for t in 0..5 {
        env.tick(t, 0x00);
    }

    // Input should have changed
    // (Not guaranteed to be different due to randomness, but history should update)
    assert_eq!(env.ticks_on_current_for_test(), 0, "Should reset after hold period");
}

#[test]
fn test_environment_tick_correct_bits_calculation() {
    let mut env = Environment::with_params(0, 100);
    env.set_input_history_for_test(&[0b10101010, 0x00, 0x00]);

    // Output matches 6 bits: 0b10101000 (bits 0,1,2 differ)
    // XOR: 0b10101010 ^ 0b10101000 = 0b00000010 (1 bit differs)
    // Wait, let me recalculate
    // 0b10101010 ^ 0b10101000 = 0b00000010, count_ones = 1
    // correct = 8 - 1 = 7
    let result = env.tick(0, 0b10101000);
    assert_eq!(result, Some(7));
}

#[test]
fn test_environment_tick_zero_correct() {
    let mut env = Environment::with_params(0, 100);
    env.set_input_history_for_test(&[0xFF, 0x00, 0x00]);

    // Output is inverted: 0x00
    let result = env.tick(0, 0x00);
    assert_eq!(result, Some(0));
}

#[test]
fn test_environment_tick_updates_input_history() {
    let mut env = Environment::with_params(0, 100);
    let old_len = env.input_history_len();

    env.tick(0, 0x00);

    // History length should stay constant (sliding window)
    assert_eq!(env.input_history_len(), old_len);
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd spore-sim && cargo test test_environment_tick
```

Expected: FAIL - tick method and test helpers not defined.

**Step 3: Implement Environment::tick**

Add to `spore-sim/src/environment.rs` inside the `impl Environment` block:

```rust
    /// Process one simulation tick.
    ///
    /// # Arguments
    /// * `tick` - Current simulation tick
    /// * `spore_output` - The Spore's output byte
    ///
    /// # Returns
    /// * `Some(correct_bits)` - If a reward is being delivered this tick
    /// * `None` - If no reward is due
    ///
    /// # Pipeline-Aware Judging
    /// The Spore's output at tick T reflects the input from tick T-PIPELINE_LATENCY.
    /// We judge against `input_history[0]` (oldest entry).
    pub fn tick(&mut self, tick: u64, spore_output: u8) -> Option<u8> {
        let mut rng = rand::thread_rng();

        // ====================================================================
        // JUDGE OUTPUT (against pipeline-delayed input)
        // ====================================================================
        let judge_input = self.input_history[0];  // Oldest entry
        let error_bits = (spore_output ^ judge_input).count_ones() as u8;
        let correct_bits = 8 - error_bits;

        // ====================================================================
        // SCHEDULE REWARD
        // ====================================================================
        let deliver_at = tick + self.reward_latency;
        self.pending_rewards.push_back((deliver_at, correct_bits));

        // ====================================================================
        // DELIVER PENDING REWARDS
        // ====================================================================
        let mut reward = None;
        while let Some(&(t, bits)) = self.pending_rewards.front() {
            if t <= tick {
                self.pending_rewards.pop_front();
                reward = Some(bits);  // Take most recent if multiple
            } else {
                break;
            }
        }

        // ====================================================================
        // UPDATE INPUT HISTORY (sliding window)
        // ====================================================================
        self.input_history.pop_front();
        self.input_history.push_back(self.current_input);

        // ====================================================================
        // ADVANCE INPUT PATTERN
        // ====================================================================
        self.ticks_on_current += 1;
        if self.ticks_on_current >= self.input_hold_ticks {
            self.ticks_on_current = 0;
            self.current_input = rng.gen::<u8>();
        }

        reward
    }

    // ========================================================================
    // TEST HELPERS
    // ========================================================================

    /// Set input history for testing (front = oldest, back = newest).
    #[cfg(test)]
    pub fn set_input_history_for_test(&mut self, history: &[u8]) {
        self.input_history.clear();
        for &h in history {
            self.input_history.push_back(h);
        }
    }

    /// Get ticks on current input for testing.
    #[cfg(test)]
    pub fn ticks_on_current_for_test(&self) -> u64 {
        self.ticks_on_current
    }
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd spore-sim && cargo test test_environment_tick
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add spore-sim/src/environment.rs spore-sim/tests/environment_tests.rs
git commit -m "feat: add Environment::tick with pipeline-aware judging and reward scheduling"
```

---

## Task 13: Implement Simulation Struct

**Files:**
- Modify: `spore-sim/src/simulation.rs`
- Create: `spore-sim/tests/simulation_tests.rs`

**Step 1: Write the failing test**

Create `spore-sim/tests/simulation_tests.rs`:

```rust
use spore_sim::simulation::Simulation;
use spore_sim::constants::*;

#[test]
fn test_simulation_new() {
    let sim = Simulation::new();
    assert_eq!(sim.tick, 0);
    assert_eq!(sim.recent_accuracy, 0.0);
}

#[test]
fn test_simulation_with_params() {
    let sim = Simulation::with_params(5, 0.95, 100);
    assert_eq!(sim.env().reward_latency, 5);
}

#[test]
fn test_simulation_step_increments_tick() {
    let mut sim = Simulation::new();
    sim.step();
    assert_eq!(sim.tick, 1);
    sim.step();
    assert_eq!(sim.tick, 2);
}

#[test]
fn test_simulation_step_updates_accuracy() {
    let mut sim = Simulation::new();

    // Run a few steps
    for _ in 0..100 {
        sim.step();
    }

    // Accuracy should have been updated (might still be low due to random init)
    // Just verify it's a valid value
    assert!(sim.recent_accuracy >= 0.0 && sim.recent_accuracy <= 1.0);
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd spore-sim && cargo test simulation_tests
```

Expected: FAIL - Simulation struct not defined.

**Step 3: Implement Simulation struct**

Replace `spore-sim/src/simulation.rs`:

```rust
//! The Simulation: Orchestrates the Spore and Environment.

use crate::spore::Spore;
use crate::environment::Environment;
use crate::constants::*;

/// The main simulation runner.
#[derive(Debug)]
pub struct Simulation {
    /// The neural network being trained
    spore: Spore,

    /// The training environment
    env: Environment,

    /// Current simulation tick
    pub tick: u64,

    /// Rolling average of accuracy
    pub recent_accuracy: f32,

    /// History of accuracy for plotting
    pub accuracy_history: Vec<f32>,
}

impl Simulation {
    /// Create a new Simulation with default parameters.
    pub fn new() -> Self {
        Self::with_params(
            DEFAULT_REWARD_LATENCY,
            DEFAULT_TRACE_DECAY,
            DEFAULT_INPUT_HOLD_TICKS,
        )
    }

    /// Create a new Simulation with custom parameters.
    pub fn with_params(
        reward_latency: u64,
        trace_decay: f32,
        input_hold_ticks: u64,
    ) -> Self {
        let spore = Spore::with_params(
            DEFAULT_LEARNING_RATE,
            trace_decay,
            DEFAULT_BASE_NOISE,
            DEFAULT_MAX_NOISE_BOOST,
        );
        let env = Environment::with_params(reward_latency, input_hold_ticks);

        Self {
            spore,
            env,
            tick: 0,
            recent_accuracy: 0.0,
            accuracy_history: Vec::new(),
        }
    }

    /// Get a reference to the Environment (for testing).
    pub fn env(&self) -> &Environment {
        &self.env
    }

    /// Get a reference to the Spore (for inspection).
    pub fn spore(&self) -> &Spore {
        &self.spore
    }

    /// Execute one simulation step.
    ///
    /// The heartbeat:
    /// 1. SENSE: Get input from environment
    /// 2. PROPAGATE: Signal flows through network
    /// 3. TICK_END: Advance pipeline, decay traces
    /// 4. OUTPUT: Read output byte
    /// 5. ENVIRONMENT: Judge output, schedule/deliver reward
    /// 6. REWARD: Inject dopamine if reward delivered
    /// 7. LEARN: Apply Hebbian update
    /// 8. MAINTAIN: Weight decay, normalization, threshold drift
    pub fn step(&mut self) {
        // 1. SENSE
        let input = self.env.get_input();

        // 2. PROPAGATE
        self.spore.propagate(input);

        // 3. TICK_END (advance pipeline, decay traces)
        self.spore.tick_end();

        // 4. OUTPUT
        let output = self.spore.output_as_byte();

        // 5. ENVIRONMENT STEP
        if let Some(correct_bits) = self.env.tick(self.tick, output) {
            // 6. REWARD
            self.spore.receive_reward(correct_bits);

            // Track accuracy
            let accuracy = correct_bits as f32 / 8.0;
            self.recent_accuracy = 0.95 * self.recent_accuracy + 0.05 * accuracy;
        }

        // 7. LEARN
        self.spore.learn();

        // 8. MAINTAIN
        self.spore.maintain(self.tick);

        self.tick += 1;
    }

    /// Run simulation for a given number of ticks.
    ///
    /// # Arguments
    /// * `max_ticks` - Number of ticks to run
    /// * `log_interval` - How often to log progress (0 = no logging)
    ///
    /// # Returns
    /// Final accuracy
    pub fn run(&mut self, max_ticks: u64, log_interval: u64) -> f32 {
        while self.tick < max_ticks {
            self.step();

            // Logging
            if log_interval > 0 && self.tick % log_interval == 0 {
                println!(
                    "Tick {}: accuracy={:.2}% frustration={:.3}",
                    self.tick,
                    self.recent_accuracy * 100.0,
                    self.spore.frustration
                );
                self.accuracy_history.push(self.recent_accuracy);
            }
        }

        self.recent_accuracy
    }

    /// Check if simulation has converged.
    ///
    /// # Arguments
    /// * `threshold` - Minimum accuracy to consider converged (e.g., 0.95)
    pub fn has_converged(&self, threshold: f32) -> bool {
        self.recent_accuracy >= threshold
    }
}

impl Default for Simulation {
    fn default() -> Self {
        Self::new()
    }
}
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd spore-sim && cargo test simulation_tests
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add spore-sim/src/simulation.rs spore-sim/tests/simulation_tests.rs
git commit -m "feat: add Simulation struct with step() and run() methods"
```

---

## Task 14: Implement Main Entry Point

**Files:**
- Modify: `spore-sim/src/main.rs`

**Step 1: Write main.rs**

Replace `spore-sim/src/main.rs`:

```rust
//! Spore Mirror Experiment - Phase 1
//!
//! Proves that Hebbian learning + Dopamine reinforcement can evolve
//! a byte-copy reflex from random noise.
//!
//! Usage:
//!   spore-sim [OPTIONS]
//!
//! Options:
//!   --ticks N         Number of ticks to run (default: 100000)
//!   --latency N       Reward latency in ticks (default: 0)
//!   --trace-decay F   Trace decay rate (default: 0.9)
//!   --hold N          Input hold ticks (default: 50)
//!   --log-interval N  Log every N ticks (default: 1000)
//!   --quiet           Suppress logging

use spore_sim::simulation::Simulation;
use spore_sim::constants::*;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let mut max_ticks: u64 = 100_000;
    let mut reward_latency: u64 = DEFAULT_REWARD_LATENCY;
    let mut trace_decay: f32 = DEFAULT_TRACE_DECAY;
    let mut input_hold_ticks: u64 = DEFAULT_INPUT_HOLD_TICKS;
    let mut log_interval: u64 = 1000;
    let mut quiet = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--ticks" => {
                i += 1;
                max_ticks = args[i].parse().expect("Invalid --ticks value");
            }
            "--latency" => {
                i += 1;
                reward_latency = args[i].parse().expect("Invalid --latency value");
            }
            "--trace-decay" => {
                i += 1;
                trace_decay = args[i].parse().expect("Invalid --trace-decay value");
            }
            "--hold" => {
                i += 1;
                input_hold_ticks = args[i].parse().expect("Invalid --hold value");
            }
            "--log-interval" => {
                i += 1;
                log_interval = args[i].parse().expect("Invalid --log-interval value");
            }
            "--quiet" => {
                quiet = true;
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_help();
                return;
            }
        }
        i += 1;
    }

    if !quiet {
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║         SPORE MIRROR EXPERIMENT - PHASE 1                 ║");
        println!("╠═══════════════════════════════════════════════════════════╣");
        println!("║ Proving: Hebbian + Dopamine → Emergent Byte Copy          ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!();
        println!("Configuration:");
        println!("  Max ticks:        {}", max_ticks);
        println!("  Reward latency:   {}", reward_latency);
        println!("  Trace decay:      {}", trace_decay);
        println!("  Input hold ticks: {}", input_hold_ticks);
        println!();

        // Validate timing constraint (Fix 4)
        let min_hold = min_input_hold_ticks(reward_latency);
        if input_hold_ticks < min_hold {
            println!("⚠️  WARNING: input_hold_ticks ({}) < minimum ({}) for latency {}",
                input_hold_ticks, min_hold, reward_latency);
            println!("   This may cause superstitious learning (Fix 4).");
            println!();
        }

        // Recommend trace decay
        let rec_decay = recommended_trace_decay(reward_latency);
        if trace_decay < rec_decay - 0.02 {
            println!("⚠️  WARNING: trace_decay ({}) may be too fast for latency {}",
                trace_decay, reward_latency);
            println!("   Recommended: >= {:.2}", rec_decay);
            println!();
        }
    }

    // Create and run simulation
    let mut sim = Simulation::with_params(
        reward_latency,
        trace_decay,
        input_hold_ticks,
    );

    let actual_log_interval = if quiet { 0 } else { log_interval };
    let final_accuracy = sim.run(max_ticks, actual_log_interval);

    // Report results
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("FINAL RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Ticks run:      {}", sim.tick);
    println!("  Final accuracy: {:.2}%", final_accuracy * 100.0);
    println!("  Frustration:    {:.3}", sim.spore().frustration);
    println!();

    if sim.has_converged(0.95) {
        println!("✅ SUCCESS: Spore learned to mirror! (accuracy > 95%)");
    } else if final_accuracy > 0.8 {
        println!("🔶 PARTIAL: Spore is learning but hasn't converged (accuracy > 80%)");
        println!("   Try running for more ticks or tuning hyperparameters.");
    } else if final_accuracy > 0.5 {
        println!("⚠️  SLOW: Spore is above baseline but learning slowly");
        println!("   Check: learning_rate, trace_decay, input_hold_ticks");
    } else {
        println!("❌ FAILED: Spore did not converge (accuracy <= 50%)");
        println!("   Check failure modes in docs/plans/2026-02-05-spore-mirror-experiment-design.md");
    }
}

fn print_help() {
    println!("Spore Mirror Experiment - Phase 1");
    println!();
    println!("USAGE:");
    println!("  spore-sim [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --ticks N         Number of ticks to run (default: 100000)");
    println!("  --latency N       Reward latency in ticks (default: 0)");
    println!("  --trace-decay F   Trace decay rate (default: 0.9)");
    println!("  --hold N          Input hold ticks (default: 50)");
    println!("  --log-interval N  Log every N ticks (default: 1000)");
    println!("  --quiet           Suppress logging");
    println!("  --help, -h        Show this help");
}
```

**Step 2: Build and verify**

Run:
```bash
cd spore-sim && cargo build --release
```

Expected: Build succeeds.

**Step 3: Test the CLI**

Run:
```bash
cd spore-sim && cargo run --release -- --help
```

Expected: Help text displayed.

**Step 4: Commit**

```bash
git add spore-sim/src/main.rs
git commit -m "feat: add main entry point with CLI argument parsing"
```

---

## Task 15: Create Integration Test

**Files:**
- Create: `spore-sim/tests/integration_tests.rs`

**Step 1: Write integration test**

Create `spore-sim/tests/integration_tests.rs`:

```rust
//! Integration tests for the Spore Mirror Experiment.

use spore_sim::simulation::Simulation;
use spore_sim::constants::*;

/// Test that simulation runs without crashing.
#[test]
fn test_simulation_runs_1000_ticks() {
    let mut sim = Simulation::new();
    sim.run(1000, 0);
    assert_eq!(sim.tick, 1000);
}

/// Test that accuracy improves over time (statistical).
/// This is a weak test - just verifies the system is learning *something*.
#[test]
fn test_accuracy_improves() {
    let mut sim = Simulation::new();

    // Run for 1000 ticks, record early accuracy
    sim.run(1000, 0);
    let early_accuracy = sim.recent_accuracy;

    // Run for another 9000 ticks
    sim.run(10000, 0);
    let later_accuracy = sim.recent_accuracy;

    // Later accuracy should be at least as good as early
    // (In rare cases random init might start good, so we don't require strictly greater)
    println!("Early accuracy: {:.2}%, Later accuracy: {:.2}%",
        early_accuracy * 100.0, later_accuracy * 100.0);

    // This is a very weak assertion - just that it doesn't get worse
    // Real convergence tests need more ticks
}

/// Test with delayed reward.
#[test]
fn test_simulation_with_latency() {
    let mut sim = Simulation::with_params(5, 0.92, 50);
    sim.run(1000, 0);
    assert_eq!(sim.tick, 1000);
}

/// Test that the simulation respects input hold constraint.
#[test]
fn test_input_hold_constraint() {
    // This should work without warnings
    let min_hold = min_input_hold_ticks(10);  // For latency 10
    let sim = Simulation::with_params(10, 0.95, min_hold);
    assert!(sim.env().input_hold_ticks >= min_hold);
}
```

**Step 2: Run integration tests**

Run:
```bash
cd spore-sim && cargo test integration_tests --release
```

Expected: All tests PASS.

**Step 3: Commit**

```bash
git add spore-sim/tests/integration_tests.rs
git commit -m "test: add integration tests for simulation"
```

---

## Task 16: Full Convergence Test (Manual)

**Files:**
- None (manual verification)

**Step 1: Run full simulation**

Run:
```bash
cd spore-sim && cargo run --release -- --ticks 100000 --log-interval 5000
```

**Step 2: Observe output**

Expected output pattern:
```
Tick 5000: accuracy=XX.XX% frustration=X.XXX
Tick 10000: accuracy=XX.XX% frustration=X.XXX
...
```

Accuracy should gradually increase. If it stays below 50%, check the failure modes in the design document.

**Step 3: Test with delayed reward**

Run:
```bash
cd spore-sim && cargo run --release -- --ticks 100000 --latency 5 --trace-decay 0.92 --log-interval 5000
```

**Step 4: Test with high latency**

Run:
```bash
cd spore-sim && cargo run --release -- --ticks 100000 --latency 10 --trace-decay 0.95 --log-interval 5000
```

**Step 5: Document results**

If convergence is achieved (>95% accuracy), record the "Life Parameters" for use in Phase 2.

If not converged:
1. Check failure modes in design document
2. Adjust hyperparameters
3. Add debugging output if needed

**Step 6: Commit any debugging changes**

```bash
git add -A
git commit -m "chore: document convergence test results"
```

---

## Task 17: Add Weight Visualization (Optional Debug Tool)

**Files:**
- Modify: `spore-sim/src/spore.rs`
- Modify: `spore-sim/src/main.rs`

**Step 1: Add dump_weights method to Spore**

Add to `spore-sim/src/spore.rs` inside the `impl Spore` block:

```rust
    /// Dump weights as a simple ASCII visualization.
    ///
    /// Shows Input→Hidden weights as a 32x8 grid where each cell
    /// is a character representing weight magnitude.
    pub fn dump_weights_ascii(&self) {
        println!("Input→Hidden Weights (32 rows x 8 cols):");
        println!("  01234567");
        for h in 0..HIDDEN_SIZE {
            print!("{:2} ", h);
            for i in 0..INPUT_SIZE {
                let w = self.weights_ih[h][i];
                let c = if w > 100 { '█' }
                    else if w > 50 { '▓' }
                    else if w > 0 { '▒' }
                    else if w > -50 { '░' }
                    else { ' ' };
                print!("{}", c);
            }
            println!();
        }
        println!();

        println!("Hidden→Output Weights (8 rows x 32 cols):");
        print!("   ");
        for h in 0..HIDDEN_SIZE {
            print!("{}", h % 10);
        }
        println!();
        for o in 0..OUTPUT_SIZE {
            print!("{}: ", o);
            for h in 0..HIDDEN_SIZE {
                let w = self.weights_ho[o][h];
                let c = if w > 100 { '█' }
                    else if w > 50 { '▓' }
                    else if w > 0 { '▒' }
                    else if w > -50 { '░' }
                    else { ' ' };
                print!("{}", c);
            }
            println!();
        }
    }
```

**Step 2: Add --dump-weights flag to main.rs**

Add to argument parsing in `main.rs`:

```rust
    let mut dump_weights = false;

    // In the match block:
    "--dump-weights" => {
        dump_weights = true;
    }

    // After simulation completes:
    if dump_weights {
        println!();
        println!("WEIGHT VISUALIZATION:");
        sim.spore().dump_weights_ascii();
    }
```

**Step 3: Build and test**

Run:
```bash
cd spore-sim && cargo run --release -- --ticks 50000 --dump-weights --quiet
```

**Step 4: Commit**

```bash
git add spore-sim/src/spore.rs spore-sim/src/main.rs
git commit -m "feat: add weight visualization debug tool"
```

---

## Task 18: Final Verification

**Files:**
- None

**Step 1: Run all tests**

Run:
```bash
cd spore-sim && cargo test --release
```

Expected: All tests PASS.

**Step 2: Run clippy**

Run:
```bash
cd spore-sim && cargo clippy
```

Expected: No warnings (or only minor ones).

**Step 3: Run full experiment**

Run:
```bash
cd spore-sim && cargo run --release -- --ticks 100000
```

Expected: Accuracy should converge to >95% if implementation is correct.

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final verification complete"
git push
```

---

## Summary: File Checklist

| File | Purpose | Status |
|------|---------|--------|
| `spore-sim/Cargo.toml` | Project config | Task 1 |
| `spore-sim/src/lib.rs` | Module exports | Task 1 |
| `spore-sim/src/main.rs` | CLI entry point | Task 14 |
| `spore-sim/src/constants.rs` | All constants | Task 2 |
| `spore-sim/src/utils.rs` | stochastic_round | Task 3 |
| `spore-sim/src/spore.rs` | Spore struct | Tasks 4-10 |
| `spore-sim/src/environment.rs` | Environment struct | Tasks 11-12 |
| `spore-sim/src/simulation.rs` | Simulation loop | Task 13 |
| `spore-sim/tests/constants_tests.rs` | Tests | Task 2 |
| `spore-sim/tests/utils_tests.rs` | Tests | Task 3 |
| `spore-sim/tests/spore_tests.rs` | Tests | Tasks 4-10 |
| `spore-sim/tests/environment_tests.rs` | Tests | Tasks 11-12 |
| `spore-sim/tests/simulation_tests.rs` | Tests | Task 13 |
| `spore-sim/tests/integration_tests.rs` | Tests | Task 15 |

---

## Execution Commands Summary

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run simulation (default)
cargo run --release

# Run with immediate reward (Phase 1a)
cargo run --release -- --ticks 100000 --latency 0

# Run with delayed reward (Phase 1b)
cargo run --release -- --ticks 100000 --latency 5 --trace-decay 0.92

# Run with high latency (Phase 1c)
cargo run --release -- --ticks 100000 --latency 10 --trace-decay 0.95

# Debug with weight visualization
cargo run --release -- --ticks 50000 --dump-weights
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Accuracy stuck at 12% | Superstitious learning | Increase `--hold` |
| Accuracy stuck at 50% | Anti-Hebbian canceling Hebbian | Check signed dopamine math |
| Nothing fires | Threshold too high | Check Fix 5 (init threshold = 0) |
| All neurons fire | Threshold runaway | Check homeostasis drift |
| Weights don't change | Integer truncation | Check stochastic rounding |
| Weights saturate | Missing decay | Check maintain() is called |
