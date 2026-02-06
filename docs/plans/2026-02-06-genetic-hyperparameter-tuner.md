# Genetic Hyperparameter Tuner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a genetic algorithm that evolves optimal hyperparameters for the Spore neural network, proving that Hebbian learning + Dopamine can achieve >95% stable byte-copy accuracy.

**Architecture:** A population of 50 Genomes (each containing 6 hyperparameters) evolves over 20 generations. Each genome is evaluated by running a Spore simulation for 20,000 ticks and scoring based on stable convergence (Option B: must hold >85% accuracy for 1000 consecutive ticks after hitting 90%). Rayon parallelizes evaluation across CPU cores. Top candidates pass a Final Exam (full 20k ticks × 3 runs, no early exit).

**Tech Stack:** Rust (std), rand 0.8, rayon 1.10, serde 1.0, serde_json 1.0

**Reference Design:** Brainstorm session results (Sections 1-5)

---

## Task 1: Update Cargo.toml with New Dependencies

**Files:**
- Modify: `spore-sim/Cargo.toml`

**Step 1: Add rayon, serde, serde_json dependencies**

Update `spore-sim/Cargo.toml`:

```toml
[package]
name = "spore-sim"
version = "0.1.0"
edition = "2021"
description = "Spore Mirror Experiment - Hebbian learning proof of concept"

[dependencies]
rand = "0.8"
rayon = "1.10"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[dev-dependencies]
# None needed yet

[[bin]]
name = "spore-sim"
path = "src/main.rs"

[lib]
name = "spore_sim"
path = "src/lib.rs"
```

**Step 2: Verify it compiles**

Run:
```bash
cd spore-sim && cargo build
```

Expected: Compiles successfully (downloads new deps).

**Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "chore: add rayon, serde, serde_json dependencies for tuner"
```

---

## Task 2: Add frustration_alpha and weight_decay_interval to Spore

**Files:**
- Modify: `spore-sim/src/spore.rs`
- Test: `spore-sim/tests/spore_tests.rs`

**Step 1: Write tests for new fields and behavior**

Add to `spore-sim/tests/spore_tests.rs`:

```rust
#[test]
fn test_spore_default_frustration_alpha() {
    let spore = Spore::new();
    assert_eq!(spore.frustration_alpha, 0.2);
}

#[test]
fn test_spore_default_weight_decay_interval() {
    let spore = Spore::new();
    assert_eq!(spore.weight_decay_interval, WEIGHT_DECAY_INTERVAL as u64);
}

#[test]
fn test_spore_with_full_params() {
    let spore = Spore::with_full_params(0.1, 0.95, 0.001, 0.03, 0.1, 150);
    assert_eq!(spore.learning_rate, 0.1);
    assert_eq!(spore.trace_decay, 0.95);
    assert_eq!(spore.base_noise, 0.001);
    assert_eq!(spore.max_noise_boost, 0.03);
    assert_eq!(spore.frustration_alpha, 0.1);
    assert_eq!(spore.weight_decay_interval, 150);
}

#[test]
fn test_receive_reward_uses_custom_frustration_alpha() {
    let mut spore = Spore::with_full_params(
        DEFAULT_LEARNING_RATE as f32,
        DEFAULT_TRACE_DECAY as f32,
        DEFAULT_BASE_NOISE as f32,
        DEFAULT_MAX_NOISE_BOOST as f32,
        0.1,  // Custom alpha
        WEIGHT_DECAY_INTERVAL as u64,
    );
    spore.frustration = 1.0;
    spore.receive_reward(8);  // 100% accuracy

    // frustration = (1.0 - 0.1) * 1.0 + 0.1 * (1.0 - 1.0) = 0.9
    assert!((spore.frustration - 0.9).abs() < 0.001);
}

#[test]
fn test_receive_reward_instant_spike_preserved() {
    // Fix 2 instant spike must work regardless of alpha
    let mut spore = Spore::with_full_params(
        DEFAULT_LEARNING_RATE as f32,
        DEFAULT_TRACE_DECAY as f32,
        DEFAULT_BASE_NOISE as f32,
        DEFAULT_MAX_NOISE_BOOST as f32,
        0.01,  // Very low alpha
        WEIGHT_DECAY_INTERVAL as u64,
    );
    spore.frustration = 0.0;
    spore.receive_reward(3);  // 37.5% < 50%

    assert_eq!(spore.frustration, 1.0, "Instant spike must still fire");
}

#[test]
fn test_maintain_uses_custom_weight_decay_interval() {
    let mut spore = Spore::with_full_params(
        DEFAULT_LEARNING_RATE as f32,
        DEFAULT_TRACE_DECAY as f32,
        DEFAULT_BASE_NOISE as f32,
        DEFAULT_MAX_NOISE_BOOST as f32,
        0.2,
        50,  // Decay every 50 ticks instead of 100
    );
    spore.weights_ih[0][0] = 64;

    // At tick 50, should decay (custom interval)
    spore.maintain(50);
    assert_eq!(spore.weights_ih[0][0], 63, "Decay should happen at custom interval");

    // Reset and verify default interval would NOT trigger
    spore.weights_ih[0][0] = 64;
    spore.maintain(75);  // Not a multiple of 50
    assert_eq!(spore.weights_ih[0][0], 64, "No decay at non-interval tick");
}
```

**Step 2: Run tests to verify they fail**

Run:
```bash
cd spore-sim && cargo test spore_tests 2>&1 | tail -20
```

Expected: FAIL (fields don't exist yet).

**Step 3: Add fields and constructor to Spore**

In `spore-sim/src/spore.rs`, add two new fields after `max_noise_boost` in the struct:

```rust
    /// Maximum noise boost at full frustration
    pub max_noise_boost: f32,

    /// EMA alpha for frustration updates (higher = more reactive)
    /// Only used when accuracy >= 50%; below 50% always spikes to 1.0
    pub frustration_alpha: f32,

    /// Ticks between weight decay applications
    pub weight_decay_interval: u64,
}
```

Add `with_full_params` constructor inside `impl Spore`:

```rust
    /// Create a new Spore with all hyperparameters specified.
    pub fn with_full_params(
        learning_rate: f32,
        trace_decay: f32,
        base_noise: f32,
        max_noise_boost: f32,
        frustration_alpha: f32,
        weight_decay_interval: u64,
    ) -> Self {
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
            thresholds_h: [INIT_THRESHOLD as i16; HIDDEN_SIZE],
            thresholds_o: [INIT_THRESHOLD as i16; OUTPUT_SIZE],
            traces_ih: [[0.0; INPUT_SIZE]; HIDDEN_SIZE],
            traces_ho: [[0.0; HIDDEN_SIZE]; OUTPUT_SIZE],
            traces_th: [0.0; HIDDEN_SIZE],
            traces_to: [0.0; OUTPUT_SIZE],
            hidden: [false; HIDDEN_SIZE],
            hidden_next: [false; HIDDEN_SIZE],
            output: [false; OUTPUT_SIZE],
            output_next: [false; OUTPUT_SIZE],
            dopamine: 0.0,
            frustration: 1.0,
            learning_rate,
            trace_decay,
            base_noise,
            max_noise_boost,
            frustration_alpha,
            weight_decay_interval,
        }
    }
```

Update existing `with_params` to delegate:

```rust
    pub fn with_params(
        learning_rate: f32,
        trace_decay: f32,
        base_noise: f32,
        max_noise_boost: f32,
    ) -> Self {
        Self::with_full_params(
            learning_rate,
            trace_decay,
            base_noise,
            max_noise_boost,
            0.2,  // Default frustration_alpha
            WEIGHT_DECAY_INTERVAL as u64,  // Default weight_decay_interval
        )
    }
```

**Step 4: Update receive_reward to use self.frustration_alpha**

In `spore-sim/src/spore.rs`, change the `receive_reward` method:

Replace:
```rust
        // Fix 2: Fast frustration response
        if accuracy < 0.5 {
            self.frustration = 1.0;  // Instant spike
        } else {
            // EMA with faster alpha (0.2 instead of 0.1)
            self.frustration = 0.8 * self.frustration + 0.2 * (1.0 - accuracy);
        }
```

With:
```rust
        // Fix 2: Fast frustration response
        if accuracy < 0.5 {
            self.frustration = 1.0;  // Instant spike (non-negotiable)
        } else {
            // Tunable EMA for fine-tuning phase
            self.frustration = (1.0 - self.frustration_alpha) * self.frustration
                + self.frustration_alpha * (1.0 - accuracy);
        }
```

**Step 5: Update maintain to use self.weight_decay_interval**

In `spore-sim/src/spore.rs`, in the `maintain` method, replace:

```rust
        if tick % (WEIGHT_DECAY_INTERVAL as u64) == 0 && tick > 0 {
```

With:
```rust
        if self.weight_decay_interval > 0 && tick % self.weight_decay_interval == 0 && tick > 0 {
```

**Step 6: Run tests**

Run:
```bash
cd spore-sim && cargo test spore_tests 2>&1
```

Expected: ALL tests PASS (both new and existing, since defaults match original behavior).

**Step 7: Commit**

```bash
git add src/spore.rs tests/spore_tests.rs
git commit -m "feat: add configurable frustration_alpha and weight_decay_interval to Spore"
```

---

## Task 3: Add Simulation::with_full_params and Return Per-Tick Accuracy from step()

**Files:**
- Modify: `spore-sim/src/simulation.rs`
- Test: `spore-sim/tests/simulation_tests.rs`

**CRITICAL CONTEXT (C1/C2 fix):** The tuner needs **instantaneous per-tick accuracy**, not the EMA
(`recent_accuracy`). The EMA has alpha=0.05, meaning it lags true accuracy by ~30-45 ticks on step
changes. This makes convergence revocation sluggish -- a crash from 95% to 30% takes ~15 ticks
before the EMA drops below 85%. The tuner MUST use per-tick accuracy for stability detection.

**Step 1: Write tests**

Add to `spore-sim/tests/simulation_tests.rs`:

```rust
#[test]
fn test_simulation_with_full_params() {
    let sim = Simulation::with_full_params(
        0,      // reward_latency
        0.95,   // trace_decay
        80,     // input_hold_ticks
        0.15,   // learning_rate
        0.03,   // max_noise_boost
        120,    // weight_decay_interval
        0.1,    // frustration_alpha
    );
    assert_eq!(sim.tick, 0);
    assert_eq!(sim.spore().learning_rate, 0.15);
    assert_eq!(sim.spore().trace_decay, 0.95);
    assert_eq!(sim.spore().max_noise_boost, 0.03);
    assert_eq!(sim.spore().weight_decay_interval, 120);
    assert_eq!(sim.spore().frustration_alpha, 0.1);
    assert_eq!(sim.spore().base_noise, DEFAULT_BASE_NOISE as f32);
    assert_eq!(sim.env().input_hold_ticks, 80);
}

#[test]
fn test_simulation_step_returns_per_tick_accuracy() {
    let mut sim = Simulation::new();
    // step() should return Some(accuracy) when a reward is delivered, None otherwise
    // With reward_latency=0, every tick delivers a reward
    let result = sim.step();
    assert!(result.is_some(), "step() should return accuracy when reward delivered");
    let acc = result.unwrap();
    assert!(acc >= 0.0 && acc <= 1.0, "Accuracy must be in [0, 1], got {}", acc);
}
```

**Step 2: Run tests to verify they fail**

Run:
```bash
cd spore-sim && cargo test test_simulation_with_full_params test_simulation_step_returns 2>&1 | tail -10
```

Expected: FAIL (methods/signatures don't exist).

**Step 3: Implement with_full_params**

Add to `spore-sim/src/simulation.rs` inside `impl Simulation`:

```rust
    /// Create a new Simulation with all tunable parameters.
    ///
    /// Used by the genetic tuner and --params CLI flag.
    pub fn with_full_params(
        reward_latency: u64,
        trace_decay: f32,
        input_hold_ticks: u64,
        learning_rate: f32,
        max_noise_boost: f32,
        weight_decay_interval: u64,
        frustration_alpha: f32,
    ) -> Self {
        let spore = Spore::with_full_params(
            learning_rate,
            trace_decay,
            DEFAULT_BASE_NOISE as f32,  // Fixed at 0.001 (not tuned)
            max_noise_boost,
            frustration_alpha,
            weight_decay_interval,
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
```

**Step 4: Change step() to return Option<f32> (per-tick accuracy)**

In `spore-sim/src/simulation.rs`, replace the `step` method:

```rust
    /// Execute one simulation step.
    ///
    /// Returns `Some(accuracy)` when a reward is delivered this tick (instantaneous
    /// per-tick accuracy as correct_bits/8.0), or `None` if no reward was due.
    ///
    /// CRITICAL: The tuner uses this return value for stability detection.
    /// Do NOT use `recent_accuracy` (EMA) for stability -- it lags by ~30-45 ticks.
    pub fn step(&mut self) -> Option<f32> {
        self.spore.tick_end();

        let input = self.env.get_input();
        self.spore.propagate(input);
        let output = self.spore.output_as_byte();

        let tick_accuracy = if let Some(correct_bits) = self.env.tick(self.tick, output) {
            self.spore.receive_reward(correct_bits);
            let accuracy = correct_bits as f32 / 8.0;
            self.recent_accuracy = 0.95 * self.recent_accuracy + 0.05 * accuracy;
            Some(accuracy)
        } else {
            None
        };

        self.spore.learn();
        self.spore.maintain(self.tick);
        self.tick += 1;

        tick_accuracy
    }
```

**Step 5: Update run() to use new step() return type**

In `spore-sim/src/simulation.rs`, update the `run` method to ignore the return value:

Replace:
```rust
    pub fn run(&mut self, max_ticks: u64, log_interval: u64) -> f32 {
        while self.tick < max_ticks {
            self.step();
```

With:
```rust
    pub fn run(&mut self, max_ticks: u64, log_interval: u64) -> f32 {
        while self.tick < max_ticks {
            let _ = self.step();
```

**Step 6: Run tests**

Run:
```bash
cd spore-sim && cargo test simulation_tests 2>&1
```

Expected: ALL tests PASS.

**Step 7: Commit**

```bash
git add src/simulation.rs tests/simulation_tests.rs
git commit -m "feat: add Simulation::with_full_params, step() returns per-tick accuracy"
```

---

## Task 4: Verify All Existing Tests Pass

**Files:**
- None (verification only)

**Step 1: Run all tests**

Run:
```bash
cd spore-sim && cargo test --release 2>&1
```

Expected: ALL tests PASS (backward compatibility preserved).

If any tests fail, fix them before proceeding.

---

## Task 5: Create tuner.rs - Genome Struct

**Files:**
- Create: `spore-sim/src/tuner.rs`
- Modify: `spore-sim/src/lib.rs`
- Create: `spore-sim/tests/tuner_tests.rs`

**Step 1: Write tests for Genome**

Create `spore-sim/tests/tuner_tests.rs`:

```rust
use spore_sim::tuner::Genome;

#[test]
fn test_genome_random_in_range() {
    for _ in 0..50 {
        let g = Genome::random();
        assert!(g.learning_rate >= 0.05 && g.learning_rate <= 0.5,
            "learning_rate {} out of range", g.learning_rate);
        assert!(g.trace_decay >= 0.85 && g.trace_decay <= 0.999,
            "trace_decay {} out of range", g.trace_decay);
        assert!(g.max_noise_boost >= 0.01 && g.max_noise_boost <= 0.05,
            "max_noise_boost {} out of range", g.max_noise_boost);
        assert!(g.weight_decay_interval >= 50 && g.weight_decay_interval <= 200,
            "weight_decay_interval {} out of range", g.weight_decay_interval);
        assert!(g.frustration_alpha >= 0.05 && g.frustration_alpha <= 0.5,
            "frustration_alpha {} out of range", g.frustration_alpha);
        assert!(g.input_hold_ticks >= 20 && g.input_hold_ticks <= 200,
            "input_hold_ticks {} out of range", g.input_hold_ticks);
    }
}

#[test]
fn test_genome_mutate_stays_in_range() {
    for _ in 0..100 {
        let mut g = Genome::random();
        g.mutate(1.0);  // Normal magnitude
        assert!(g.learning_rate >= 0.05 && g.learning_rate <= 0.5);
        assert!(g.trace_decay >= 0.85 && g.trace_decay <= 0.999);
        assert!(g.max_noise_boost >= 0.01 && g.max_noise_boost <= 0.05);
        assert!(g.weight_decay_interval >= 50 && g.weight_decay_interval <= 200);
        assert!(g.frustration_alpha >= 0.05 && g.frustration_alpha <= 0.5);
        assert!(g.input_hold_ticks >= 20 && g.input_hold_ticks <= 200);
    }
}

#[test]
fn test_genome_mutate_boosted_stays_in_range() {
    for _ in 0..100 {
        let mut g = Genome::random();
        g.mutate(2.0);  // Boosted magnitude
        assert!(g.learning_rate >= 0.05 && g.learning_rate <= 0.5);
        assert!(g.trace_decay >= 0.85 && g.trace_decay <= 0.999);
        assert!(g.max_noise_boost >= 0.01 && g.max_noise_boost <= 0.05);
        assert!(g.weight_decay_interval >= 50 && g.weight_decay_interval <= 200);
        assert!(g.frustration_alpha >= 0.05 && g.frustration_alpha <= 0.5);
        assert!(g.input_hold_ticks >= 20 && g.input_hold_ticks <= 200);
    }
}

#[test]
fn test_genome_mutate_changes_values() {
    // Over many mutations, at least SOME values should change
    let original = Genome::random();
    let mut changed_count = 0;

    for _ in 0..100 {
        let mut g = original.clone();
        g.mutate(1.0);
        if (g.learning_rate - original.learning_rate).abs() > 0.0001 {
            changed_count += 1;
        }
    }

    // With 10% mutation probability per gene, ~10% should change
    assert!(changed_count > 0, "Mutation should change values sometimes");
}

#[test]
fn test_genome_crossover_combines_parents() {
    // Create two very different parents
    let mut parent_a = Genome::random();
    let mut parent_b = Genome::random();
    parent_a.learning_rate = 0.05;
    parent_b.learning_rate = 0.5;
    parent_a.trace_decay = 0.85;
    parent_b.trace_decay = 0.999;

    // Over many crossovers, child should get genes from both parents
    let mut got_a_lr = false;
    let mut got_b_lr = false;

    for _ in 0..100 {
        let child = Genome::crossover(&parent_a, &parent_b);
        if (child.learning_rate - 0.05).abs() < 0.001 {
            got_a_lr = true;
        }
        if (child.learning_rate - 0.5).abs() < 0.001 {
            got_b_lr = true;
        }
    }

    assert!(got_a_lr && got_b_lr, "Crossover should use genes from both parents");
}

#[test]
fn test_genome_serialization() {
    let g = Genome::random();
    let json = serde_json::to_string(&g).unwrap();
    let g2: Genome = serde_json::from_str(&json).unwrap();
    assert!((g.learning_rate - g2.learning_rate).abs() < 0.0001);
    assert!((g.trace_decay - g2.trace_decay).abs() < 0.0001);
    assert_eq!(g.weight_decay_interval, g2.weight_decay_interval);
    assert_eq!(g.input_hold_ticks, g2.input_hold_ticks);
}
```

**Step 2: Run tests to verify they fail**

Run:
```bash
cd spore-sim && cargo test tuner_tests 2>&1 | tail -5
```

Expected: FAIL (module doesn't exist).

**Step 3: Create tuner.rs with Genome struct**

Create `spore-sim/src/tuner.rs`:

```rust
//! Genetic hyperparameter tuner for the Spore neural network.
//!
//! Evolves optimal hyperparameters using a genetic algorithm:
//! - Population of 50 Genomes
//! - 20 generations of evolution
//! - Crossover + mutation + diversity injection
//! - Parallel evaluation with rayon
//! - Final Exam on top candidates (full marathon, 3 runs)

use rand::Rng;
use serde::{Serialize, Deserialize};

/// A genome encoding 6 tunable hyperparameters for the Spore network.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    /// Learning rate for Hebbian weight updates. Range: 0.05 - 0.5
    pub learning_rate: f32,

    /// Eligibility trace decay per tick. Range: 0.85 - 0.999
    /// Exponentially sensitive - small changes have large effects.
    pub trace_decay: f32,

    /// Maximum noise boost at full frustration. Range: 0.01 - 0.05
    pub max_noise_boost: f32,

    /// Ticks between weight decay applications. Range: 50 - 200
    pub weight_decay_interval: u64,

    /// EMA alpha for frustration updates (>50% accuracy). Range: 0.05 - 0.5
    pub frustration_alpha: f32,

    /// Ticks to hold each input pattern. Range: 20 - 200
    pub input_hold_ticks: u64,
}

impl Genome {
    /// Create a random genome with all genes in valid ranges.
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            learning_rate: rng.gen_range(0.05..=0.5),
            trace_decay: rng.gen_range(0.85..=0.999),
            max_noise_boost: rng.gen_range(0.01..=0.05),
            weight_decay_interval: rng.gen_range(50..=200),
            frustration_alpha: rng.gen_range(0.05..=0.5),
            input_hold_ticks: rng.gen_range(20..=200),
        }
    }

    /// Mutate this genome in-place.
    ///
    /// Each gene has a 10% independent chance of mutation.
    /// `magnitude_mult` scales the STEP SIZE (not probability).
    /// Use magnitude_mult=2.0 to escape local minima.
    pub fn mutate(&mut self, magnitude_mult: f32) {
        let mut rng = rand::thread_rng();

        if rng.gen::<f32>() < 0.1 {
            self.learning_rate += rng.gen_range(-0.02..=0.02) * magnitude_mult;
            self.learning_rate = self.learning_rate.clamp(0.05, 0.5);
        }
        if rng.gen::<f32>() < 0.1 {
            // Fine-grained: trace_decay is exponentially sensitive
            self.trace_decay += rng.gen_range(-0.005..=0.005) * magnitude_mult;
            self.trace_decay = self.trace_decay.clamp(0.85, 0.999);
        }
        if rng.gen::<f32>() < 0.1 {
            self.max_noise_boost += rng.gen_range(-0.005..=0.005) * magnitude_mult;
            self.max_noise_boost = self.max_noise_boost.clamp(0.01, 0.05);
        }
        if rng.gen::<f32>() < 0.1 {
            let delta = (rng.gen_range(-10..=10) as f32 * magnitude_mult) as i64;
            self.weight_decay_interval = (self.weight_decay_interval as i64 + delta)
                .clamp(50, 200) as u64;
        }
        if rng.gen::<f32>() < 0.1 {
            self.frustration_alpha += rng.gen_range(-0.02..=0.02) * magnitude_mult;
            self.frustration_alpha = self.frustration_alpha.clamp(0.05, 0.5);
        }
        if rng.gen::<f32>() < 0.1 {
            let delta = (rng.gen_range(-10..=10) as f32 * magnitude_mult) as i64;
            self.input_hold_ticks = (self.input_hold_ticks as i64 + delta)
                .clamp(20, 200) as u64;
        }
    }

    /// Sexual reproduction: combine genes from two parents.
    ///
    /// Each gene is independently selected from parent_a or parent_b
    /// with equal probability (uniform crossover).
    pub fn crossover(parent_a: &Genome, parent_b: &Genome) -> Genome {
        let mut rng = rand::thread_rng();
        Genome {
            learning_rate: if rng.gen() { parent_a.learning_rate } else { parent_b.learning_rate },
            trace_decay: if rng.gen() { parent_a.trace_decay } else { parent_b.trace_decay },
            max_noise_boost: if rng.gen() { parent_a.max_noise_boost } else { parent_b.max_noise_boost },
            weight_decay_interval: if rng.gen() { parent_a.weight_decay_interval } else { parent_b.weight_decay_interval },
            frustration_alpha: if rng.gen() { parent_a.frustration_alpha } else { parent_b.frustration_alpha },
            input_hold_ticks: if rng.gen() { parent_a.input_hold_ticks } else { parent_b.input_hold_ticks },
        }
    }
}
```

**Step 4: Add module to lib.rs**

Update `spore-sim/src/lib.rs`:

```rust
pub mod constants;
pub mod utils;
pub mod spore;
pub mod environment;
pub mod simulation;
pub mod tuner;
```

**Step 5: Run tests**

Run:
```bash
cd spore-sim && cargo test tuner_tests 2>&1
```

Expected: ALL tests PASS.

**Step 6: Commit**

```bash
git add src/tuner.rs src/lib.rs tests/tuner_tests.rs
git commit -m "feat: add Genome struct with mutation, crossover, and serialization"
```

---

## Task 6: Add Evaluation Functions to tuner.rs

**Files:**
- Modify: `spore-sim/src/tuner.rs`
- Modify: `spore-sim/tests/tuner_tests.rs`

**Step 1: Write tests for evaluation**

Add to `spore-sim/tests/tuner_tests.rs`:

```rust
use spore_sim::tuner::{Genome, EvalResult, evaluate_fast, evaluate_full};

#[test]
fn test_eval_result_score_is_finite() {
    let g = Genome::random();
    let result = evaluate_fast(&g, 1000);
    assert!(result.score.is_finite(), "Score must be finite, got {}", result.score);
    assert!(result.final_accuracy >= 0.0 && result.final_accuracy <= 1.0);
}

#[test]
fn test_evaluate_fast_returns_valid_result() {
    let g = Genome::random();
    let result = evaluate_fast(&g, 2000);
    assert!(result.score.is_finite());
    assert!(result.final_accuracy >= 0.0 && result.final_accuracy <= 1.0);
    // Stable should be bool (always valid)
    let _ = result.stable;
}

#[test]
fn test_evaluate_full_returns_valid_result() {
    let g = Genome::random();
    let result = evaluate_full(&g, 2000);
    assert!(result.score.is_finite());
    assert!(result.final_accuracy >= 0.0 && result.final_accuracy <= 1.0);
}

#[test]
fn test_evaluate_stable_genome_scores_higher() {
    // Run two evaluations: one that's likely stable-ish, one that's likely not
    // At minimum, verify score formula: stable base (2000) always beats unstable (500)
    // for the same accuracy level.
    //
    // Proof: For accuracy a, stable score = a*2000 - t/100. Unstable = a*500.
    // stable > unstable when a*2000 - t/100 > a*500 => a*1500 > t/100 => t < a*150000
    // For any convergence before tick 142500 at 95% accuracy, stable wins.
    let a = 0.95_f32;
    let t = 5000_u64;
    let stable_score = a * 2000.0 - (t as f32 / 100.0);
    let unstable_score = a * 500.0;
    assert!(stable_score > unstable_score,
        "Stable score {:.1} should beat unstable {:.1}", stable_score, unstable_score);

    // Also verify degenerate case: very late convergence still beats unstable
    let late_t = 19000_u64;
    let late_stable_score = a * 2000.0 - (late_t as f32 / 100.0);
    assert!(late_stable_score > unstable_score,
        "Late stable {:.1} should still beat unstable {:.1}", late_stable_score, unstable_score);
}

#[test]
fn test_evaluate_fast_different_runs_may_differ() {
    // Multi-run robustness: evaluate_fast runs 3 times and takes worst.
    // Verify it returns a valid result (we can't test the "worst" logic deterministically
    // without seeded RNG, but we can verify the contract).
    let g = Genome::random();
    let r1 = evaluate_fast(&g, 1000);
    let r2 = evaluate_fast(&g, 1000);
    // Both should be valid, may differ due to different random sequences
    assert!(r1.score.is_finite());
    assert!(r2.score.is_finite());
}
```

**Step 2: Run tests to verify they fail**

Run:
```bash
cd spore-sim && cargo test tuner_tests 2>&1 | tail -10
```

Expected: FAIL (EvalResult, evaluate_fast, evaluate_full don't exist).

**Step 3: Implement EvalResult and evaluation functions**

Add to `spore-sim/src/tuner.rs`:

```rust
use crate::simulation::Simulation;
use crate::constants::DEFAULT_BASE_NOISE;

/// Result of evaluating a single Genome.
#[derive(Clone, Debug)]
pub struct EvalResult {
    /// Fitness score (higher = better)
    pub score: f32,

    /// Mean accuracy since convergence (or final accuracy if never converged)
    pub final_accuracy: f32,

    /// Tick at which stable convergence was achieved (None = never)
    pub convergence_tick: Option<u64>,

    /// Whether the genome achieved stable convergence
    pub stable: bool,
}

/// Fast evaluation for use during evolution loop.
///
/// Runs the genome 3 times (different random sequences) and returns the WORST result.
/// Includes early exit: if stable for 2000+ ticks at 98%+, stop.
pub fn evaluate_fast(genome: &Genome, ticks: u64) -> EvalResult {
    let results: Vec<EvalResult> = (0..3)
        .map(|_| evaluate_single(genome, ticks, true))
        .collect();

    results.into_iter()
        .min_by(|a, b| a.score.total_cmp(&b.score))
        .unwrap()
}

/// Full marathon evaluation for Final Exam.
///
/// Runs the genome 3 times with NO early exit. Must survive all ticks.
/// Returns the WORST result (robustness check).
pub fn evaluate_full(genome: &Genome, ticks: u64) -> EvalResult {
    let results: Vec<EvalResult> = (0..3)
        .map(|_| evaluate_single(genome, ticks, false))
        .collect();

    results.into_iter()
        .min_by(|a, b| a.score.total_cmp(&b.score))
        .unwrap()
}

/// Evaluate a single genome for a given number of ticks.
///
/// IMPORTANT: Uses INSTANTANEOUS per-tick accuracy from sim.step() return value,
/// NOT the EMA (sim.recent_accuracy). The EMA lags by ~30-45 ticks and would make
/// convergence revocation sluggish. Per-tick accuracy gives honest, real-time signal.
///
/// Stability detection (Option B):
/// - Track when accuracy first crosses 90%
/// - If accuracy stays >= 85% for 1000 consecutive ticks → stable convergence
/// - Dip below 85% → reset timer AND revoke convergence
///
/// Score:
/// - Stable: mean_accuracy_since_converge * 2000 - (convergence_tick / 100)
/// - Unstable: recent_accuracy * 500
fn evaluate_single(genome: &Genome, ticks: u64, allow_early_exit: bool) -> EvalResult {
    let mut sim = Simulation::with_full_params(
        0,  // reward_latency (fixed for tuner)
        genome.trace_decay,
        genome.input_hold_ticks,
        genome.learning_rate,
        genome.max_noise_boost,
        genome.weight_decay_interval,
        genome.frustration_alpha,
    );

    let mut stability_window_start: Option<u64> = None;
    let mut convergence_tick: Option<u64> = None;
    let mut accuracy_sum_since_converge: f32 = 0.0;
    let mut ticks_since_converge: u64 = 0;
    let mut last_accuracy: f32 = 0.0;

    for tick in 0..ticks {
        // step() returns Some(accuracy) when reward delivered, None otherwise
        // With reward_latency=0, this is always Some
        let tick_accuracy = sim.step();

        // Use per-tick accuracy for stability detection (NOT the EMA)
        let acc = match tick_accuracy {
            Some(a) => { last_accuracy = a; a }
            None => last_accuracy,  // Hold last known accuracy between rewards
        };

        // Stability detection
        if acc >= 0.90 {
            if stability_window_start.is_none() {
                stability_window_start = Some(tick);
            }
            if tick - stability_window_start.unwrap() >= 1000 && convergence_tick.is_none() {
                convergence_tick = Some(stability_window_start.unwrap());
            }
        } else if acc < 0.85 {
            stability_window_start = None;
            convergence_tick = None;  // REVOKE on crash
            accuracy_sum_since_converge = 0.0;
            ticks_since_converge = 0;
        }
        // 0.85 <= acc < 0.90: timer continues, no action

        // Track integral accuracy after convergence
        if convergence_tick.is_some() {
            accuracy_sum_since_converge += acc;
            ticks_since_converge += 1;

            // Early exit (fast eval only): stable for 2000+ ticks at 98%+
            if allow_early_exit && ticks_since_converge >= 2000 {
                let mean_acc = accuracy_sum_since_converge / ticks_since_converge as f32;
                if mean_acc >= 0.98 {
                    return EvalResult {
                        score: mean_acc * 2000.0 - (convergence_tick.unwrap() as f32 / 100.0),
                        final_accuracy: mean_acc,
                        convergence_tick,
                        stable: true,
                    };
                }
            }
        }
    }

    // End of run scoring
    let stable = convergence_tick.is_some() && ticks_since_converge > 0;
    let mean_acc = if ticks_since_converge > 0 {
        accuracy_sum_since_converge / ticks_since_converge as f32
    } else {
        sim.recent_accuracy  // Fall back to EMA only for unstable genomes
    };

    let score = if stable {
        mean_acc * 2000.0 - (convergence_tick.unwrap() as f32 / 100.0)
    } else {
        sim.recent_accuracy * 500.0
    };

    EvalResult {
        score,
        final_accuracy: mean_acc,
        convergence_tick,
        stable,
    }
}
```

**Step 4: Run tests**

Run:
```bash
cd spore-sim && cargo test tuner_tests 2>&1
```

Expected: ALL tests PASS.

**Step 5: Commit**

```bash
git add src/tuner.rs tests/tuner_tests.rs
git commit -m "feat: add evaluation functions with stability detection and multi-run scoring"
```

---

## Task 7: Add Evolution Loop to tuner.rs

**Files:**
- Modify: `spore-sim/src/tuner.rs`
- Modify: `spore-sim/tests/tuner_tests.rs`

**Step 1: Write test for evolution loop**

Add to `spore-sim/tests/tuner_tests.rs`:

```rust
use spore_sim::tuner::TunerConfig;

#[test]
fn test_tuner_config_default() {
    let config = TunerConfig::default();
    assert_eq!(config.population_size, 50);
    assert_eq!(config.generations, 20);
    assert_eq!(config.elite_count, 10);
    assert_eq!(config.ticks_per_eval, 20_000);
}

#[test]
fn test_tune_tiny_run() {
    // Minimal tuner run to verify it doesn't crash
    let config = TunerConfig {
        population_size: 6,
        generations: 2,
        elite_count: 2,
        ticks_per_eval: 500,
        finalist_count: 2,
    };
    let (best_genome, best_result) = spore_sim::tuner::tune(&config);
    assert!(best_result.score.is_finite());
    assert!(best_genome.learning_rate >= 0.05 && best_genome.learning_rate <= 0.5);
    assert!(best_genome.trace_decay >= 0.85 && best_genome.trace_decay <= 0.999);
}
```

**Step 2: Run tests to verify they fail**

Run:
```bash
cd spore-sim && cargo test test_tune_tiny_run 2>&1 | tail -5
```

Expected: FAIL (TunerConfig, tune don't exist).

**Step 3: Implement TunerConfig and tune()**

Add to `spore-sim/src/tuner.rs`:

```rust
use rayon::prelude::*;

/// Configuration for the genetic tuner.
pub struct TunerConfig {
    /// Number of genomes per generation
    pub population_size: usize,

    /// Number of evolution generations
    pub generations: usize,

    /// Number of elite genomes to preserve (cached, not re-evaluated)
    pub elite_count: usize,

    /// Simulation ticks per evaluation
    pub ticks_per_eval: u64,

    /// Number of finalists for the Final Exam
    pub finalist_count: usize,
}

impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            generations: 20,
            elite_count: 10,
            ticks_per_eval: 20_000,
            finalist_count: 15,
        }
    }
}

/// Run the genetic hyperparameter tuner.
///
/// Returns the best genome and its evaluation result.
///
/// Algorithm:
/// 1. Generate random initial population
/// 2. For each generation:
///    a. Evaluate all new genomes in parallel (rayon)
///    b. Merge with cached elite scores
///    c. Select top N as new elites
///    d. Breed next generation: crossover + mutation + diversity injection
/// 3. Final Exam: top candidates run full marathon (no early exit)
pub fn tune(config: &TunerConfig) -> (Genome, EvalResult) {
    let mut rng = rand::thread_rng();

    // Initial population: all random
    let mut population: Vec<Genome> = (0..config.population_size)
        .map(|_| Genome::random())
        .collect();

    // Elite cache: (genome, score) - NOT re-evaluated each generation
    let mut elite_cache: Vec<(Genome, EvalResult)> = Vec::new();

    let mut best_score: f32 = f32::NEG_INFINITY;
    let mut gens_without_improvement: usize = 0;

    for gen in 0..config.generations {
        // PARALLEL: Evaluate all new genomes
        let new_scored: Vec<(Genome, EvalResult)> = population
            .par_iter()
            .map(|g| (g.clone(), evaluate_fast(g, config.ticks_per_eval)))
            .collect();

        // Merge with cached elites
        let mut all_scored: Vec<(Genome, EvalResult)> = elite_cache.clone();
        all_scored.extend(new_scored);

        // Safe sort (handles NaN)
        all_scored.sort_by(|a, b| b.1.score.total_cmp(&a.1.score));

        // Truncate to reasonable size
        all_scored.truncate(config.population_size + config.elite_count);

        let gen_best = all_scored[0].1.score;

        // Adaptive mutation tracking
        if gen_best > best_score {
            best_score = gen_best;
            gens_without_improvement = 0;
        } else {
            gens_without_improvement += 1;
        }
        let magnitude_mult = if gens_without_improvement >= 3 { 2.0 } else { 1.0 };

        // Progress report (after collect, not during parallel work)
        let stable_count = all_scored.iter().filter(|s| s.1.stable).count();
        let eval_count = all_scored.len().min(config.population_size);
        let avg_score: f32 = all_scored.iter().take(eval_count)
            .map(|s| s.1.score).sum::<f32>() / eval_count as f32;
        eprintln!("Gen {:2}: best={:.1} avg={:.1} stable={}/{}{}",
            gen, gen_best, avg_score, stable_count, eval_count,
            if magnitude_mult > 1.0 { " [BOOST]" } else { "" });

        // Update elite cache (top N with their known scores)
        elite_cache = all_scored[0..config.elite_count.min(all_scored.len())].to_vec();

        // BREED next generation: only new children (elites cached separately)
        let elites: Vec<&Genome> = elite_cache.iter().map(|(g, _)| g).collect();
        let diversity_count = config.population_size / 5;  // 20% fresh randoms
        let children_count = config.population_size - diversity_count;

        population = Vec::with_capacity(config.population_size);

        // Children from crossover + mutation
        for _ in 0..children_count {
            let parent_a = &elites[rng.gen_range(0..elites.len())];
            let parent_b = &elites[rng.gen_range(0..elites.len())];
            let mut child = Genome::crossover(parent_a, parent_b);
            child.mutate(magnitude_mult);
            population.push(child);
        }

        // Diversity injection: fresh randoms
        for _ in 0..diversity_count {
            population.push(Genome::random());
        }
    }

    // FINAL EXAM: Top candidates run full marathon (no early exit)
    let finalist_count = config.finalist_count.min(elite_cache.len());
    eprintln!("\n=== FINAL EXAM (Top {}, Full Marathon, 3 Runs) ===", finalist_count);

    let finalist_genomes: Vec<Genome> = elite_cache[0..finalist_count]
        .iter()
        .map(|(g, _)| g.clone())
        .collect();

    let finalists: Vec<(Genome, EvalResult)> = finalist_genomes
        .par_iter()
        .map(|g| (g.clone(), evaluate_full(g, config.ticks_per_eval)))
        .collect();

    // Print results AFTER parallel work (no race condition)
    for (i, (genome, result)) in finalists.iter().enumerate() {
        eprintln!("  #{:2}: score={:.1} acc={:.2}% stable={} conv@{:?} lr={:.3} td={:.4}",
            i + 1, result.score, result.final_accuracy * 100.0,
            result.stable, result.convergence_tick,
            genome.learning_rate, genome.trace_decay);
    }

    finalists
        .into_iter()
        .max_by(|a, b| a.1.score.total_cmp(&b.1.score))
        .unwrap()
}
```

**Step 4: Run tests**

Run:
```bash
cd spore-sim && cargo test tuner_tests 2>&1
```

Expected: ALL tests PASS.

**Step 5: Commit**

```bash
git add src/tuner.rs tests/tuner_tests.rs
git commit -m "feat: add genetic evolution loop with parallel evaluation and final exam"
```

---

## Task 8: Update main.rs with --tune, --params, and New Flags

**Files:**
- Modify: `spore-sim/src/main.rs`

**Step 1: Implement CLI updates**

Replace the entire `spore-sim/src/main.rs` with:

```rust
//! Spore Mirror Experiment - Phase 1
//!
//! Proves that Hebbian learning + Dopamine reinforcement can evolve
//! a byte-copy reflex from random noise.
//!
//! Usage:
//!   spore-sim [OPTIONS]
//!
//! Modes:
//!   (default)         Run a single simulation
//!   --tune            Run genetic hyperparameter tuner
//!
//! Simulation Options:
//!   --ticks N         Number of ticks to run (default: 100000)
//!   --latency N       Reward latency in ticks (default: 0)
//!   --trace-decay F   Trace decay rate (default: 0.9)
//!   --hold N          Input hold ticks (default: 50)
//!   --learning-rate F Learning rate (default: 0.5)
//!   --noise-boost F   Max noise boost (default: 0.05)
//!   --decay-interval N Weight decay interval (default: 100)
//!   --frustration-alpha F Frustration EMA alpha (default: 0.2)
//!   --log-interval N  Log every N ticks (default: 1000)
//!   --quiet           Suppress logging
//!   --dump-weights    Show ASCII weight visualization at end
//!   --params FILE     Load parameters from JSON file
//!
//! Tuner Options:
//!   --tune            Run genetic hyperparameter tuner
//!   --population N    Tuner population size (default: 50)
//!   --generations N   Tuner generations (default: 20)
//!   --output FILE     Output JSON file (default: best_params.json)

use spore_sim::simulation::Simulation;
use spore_sim::tuner::{self, Genome, TunerConfig};
use spore_sim::constants::*;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Check for --tune mode
    if args.iter().any(|a| a == "--tune") {
        run_tuner(&args);
        return;
    }

    // Normal simulation mode
    run_simulation(&args);
}

fn run_simulation(args: &[String]) {
    // Defaults
    let mut max_ticks: u64 = 100_000;
    let mut reward_latency: u64 = DEFAULT_REWARD_LATENCY as u64;
    let mut trace_decay: f32 = DEFAULT_TRACE_DECAY as f32;
    let mut input_hold_ticks: u64 = DEFAULT_INPUT_HOLD_TICKS as u64;
    let mut learning_rate: f32 = DEFAULT_LEARNING_RATE as f32;
    let mut max_noise_boost: f32 = DEFAULT_MAX_NOISE_BOOST as f32;
    let mut weight_decay_interval: u64 = WEIGHT_DECAY_INTERVAL as u64;
    let mut frustration_alpha: f32 = 0.2;
    let mut log_interval: u64 = 1000;
    let mut quiet = false;
    let mut dump_weights = false;

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
            "--learning-rate" => {
                i += 1;
                learning_rate = args[i].parse().expect("Invalid --learning-rate value");
            }
            "--noise-boost" => {
                i += 1;
                max_noise_boost = args[i].parse().expect("Invalid --noise-boost value");
            }
            "--decay-interval" => {
                i += 1;
                weight_decay_interval = args[i].parse().expect("Invalid --decay-interval value");
            }
            "--frustration-alpha" => {
                i += 1;
                frustration_alpha = args[i].parse().expect("Invalid --frustration-alpha value");
            }
            "--log-interval" => {
                i += 1;
                log_interval = args[i].parse().expect("Invalid --log-interval value");
            }
            "--quiet" => {
                quiet = true;
            }
            "--dump-weights" => {
                dump_weights = true;
            }
            "--params" => {
                i += 1;
                let json = fs::read_to_string(&args[i])
                    .unwrap_or_else(|e| panic!("Failed to read {}: {}", args[i], e));
                let genome: Genome = serde_json::from_str(&json)
                    .unwrap_or_else(|e| panic!("Failed to parse {}: {}", args[i], e));
                // Override with loaded values
                learning_rate = genome.learning_rate;
                trace_decay = genome.trace_decay;
                max_noise_boost = genome.max_noise_boost;
                weight_decay_interval = genome.weight_decay_interval;
                frustration_alpha = genome.frustration_alpha;
                input_hold_ticks = genome.input_hold_ticks;
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
        println!("========================================");
        println!("  SPORE MIRROR EXPERIMENT - PHASE 1    ");
        println!("========================================");
        println!(" Proving: Hebbian + Dopamine = Emergent Byte Copy");
        println!();
        println!("Configuration:");
        println!("  Max ticks:           {}", max_ticks);
        println!("  Reward latency:      {}", reward_latency);
        println!("  Learning rate:       {}", learning_rate);
        println!("  Trace decay:         {}", trace_decay);
        println!("  Max noise boost:     {}", max_noise_boost);
        println!("  Weight decay intv:   {}", weight_decay_interval);
        println!("  Frustration alpha:   {}", frustration_alpha);
        println!("  Input hold ticks:    {}", input_hold_ticks);
        println!();

        // Validate timing constraint (Fix 4)
        let min_hold = min_input_hold_ticks(reward_latency as usize);
        if (input_hold_ticks as usize) < min_hold {
            println!("WARNING: input_hold_ticks ({}) < minimum ({}) for latency {}",
                input_hold_ticks, min_hold, reward_latency);
            println!("   This may cause superstitious learning (Fix 4).");
            println!();
        }

        // Recommend trace decay
        let rec_decay = recommended_trace_decay(reward_latency as usize);
        if (trace_decay as f64) < rec_decay - 0.02 {
            println!("WARNING: trace_decay ({}) may be too fast for latency {}",
                trace_decay, reward_latency);
            println!("   Recommended: >= {:.2}", rec_decay);
            println!();
        }
    }

    // Create and run simulation
    let mut sim = Simulation::with_full_params(
        reward_latency,
        trace_decay,
        input_hold_ticks,
        learning_rate,
        max_noise_boost,
        weight_decay_interval,
        frustration_alpha,
    );

    let actual_log_interval = if quiet { 0 } else { log_interval };
    let final_accuracy = sim.run(max_ticks, actual_log_interval);

    // Dump weights if requested
    if dump_weights {
        println!();
        println!("WEIGHT VISUALIZATION:");
        sim.spore().dump_weights_ascii();
    }

    // Report results
    println!();
    println!("========================================");
    println!("FINAL RESULTS");
    println!("========================================");
    println!("  Ticks run:      {}", sim.tick);
    println!("  Final accuracy: {:.2}%", final_accuracy * 100.0);
    println!("  Frustration:    {:.3}", sim.spore().frustration);
    println!();

    if sim.has_converged(0.95) {
        println!("SUCCESS: Spore learned to mirror! (accuracy > 95%)");
    } else if final_accuracy > 0.8 {
        println!("PARTIAL: Spore is learning but hasn't converged (accuracy > 80%)");
        println!("   Try running for more ticks or tuning hyperparameters.");
    } else if final_accuracy > 0.5 {
        println!("SLOW: Spore is above baseline but learning slowly");
        println!("   Check: learning_rate, trace_decay, input_hold_ticks");
    } else {
        println!("FAILED: Spore did not converge (accuracy <= 50%)");
        println!("   Check failure modes in docs/plans/");
    }
}

fn run_tuner(args: &[String]) {
    let mut config = TunerConfig::default();
    let mut output_file = String::from("best_params.json");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--tune" => {}  // Already handled
            "--population" => {
                i += 1;
                config.population_size = args[i].parse().expect("Invalid --population");
            }
            "--generations" => {
                i += 1;
                config.generations = args[i].parse().expect("Invalid --generations");
            }
            "--ticks" => {
                i += 1;
                config.ticks_per_eval = args[i].parse().expect("Invalid --ticks");
            }
            "--output" | "-o" => {
                i += 1;
                output_file = args[i].clone();
            }
            _ => {}  // Ignore unknown args in tune mode
        }
        i += 1;
    }

    println!("========================================");
    println!("  SPORE GENETIC HYPERPARAMETER TUNER   ");
    println!("========================================");
    println!();
    println!("Configuration:");
    println!("  Population:    {}", config.population_size);
    println!("  Generations:   {}", config.generations);
    println!("  Ticks/eval:    {}", config.ticks_per_eval);
    println!("  Finalists:     {}", config.finalist_count);
    println!("  Output file:   {}", output_file);
    println!();
    println!("Starting evolution...");
    println!();

    let start = std::time::Instant::now();
    let (best_genome, best_result) = tuner::tune(&config);
    let elapsed = start.elapsed();

    println!();
    println!("========================================");
    println!("  EVOLUTION COMPLETE");
    println!("========================================");
    println!("  Time elapsed:  {:.1}s", elapsed.as_secs_f32());
    println!("  Best score:    {:.1}", best_result.score);
    println!("  Mean accuracy: {:.2}%", best_result.final_accuracy * 100.0);
    println!("  Stable:        {}", best_result.stable);
    println!("  Converged at:  {:?}", best_result.convergence_tick);
    println!();
    println!("OPTIMAL PARAMETERS:");
    println!("  learning_rate:         {:.4}", best_genome.learning_rate);
    println!("  trace_decay:           {:.4}", best_genome.trace_decay);
    println!("  max_noise_boost:       {:.5}", best_genome.max_noise_boost);
    println!("  weight_decay_interval: {}", best_genome.weight_decay_interval);
    println!("  frustration_alpha:     {:.4}", best_genome.frustration_alpha);
    println!("  input_hold_ticks:      {}", best_genome.input_hold_ticks);
    println!();

    // Save to JSON
    let json = serde_json::to_string_pretty(&best_genome)
        .expect("Failed to serialize genome");
    fs::write(&output_file, &json)
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", output_file, e));

    println!("Saved to: {}", output_file);
    println!();
    println!("To run with these parameters:");
    println!("  cargo run --release -- --params {}", output_file);
}

fn print_help() {
    println!("Spore Mirror Experiment - Phase 1");
    println!();
    println!("USAGE:");
    println!("  spore-sim [OPTIONS]");
    println!("  spore-sim --tune [TUNER OPTIONS]");
    println!();
    println!("SIMULATION OPTIONS:");
    println!("  --ticks N             Number of ticks to run (default: 100000)");
    println!("  --latency N           Reward latency in ticks (default: 0)");
    println!("  --trace-decay F       Trace decay rate (default: 0.9)");
    println!("  --hold N              Input hold ticks (default: 50)");
    println!("  --learning-rate F     Learning rate (default: 0.5)");
    println!("  --noise-boost F       Max noise boost (default: 0.05)");
    println!("  --decay-interval N    Weight decay interval (default: 100)");
    println!("  --frustration-alpha F Frustration EMA alpha (default: 0.2)");
    println!("  --log-interval N      Log every N ticks (default: 1000)");
    println!("  --quiet               Suppress logging");
    println!("  --dump-weights        Show ASCII weight visualization at end");
    println!("  --params FILE         Load parameters from JSON file");
    println!("  --help, -h            Show this help");
    println!();
    println!("TUNER OPTIONS:");
    println!("  --tune                Run genetic hyperparameter tuner");
    println!("  --population N        Tuner population size (default: 50)");
    println!("  --generations N       Tuner generations (default: 20)");
    println!("  --ticks N             Ticks per evaluation (default: 20000)");
    println!("  --output FILE, -o     Output JSON file (default: best_params.json)");
}
```

**Step 2: Build and test**

Run:
```bash
cd spore-sim && cargo build --release 2>&1
```

Expected: Compiles successfully.

**Step 3: Test help text**

Run:
```bash
cd spore-sim && cargo run --release -- --help 2>&1
```

Expected: Shows combined help text with simulation and tuner options.

**Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat: add --tune, --params, and all hyperparameter CLI flags"
```

---

## Task 9: Run All Tests

**Files:**
- None (verification only)

**Step 1: Run all tests**

Run:
```bash
cd spore-sim && cargo test --release 2>&1
```

Expected: ALL tests PASS.

If any tests fail, fix them before proceeding.

**Step 2: Run clippy**

Run:
```bash
cd spore-sim && cargo clippy 2>&1
```

Expected: No errors (warnings are acceptable).

---

## Task 10: Run the Tuner

**Files:**
- None (execution only)

**Step 1: Run tuner with default settings**

Run:
```bash
cd spore-sim && cargo run --release -- --tune 2>&1
```

Expected output pattern:
```
Gen  0: best=XXX.X avg=XXX.X stable=X/50
Gen  1: best=XXX.X avg=XXX.X stable=X/50
...
Gen 19: best=XXX.X avg=XXX.X stable=X/50

=== FINAL EXAM (Top 15, Full Marathon, 3 Runs) ===
  # 1: score=XXX.X acc=XX.XX% stable=true conv@Some(XXXX)
  ...

EVOLUTION COMPLETE
  Best score:    XXX.X
  Mean accuracy: XX.XX%
  Stable:        true/false
  ...

Saved to: best_params.json
```

**Step 2: Verify output file**

Run:
```bash
cat best_params.json
```

Expected: Valid JSON with 6 parameters.

---

## Task 11: Verify Best Parameters

**Files:**
- None (verification only)

**Step 1: Run simulation with discovered parameters**

Run:
```bash
cd spore-sim && cargo run --release -- --params best_params.json --ticks 100000 2>&1
```

Expected: Accuracy should be higher and more stable than with default parameters.

**Step 2: Run with dump-weights**

Run:
```bash
cd spore-sim && cargo run --release -- --params best_params.json --ticks 50000 --dump-weights --quiet 2>&1
```

Expected: Weight visualization shows structured patterns (not uniform noise).

**Step 3: Add best_params.json to .gitignore (generated output, not committed)**

Add `best_params.json` to `spore-sim/.gitignore`:

```
/target/
Cargo.lock
best_params.json
```

**Step 4: Final commit**

```bash
git add .gitignore
git commit -m "chore: add best_params.json to gitignore (generated output)"
```

---

## Summary: File Checklist

| File | Purpose | Task |
|------|---------|------|
| `spore-sim/Cargo.toml` | Add rayon, serde, serde_json | Task 1 |
| `spore-sim/src/spore.rs` | Add frustration_alpha, weight_decay_interval | Task 2 |
| `spore-sim/src/simulation.rs` | Add with_full_params constructor | Task 3 |
| `spore-sim/src/tuner.rs` | Genome, evaluate, tune (NEW) | Tasks 5-7 |
| `spore-sim/src/lib.rs` | Add tuner module | Task 5 |
| `spore-sim/src/main.rs` | --tune, --params, new flags | Task 8 |
| `spore-sim/tests/spore_tests.rs` | Tests for new Spore params | Task 2 |
| `spore-sim/tests/simulation_tests.rs` | Test with_full_params | Task 3 |
| `spore-sim/tests/tuner_tests.rs` | Genome, eval, tune tests (NEW) | Tasks 5-7 |

## Key Design Decisions

1. **base_noise fixed at 0.001** - not tuned (only 5% of total noise signal)
2. **trace_decay mutation ±0.005** - fine-grained due to exponential sensitivity
3. **weight_decay_interval range 50-200** - avoids "Alzheimer's cliff" at low values
4. **max_noise_boost capped at 0.05** - above this causes seizure-like behavior
5. **Stability = 1000 ticks above 85% after hitting 90%** - Option B (Habit over Luck)
6. **Convergence revoked on crash below 85%** - prevents false stability claims
7. **Integral accuracy** - mean_accuracy_since_converge, not final_accuracy (prevents jitter kills)
8. **Early exit only in fast eval** - Final Exam runs full marathon
9. **Multi-run (3x) worst score** - robustness check against lucky random sequences
10. **Elite cache** - don't re-evaluate known-good genomes
11. **20% diversity injection** - prevents genetic drift / in-breeding
12. **Crossover + mutation** - true GA, not hill climbing
13. **Adaptive mutation magnitude** - 2x step size after 3 stagnant generations
