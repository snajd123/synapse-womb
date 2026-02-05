# Spore Mirror Experiment: Phase 1 Design

**Status:** Ready for Implementation
**Date:** 2026-02-05
**Goal:** Prove that Hebbian learning + Dopamine reinforcement can evolve a "copy byte" reflex from random noise

---

## 1. The Experiment

### 1.1 Hypothesis

A neural substrate with random initial weights can self-organize into an identity function (Input → Output) using only:
- Eligibility traces (local memory of "what just fired")
- Dopamine signal (global reward for correct output)
- Noise injection (exploration when frustrated)

No backpropagation. No gradient descent. Pure reinforcement at the synapse level.

### 1.2 Success Criteria

- **Convergence:** >95% accuracy sustained for 1000+ ticks
- **Speed:** Convergence in <100,000 ticks (ideally <10,000)
- **Robustness:** Works with reward latency of 0, 5, and 10 ticks

### 1.3 Why This Matters

If this works for 8 bits, the same mechanism scales to:
- "Copy network packet to memory" (Network Driver)
- "Copy memory to screen" (Video Driver)
- "Copy encrypted input to decrypted output" (Crypto Engine)

This is the foundation of the Synapse "Zero-Code Driver Synthesis" claim.

---

## 2. Architecture

### 2.1 Network Topology: 8-32-8

```
Input (8 neurons) → Hidden (32 neurons) → Output (8 neurons)
     256 synapses       256 synapses
              = 512 total synapses
```

**Why this size:**
1. **Redundancy:** 32 hidden neurons allow multiple competing pathways per bit
2. **Debuggability:** 512 synapses fits on one screen for visualization
3. **Non-linearity:** Forces actual learning, not just pass-through

### 2.2 Pipeline Model

The Spore is **pipelined** with 1-tick delay between layers:

```
Tick N:   Input arrives → Hidden computes (writes to hidden_next)
Tick N+1: hidden_next → hidden, Output computes (writes to output_next)
Tick N+2: output_next → output (result visible)
```

**Pipeline latency = 2 ticks**

This enables true parallelism and is more biologically realistic.

---

## 3. Data Structures

### 3.1 The Spore

```rust
const DEFAULT_THRESHOLD: i16 = 50;     // Homeostasis target (not init value!)
const INIT_THRESHOLD: i16 = 0;        // Fix 5: Start "trigger happy"
const PIPELINE_LATENCY: usize = 2;
const MAX_WEIGHT_SUM: i32 = 400;      // Per-neuron weight budget (Fix 1)
const BASELINE_ACCURACY: f32 = 0.5;   // Below this = punishment (Fix 3)

struct Spore {
    // Network topology: 8-32-8
    weights_ih: [[i16; 8]; 32],   // Input → Hidden (256 weights)
    weights_ho: [[i16; 32]; 8],   // Hidden → Output (256 weights)

    // Per-neuron thresholds (learnable)
    // Fix 5: Initialize to INIT_THRESHOLD (0), NOT DEFAULT_THRESHOLD (50)
    thresholds_h: [i16; 32],      // init: [0; 32] (trigger happy)
    thresholds_o: [i16; 8],       // init: [0; 8]  (trigger happy)

    // Eligibility traces (same shape as weights)
    traces_ih: [[f32; 8]; 32],
    traces_ho: [[f32; 32]; 8],
    traces_th: [f32; 32],         // Threshold traces for hidden
    traces_to: [f32; 8],          // Threshold traces for output

    // Double-buffered neuron state (for pipeline)
    hidden: [bool; 32],           // State from LAST tick
    hidden_next: [bool; 32],      // State being computed THIS tick
    output: [bool; 8],
    output_next: [bool; 8],

    // Learning state
    dopamine: f32,                // Current dopamine level
    frustration: f32,             // Rolling average of failure (drives noise)

    // Constants (tunable)
    learning_rate: f32,
    trace_decay: f32,
    base_noise: f32,
    max_noise_boost: f32,
}
```

### 3.2 The Environment

```rust
struct Environment {
    reward_latency: u64,
    pending_rewards: VecDeque<(u64, u8)>,  // (tick_to_deliver, correct_bits)

    // Input history for pipeline-aware judging
    input_history: VecDeque<u8>,

    current_input: u8,
    input_hold_ticks: u64,        // How long to hold each input
    ticks_on_current: u64,
}
```

### 3.3 Initialization: "The Spark" (Fix 5)

**The Bootstrap Problem:** Random weights average to 0. If threshold = 50, nothing fires. No traces. No learning. Dead Spore.

**The Solution:** Start "trigger happy" and prune to precision.

```rust
fn new_spore() -> Spore {
    Spore {
        // Random weights in [-50, 50]
        weights_ih: random_weights(),
        weights_ho: random_weights(),

        // Fix 5: Thresholds start at 0, NOT 50
        // This makes neurons fire at everything (epilepsy)
        thresholds_h: [INIT_THRESHOLD; 32],  // [0; 32]
        thresholds_o: [INIT_THRESHOLD; 8],   // [0; 8]

        // ... rest initialized to zero/false ...
        frustration: 1.0,  // Start fully frustrated (exploring)
    }
}
```

**The Lifecycle:**
1. **Birth:** Everything fires (epileptic chaos)
2. **Learning:** Useful paths get rewarded → thresholds drop further, weights strengthen
3. **Homeostasis:** Thresholds drift UP toward 50 → only strong pathways survive
4. **Maturity:** Precise firing, low noise, high accuracy

This is "Start Hot, Cool to Precision" - the opposite of trying to wake a dead network.

---

## 4. Mechanics

### 4.1 Activation Function

**Hard Threshold + Noise Injection:**

```rust
let fires = sum > threshold || random::<f32>() < noise_rate;
```

- Binary output (true/false)
- Spontaneous firing provides exploration
- Noise rate scales with frustration

### 4.2 Noise (Frustration-Driven Exploration)

```rust
// Fix 2: Fast frustration response - biology doesn't wait to feel pain
if accuracy < 0.5 {
    frustration = 1.0;  // Instant spike on bad result
} else {
    frustration = 0.8 * frustration + 0.2 * (1.0 - accuracy);  // Faster EMA (0.2 not 0.1)
}
noise_rate = base_noise + (frustration * max_noise_boost);
```

- When succeeding: frustration decays, network is "calm"
- When failing badly (< 50%): frustration spikes immediately to max
- EMA window shortened to react within ~5 ticks, not 10

### 4.3 Reward Signal

**Proportional with quadratic scaling:**

```rust
let accuracy = correct_bits as f32 / 8.0;
let reward = accuracy * accuracy;  // (correct/8)²
```

| Correct Bits | Reward |
|--------------|--------|
| 8/8          | 1.00   |
| 7/8          | 0.77   |
| 6/8          | 0.56   |
| 4/8          | 0.25   |

Quadratic scaling prevents "7/8 is good enough" local minima.

### 4.4 Eligibility Traces

```rust
// When a synapse participates in firing:
trace = 1.0;

// Every tick:
trace *= trace_decay;  // 0.9 default
```

Traces are the "chemical residue" - memory of what just fired.

### 4.5 Hebbian Learning (with Anti-Hebbian)

```rust
fn receive_reward(&mut self, correct_bits: u8) {
    let accuracy = correct_bits as f32 / 8.0;

    // Fix 3: Signed dopamine - below baseline = punishment
    // dopamine ranges from -0.25 (0% accuracy) to +1.0 (100% accuracy)
    let reward = accuracy * accuracy;                    // 0.0 to 1.0
    let baseline_reward = BASELINE_ACCURACY * BASELINE_ACCURACY;  // 0.25
    self.dopamine = reward - baseline_reward;            // -0.25 to +0.75

    // Frustration update (Fix 2: instant spike on bad result)
    if accuracy < 0.5 {
        self.frustration = 1.0;
    } else {
        self.frustration = 0.8 * self.frustration + 0.2 * (1.0 - accuracy);
    }
}

fn learn(&mut self, tick: u64) {
    if self.dopamine.abs() < 0.001 { return; }

    let lr_scaled = self.learning_rate * 100.0;
    let d = self.dopamine;  // Can be negative!

    for each synapse {
        let change = lr_scaled * d * trace;
        let delta = stochastic_round(change);
        // Fix 3: saturating_add works for negative delta too
        weight = weight.saturating_add(delta);
    }

    for each threshold {
        let t_delta = stochastic_round(lr_scaled * d * t_trace);
        // Positive d = lower threshold (eager), Negative d = raise threshold (stubborn)
        threshold = threshold.saturating_sub(t_delta);
    }

    self.dopamine = 0.0;
}

fn stochastic_round(value: f32) -> i16 {
    let sign = value.signum() as i16;
    let abs_val = value.abs();
    let floor = abs_val as i16;
    let frac = abs_val.fract();
    sign * (floor + if random::<f32>() < frac { 1 } else { 0 })
}
```

**Key fixes applied:**
- Stochastic rounding prevents small deltas from truncating to zero
- Dopamine consumed after learning (atomic updates)
- **Fix 3:** Signed dopamine - accuracy below 50% baseline actively weakens synapses (Anti-Hebbian)

### 4.6 Homeostasis (Preventing Runaway)

```rust
fn maintain(&mut self, tick: u64) {
    // Weight decay every 100 ticks (~1.5% decay)
    if tick % 100 == 0 {
        for w in all_weights {
            *w -= *w >> 6;
        }
    }

    // Fix 1: Weight Normalization (per-neuron budget)
    // Each neuron has a finite "synaptic energy" to distribute
    for h in 0..32 {
        let sum: i32 = self.weights_ih[h].iter().map(|&w| w.abs() as i32).sum();
        if sum > MAX_WEIGHT_SUM {
            let scale = MAX_WEIGHT_SUM as f32 / sum as f32;
            for w in &mut self.weights_ih[h] {
                *w = (*w as f32 * scale) as i16;
            }
        }
    }
    for o in 0..8 {
        let sum: i32 = self.weights_ho[o].iter().map(|&w| w.abs() as i32).sum();
        if sum > MAX_WEIGHT_SUM {
            let scale = MAX_WEIGHT_SUM as f32 / sum as f32;
            for w in &mut self.weights_ho[o] {
                *w = (*w as f32 * scale) as i16;
            }
        }
    }

    // Threshold drift back to default (every tick)
    for t in all_thresholds {
        if *t < DEFAULT_THRESHOLD { *t += 1; }
        if *t > DEFAULT_THRESHOLD { *t -= 1; }
    }
}
```

**Prevents:**
- "Screaming neurons" (thresholds hitting minimum)
- "Crystallized network" (weights saturating at maximum)
- **Fix 1:** "Hard-wired neurons" - per-neuron weight budget prevents sum from growing unbounded

---

## 5. The Simulation Loop

```rust
fn run(&mut self, max_ticks: u64) {
    while self.tick < max_ticks {
        // 1. SENSE
        let input = self.env.get_input();

        // 2. PROPAGATE
        self.spore.propagate(input);

        // 3. ADVANCE PIPELINE (swap buffers, decay traces)
        self.spore.tick_end();

        // 4. READ OUTPUT
        let output = self.spore.output_as_byte();

        // 5. ENVIRONMENT STEP (judge against input[tick-2], schedule reward)
        if let Some(correct_bits) = self.env.tick(self.tick, output) {
            // 6. INJECT DOPAMINE
            self.spore.receive_reward(correct_bits);

            // Track accuracy
            let accuracy = correct_bits as f32 / 8.0;
            self.recent_accuracy = 0.95 * self.recent_accuracy + 0.05 * accuracy;
        }

        // 7. LEARN (consumes dopamine)
        self.spore.learn(self.tick);

        // 8. MAINTAIN (weight decay, threshold homeostasis)
        self.spore.maintain(self.tick);

        self.tick += 1;
    }
}
```

### 5.1 Pipeline-Aware Judging

The environment compares output against `input[tick - PIPELINE_LATENCY]`, not current input:

```rust
fn tick(&mut self, tick: u64, spore_output: u8) -> Option<u8> {
    // Judge against the input the spore was ACTUALLY responding to
    let judge_input = self.input_history[0];  // Oldest in queue

    let error_bits = (spore_output ^ judge_input).count_ones() as u8;
    let correct_bits = 8 - error_bits;

    // ... schedule reward, update input history ...
}
```

---

## 6. Tunable Parameters

### 6.1 Starting Values

| Parameter | Value | Notes |
|-----------|-------|-------|
| `learning_rate` | 0.5 | Scaled by 100x internally |
| `trace_decay` | 0.9 | Increase for higher latency |
| `base_noise` | 0.001 | 0.1% spontaneous firing |
| `max_noise_boost` | 0.05 | Up to 5% at max frustration |
| `reward_latency` | 0 | Start immediate, then increase |
| `input_hold_ticks` | 50 | **Fix 4:** Must be >> PIPELINE + REWARD_LATENCY |

### 6.2 Input Hold Timing Constraint (Fix 4)

**Rule:** `input_hold_ticks >= 2 * (PIPELINE_LATENCY + reward_latency) + 10`

This ensures a "clear air" window where reward and input are perfectly aligned.

| reward_latency | min input_hold_ticks | Reasoning |
|----------------|---------------------|-----------|
| 0              | 14                  | 2*(2+0)+10 |
| 5              | 24                  | 2*(2+5)+10 |
| 10             | 34                  | 2*(2+10)+10 |

**Why this matters:** If input changes while rewards for the *previous* input are still arriving, the Spore learns superstitious associations (Input B → Output A's rewards).

### 6.3 Trace Decay Tuning Table

Rule: `trace^(pipeline + reward_latency)` should be > 0.5

| reward_latency | recommended trace_decay | trace after full delay |
|----------------|-------------------------|------------------------|
| 0              | 0.90                    | 0.9² = 0.81            |
| 5              | 0.92                    | 0.92⁷ = 0.56           |
| 10             | 0.95                    | 0.95¹² = 0.54          |
| 20             | 0.97                    | 0.97²² = 0.51          |

```rust
fn recommended_trace_decay(reward_latency: u64) -> f32 {
    let total_delay = PIPELINE_LATENCY as u64 + reward_latency;
    0.5_f32.powf(1.0 / total_delay as f32)
}
```

---

## 7. Experimental Protocol

### Phase 1a: Sanity Check (Immediate Reward)

1. Set `reward_latency = 0`
2. Run for 100,000 ticks
3. **Pass:** accuracy > 95% within 50,000 ticks
4. **Fail:** Tune `learning_rate`, check for bugs

### Phase 1b: Trace Stress Test (Delayed Reward)

1. Keep parameters from 1a
2. Set `reward_latency = 5`, adjust `trace_decay` per table
3. **Pass:** Still converges (may take longer)
4. **Fail:** Trace decay too fast, increase it

### Phase 1c: Robustness Test

1. Set `reward_latency = 10`
2. Adjust `trace_decay = 0.95`
3. **Pass:** Converges within 100,000 ticks
4. Record final parameters as "Life Parameters"

---

## 8. Visualization (Debug Tools)

### 8.1 Weight Matrix Heatmap

Display the 8×32 and 32×8 weight matrices as heatmaps. Watch for:
- **Success:** 8 distinct "lines" forming (one pathway per bit)
- **Failure:** Uniform noise or solid blocks

### 8.2 Convergence Plot

Plot `recent_accuracy` over time. Look for:
- **Success:** Rising curve that plateaus at >95%
- **Failure:** Flat line, oscillation, or slow rise that never plateaus

### 8.3 Frustration Monitor

Track `frustration` over time:
- Should be high initially (exploring)
- Should drop as accuracy increases (calming down)

---

## 9. Next Steps

After Phase 1 succeeds:

1. **Phase 2 (Implant):** Port to `no_std` Rust, run on QEMU RISC-V
2. **Phase 3 (Scale):** Extend to 16-bit, 32-bit patterns
3. **Phase 4 (Integrate):** Connect to TensorFS and the Fog

---

## Appendix: Known Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Weights never change | Integer truncation | Check stochastic rounding, increase LR |
| All neurons always fire | Threshold runaway | Check homeostasis is running |
| Weights hit max and stick | No weight decay | Check maintain() is being called |
| Accuracy stuck at ~40% | Judging wrong input | Check pipeline-aware comparison |
| Accuracy oscillates wildly | Trace too short for latency | Increase trace_decay |
| Network goes silent | Thresholds too high | Lower DEFAULT_THRESHOLD or check noise injection |
| Neurons become "hard-wired 1s" | Weight sum too large | **Fix 1:** Check weight normalization in maintain() |
| Exploration kicks in too late | Frustration EMA too slow | **Fix 2:** Check instant spike on accuracy < 0.5 |
| Bad pathways persist forever | Positive-only learning | **Fix 3:** Check signed dopamine (anti-Hebbian) |
| Accuracy stuck at ~12% (noise) | Superstitious learning | **Fix 4:** Increase input_hold_ticks (must be >> latency) |
| Nothing fires, accuracy = random | Dead Spore bootstrap | **Fix 5:** Init thresholds to 0, not 50 (start hot) |
