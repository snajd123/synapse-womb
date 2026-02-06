//! The Spore: Smallest unit of Narrow Intent.
//!
//! A tiny 8-4-1 neural network that learns to produce one output bit.
//! - Hard threshold activation (f32 weights, binary decisions)
//! - Tunable bias per neuron
//! - Signed dopamine: +1.0 (correct) or -cortisol_strength (wrong)
//! - Per-Spore frustration with Fix 2 instant spike
//! - Eligibility traces for temporal credit assignment

use rand::Rng;
use crate::constants::*;
use crate::utils::random_weight;

/// A single Spore: an 8→4→1 network responsible for one output bit.
#[derive(Debug, Clone)]
pub struct Spore {
    // ========================================================================
    // WEIGHTS (f32, learnable)
    // ========================================================================

    /// Input → Hidden weights. Shape: [HIDDEN_SIZE][INPUT_SIZE] = [4][8]
    pub weights_ih: [[f32; INPUT_SIZE]; HIDDEN_SIZE],

    /// Hidden → Output weights. Shape: [HIDDEN_SIZE] = [4]
    pub weights_ho: [f32; HIDDEN_SIZE],

    // ========================================================================
    // BIASES (f32, learnable, per-neuron excitability)
    // ========================================================================

    /// Hidden layer biases. Shape: [HIDDEN_SIZE] = [4]
    pub bias_h: [f32; HIDDEN_SIZE],

    /// Output neuron bias.
    pub bias_o: f32,

    // ========================================================================
    // ELIGIBILITY TRACES
    // ========================================================================

    /// Traces for Input → Hidden synapses. Shape: [HIDDEN_SIZE][INPUT_SIZE]
    pub traces_ih: [[f32; INPUT_SIZE]; HIDDEN_SIZE],

    /// Traces for Hidden → Output synapses. Shape: [HIDDEN_SIZE]
    pub traces_ho: [f32; HIDDEN_SIZE],

    /// Bias traces for hidden neurons (1.0 if fired, decays).
    pub trace_bias_h: [f32; HIDDEN_SIZE],

    /// Bias trace for output neuron (1.0 if fired, decays).
    pub trace_bias_o: f32,

    // ========================================================================
    // NEURON STATE (no double buffering — fires in one tick)
    // ========================================================================

    /// Hidden layer activations (current tick).
    pub hidden: [bool; HIDDEN_SIZE],

    /// Output bit (current tick).
    pub output: bool,

    // ========================================================================
    // LEARNING STATE
    // ========================================================================

    /// Current reward signal: +1.0 (dopamine) or -cortisol_strength.
    /// Consumed after learn().
    pub dopamine: f32,

    /// Frustration level (drives noise injection).
    /// 0.0 = calm, 1.0 = frantic.
    pub frustration: f32,

    /// Per-Spore rolling accuracy (EMA).
    pub recent_accuracy: f32,

    /// Ticks this Spore has been alive (for rejuvenation grace period).
    pub ticks_alive: u64,

    /// Output firing rate EMA for activity homeostasis.
    /// Tracks how often this Spore's output neuron fires.
    pub firing_rate: f32,

    // ========================================================================
    // HYPERPARAMETERS
    // ========================================================================

    pub learning_rate: f32,
    pub trace_decay: f32,
    pub base_noise: f32,
    pub max_noise_boost: f32,
    pub frustration_alpha: f32,
    pub weight_decay_interval: u64,
    pub cortisol_strength: f32,
    pub target_rate: f32,
    pub homeostasis_rate: f32,
}

impl Spore {
    /// Create a new Spore with specified hyperparameters.
    pub fn new(
        learning_rate: f32,
        trace_decay: f32,
        base_noise: f32,
        max_noise_boost: f32,
        frustration_alpha: f32,
        weight_decay_interval: u64,
        cortisol_strength: f32,
    ) -> Self {
        let mut weights_ih = [[0.0_f32; INPUT_SIZE]; HIDDEN_SIZE];
        let mut weights_ho = [0.0_f32; HIDDEN_SIZE];

        for j in 0..HIDDEN_SIZE {
            for i in 0..INPUT_SIZE {
                weights_ih[j][i] = random_weight();
            }
            weights_ho[j] = random_weight();
        }

        Self {
            weights_ih,
            weights_ho,
            bias_h: [INITIAL_BIAS; HIDDEN_SIZE],
            bias_o: INITIAL_BIAS,
            traces_ih: [[0.0; INPUT_SIZE]; HIDDEN_SIZE],
            traces_ho: [0.0; HIDDEN_SIZE],
            trace_bias_h: [0.0; HIDDEN_SIZE],
            trace_bias_o: 0.0,
            hidden: [false; HIDDEN_SIZE],
            output: false,
            dopamine: 0.0,
            frustration: 1.0, // Start fully frustrated (explore)
            recent_accuracy: 0.0,
            ticks_alive: 0,
            firing_rate: 0.0,
            learning_rate,
            trace_decay,
            base_noise,
            max_noise_boost,
            frustration_alpha,
            weight_decay_interval,
            cortisol_strength,
            target_rate: DEFAULT_TARGET_RATE,
            homeostasis_rate: DEFAULT_HOMEOSTASIS_RATE,
        }
    }

    /// Create a Spore with default hyperparameters.
    pub fn default_params() -> Self {
        Self::new(
            DEFAULT_LEARNING_RATE,
            DEFAULT_TRACE_DECAY,
            DEFAULT_BASE_NOISE,
            DEFAULT_MAX_NOISE_BOOST,
            DEFAULT_FRUSTRATION_ALPHA,
            DEFAULT_WEIGHT_DECAY_INTERVAL,
            DEFAULT_CORTISOL_STRENGTH,
        )
    }

    /// Forward pass: compute hidden layer and output bit.
    ///
    /// Hard threshold everywhere: `sum + bias > 0.0` → fires.
    /// Noise injection based on frustration level.
    /// Sets eligibility traces for active synapses.
    pub fn fire(&mut self, inputs: &[f32; INPUT_SIZE]) -> bool {
        let mut rng = rand::thread_rng();
        let noise_rate = self.base_noise + (self.frustration * self.max_noise_boost);

        // ====================================================================
        // HIDDEN LAYER (hard threshold)
        // ====================================================================
        for j in 0..HIDDEN_SIZE {
            let mut sum = self.bias_h[j];
            for i in 0..INPUT_SIZE {
                sum += inputs[i] * self.weights_ih[j][i];
            }

            self.hidden[j] = sum > 0.0 || rng.gen::<f32>() < noise_rate;

            // Set traces for active synapses
            if self.hidden[j] {
                for i in 0..INPUT_SIZE {
                    if inputs[i] > 0.5 {
                        self.traces_ih[j][i] = 1.0;
                    }
                }
                self.trace_bias_h[j] = 1.0;
            }
        }

        // ====================================================================
        // OUTPUT (hard threshold)
        // ====================================================================
        let mut sum = self.bias_o;
        for j in 0..HIDDEN_SIZE {
            if self.hidden[j] {
                sum += self.weights_ho[j];
            }
        }

        self.output = sum > 0.0 || rng.gen::<f32>() < noise_rate;

        // Set output traces
        if self.output {
            for j in 0..HIDDEN_SIZE {
                if self.hidden[j] {
                    self.traces_ho[j] = 1.0;
                }
            }
            self.trace_bias_o = 1.0;
        }

        // Update firing rate EMA for activity homeostasis
        let fired = if self.output { 1.0_f32 } else { 0.0 };
        self.firing_rate = 0.99 * self.firing_rate + 0.01 * fired;

        self.output
    }

    /// Receive per-bit boolean reward.
    ///
    /// - correct=true  → dopamine = +1.0 (strengthen active traces)
    /// - correct=false → dopamine = -cortisol_strength (weaken active traces)
    ///
    /// Updates per-Spore frustration using `recent_accuracy` (EMA), NOT per-tick:
    /// - recent_accuracy < 50% → instant spike to 1.0 (Fix 2, non-negotiable)
    /// - recent_accuracy >= 50% → EMA with frustration_alpha
    ///
    /// IMPORTANT: We check recent_accuracy (EMA) for the spike, not per-tick accuracy.
    /// Per-tick accuracy for a 1-bit output is binary (0.0 or 1.0). If we spiked on
    /// every wrong tick, a Spore at 90% accuracy would spike every ~10th tick and
    /// frustration would never settle. Using EMA preserves Fix 2's intent: "if I'm
    /// *generally* failing, panic."
    pub fn receive_reward(&mut self, correct: bool) {
        let accuracy = if correct { 1.0_f32 } else { 0.0 };

        // Update per-Spore accuracy EMA (BEFORE frustration check)
        self.recent_accuracy = 0.95 * self.recent_accuracy + 0.05 * accuracy;

        // Frustration update (Fix 2: spike based on TREND, not single tick)
        if self.recent_accuracy < BASELINE_ACCURACY {
            self.frustration = 1.0;
        } else {
            self.frustration = (1.0 - self.frustration_alpha) * self.frustration
                + self.frustration_alpha * (1.0 - accuracy);
        }

        // Signed dopamine
        self.dopamine = if correct { 1.0 } else { -self.cortisol_strength };
    }

    /// Apply Hebbian learning using signed dopamine and traces.
    ///
    /// Learning rate is gated by accuracy: `effective_lr = lr * (1 - recent_accuracy)`.
    /// Converged Spores (high accuracy) learn slowly, preventing catastrophic forgetting.
    /// Newborn Spores (accuracy=0) learn at full speed.
    ///
    /// `weight += effective_lr * dopamine * trace`
    /// `bias   += effective_lr * dopamine * trace_b`
    ///
    /// Dopamine is consumed (set to 0) after learning.
    pub fn learn(&mut self) {
        if self.dopamine.abs() < 0.001 {
            return;
        }

        let d = self.dopamine;
        let lr = self.learning_rate * (1.0 - self.recent_accuracy);

        // Update Input → Hidden weights and biases
        for j in 0..HIDDEN_SIZE {
            for i in 0..INPUT_SIZE {
                self.weights_ih[j][i] += lr * d * self.traces_ih[j][i];
            }
            self.bias_h[j] += lr * d * self.trace_bias_h[j];
        }

        // Update Hidden → Output weights and bias
        for j in 0..HIDDEN_SIZE {
            self.weights_ho[j] += lr * d * self.traces_ho[j];
        }
        self.bias_o += lr * d * self.trace_bias_o;

        // Consume dopamine
        self.dopamine = 0.0;
    }

    /// Maintenance: decay traces, apply weight decay, activity homeostasis.
    ///
    /// - Traces decay by trace_decay each tick
    /// - Weights decay by *0.99 every weight_decay_interval ticks
    /// - Biases are NOT decayed (structural property)
    /// - Activity homeostasis: nudge bias_o toward target firing rate
    /// - ticks_alive incremented for rejuvenation tracking
    pub fn maintain(&mut self, tick: u64) {
        self.ticks_alive += 1;

        // Activity homeostasis: keep output neuron at target firing rate
        let diff = self.target_rate - self.firing_rate;
        self.bias_o += self.homeostasis_rate * diff;

        // Trace decay
        for row in &mut self.traces_ih {
            for t in row {
                *t *= self.trace_decay;
            }
        }
        for t in &mut self.traces_ho {
            *t *= self.trace_decay;
        }
        for t in &mut self.trace_bias_h {
            *t *= self.trace_decay;
        }
        self.trace_bias_o *= self.trace_decay;

        // Weight decay (every weight_decay_interval ticks)
        if self.weight_decay_interval > 0
            && tick % self.weight_decay_interval == 0
            && tick > 0
        {
            for row in &mut self.weights_ih {
                for w in row {
                    *w *= 0.99;
                }
            }
            for w in &mut self.weights_ho {
                *w *= 0.99;
            }
            // Biases NOT decayed
        }
    }

    /// Reset this Spore to random initial state (rejuvenation).
    ///
    /// Same slot, new brain. Weights, biases, traces, frustration all reset.
    pub fn reset(&mut self) {
        for j in 0..HIDDEN_SIZE {
            for i in 0..INPUT_SIZE {
                self.weights_ih[j][i] = random_weight();
            }
            self.weights_ho[j] = random_weight();
        }

        self.bias_h = [INITIAL_BIAS; HIDDEN_SIZE];
        self.bias_o = INITIAL_BIAS;
        self.traces_ih = [[0.0; INPUT_SIZE]; HIDDEN_SIZE];
        self.traces_ho = [0.0; HIDDEN_SIZE];
        self.trace_bias_h = [0.0; HIDDEN_SIZE];
        self.trace_bias_o = 0.0;
        self.hidden = [false; HIDDEN_SIZE];
        self.output = false;
        self.dopamine = 0.0;
        self.frustration = 1.0;
        self.recent_accuracy = 0.0;
        self.ticks_alive = 0;
        self.firing_rate = 0.0;
    }
}
