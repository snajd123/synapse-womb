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
        let baseline_reward = BASELINE_ACCURACY as f32 * BASELINE_ACCURACY as f32;
        self.dopamine = reward - baseline_reward;

        // Fix 2: Fast frustration response
        if accuracy < 0.5 {
            self.frustration = 1.0;  // Instant spike
        } else {
            // EMA with faster alpha (0.2 instead of 0.1)
            self.frustration = 0.8 * self.frustration + 0.2 * (1.0 - accuracy);
        }
    }

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
        if tick % (WEIGHT_DECAY_INTERVAL as u64) == 0 && tick > 0 {
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
        let default_t = DEFAULT_THRESHOLD as i16;
        for t in &mut self.thresholds_h {
            if *t < default_t {
                *t += 1;
            } else if *t > default_t {
                *t -= 1;
            }
        }
        for t in &mut self.thresholds_o {
            if *t < default_t {
                *t += 1;
            } else if *t > default_t {
                *t -= 1;
            }
        }
    }

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
}

impl Default for Spore {
    fn default() -> Self {
        Self::new()
    }
}
