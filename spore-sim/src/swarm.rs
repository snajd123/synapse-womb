//! The Swarm: A colony of specialist Spores.
//!
//! Manages N independent Spores, each responsible for one output bit.
//! - Distributes all inputs to every Spore simultaneously
//! - Collects single-bit outputs and assembles into output byte/word
//! - Delivers per-bit boolean reward (Dopamine or Cortisol)
//! - Rejuvenates stuck Spores (reset weights if accuracy ~random for too long)

use crate::spore::Spore;
use crate::constants::*;

/// A Swarm of N Spores, one per output bit.
#[derive(Debug, Clone)]
pub struct Swarm {
    /// The colony of Spores.
    pub spores: Vec<Spore>,
}

impl Swarm {
    /// Spawn a new Swarm with `n` identical seed Spores.
    pub fn new(
        n: usize,
        learning_rate: f32,
        trace_decay: f32,
        base_noise: f32,
        max_noise_boost: f32,
        frustration_alpha: f32,
        weight_decay_interval: u64,
        cortisol_strength: f32,
    ) -> Self {
        let spores = (0..n)
            .map(|_| Spore::new(
                learning_rate,
                trace_decay,
                base_noise,
                max_noise_boost,
                frustration_alpha,
                weight_decay_interval,
                cortisol_strength,
            ))
            .collect();

        Self { spores }
    }

    /// Number of Spores (output bits).
    pub fn size(&self) -> usize {
        self.spores.len()
    }

    /// Execute one tick: fire all Spores, compare to targets, reward, learn, maintain.
    ///
    /// Returns mean accuracy across all Spores for this tick.
    pub fn tick(&mut self, inputs: &[f32; INPUT_SIZE], targets: &[bool], tick: u64) -> f32 {
        let n_bits = targets.len();
        let mut correct_count: usize = 0;

        for (i, spore) in self.spores.iter_mut().enumerate() {
            // 1. Fire
            let output = spore.fire(inputs);

            // 2. Per-bit credit assignment (Spore i targets bit i % n_bits)
            let correct = output == targets[i % n_bits];
            if correct {
                correct_count += 1;
            }

            // 3. Reward
            spore.receive_reward(correct);

            // 4. Learn
            spore.learn();

            // 5. Maintain
            spore.maintain(tick);
        }

        // 6. Rejuvenate stuck Spores
        self.rejuvenate();

        correct_count as f32 / self.spores.len() as f32
    }

    /// Rejuvenate any Spore stuck at ~random accuracy after grace period.
    ///
    /// If recent_accuracy < REJUVENATION_THRESHOLD and ticks_alive > REJUVENATION_GRACE_TICKS,
    /// reset the Spore to random weights (same slot, new brain).
    pub fn rejuvenate(&mut self) {
        for spore in &mut self.spores {
            if spore.ticks_alive > REJUVENATION_GRACE_TICKS
                && spore.recent_accuracy < REJUVENATION_THRESHOLD
            {
                spore.reset();
            }
        }
    }

    /// Consensus accuracy: mean of best `recent_accuracy` per bit.
    ///
    /// For each of the 8 output bits, find the Spore with the highest
    /// `recent_accuracy` assigned to that bit. The system accuracy is the
    /// mean of these 8 best values.
    pub fn accuracy(&self) -> f32 {
        if self.spores.is_empty() {
            return 0.0;
        }
        let n_bits = 8;
        let mut sum = 0.0_f32;
        for k in 0..n_bits {
            let best = self.spores.iter()
                .enumerate()
                .filter(|(i, _)| i % n_bits == k)
                .map(|(_, s)| s.recent_accuracy)
                .max_by(|a, b| a.total_cmp(b))
                .unwrap_or(0.0);
            sum += best;
        }
        sum / n_bits as f32
    }

    /// Get the assembled output as a byte using consensus mapping.
    ///
    /// For each bit k (0..7), find the Spore assigned to k with the
    /// highest `recent_accuracy` and use its output. This is
    /// "best expert wins" — system output equals the best Spore per bit.
    pub fn output_byte(&self) -> u8 {
        let mut byte = 0u8;
        for k in 0..8 {
            let best_spore = self.spores.iter()
                .enumerate()
                .filter(|(i, _)| i % 8 == k)
                .max_by(|(i_a, a), (i_b, b)| a.recent_accuracy.total_cmp(&b.recent_accuracy).then(i_b.cmp(i_a)))
                .map(|(_, s)| s);

            if let Some(spore) = best_spore {
                if spore.output {
                    byte |= 1 << k;
                }
            }
        }
        byte
    }
}
