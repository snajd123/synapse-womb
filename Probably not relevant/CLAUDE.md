# CLAUDE.md — Project Rules for Synapse

## Superpowers Skills

The rules of all Superpowers skills must always be adhered to. When a Superpowers skill applies to the current task, it MUST be invoked before taking action. No exceptions.

## Code Review

When using the `superpowers:requesting-code-review` skill, the code review agent MUST use the **Opus 4.6** model (`model: "opus"`), not Haiku or Sonnet. Code review requires deep reasoning and architectural awareness — never downgrade.

## Project Structure

- **spore-sim/**: Rust crate implementing the SWARM V2 Mirror Experiment
  - N x (8→4→1) Spores with per-bit credit assignment (Dopamine + Cortisol)
  - Hard threshold activation, f32 weights, tunable bias per neuron
  - Genetic hyperparameter tuner (`src/tuner.rs`)
  - Run `cargo test --release` from `spore-sim/` to verify
  - Run `cargo run --release -- --tune` to evolve optimal parameters

## Key Invariants

- **Fix 2 (instant spike)**: `frustration = 1.0` when accuracy < 50% is non-negotiable. Never gate this behind a tunable parameter.
- **base_noise fixed at 0.001**: Not tuned — hardcoded in `Simulation::with_full_params`.
- **Per-tick accuracy**: The tuner uses instantaneous per-tick accuracy from `step()` return value, NOT the EMA `recent_accuracy`.
- **Convergence revocation**: If accuracy drops below 85%, convergence status is revoked.
- **Signed dopamine**: Correct → +1.0, Wrong → -cortisol_strength. Single formula for both weights and biases.
- **Accuracy-Gated Learning Rate**: `effective_lr = lr * (1 - recent_accuracy)`. Converged Spores learn slowly; newborn Spores learn at full speed. Prevents catastrophic forgetting.
- **Bias NOT decayed**: Biases are structural properties, not transient signals. Weight decay applies only to weights.
- **Positive initial bias (INITIAL_BIAS = 0.5)**: Prevents "Initial Blackout" — neurons must fire from tick 1 so traces exist for learning. Without this, unlucky random weights → never fires → no traces → learn() does `weight += LR * reward * 0.0` forever.
- **Target Activity Homeostasis**: `firing_rate` EMA tracks output activity; `maintain()` nudges `bias_o` toward `target_rate` (10%). Silent → bias up. Overactive → bias down. No Spore can stay dead.
- **Winner-Take-All hidden layer**: Only the hidden neuron with the highest sum fires via threshold. Suppressed neurons can fire via noise (exploration). Prevents hidden neuron homogeneity — forces specialization.
- **Hard threshold everywhere**: Hidden and output neurons use `sum + bias > 0.0`. No sigmoid.
- **Spore isolation**: Spores do not share weights, traces, or state. Per-bit credit ensures zero interference.
- **Redundancy and Consensus**: `DEFAULT_SWARM_SIZE = 32` (4 Spores per bit). Each Spore `i` targets bit `i % 8`. Output and accuracy use "best expert wins" — the Spore with highest `recent_accuracy` per bit determines the output. Probability amplification: P(bit fails) = P(stuck)^4.
