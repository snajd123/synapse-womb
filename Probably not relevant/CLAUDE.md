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
- **Bias NOT decayed**: Biases are structural properties, not transient signals. Weight decay applies only to weights.
- **Hard threshold everywhere**: Hidden and output neurons use `sum + bias > 0.0`. No sigmoid.
- **Spore isolation**: Spores do not share weights, traces, or state. Per-bit credit ensures zero interference.
