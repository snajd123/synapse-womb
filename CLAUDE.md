# CLAUDE.md — Project Rules for Synapse

## Superpowers Skills

The rules of all Superpowers skills must always be adhered to. When a Superpowers skill applies to the current task, it MUST be invoked before taking action. No exceptions.

## Code Review

When using the `superpowers:requesting-code-review` skill, the code review agent MUST use the **Opus 4.6** model (`model: "opus"`), not Haiku or Sonnet. Code review requires deep reasoning and architectural awareness — never downgrade.

## Project Structure

- **spore-sim/**: Rust crate implementing the Spore Mirror Experiment (Phase 1)
  - 8-32-8 neural network with Hebbian learning + Dopamine reinforcement
  - Genetic hyperparameter tuner (`src/tuner.rs`)
  - Run `cargo test --release` from `spore-sim/` to verify
  - Run `cargo run --release -- --tune` to evolve optimal parameters

## Key Invariants

- **Fix 2 (instant spike)**: `frustration = 1.0` when accuracy < 50% is non-negotiable. Never gate this behind a tunable parameter.
- **base_noise fixed at 0.001**: Not tuned — only 5% of total noise signal. Hardcoded in `Simulation::with_full_params`.
- **Per-tick accuracy**: The tuner uses instantaneous per-tick accuracy from `step()` return value, NOT the EMA `recent_accuracy` (which lags by ~30-45 ticks).
- **Convergence revocation**: If accuracy drops below 85%, convergence status is revoked (set to None).
