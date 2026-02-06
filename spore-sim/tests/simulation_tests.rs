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
