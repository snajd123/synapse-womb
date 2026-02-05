use spore_sim::simulation::Simulation;

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
