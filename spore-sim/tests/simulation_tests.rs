use spore_sim::simulation::Simulation;

#[test]
fn test_simulation_new() {
    let sim = Simulation::new();
    assert_eq!(sim.tick, 0);
    assert_eq!(sim.recent_accuracy, 0.0);
}

#[test]
fn test_simulation_with_full_params() {
    let sim = Simulation::with_full_params(8, 80, 0.15, 0.95, 0.03, 120, 0.1, 0.4);
    assert_eq!(sim.tick, 0);
    assert_eq!(sim.swarm().size(), 8);
    assert_eq!(sim.env().input_hold_ticks, 80);
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
fn test_simulation_step_returns_accuracy() {
    let mut sim = Simulation::new();
    let result = sim.step();
    assert!(result.is_some());
    let acc = result.unwrap();
    assert!(acc >= 0.0 && acc <= 1.0, "Accuracy {} out of range", acc);
}

#[test]
fn test_simulation_step_updates_ema() {
    let mut sim = Simulation::new();
    for _ in 0..100 {
        sim.step();
    }
    assert!(sim.recent_accuracy >= 0.0 && sim.recent_accuracy <= 1.0);
}

#[test]
fn test_simulation_custom_swarm_size() {
    let sim = Simulation::with_full_params(32, 50, 0.1, 0.9, 0.05, 100, 0.2, 0.3);
    assert_eq!(sim.swarm().size(), 32);
}
