//! Integration tests for the SWARM V2 Mirror Experiment.

use spore_sim::simulation::Simulation;

#[test]
fn test_simulation_runs_1000_ticks() {
    let mut sim = Simulation::new();
    sim.run(1000, 0);
    assert_eq!(sim.tick, 1000);
}

#[test]
fn test_swarm_accuracy_valid_after_run() {
    let mut sim = Simulation::new();
    sim.run(5000, 0);
    assert!(sim.recent_accuracy >= 0.0 && sim.recent_accuracy <= 1.0);
}

#[test]
fn test_simulation_runs_10000_ticks_without_crash() {
    let mut sim = Simulation::new();
    sim.run(10_000, 0);
    assert_eq!(sim.tick, 10_000);
}
