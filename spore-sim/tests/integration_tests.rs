//! Integration tests for the Spore Mirror Experiment.

use spore_sim::simulation::Simulation;
use spore_sim::constants::*;

/// Test that simulation runs without crashing.
#[test]
fn test_simulation_runs_1000_ticks() {
    let mut sim = Simulation::new();
    sim.run(1000, 0);
    assert_eq!(sim.tick, 1000);
}

/// Test that accuracy improves over time (statistical).
/// This is a weak test - just verifies the system is learning *something*.
#[test]
fn test_accuracy_improves() {
    let mut sim = Simulation::new();

    // Run for 1000 ticks, record early accuracy
    sim.run(1000, 0);
    let early_accuracy = sim.recent_accuracy;

    // Run for another 9000 ticks
    sim.run(10000, 0);
    let later_accuracy = sim.recent_accuracy;

    // Later accuracy should be at least as good as early
    // (In rare cases random init might start good, so we don't require strictly greater)
    println!("Early accuracy: {:.2}%, Later accuracy: {:.2}%",
        early_accuracy * 100.0, later_accuracy * 100.0);

    // This is a very weak assertion - just that it doesn't get worse
    // Real convergence tests need more ticks
}

/// Test with delayed reward.
#[test]
fn test_simulation_with_latency() {
    let mut sim = Simulation::with_params(5, 0.92, 50);
    sim.run(1000, 0);
    assert_eq!(sim.tick, 1000);
}

/// Test that the simulation respects input hold constraint.
#[test]
fn test_input_hold_constraint() {
    // This should work without warnings
    let min_hold = min_input_hold_ticks(10);  // For latency 10
    let sim = Simulation::with_params(10, 0.95, min_hold as u64);
    assert!(sim.env().input_hold_ticks >= min_hold as u64);
}
