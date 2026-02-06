use spore_sim::constants::*;

#[test]
fn test_topology_constants() {
    assert_eq!(INPUT_SIZE, 8);
    assert_eq!(HIDDEN_SIZE, 4);
}

#[test]
fn test_learning_defaults() {
    assert!((DEFAULT_LEARNING_RATE - 0.1).abs() < 0.001);
    assert!((DEFAULT_TRACE_DECAY - 0.9).abs() < 0.001);
    assert!((DEFAULT_BASE_NOISE - 0.001).abs() < 0.0001);
    assert!((DEFAULT_MAX_NOISE_BOOST - 0.05).abs() < 0.001);
    assert!((DEFAULT_CORTISOL_STRENGTH - 0.3).abs() < 0.001);
}

#[test]
fn test_environment_defaults() {
    assert_eq!(DEFAULT_INPUT_HOLD_TICKS, 50);
    assert_eq!(DEFAULT_SWARM_SIZE, 32);
}

#[test]
fn test_rejuvenation_constants() {
    assert!((REJUVENATION_THRESHOLD - 0.55).abs() < 0.01);
    assert_eq!(REJUVENATION_GRACE_TICKS, 10_000);
}
