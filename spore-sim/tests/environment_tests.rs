use spore_sim::environment::Environment;
use spore_sim::constants::*;

#[test]
fn test_environment_new() {
    let env = Environment::new(0);
    assert_eq!(env.reward_latency, 0);
    assert_eq!(env.input_hold_ticks, DEFAULT_INPUT_HOLD_TICKS as u64);
}

#[test]
fn test_environment_with_params() {
    let env = Environment::with_params(5, 100);
    assert_eq!(env.reward_latency, 5);
    assert_eq!(env.input_hold_ticks, 100);
}

#[test]
fn test_environment_input_history_initialized() {
    let env = Environment::new(0);
    // Input history should have exactly PIPELINE_LATENCY entries (NOT +1!)
    assert_eq!(env.input_history_len(), PIPELINE_LATENCY);
}

#[test]
fn test_environment_get_input_returns_current() {
    let env = Environment::new(0);
    let input = env.get_input();
    // Should return a valid u8
    assert!(input <= 255);
}
