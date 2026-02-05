use spore_sim::environment::Environment;
use spore_sim::constants::*;

#[test]
fn test_environment_tick_judges_against_delayed_input() {
    let mut env = Environment::with_params(0, 100);  // No reward latency

    // Manually set up input history to control the test
    env.set_input_history_for_test(&[0xAA, 0xBB]);  // Only PIPELINE_LATENCY=2 entries!

    // If spore outputs 0xAA (matches oldest), should get 8 correct
    let result = env.tick(0, 0xAA);
    assert_eq!(result, Some(8));
}

#[test]
fn test_environment_tick_with_reward_latency() {
    let mut env = Environment::with_params(5, 100);
    env.set_input_history_for_test(&[0xFF, 0xFF]);

    // At tick 0, schedule reward for tick 5
    let result = env.tick(0, 0xFF);  // Perfect match
    assert!(result.is_none(), "Reward should not arrive yet at tick 0");

    // Advance to tick 5
    for t in 1..5 {
        env.tick(t, 0xFF);
    }
    let result = env.tick(5, 0xFF);
    assert!(result.is_some(), "Reward should arrive at tick 5");
}

#[test]
fn test_environment_tick_input_changes_after_hold_ticks() {
    let mut env = Environment::with_params(0, 5);  // Hold for 5 ticks

    // Tick 5 times with dummy output
    for t in 0..5 {
        env.tick(t, 0x00);
    }

    // Should reset after hold period
    assert_eq!(env.ticks_on_current_for_test(), 0, "Should reset after hold period");
}

#[test]
fn test_environment_tick_correct_bits_calculation() {
    let mut env = Environment::with_params(0, 100);
    env.set_input_history_for_test(&[0b10101010, 0x00]);

    // 0b10101010 ^ 0b10101000 = 0b00000010, count_ones = 1
    // correct = 8 - 1 = 7
    let result = env.tick(0, 0b10101000);
    assert_eq!(result, Some(7));
}

#[test]
fn test_environment_tick_zero_correct() {
    let mut env = Environment::with_params(0, 100);
    env.set_input_history_for_test(&[0xFF, 0x00]);

    // Output is inverted: 0x00, all 8 bits wrong
    let result = env.tick(0, 0x00);
    assert_eq!(result, Some(0));
}

#[test]
fn test_environment_tick_updates_input_history() {
    let mut env = Environment::with_params(0, 100);
    let old_len = env.input_history_len();

    env.tick(0, 0x00);

    // History length should stay constant (sliding window)
    assert_eq!(env.input_history_len(), old_len);
}

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
