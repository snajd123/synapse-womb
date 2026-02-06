use spore_sim::environment::Environment;
use spore_sim::constants::*;

#[test]
fn test_environment_new() {
    let env = Environment::new(50);
    assert_eq!(env.input_hold_ticks, 50);
    assert_eq!(env.ticks_on_current_for_test(), 0);
}

#[test]
fn test_environment_default() {
    let env = Environment::default();
    assert_eq!(env.input_hold_ticks, DEFAULT_INPUT_HOLD_TICKS);
}

#[test]
fn test_get_input_returns_u8() {
    let env = Environment::new(50);
    let input = env.get_input();
    let _ = input; // Always valid u8
}

#[test]
fn test_get_input_f32_matches_u8() {
    let env = Environment::new(50);
    let byte = env.get_input();
    let f32s = env.get_input_f32();

    // Positive channels (0-7)
    for i in 0..8 {
        let expected = ((byte >> i) & 1) as f32;
        assert_eq!(f32s[i], expected, "Positive bit {} mismatch", i);
    }
    // Complementary channels (8-15)
    for i in 0..8 {
        let expected = 1.0 - ((byte >> i) & 1) as f32;
        assert_eq!(f32s[i + 8], expected, "Complementary bit {} mismatch", i);
    }
}

#[test]
fn test_get_target_bits_mirrors_input() {
    let env = Environment::new(50);
    let byte = env.get_input();
    let targets = env.get_target_bits();

    assert_eq!(targets.len(), 8);
    for i in 0..8 {
        let expected = (byte >> i) & 1 == 1;
        assert_eq!(targets[i], expected, "Target bit {} mismatch", i);
    }
}

#[test]
fn test_advance_switches_after_hold_ticks() {
    let mut env = Environment::new(5);

    for _ in 0..4 {
        env.advance();
    }
    assert_eq!(env.ticks_on_current_for_test(), 4);

    env.advance(); // 5th advance → reset
    assert_eq!(env.ticks_on_current_for_test(), 0);
}

#[test]
fn test_advance_holds_input_during_hold_period() {
    let mut env = Environment::new(100);
    let first_input = env.get_input();

    // Advance 99 times — should still be same input
    for _ in 0..99 {
        env.advance();
    }
    assert_eq!(env.get_input(), first_input);
}
