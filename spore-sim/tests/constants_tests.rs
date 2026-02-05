use spore_sim::constants::*;

#[test]
fn test_constants_exist_and_have_correct_values() {
    // Network topology
    assert_eq!(INPUT_SIZE, 8);
    assert_eq!(HIDDEN_SIZE, 32);
    assert_eq!(OUTPUT_SIZE, 8);

    // Thresholds
    assert_eq!(DEFAULT_THRESHOLD, 50);
    assert_eq!(INIT_THRESHOLD, 0);

    // Pipeline
    assert_eq!(PIPELINE_LATENCY, 2);

    // Homeostasis
    assert_eq!(MAX_WEIGHT_SUM, 400);
    assert_eq!(WEIGHT_DECAY_INTERVAL, 100);

    // Learning
    assert!((BASELINE_ACCURACY - 0.5).abs() < 0.001);
    assert!((DEFAULT_LEARNING_RATE - 0.5).abs() < 0.001);
    assert!((DEFAULT_TRACE_DECAY - 0.9).abs() < 0.001);
    assert!((DEFAULT_BASE_NOISE - 0.001).abs() < 0.0001);
    assert!((DEFAULT_MAX_NOISE_BOOST - 0.05).abs() < 0.001);

    // Environment
    assert_eq!(DEFAULT_INPUT_HOLD_TICKS, 50);
    assert_eq!(DEFAULT_REWARD_LATENCY, 0);

    // Weight initialization range
    assert_eq!(WEIGHT_INIT_MIN, -50);
    assert_eq!(WEIGHT_INIT_MAX, 50);
}

#[test]
fn test_timing_constraint_helper() {
    // Rule: input_hold_ticks >= 2 * (PIPELINE_LATENCY + reward_latency) + 10
    assert_eq!(min_input_hold_ticks(0), 14);   // 2*(2+0)+10
    assert_eq!(min_input_hold_ticks(5), 24);   // 2*(2+5)+10
    assert_eq!(min_input_hold_ticks(10), 34);  // 2*(2+10)+10
}

#[test]
fn test_recommended_trace_decay() {
    // At latency 0: trace^2 should be > 0.5, so decay ~0.71 minimum
    // We use 0.9 which gives 0.81
    let decay_0 = recommended_trace_decay(0);
    assert!(decay_0 > 0.7 && decay_0 < 0.95);

    // At latency 10: trace^12 should be > 0.5
    // With high latency, decay approaches 0.99 (our cap to ensure some decay)
    let decay_10 = recommended_trace_decay(10);
    assert!(decay_10 > 0.9 && decay_10 <= 0.99);
}
