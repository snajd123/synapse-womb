use spore_sim::spore::Spore;
use spore_sim::constants::*;

#[test]
fn test_spore_struct_exists() {
    let spore = Spore::new();
    // Just verify it compiles and has the expected fields
    assert_eq!(spore.weights_ih.len(), HIDDEN_SIZE);
    assert_eq!(spore.weights_ih[0].len(), INPUT_SIZE);
    assert_eq!(spore.weights_ho.len(), OUTPUT_SIZE);
    assert_eq!(spore.weights_ho[0].len(), HIDDEN_SIZE);
}

#[test]
fn test_spore_thresholds_init_to_zero() {
    // Fix 5: Thresholds must start at 0, not DEFAULT_THRESHOLD
    let spore = Spore::new();
    for t in &spore.thresholds_h {
        assert_eq!(*t, INIT_THRESHOLD as i16, "Hidden thresholds should init to 0");
    }
    for t in &spore.thresholds_o {
        assert_eq!(*t, INIT_THRESHOLD as i16, "Output thresholds should init to 0");
    }
}

#[test]
fn test_spore_traces_init_to_zero() {
    let spore = Spore::new();
    for row in &spore.traces_ih {
        for t in row {
            assert_eq!(*t, 0.0);
        }
    }
    for row in &spore.traces_ho {
        for t in row {
            assert_eq!(*t, 0.0);
        }
    }
}

#[test]
fn test_spore_weights_in_valid_range() {
    let spore = Spore::new();
    for row in &spore.weights_ih {
        for w in row {
            assert!(*w >= WEIGHT_INIT_MIN as i16 && *w <= WEIGHT_INIT_MAX as i16);
        }
    }
    for row in &spore.weights_ho {
        for w in row {
            assert!(*w >= WEIGHT_INIT_MIN as i16 && *w <= WEIGHT_INIT_MAX as i16);
        }
    }
}

#[test]
fn test_spore_frustration_starts_at_one() {
    // Start fully frustrated to encourage exploration
    let spore = Spore::new();
    assert_eq!(spore.frustration, 1.0);
}

#[test]
fn test_spore_dopamine_starts_at_zero() {
    let spore = Spore::new();
    assert_eq!(spore.dopamine, 0.0);
}

#[test]
fn test_spore_activations_start_false() {
    let spore = Spore::new();
    for h in &spore.hidden {
        assert!(!*h);
    }
    for h in &spore.hidden_next {
        assert!(!*h);
    }
    for o in &spore.output {
        assert!(!*o);
    }
    for o in &spore.output_next {
        assert!(!*o);
    }
}

#[test]
fn test_propagate_updates_hidden_next() {
    let mut spore = Spore::new();
    // Set thresholds very low so neurons fire easily
    spore.thresholds_h = [-1000; HIDDEN_SIZE];
    spore.base_noise = 0.0;  // Disable noise for deterministic test

    // All 1s input
    spore.propagate(0xFF);

    // With very low thresholds, all hidden_next should fire
    for h in &spore.hidden_next {
        assert!(*h, "Hidden neurons should fire with low threshold");
    }
}

#[test]
fn test_propagate_updates_output_next_from_hidden() {
    let mut spore = Spore::new();
    // Set all thresholds very low
    spore.thresholds_h = [-1000; HIDDEN_SIZE];
    spore.thresholds_o = [-1000; OUTPUT_SIZE];
    spore.base_noise = 0.0;

    // Force hidden layer to be all true (from previous tick)
    spore.hidden = [true; HIDDEN_SIZE];

    // Propagate (output_next should now fire based on hidden)
    spore.propagate(0xFF);

    for o in &spore.output_next {
        assert!(*o, "Output neurons should fire with low threshold and all hidden firing");
    }
}

#[test]
fn test_propagate_high_threshold_nothing_fires() {
    let mut spore = Spore::new();
    // Set thresholds very high
    spore.thresholds_h = [10000; HIDDEN_SIZE];
    spore.thresholds_o = [10000; OUTPUT_SIZE];
    spore.base_noise = 0.0;  // Disable noise
    spore.frustration = 0.0; // No frustration boost

    spore.propagate(0xFF);

    for h in &spore.hidden_next {
        assert!(!*h, "Hidden neurons should not fire with very high threshold");
    }
}

#[test]
fn test_propagate_sets_traces_on_firing() {
    let mut spore = Spore::new();
    spore.thresholds_h = [-1000; HIDDEN_SIZE];
    spore.base_noise = 0.0;

    // Input with bit 0 set
    spore.propagate(0x01);

    // Hidden neurons should fire (low threshold)
    // Traces for input bit 0 -> hidden should be set to 1.0
    for h in 0..HIDDEN_SIZE {
        if spore.hidden_next[h] {
            assert_eq!(spore.traces_ih[h][0], 1.0, "Trace should be 1.0 for firing synapse");
        }
    }
}

#[test]
fn test_propagate_does_not_set_trace_for_zero_input() {
    let mut spore = Spore::new();
    spore.thresholds_h = [-1000; HIDDEN_SIZE];
    spore.base_noise = 0.0;

    // Zero all traces first
    spore.traces_ih = [[0.0; INPUT_SIZE]; HIDDEN_SIZE];

    // Input with only bit 0 set (0x01)
    spore.propagate(0x01);

    // Traces for bit 1 (which is 0 in input) should remain 0
    for h in 0..HIDDEN_SIZE {
        // Bit 1 is not set in input, so trace should stay 0
        assert_eq!(spore.traces_ih[h][1], 0.0, "Trace should be 0 for non-firing input");
    }
}

#[test]
fn test_tick_end_advances_pipeline() {
    let mut spore = Spore::new();

    // Set some values in hidden_next and output_next
    spore.hidden_next[0] = true;
    spore.hidden_next[5] = true;
    spore.output_next[3] = true;

    spore.tick_end();

    // After tick_end, hidden should equal what hidden_next was
    assert!(spore.hidden[0]);
    assert!(spore.hidden[5]);
    assert!(!spore.hidden[1]);  // Was false

    // Same for output
    assert!(spore.output[3]);
    assert!(!spore.output[0]);  // Was false
}

#[test]
fn test_tick_end_decays_traces() {
    let mut spore = Spore::new();
    spore.trace_decay = 0.9;

    // Set some traces
    spore.traces_ih[0][0] = 1.0;
    spore.traces_ho[0][0] = 0.5;
    spore.traces_th[0] = 1.0;
    spore.traces_to[0] = 0.8;

    spore.tick_end();

    // Traces should decay by trace_decay
    assert!((spore.traces_ih[0][0] - 0.9).abs() < 0.001);
    assert!((spore.traces_ho[0][0] - 0.45).abs() < 0.001);
    assert!((spore.traces_th[0] - 0.9).abs() < 0.001);
    assert!((spore.traces_to[0] - 0.72).abs() < 0.001);
}

#[test]
fn test_tick_end_multiple_decays() {
    let mut spore = Spore::new();
    spore.trace_decay = 0.9;
    spore.traces_ih[0][0] = 1.0;

    // After 10 ticks, trace should be 0.9^10 ≈ 0.349
    for _ in 0..10 {
        spore.tick_end();
    }

    let expected = 0.9_f32.powi(10);
    assert!((spore.traces_ih[0][0] - expected).abs() < 0.001);
}

#[test]
fn test_output_as_byte_all_zeros() {
    let mut spore = Spore::new();
    spore.output = [false; OUTPUT_SIZE];
    assert_eq!(spore.output_as_byte(), 0x00);
}

#[test]
fn test_output_as_byte_all_ones() {
    let mut spore = Spore::new();
    spore.output = [true; OUTPUT_SIZE];
    assert_eq!(spore.output_as_byte(), 0xFF);
}

#[test]
fn test_output_as_byte_specific_pattern() {
    let mut spore = Spore::new();
    // Set bits 0, 2, 4, 6 (0b01010101 = 0x55)
    spore.output = [true, false, true, false, true, false, true, false];
    assert_eq!(spore.output_as_byte(), 0x55);
}

#[test]
fn test_output_as_byte_another_pattern() {
    let mut spore = Spore::new();
    // Set bits 1, 3, 5, 7 (0b10101010 = 0xAA)
    spore.output = [false, true, false, true, false, true, false, true];
    assert_eq!(spore.output_as_byte(), 0xAA);
}

#[test]
fn test_receive_reward_perfect_accuracy() {
    let mut spore = Spore::new();
    spore.receive_reward(8);  // 8/8 correct

    // dopamine = (8/8)² - (0.5)² = 1.0 - 0.25 = 0.75
    assert!((spore.dopamine - 0.75).abs() < 0.001);
}

#[test]
fn test_receive_reward_zero_accuracy() {
    let mut spore = Spore::new();
    spore.receive_reward(0);  // 0/8 correct

    // dopamine = (0/8)² - (0.5)² = 0 - 0.25 = -0.25
    assert!((spore.dopamine - (-0.25)).abs() < 0.001);
}

#[test]
fn test_receive_reward_baseline_accuracy() {
    let mut spore = Spore::new();
    spore.receive_reward(4);  // 4/8 = 50% = baseline

    // dopamine = (4/8)² - (0.5)² = 0.25 - 0.25 = 0
    assert!(spore.dopamine.abs() < 0.001);
}

#[test]
fn test_receive_reward_frustration_spikes_on_low_accuracy() {
    let mut spore = Spore::new();
    spore.frustration = 0.5;  // Start at some value

    spore.receive_reward(3);  // 3/8 = 37.5% < 50%

    // Fix 2: Frustration should spike to 1.0 immediately
    assert_eq!(spore.frustration, 1.0);
}

#[test]
fn test_receive_reward_frustration_decays_on_high_accuracy() {
    let mut spore = Spore::new();
    spore.frustration = 1.0;

    spore.receive_reward(8);  // 100% accuracy

    // frustration = 0.8 * 1.0 + 0.2 * (1.0 - 1.0) = 0.8
    assert!((spore.frustration - 0.8).abs() < 0.001);
}

#[test]
fn test_receive_reward_frustration_ema_on_medium_accuracy() {
    let mut spore = Spore::new();
    spore.frustration = 0.5;

    spore.receive_reward(6);  // 6/8 = 75% > 50%

    // frustration = 0.8 * 0.5 + 0.2 * (1.0 - 0.75) = 0.4 + 0.05 = 0.45
    assert!((spore.frustration - 0.45).abs() < 0.001);
}

#[test]
fn test_learn_increases_weights_on_positive_dopamine() {
    let mut spore = Spore::new();
    spore.dopamine = 0.5;
    spore.learning_rate = 0.5;

    // Set a trace
    spore.traces_ih[0][0] = 1.0;
    let original_weight = spore.weights_ih[0][0];

    // Run many learn cycles to accumulate stochastic changes
    for _ in 0..100 {
        spore.dopamine = 0.5;
        spore.traces_ih[0][0] = 1.0;
        spore.learn();
    }

    // Weight should have increased (statistically)
    assert!(spore.weights_ih[0][0] > original_weight,
        "Weight should increase with positive dopamine. Was {}, now {}",
        original_weight, spore.weights_ih[0][0]);
}

#[test]
fn test_learn_decreases_weights_on_negative_dopamine() {
    let mut spore = Spore::new();
    spore.learning_rate = 0.5;

    // Set a trace and a starting weight
    spore.traces_ih[0][0] = 1.0;
    spore.weights_ih[0][0] = 50;  // Start positive
    let original_weight = spore.weights_ih[0][0];

    // Run many learn cycles with negative dopamine
    for _ in 0..100 {
        spore.dopamine = -0.25;
        spore.traces_ih[0][0] = 1.0;
        spore.learn();
    }

    // Weight should have decreased
    assert!(spore.weights_ih[0][0] < original_weight,
        "Weight should decrease with negative dopamine. Was {}, now {}",
        original_weight, spore.weights_ih[0][0]);
}

#[test]
fn test_learn_consumes_dopamine() {
    let mut spore = Spore::new();
    spore.dopamine = 0.5;
    spore.traces_ih[0][0] = 1.0;

    spore.learn();

    assert_eq!(spore.dopamine, 0.0, "Dopamine should be consumed after learn");
}

#[test]
fn test_learn_does_nothing_with_zero_dopamine() {
    let mut spore = Spore::new();
    spore.dopamine = 0.0;
    spore.traces_ih[0][0] = 1.0;
    let original_weight = spore.weights_ih[0][0];

    spore.learn();

    assert_eq!(spore.weights_ih[0][0], original_weight,
        "Weight should not change with zero dopamine");
}

#[test]
fn test_learn_does_nothing_with_zero_trace() {
    let mut spore = Spore::new();
    spore.dopamine = 0.5;
    spore.traces_ih[0][0] = 0.0;  // No trace
    let original_weight = spore.weights_ih[0][0];

    spore.learn();

    assert_eq!(spore.weights_ih[0][0], original_weight,
        "Weight should not change with zero trace");
}

#[test]
fn test_learn_threshold_decreases_on_positive_dopamine() {
    let mut spore = Spore::new();
    spore.learning_rate = 0.5;
    spore.thresholds_h[0] = 50;

    // Run many cycles
    for _ in 0..100 {
        spore.dopamine = 0.5;
        spore.traces_th[0] = 1.0;
        spore.learn();
    }

    // Threshold should decrease (neuron becomes more eager)
    assert!(spore.thresholds_h[0] < 50,
        "Threshold should decrease with positive dopamine");
}

#[test]
fn test_learn_threshold_increases_on_negative_dopamine() {
    let mut spore = Spore::new();
    spore.learning_rate = 0.5;
    spore.thresholds_h[0] = 0;

    for _ in 0..100 {
        spore.dopamine = -0.25;
        spore.traces_th[0] = 1.0;
        spore.learn();
    }

    // Threshold should increase (neuron becomes more stubborn)
    assert!(spore.thresholds_h[0] > 0,
        "Threshold should increase with negative dopamine");
}
