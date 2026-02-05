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
