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
