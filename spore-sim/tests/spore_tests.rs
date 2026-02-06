use spore_sim::spore::Spore;
use spore_sim::constants::*;

// =============================================================================
// Construction
// =============================================================================

#[test]
fn test_spore_construction() {
    let s = Spore::default_params();
    assert_eq!(s.weights_ih.len(), HIDDEN_SIZE);
    assert_eq!(s.weights_ih[0].len(), INPUT_SIZE);
    assert_eq!(s.weights_ho.len(), HIDDEN_SIZE);
    assert_eq!(s.bias_h, [INITIAL_BIAS; HIDDEN_SIZE]);
    assert_eq!(s.bias_o, INITIAL_BIAS);
}

#[test]
fn test_spore_initial_state() {
    let s = Spore::default_params();
    assert_eq!(s.frustration, 1.0, "Start fully frustrated");
    assert_eq!(s.dopamine, 0.0);
    assert_eq!(s.recent_accuracy, 0.0);
    assert_eq!(s.ticks_alive, 0);
    assert_eq!(s.firing_rate, 0.0);
    assert_eq!(s.output, false);
    assert_eq!(s.hidden, [false; HIDDEN_SIZE]);
}

#[test]
fn test_spore_weights_in_range() {
    let s = Spore::default_params();
    for j in 0..HIDDEN_SIZE {
        for i in 0..INPUT_SIZE {
            assert!(s.weights_ih[j][i].abs() <= WEIGHT_INIT_RANGE,
                "weight_ih out of range: {}", s.weights_ih[j][i]);
        }
        assert!(s.weights_ho[j].abs() <= WEIGHT_INIT_RANGE,
            "weight_ho out of range: {}", s.weights_ho[j]);
    }
}

#[test]
fn test_spore_custom_params() {
    let s = Spore::new(0.2, 0.95, 0.002, 0.03, 0.1, 150, 0.4);
    assert_eq!(s.learning_rate, 0.2);
    assert_eq!(s.trace_decay, 0.95);
    assert_eq!(s.base_noise, 0.002);
    assert_eq!(s.max_noise_boost, 0.03);
    assert_eq!(s.frustration_alpha, 0.1);
    assert_eq!(s.weight_decay_interval, 150);
    assert_eq!(s.cortisol_strength, 0.4);
}

// =============================================================================
// fire()
// =============================================================================

#[test]
fn test_fire_returns_bool() {
    let mut s = Spore::default_params();
    let inputs = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let output = s.fire(&inputs);
    let _ = output;
}

#[test]
fn test_fire_with_strong_positive_weights_fires() {
    let mut s = Spore::default_params();
    s.base_noise = 0.0;
    s.frustration = 0.0;

    for j in 0..HIDDEN_SIZE {
        for i in 0..INPUT_SIZE {
            s.weights_ih[j][i] = 10.0;
        }
        s.weights_ho[j] = 10.0;
        s.bias_h[j] = 10.0;
    }
    s.bias_o = 10.0;

    let inputs = [1.0; INPUT_SIZE];
    let output = s.fire(&inputs);
    assert!(output, "Should fire with strongly positive weights");
}

#[test]
fn test_fire_with_zero_inputs_uses_bias() {
    let mut s = Spore::default_params();
    s.base_noise = 0.0;
    s.frustration = 0.0;

    for j in 0..HIDDEN_SIZE {
        s.bias_h[j] = 10.0;
    }
    s.bias_o = 10.0;

    let inputs = [0.0; INPUT_SIZE];
    let output = s.fire(&inputs);
    assert!(output, "Positive bias should cause firing with zero input");
}

#[test]
fn test_fire_sets_traces_on_active_synapses() {
    let mut s = Spore::default_params();
    s.base_noise = 0.0;
    s.frustration = 0.0;

    // Give hidden neuron 0 the highest sum so it wins WTA
    s.bias_h[0] = 100.0;
    for j in 1..HIDDEN_SIZE {
        s.bias_h[j] = -100.0; // suppress others
    }
    s.bias_o = 100.0;

    let mut inputs = [0.0; INPUT_SIZE];
    inputs[0] = 1.0;

    s.fire(&inputs);

    // Only the winner (H0) should have traces
    assert_eq!(s.traces_ih[0][0], 1.0, "Winner's active input should have trace 1.0");
    assert_eq!(s.traces_ih[0][1], 0.0, "Winner's inactive input should have trace 0.0");
    assert_eq!(s.trace_bias_h[0], 1.0, "Winner should have bias trace 1.0");

    // Suppressed neurons should NOT have traces (no noise)
    for j in 1..HIDDEN_SIZE {
        for i in 0..INPUT_SIZE {
            assert_eq!(s.traces_ih[j][i], 0.0,
                "Suppressed H{} should have no traces", j);
        }
        assert_eq!(s.trace_bias_h[j], 0.0,
            "Suppressed H{} should have no bias trace", j);
    }
    assert_eq!(s.trace_bias_o, 1.0, "Fired output should have bias trace 1.0");
}

#[test]
fn test_fire_no_traces_when_not_firing() {
    let mut s = Spore::default_params();
    s.base_noise = 0.0;
    s.frustration = 0.0;

    for j in 0..HIDDEN_SIZE {
        s.bias_h[j] = -100.0;
    }
    s.bias_o = -100.0;

    let inputs = [1.0; INPUT_SIZE];
    s.fire(&inputs);

    for j in 0..HIDDEN_SIZE {
        for i in 0..INPUT_SIZE {
            assert_eq!(s.traces_ih[j][i], 0.0);
        }
        assert_eq!(s.trace_bias_h[j], 0.0);
    }
    assert_eq!(s.trace_bias_o, 0.0);
}

// =============================================================================
// Winner-Take-All (Lateral Inhibition)
// =============================================================================

#[test]
fn test_fire_wta_only_winner_fires() {
    let mut s = Spore::default_params();
    s.base_noise = 0.0;
    s.frustration = 0.0;

    // Make neuron 2 the clear winner
    s.bias_h = [-100.0; HIDDEN_SIZE];
    s.bias_h[2] = 10.0;
    s.bias_o = 100.0;

    let inputs = [1.0; INPUT_SIZE];
    s.fire(&inputs);

    assert!(s.hidden[2], "Winner (H2) should fire");
    assert!(!s.hidden[0], "Non-winner H0 should NOT fire (no noise)");
    assert!(!s.hidden[1], "Non-winner H1 should NOT fire (no noise)");
    assert!(!s.hidden[3], "Non-winner H3 should NOT fire (no noise)");
}

#[test]
fn test_fire_wta_suppressed_neurons_no_traces() {
    let mut s = Spore::default_params();
    s.base_noise = 0.0;
    s.frustration = 0.0;

    // Make neuron 0 the winner, suppress others
    for j in 0..HIDDEN_SIZE {
        for i in 0..INPUT_SIZE {
            s.weights_ih[j][i] = 0.0;
        }
    }
    s.weights_ih[0][0] = 10.0;
    s.bias_h = [0.0; HIDDEN_SIZE];
    s.bias_h[0] = 5.0;
    s.bias_o = 100.0;

    let inputs = [1.0; INPUT_SIZE];
    s.fire(&inputs);

    assert_eq!(s.traces_ih[0][0], 1.0, "Winner should have trace on active input");
    assert_eq!(s.trace_bias_h[0], 1.0, "Winner should have bias trace");

    for j in 1..HIDDEN_SIZE {
        assert_eq!(s.trace_bias_h[j], 0.0,
            "Suppressed H{} should have no bias trace", j);
    }
}

#[test]
fn test_fire_wta_negative_winner_stays_silent() {
    let mut s = Spore::default_params();
    s.base_noise = 0.0;
    s.frustration = 0.0;

    s.bias_h = [-10.0; HIDDEN_SIZE];
    for j in 0..HIDDEN_SIZE {
        for i in 0..INPUT_SIZE {
            s.weights_ih[j][i] = -1.0;
        }
    }

    let inputs = [1.0; INPUT_SIZE];
    s.fire(&inputs);

    for j in 0..HIDDEN_SIZE {
        assert!(!s.hidden[j], "H{} should NOT fire when all sums are negative", j);
    }
}

#[test]
fn test_fire_wta_different_winners_for_different_inputs() {
    let mut s = Spore::default_params();
    s.base_noise = 0.0;
    s.frustration = 0.0;

    for j in 0..HIDDEN_SIZE {
        for i in 0..INPUT_SIZE {
            s.weights_ih[j][i] = 0.0;
        }
        s.bias_h[j] = 0.0;
    }
    s.weights_ih[0][0] = 10.0;
    s.weights_ih[1][1] = 10.0;
    s.bias_o = 100.0;

    let mut inputs_a = [0.0; INPUT_SIZE];
    inputs_a[0] = 1.0;
    s.fire(&inputs_a);
    assert!(s.hidden[0], "H0 should win when input[0]=1");
    assert!(!s.hidden[1], "H1 should NOT win when input[0]=1");

    s.traces_ih = [[0.0; INPUT_SIZE]; HIDDEN_SIZE];
    s.traces_ho = [0.0; HIDDEN_SIZE];
    s.trace_bias_h = [0.0; HIDDEN_SIZE];
    s.trace_bias_o = 0.0;

    let mut inputs_b = [0.0; INPUT_SIZE];
    inputs_b[1] = 1.0;
    s.fire(&inputs_b);
    assert!(s.hidden[1], "H1 should win when input[1]=1");
    assert!(!s.hidden[0], "H0 should NOT win when input[1]=1");
}

// =============================================================================
// receive_reward()
// =============================================================================

#[test]
fn test_reward_correct_sets_positive_dopamine() {
    let mut s = Spore::default_params();
    s.receive_reward(true);
    assert_eq!(s.dopamine, 1.0);
}

#[test]
fn test_reward_wrong_sets_negative_dopamine() {
    let mut s = Spore::default_params();
    s.receive_reward(false);
    assert_eq!(s.dopamine, -DEFAULT_CORTISOL_STRENGTH);
}

#[test]
fn test_reward_updates_accuracy_ema() {
    let mut s = Spore::default_params();
    s.recent_accuracy = 0.0;
    s.receive_reward(true);
    assert!((s.recent_accuracy - 0.05).abs() < 0.001);
}

#[test]
fn test_reward_frustration_spike_when_generally_failing() {
    let mut s = Spore::default_params();
    s.frustration = 0.5;
    s.recent_accuracy = 0.3;
    s.receive_reward(false);
    assert_eq!(s.frustration, 1.0, "Fix 2: spike when recent_accuracy < 50%");
}

#[test]
fn test_reward_frustration_no_spike_when_generally_succeeding() {
    let mut s = Spore::default_params();
    s.frustration = 0.3;
    s.recent_accuracy = 0.90;
    s.receive_reward(false);
    assert!(s.frustration < 1.0, "Should NOT spike when recent_accuracy > 50%");
    assert!((s.frustration - 0.44).abs() < 0.01);
}

#[test]
fn test_reward_frustration_decays_on_correct() {
    let mut s = Spore::default_params();
    s.frustration = 1.0;
    s.recent_accuracy = 0.8;
    s.receive_reward(true);
    assert!((s.frustration - 0.8).abs() < 0.001);
}

#[test]
fn test_reward_custom_cortisol_strength() {
    let mut s = Spore::new(0.1, 0.9, 0.001, 0.05, 0.2, 100, 0.7);
    s.receive_reward(false);
    assert_eq!(s.dopamine, -0.7);
}

// =============================================================================
// learn()
// =============================================================================

#[test]
fn test_learn_strengthens_on_dopamine() {
    let mut s = Spore::default_params();
    s.traces_ih[0][0] = 1.0;
    let original = s.weights_ih[0][0];
    s.dopamine = 1.0;
    s.learn();
    assert!(s.weights_ih[0][0] > original,
        "Weight should increase. Was {}, now {}", original, s.weights_ih[0][0]);
}

#[test]
fn test_learn_weakens_on_cortisol() {
    let mut s = Spore::default_params();
    s.weights_ih[0][0] = 0.3;
    s.traces_ih[0][0] = 1.0;
    s.dopamine = -0.3;
    s.learn();
    assert!(s.weights_ih[0][0] < 0.3,
        "Weight should decrease with cortisol. Now {}", s.weights_ih[0][0]);
}

#[test]
fn test_learn_updates_bias() {
    let mut s = Spore::default_params();
    s.trace_bias_h[0] = 1.0;
    s.dopamine = 1.0;
    let original_bias = s.bias_h[0];
    s.learn();
    assert!(s.bias_h[0] > original_bias, "Bias should increase with dopamine");
}

#[test]
fn test_learn_consumes_dopamine() {
    let mut s = Spore::default_params();
    s.dopamine = 1.0;
    s.traces_ih[0][0] = 1.0;
    s.learn();
    assert_eq!(s.dopamine, 0.0);
}

#[test]
fn test_learn_no_change_with_zero_dopamine() {
    let mut s = Spore::default_params();
    s.dopamine = 0.0;
    s.traces_ih[0][0] = 1.0;
    let original = s.weights_ih[0][0];
    s.learn();
    assert_eq!(s.weights_ih[0][0], original);
}

#[test]
fn test_learn_no_change_with_zero_trace() {
    let mut s = Spore::default_params();
    s.dopamine = 1.0;
    s.traces_ih[0][0] = 0.0;
    let original = s.weights_ih[0][0];
    s.learn();
    assert_eq!(s.weights_ih[0][0], original);
}

#[test]
fn test_learn_gated_by_high_accuracy() {
    let mut s = Spore::default_params();
    s.weights_ih[0][0] = 0.0;
    s.traces_ih[0][0] = 1.0;
    s.dopamine = 1.0;
    s.recent_accuracy = 0.9;
    s.learn();
    // effective_lr = 0.1 * (1.0 - 0.9) = 0.01
    // weight change = 0.01 * 1.0 * 1.0 = 0.01
    assert!((s.weights_ih[0][0] - 0.01).abs() < 0.001,
        "At 90% accuracy, weight change should be ~0.01, got {}", s.weights_ih[0][0]);
}

#[test]
fn test_learn_ungated_at_zero_accuracy() {
    let mut s = Spore::default_params();
    s.weights_ih[0][0] = 0.0;
    s.traces_ih[0][0] = 1.0;
    s.dopamine = 1.0;
    s.recent_accuracy = 0.0;
    s.learn();
    // effective_lr = 0.1 * (1.0 - 0.0) = 0.1
    // weight change = 0.1 * 1.0 * 1.0 = 0.1
    assert!((s.weights_ih[0][0] - 0.1).abs() < 0.001,
        "At 0% accuracy, weight change should be ~0.1, got {}", s.weights_ih[0][0]);
}

#[test]
fn test_learn_cortisol_barely_moves_converged_spore() {
    let mut s = Spore::default_params();
    s.weights_ih[0][0] = 1.0;
    s.traces_ih[0][0] = 1.0;
    s.dopamine = -DEFAULT_CORTISOL_STRENGTH; // cortisol
    s.recent_accuracy = 0.95;
    s.learn();
    // effective_lr = 0.1 * (1.0 - 0.95) = 0.005
    // weight change = 0.005 * -0.3 * 1.0 = -0.0015
    // new weight = 1.0 - 0.0015 = 0.9985
    assert!(s.weights_ih[0][0] > 0.99,
        "Converged Spore should barely lose weight from cortisol, got {}", s.weights_ih[0][0]);
}

// =============================================================================
// maintain()
// =============================================================================

#[test]
fn test_maintain_decays_traces() {
    let mut s = Spore::default_params();
    s.traces_ih[0][0] = 1.0;
    s.traces_ho[0] = 1.0;
    s.trace_bias_h[0] = 1.0;
    s.trace_bias_o = 1.0;

    s.maintain(1);

    assert!((s.traces_ih[0][0] - 0.9).abs() < 0.001);
    assert!((s.traces_ho[0] - 0.9).abs() < 0.001);
    assert!((s.trace_bias_h[0] - 0.9).abs() < 0.001);
    assert!((s.trace_bias_o - 0.9).abs() < 0.001);
}

#[test]
fn test_maintain_weight_decay_at_interval() {
    let mut s = Spore::default_params();
    s.weights_ih[0][0] = 1.0;
    s.maintain(DEFAULT_WEIGHT_DECAY_INTERVAL);
    assert!((s.weights_ih[0][0] - 0.99).abs() < 0.001);
}

#[test]
fn test_maintain_no_weight_decay_off_interval() {
    let mut s = Spore::default_params();
    s.weights_ih[0][0] = 1.0;
    s.maintain(50);
    assert_eq!(s.weights_ih[0][0], 1.0);
}

#[test]
fn test_maintain_bias_not_decayed() {
    let mut s = Spore::default_params();
    s.bias_h[0] = 1.0;
    s.bias_o = 1.0;
    // Set firing_rate = target_rate so homeostasis doesn't nudge bias_o
    s.firing_rate = s.target_rate;
    s.maintain(DEFAULT_WEIGHT_DECAY_INTERVAL);
    assert_eq!(s.bias_h[0], 1.0, "Hidden bias should NOT be decayed");
    assert_eq!(s.bias_o, 1.0, "Output bias should NOT be decayed when at target rate");
}

#[test]
fn test_maintain_increments_ticks_alive() {
    let mut s = Spore::default_params();
    assert_eq!(s.ticks_alive, 0);
    s.maintain(0);
    assert_eq!(s.ticks_alive, 1);
    s.maintain(1);
    assert_eq!(s.ticks_alive, 2);
}

// =============================================================================
// Activity Homeostasis
// =============================================================================

#[test]
fn test_fire_updates_firing_rate_ema() {
    let mut s = Spore::default_params();
    s.base_noise = 0.0;
    s.frustration = 0.0;
    // Force output to fire
    for j in 0..HIDDEN_SIZE {
        s.weights_ih[j] = [10.0; INPUT_SIZE];
        s.weights_ho[j] = 10.0;
        s.bias_h[j] = 10.0;
    }
    s.bias_o = 10.0;

    assert_eq!(s.firing_rate, 0.0);
    s.fire(&[1.0; INPUT_SIZE]);
    // rate = 0.99 * 0.0 + 0.01 * 1.0 = 0.01
    assert!((s.firing_rate - 0.01).abs() < 0.001);
}

#[test]
fn test_fire_firing_rate_stays_zero_when_silent() {
    let mut s = Spore::default_params();
    s.base_noise = 0.0;
    s.frustration = 0.0;
    // Force output to NOT fire
    for j in 0..HIDDEN_SIZE {
        s.bias_h[j] = -100.0;
    }
    s.bias_o = -100.0;

    s.fire(&[0.0; INPUT_SIZE]);
    // rate = 0.99 * 0.0 + 0.01 * 0.0 = 0.0
    assert_eq!(s.firing_rate, 0.0);
}

#[test]
fn test_homeostasis_increases_bias_when_silent() {
    let mut s = Spore::default_params();
    s.firing_rate = 0.0; // Dead neuron
    let original_bias = s.bias_o;
    s.maintain(1);
    // diff = 0.1 - 0.0 = 0.1, bias_o += 0.01 * 0.1 = 0.001
    assert!(s.bias_o > original_bias, "Homeostasis should increase bias_o for silent Spore");
    assert!((s.bias_o - original_bias - 0.001).abs() < 0.0001);
}

#[test]
fn test_homeostasis_decreases_bias_when_overactive() {
    let mut s = Spore::default_params();
    s.firing_rate = 0.5; // Firing 50% of the time (target is 10%)
    let original_bias = s.bias_o;
    s.maintain(1);
    // diff = 0.1 - 0.5 = -0.4, bias_o += 0.01 * -0.4 = -0.004
    assert!(s.bias_o < original_bias, "Homeostasis should decrease bias_o for overactive Spore");
    assert!((s.bias_o - original_bias + 0.004).abs() < 0.0001);
}

#[test]
fn test_homeostasis_no_change_at_target() {
    let mut s = Spore::default_params();
    s.firing_rate = DEFAULT_TARGET_RATE; // Exactly at target
    let original_bias = s.bias_o;
    s.maintain(1);
    assert_eq!(s.bias_o, original_bias, "No adjustment needed at target rate");
}

// =============================================================================
// reset()
// =============================================================================

#[test]
fn test_reset_clears_state() {
    let mut s = Spore::default_params();
    s.ticks_alive = 50000;
    s.frustration = 0.1;
    s.recent_accuracy = 0.99;
    s.dopamine = 0.5;
    s.traces_ih[0][0] = 0.8;
    s.bias_h[0] = 5.0;

    s.firing_rate = 0.8;

    s.reset();

    assert_eq!(s.ticks_alive, 0);
    assert_eq!(s.frustration, 1.0);
    assert_eq!(s.recent_accuracy, 0.0);
    assert_eq!(s.dopamine, 0.0);
    assert_eq!(s.traces_ih[0][0], 0.0);
    assert_eq!(s.bias_h[0], INITIAL_BIAS);
    assert_eq!(s.bias_o, INITIAL_BIAS);
    assert_eq!(s.firing_rate, 0.0);
    assert_eq!(s.output, false);
}

#[test]
fn test_reset_randomizes_weights() {
    let mut s = Spore::default_params();
    s.weights_ih[0][0] = 999.0;
    s.reset();
    assert!(s.weights_ih[0][0].abs() <= WEIGHT_INIT_RANGE,
        "Weights should be re-randomized, got {}", s.weights_ih[0][0]);
}
