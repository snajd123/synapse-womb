use spore_sim::swarm::Swarm;
use spore_sim::constants::*;

#[test]
fn test_swarm_construction() {
    let swarm = Swarm::new(32, 0.1, 0.9, 0.001, 0.05, 0.2, 100, 0.3);
    assert_eq!(swarm.size(), 32);
    assert_eq!(swarm.spores.len(), 32);
}

#[test]
fn test_swarm_different_sizes() {
    let s8 = Swarm::new(8, 0.1, 0.9, 0.001, 0.05, 0.2, 100, 0.3);
    let s32 = Swarm::new(32, 0.1, 0.9, 0.001, 0.05, 0.2, 100, 0.3);
    assert_eq!(s8.size(), 8);
    assert_eq!(s32.size(), 32);
}

#[test]
fn test_swarm_tick_returns_valid_accuracy() {
    let mut swarm = Swarm::new(32, 0.1, 0.9, 0.001, 0.05, 0.2, 100, 0.3);
    let inputs = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let targets = vec![true, false, true, false, true, false, true, false];
    let accuracy = swarm.tick(&inputs, &targets, 0);
    assert!(accuracy >= 0.0 && accuracy <= 1.0, "Accuracy {} out of range", accuracy);
}

#[test]
fn test_swarm_tick_perfect_score_possible() {
    let mut swarm = Swarm::new(32, 0.1, 0.9, 0.0, 0.0, 0.2, 100, 0.3);
    for spore in &mut swarm.spores {
        spore.base_noise = 0.0;
        spore.frustration = 0.0;
        for j in 0..HIDDEN_SIZE {
            spore.bias_h[j] = 100.0;
        }
        spore.bias_o = 100.0;
    }

    let inputs = [0.0; INPUT_SIZE];
    let targets = vec![true; 8];
    let accuracy = swarm.tick(&inputs, &targets, 0);
    assert_eq!(accuracy, 1.0, "Should be perfect when all outputs match targets");
}

#[test]
fn test_swarm_output_byte() {
    let mut swarm = Swarm::new(8, 0.1, 0.9, 0.001, 0.05, 0.2, 100, 0.3);
    // Manually set outputs
    swarm.spores[0].output = true;
    swarm.spores[1].output = false;
    swarm.spores[2].output = true;
    swarm.spores[3].output = false;
    swarm.spores[4].output = true;
    swarm.spores[5].output = false;
    swarm.spores[6].output = true;
    swarm.spores[7].output = false;
    assert_eq!(swarm.output_byte(), 0b01010101);
}

#[test]
fn test_swarm_rejuvenation_resets_stuck_spore() {
    let mut swarm = Swarm::new(2, 0.1, 0.9, 0.001, 0.05, 0.2, 100, 0.3);

    // Simulate a stuck Spore: low accuracy, past grace period
    swarm.spores[0].recent_accuracy = 0.50;
    swarm.spores[0].ticks_alive = REJUVENATION_GRACE_TICKS + 1;

    // Spore 1 is fine
    swarm.spores[1].recent_accuracy = 0.95;
    swarm.spores[1].ticks_alive = REJUVENATION_GRACE_TICKS + 1;

    swarm.rejuvenate();

    // Spore 0 should be reset
    assert_eq!(swarm.spores[0].ticks_alive, 0, "Stuck Spore should be rejuvenated");
    assert_eq!(swarm.spores[0].frustration, 1.0, "Rejuvenated Spore starts frustrated");

    // Spore 1 should be untouched
    assert!(swarm.spores[1].ticks_alive > 0, "Good Spore should not be reset");
}

#[test]
fn test_swarm_rejuvenation_respects_grace_period() {
    let mut swarm = Swarm::new(1, 0.1, 0.9, 0.001, 0.05, 0.2, 100, 0.3);

    // Low accuracy but within grace period
    swarm.spores[0].recent_accuracy = 0.40;
    swarm.spores[0].ticks_alive = 5_000; // < REJUVENATION_GRACE_TICKS

    swarm.rejuvenate();

    assert_eq!(swarm.spores[0].ticks_alive, 5_000, "Should not rejuvenate during grace period");
}

#[test]
fn test_swarm_accuracy() {
    let mut swarm = Swarm::new(8, 0.1, 0.9, 0.001, 0.05, 0.2, 100, 0.3);
    swarm.spores[0].recent_accuracy = 1.0;
    swarm.spores[1].recent_accuracy = 0.5;
    swarm.spores[2].recent_accuracy = 0.75;
    swarm.spores[3].recent_accuracy = 0.25;
    swarm.spores[4].recent_accuracy = 0.8;
    swarm.spores[5].recent_accuracy = 0.6;
    swarm.spores[6].recent_accuracy = 0.9;
    swarm.spores[7].recent_accuracy = 0.7;
    let acc = swarm.accuracy();
    let expected = (1.0 + 0.5 + 0.75 + 0.25 + 0.8 + 0.6 + 0.9 + 0.7) / 8.0;
    assert!((acc - expected).abs() < 0.001, "Expected {}, got {}", expected, acc);
}

#[test]
fn test_swarm_per_bit_credit_isolation() {
    // Verify that rewarding Spore 0 doesn't affect Spore 1
    let mut swarm = Swarm::new(2, 0.1, 0.9, 0.0, 0.0, 0.2, 100, 0.3);

    // Force deterministic: no noise
    for spore in &mut swarm.spores {
        spore.base_noise = 0.0;
        spore.frustration = 0.0;
    }

    // Set up Spore 0 with a known trace
    swarm.spores[0].traces_ih[0][0] = 1.0;
    swarm.spores[0].dopamine = 1.0;
    let s0_weight_before = swarm.spores[0].weights_ih[0][0];
    let s1_weight_before = swarm.spores[1].weights_ih[0][0];

    // Learn only Spore 0
    swarm.spores[0].learn();

    assert!(swarm.spores[0].weights_ih[0][0] != s0_weight_before,
        "Spore 0 should have learned");
    assert_eq!(swarm.spores[1].weights_ih[0][0], s1_weight_before,
        "Spore 1 should be unaffected");
}

#[test]
fn test_swarm_tick_target_mapping_modulo() {
    let mut swarm = Swarm::new(16, 0.1, 0.9, 0.0, 0.0, 0.2, 100, 0.3);
    for spore in &mut swarm.spores {
        spore.base_noise = 0.0;
        spore.frustration = 0.0;
    }

    // Force all Spores to output true by setting high biases
    for spore in &mut swarm.spores {
        for j in 0..HIDDEN_SIZE {
            spore.bias_h[j] = 100.0;
        }
        spore.bias_o = 100.0;
    }

    // targets only has 8 elements — tick should use i % 8
    let inputs = [0.0; INPUT_SIZE];
    let targets = vec![true; 8];
    let accuracy = swarm.tick(&inputs, &targets, 0);

    assert_eq!(accuracy, 1.0, "All Spores should match targets[i % 8]");
}

#[test]
fn test_swarm_output_byte_consensus_best_accuracy_wins() {
    let mut swarm = Swarm::new(16, 0.1, 0.9, 0.001, 0.05, 0.2, 100, 0.3);

    // For bit 0: Spores 0 and 8 compete
    swarm.spores[0].output = false;
    swarm.spores[0].recent_accuracy = 0.3;
    swarm.spores[8].output = true;
    swarm.spores[8].recent_accuracy = 0.9;

    // For bits 1-7: set all Spores to output=false
    for i in 1..16 {
        if i != 8 {
            swarm.spores[i].output = false;
        }
    }

    let byte = swarm.output_byte();
    assert_eq!(byte & 1, 1, "Bit 0 should be 1 (best-accuracy Spore 8 output)");
}

#[test]
fn test_swarm_output_byte_consensus_tiebreak_uses_first() {
    let mut swarm = Swarm::new(16, 0.1, 0.9, 0.001, 0.05, 0.2, 100, 0.3);

    swarm.spores[0].output = true;
    swarm.spores[0].recent_accuracy = 0.5;
    swarm.spores[8].output = false;
    swarm.spores[8].recent_accuracy = 0.5;

    for i in 1..16 {
        if i != 8 {
            swarm.spores[i].output = false;
        }
    }

    let byte = swarm.output_byte();
    assert_eq!(byte & 1, 1, "Bit 0 should be 1 (tiebreak favors first Spore)");
}

#[test]
fn test_swarm_accuracy_consensus_best_per_bit() {
    let mut swarm = Swarm::new(16, 0.1, 0.9, 0.001, 0.05, 0.2, 100, 0.3);

    swarm.spores[0].recent_accuracy = 0.3;
    swarm.spores[8].recent_accuracy = 0.9;

    for k in 1..8 {
        swarm.spores[k].recent_accuracy = 0.8;
        swarm.spores[k + 8].recent_accuracy = 0.2;
    }

    let acc = swarm.accuracy();
    let expected = (0.9 + 7.0 * 0.8) / 8.0;
    assert!((acc - expected).abs() < 0.001,
        "Consensus accuracy should be {}, got {}", expected, acc);
}
