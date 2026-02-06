use spore_sim::tuner::Genome;

#[test]
fn test_genome_random_in_range() {
    for _ in 0..50 {
        let g = Genome::random();
        assert!(g.learning_rate >= 0.05 && g.learning_rate <= 0.5,
            "learning_rate {} out of range", g.learning_rate);
        assert!(g.trace_decay >= 0.85 && g.trace_decay <= 0.999,
            "trace_decay {} out of range", g.trace_decay);
        assert!(g.max_noise_boost >= 0.01 && g.max_noise_boost <= 0.05,
            "max_noise_boost {} out of range", g.max_noise_boost);
        assert!(g.weight_decay_interval >= 50 && g.weight_decay_interval <= 200,
            "weight_decay_interval {} out of range", g.weight_decay_interval);
        assert!(g.frustration_alpha >= 0.05 && g.frustration_alpha <= 0.5,
            "frustration_alpha {} out of range", g.frustration_alpha);
        assert!(g.input_hold_ticks >= 20 && g.input_hold_ticks <= 200,
            "input_hold_ticks {} out of range", g.input_hold_ticks);
    }
}

#[test]
fn test_genome_mutate_stays_in_range() {
    for _ in 0..100 {
        let mut g = Genome::random();
        g.mutate(1.0);  // Normal magnitude
        assert!(g.learning_rate >= 0.05 && g.learning_rate <= 0.5);
        assert!(g.trace_decay >= 0.85 && g.trace_decay <= 0.999);
        assert!(g.max_noise_boost >= 0.01 && g.max_noise_boost <= 0.05);
        assert!(g.weight_decay_interval >= 50 && g.weight_decay_interval <= 200);
        assert!(g.frustration_alpha >= 0.05 && g.frustration_alpha <= 0.5);
        assert!(g.input_hold_ticks >= 20 && g.input_hold_ticks <= 200);
    }
}

#[test]
fn test_genome_mutate_boosted_stays_in_range() {
    for _ in 0..100 {
        let mut g = Genome::random();
        g.mutate(2.0);  // Boosted magnitude
        assert!(g.learning_rate >= 0.05 && g.learning_rate <= 0.5);
        assert!(g.trace_decay >= 0.85 && g.trace_decay <= 0.999);
        assert!(g.max_noise_boost >= 0.01 && g.max_noise_boost <= 0.05);
        assert!(g.weight_decay_interval >= 50 && g.weight_decay_interval <= 200);
        assert!(g.frustration_alpha >= 0.05 && g.frustration_alpha <= 0.5);
        assert!(g.input_hold_ticks >= 20 && g.input_hold_ticks <= 200);
    }
}

#[test]
fn test_genome_mutate_changes_values() {
    // Over many mutations, at least SOME values should change
    let original = Genome::random();
    let mut changed_count = 0;

    for _ in 0..100 {
        let mut g = original.clone();
        g.mutate(1.0);
        if (g.learning_rate - original.learning_rate).abs() > 0.0001 {
            changed_count += 1;
        }
    }

    // With 10% mutation probability per gene, ~10% should change
    assert!(changed_count > 0, "Mutation should change values sometimes");
}

#[test]
fn test_genome_crossover_combines_parents() {
    // Create two very different parents
    let mut parent_a = Genome::random();
    let mut parent_b = Genome::random();
    parent_a.learning_rate = 0.05;
    parent_b.learning_rate = 0.5;
    parent_a.trace_decay = 0.85;
    parent_b.trace_decay = 0.999;

    // Over many crossovers, child should get genes from both parents
    let mut got_a_lr = false;
    let mut got_b_lr = false;

    for _ in 0..100 {
        let child = Genome::crossover(&parent_a, &parent_b);
        if (child.learning_rate - 0.05).abs() < 0.001 {
            got_a_lr = true;
        }
        if (child.learning_rate - 0.5).abs() < 0.001 {
            got_b_lr = true;
        }
    }

    assert!(got_a_lr && got_b_lr, "Crossover should use genes from both parents");
}

#[test]
fn test_genome_serialization() {
    let g = Genome::random();
    let json = serde_json::to_string(&g).unwrap();
    let g2: Genome = serde_json::from_str(&json).unwrap();
    assert!((g.learning_rate - g2.learning_rate).abs() < 0.0001);
    assert!((g.trace_decay - g2.trace_decay).abs() < 0.0001);
    assert_eq!(g.weight_decay_interval, g2.weight_decay_interval);
    assert_eq!(g.input_hold_ticks, g2.input_hold_ticks);
}
