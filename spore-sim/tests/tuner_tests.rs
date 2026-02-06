use spore_sim::tuner::{Genome, TunerConfig, evaluate_fast, evaluate_full};

#[test]
fn test_genome_random_in_range() {
    for _ in 0..50 {
        let g = Genome::random();
        assert!(g.learning_rate >= 0.05 && g.learning_rate <= 0.5);
        assert!(g.trace_decay >= 0.85 && g.trace_decay <= 0.999);
        assert!(g.max_noise_boost >= 0.01 && g.max_noise_boost <= 0.05);
        assert!(g.weight_decay_interval >= 50 && g.weight_decay_interval <= 200);
        assert!(g.frustration_alpha >= 0.05 && g.frustration_alpha <= 0.5);
        assert!(g.input_hold_ticks >= 20 && g.input_hold_ticks <= 200);
        assert!(g.cortisol_strength >= 0.5 && g.cortisol_strength <= 1.5,
            "cortisol_strength {} out of range [0.5, 1.5]", g.cortisol_strength);
    }
}

#[test]
fn test_genome_mutate_stays_in_range() {
    for _ in 0..100 {
        let mut g = Genome::random();
        g.mutate(1.0);
        assert!(g.learning_rate >= 0.05 && g.learning_rate <= 0.5);
        assert!(g.trace_decay >= 0.85 && g.trace_decay <= 0.999);
        assert!(g.max_noise_boost >= 0.01 && g.max_noise_boost <= 0.05);
        assert!(g.weight_decay_interval >= 50 && g.weight_decay_interval <= 200);
        assert!(g.frustration_alpha >= 0.05 && g.frustration_alpha <= 0.5);
        assert!(g.input_hold_ticks >= 20 && g.input_hold_ticks <= 200);
        assert!(g.cortisol_strength >= 0.5 && g.cortisol_strength <= 1.5);
    }
}

#[test]
fn test_genome_mutate_boosted_stays_in_range() {
    for _ in 0..100 {
        let mut g = Genome::random();
        g.mutate(2.0);
        assert!(g.cortisol_strength >= 0.5 && g.cortisol_strength <= 1.5);
    }
}

#[test]
fn test_genome_mutate_changes_values() {
    let original = Genome::random();
    let mut changed_count = 0;
    for _ in 0..100 {
        let mut g = original.clone();
        g.mutate(1.0);
        if (g.learning_rate - original.learning_rate).abs() > 0.0001 {
            changed_count += 1;
        }
    }
    assert!(changed_count > 0, "Mutation should change values sometimes");
}

#[test]
fn test_genome_crossover_combines_parents() {
    let mut parent_a = Genome::random();
    let mut parent_b = Genome::random();
    parent_a.cortisol_strength = 0.6;
    parent_b.cortisol_strength = 1.4;

    let mut got_a = false;
    let mut got_b = false;

    for _ in 0..100 {
        let child = Genome::crossover(&parent_a, &parent_b);
        if (child.cortisol_strength - 0.6).abs() < 0.001 { got_a = true; }
        if (child.cortisol_strength - 1.4).abs() < 0.001 { got_b = true; }
    }

    assert!(got_a && got_b, "Crossover should use cortisol from both parents");
}

#[test]
fn test_genome_serialization() {
    let g = Genome::random();
    let json = serde_json::to_string(&g).unwrap();
    let g2: Genome = serde_json::from_str(&json).unwrap();
    assert!((g.learning_rate - g2.learning_rate).abs() < 0.0001);
    assert!((g.cortisol_strength - g2.cortisol_strength).abs() < 0.0001);
    assert_eq!(g.weight_decay_interval, g2.weight_decay_interval);
}

#[test]
fn test_eval_result_score_is_finite() {
    let g = Genome::random();
    let result = evaluate_fast(&g, 1000);
    assert!(result.score.is_finite(), "Score must be finite, got {}", result.score);
    assert!(result.final_accuracy >= 0.0 && result.final_accuracy <= 1.0);
}

#[test]
fn test_evaluate_fast_returns_valid_result() {
    let g = Genome::random();
    let result = evaluate_fast(&g, 2000);
    assert!(result.score.is_finite());
    assert!(result.final_accuracy >= 0.0 && result.final_accuracy <= 1.0);
}

#[test]
fn test_evaluate_full_returns_valid_result() {
    let g = Genome::random();
    let result = evaluate_full(&g, 2000);
    assert!(result.score.is_finite());
    assert!(result.final_accuracy >= 0.0 && result.final_accuracy <= 1.0);
}

#[test]
fn test_evaluate_stable_genome_scores_higher() {
    let a = 0.95_f32;
    let t = 5000_u64;
    let stable_score = a * 2000.0 - (t as f32 / 100.0);
    let unstable_score = a * 500.0;
    assert!(stable_score > unstable_score);
}

#[test]
fn test_tuner_config_default() {
    let config = TunerConfig::default();
    assert_eq!(config.population_size, 50);
    assert_eq!(config.generations, 20);
    assert_eq!(config.elite_count, 10);
    assert_eq!(config.ticks_per_eval, 20_000);
}

#[test]
fn test_tune_tiny_run() {
    let config = TunerConfig {
        population_size: 6,
        generations: 2,
        elite_count: 2,
        ticks_per_eval: 500,
        finalist_count: 2,
    };
    let (best_genome, best_result) = spore_sim::tuner::tune(&config);
    assert!(best_result.score.is_finite());
    assert!(best_genome.cortisol_strength >= 0.5 && best_genome.cortisol_strength <= 1.5);
}
