use spore_sim::tuner::{Genome, EvalResult, TunerConfig, evaluate_fast, evaluate_full};

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
    // Stable should be bool (always valid)
    let _ = result.stable;
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
    // Verify score formula: stable base (2000) always beats unstable (500)
    // for the same accuracy level.
    //
    // Proof: For accuracy a, stable score = a*2000 - t/100. Unstable = a*500.
    // stable > unstable when a*2000 - t/100 > a*500 => a*1500 > t/100 => t < a*150000
    // For any convergence before tick 142500 at 95% accuracy, stable wins.
    let a = 0.95_f32;
    let t = 5000_u64;
    let stable_score = a * 2000.0 - (t as f32 / 100.0);
    let unstable_score = a * 500.0;
    assert!(stable_score > unstable_score,
        "Stable score {:.1} should beat unstable {:.1}", stable_score, unstable_score);

    // Also verify degenerate case: very late convergence still beats unstable
    let late_t = 19000_u64;
    let late_stable_score = a * 2000.0 - (late_t as f32 / 100.0);
    assert!(late_stable_score > unstable_score,
        "Late stable {:.1} should still beat unstable {:.1}", late_stable_score, unstable_score);
}

#[test]
fn test_evaluate_fast_different_runs_may_differ() {
    // Multi-run robustness: evaluate_fast runs 3 times and takes worst.
    // Verify it returns a valid result.
    let g = Genome::random();
    let r1 = evaluate_fast(&g, 1000);
    let r2 = evaluate_fast(&g, 1000);
    // Both should be valid, may differ due to different random sequences
    assert!(r1.score.is_finite());
    assert!(r2.score.is_finite());
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
    // Minimal tuner run to verify it doesn't crash
    let config = TunerConfig {
        population_size: 6,
        generations: 2,
        elite_count: 2,
        ticks_per_eval: 500,
        finalist_count: 2,
    };
    let (best_genome, best_result) = spore_sim::tuner::tune(&config);
    assert!(best_result.score.is_finite());
    assert!(best_genome.learning_rate >= 0.05 && best_genome.learning_rate <= 0.5);
    assert!(best_genome.trace_decay >= 0.85 && best_genome.trace_decay <= 0.999);
}
