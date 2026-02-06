use spore_sim::utils::random_weight;
use spore_sim::constants::WEIGHT_INIT_RANGE;

#[test]
fn test_random_weight_in_range() {
    for _ in 0..1000 {
        let w = random_weight();
        assert!(w >= -WEIGHT_INIT_RANGE && w <= WEIGHT_INIT_RANGE,
            "Weight {} outside range", w);
    }
}

#[test]
fn test_random_weight_distribution() {
    let mut seen_positive = false;
    let mut seen_negative = false;
    for _ in 0..100 {
        let w = random_weight();
        if w > 0.0 { seen_positive = true; }
        if w < 0.0 { seen_negative = true; }
    }
    assert!(seen_positive && seen_negative, "Should produce both positive and negative");
}
