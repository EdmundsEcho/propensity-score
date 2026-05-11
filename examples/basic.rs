//! Minimal end-to-end example: fit → predict → AUC → decile bin.
//!
//! Run with:
//!     cargo run --example basic
//!
//! The toy data is 6 subjects, 1 feature, clearly separable. The point is
//! to show the API surface — `fit`, `Findings::predict`, `Findings::auc`,
//! and `Findings::bin` — not to demonstrate model quality.

use nalgebra::{DMatrix, DVector};
use propensity_score::{fit, BinConfig, BinStrategy, Config};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 6 subjects, 1 feature. Positive feature → class 1, negative → class 0.
    let features = DMatrix::from_row_slice(
        6,
        1,
        &[3.0, 2.0, 1.0, -1.0, -2.0, -3.0],
    );
    let target = DVector::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);

    let findings = fit(&features, &target, &Config::default())?;
    println!(
        "fit: intercept = {:.3}, weights = {:?}, iter = {}, cost = {:.5}",
        findings.intercept,
        findings.weights.iter().collect::<Vec<_>>(),
        findings.iterations,
        findings.cost,
    );

    let scores = findings.predict(&features);
    println!("\nscores per subject:");
    for (i, s) in scores.iter().enumerate() {
        println!("  subject {i}: {s:.3} (target = {})", target[i]);
    }

    let auc = findings.auc(&scores, &target);
    println!("\nAUC = {auc:.3}");

    // Decile binning. With only 6 subjects this just spreads them out;
    // the realistic use case is N >> bin count.
    let bins = findings.bin(
        &scores,
        &BinConfig {
            count: 5,
            strategy: BinStrategy::Quantile,
        },
    );
    println!("\nquintile bins per subject: {bins:?}");

    Ok(())
}
