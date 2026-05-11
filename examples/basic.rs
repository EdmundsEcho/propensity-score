//! Realistic-shape example: estimate the propensity that a subject
//! received a marketing campaign, given basic demographics.
//!
//! Run with:
//!     cargo run --example basic
//!
//! The data is 15 synthetic subjects with a planted pattern (California
//! residents + older people skew toward being treated). The example shows
//! how to:
//!   1. Encode categorical features:
//!      - `sex` as a 0/1 indicator (single column)
//!      - `state` as one-hot indicators with one level dropped (NY here
//!        is the *reference* category — both indicators are 0)
//!   2. Z-score a numeric feature (`age`) so gradient descent is
//!      well-conditioned (raw magnitudes in the dozens cause the solver
//!      to lurch).
//!   3. Fit the logistic-regression model and inspect the AUC.
//!   4. Bin propensity scores for downstream matching.

use nalgebra::{DMatrix, DVector};
use propensity_score::{fit, BinConfig, BinStrategy, Config};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // (sex_M, state_CA, state_TX, age_years, treated)
    // state NY is the reference: state_CA = state_TX = 0 means "NY".
    let rows: [(f64, f64, f64, f64, f64); 15] = [
        (0.0, 1.0, 0.0, 55.0, 1.0),
        (0.0, 1.0, 0.0, 62.0, 1.0),
        (1.0, 1.0, 0.0, 58.0, 1.0),
        (1.0, 1.0, 0.0, 71.0, 1.0),
        (0.0, 1.0, 0.0, 49.0, 1.0),
        (0.0, 0.0, 0.0, 45.0, 0.0),
        (1.0, 0.0, 0.0, 38.0, 0.0),
        (1.0, 0.0, 0.0, 52.0, 0.0),
        (0.0, 0.0, 0.0, 41.0, 0.0),
        (1.0, 0.0, 0.0, 33.0, 0.0),
        (0.0, 0.0, 1.0, 47.0, 0.0),
        (1.0, 0.0, 1.0, 39.0, 0.0),
        (0.0, 0.0, 1.0, 51.0, 1.0),
        (1.0, 0.0, 1.0, 64.0, 1.0),
        (0.0, 0.0, 1.0, 56.0, 1.0),
    ];
    let n = rows.len();

    // Z-score age before fitting. Without this the solver lurches because
    // age (~30-70) dwarfs the 0/1 indicators in gradient magnitude.
    let age_mean = rows.iter().map(|r| r.3).sum::<f64>() / n as f64;
    let age_sd = (rows.iter().map(|r| (r.3 - age_mean).powi(2)).sum::<f64>()
        / n as f64)
        .sqrt()
        .max(1e-9);

    let mut features = DMatrix::zeros(n, 4);
    let mut treatment = DVector::zeros(n);
    for (i, r) in rows.iter().enumerate() {
        features[(i, 0)] = r.0; // sex_M
        features[(i, 1)] = r.1; // state_CA
        features[(i, 2)] = r.2; // state_TX
        features[(i, 3)] = (r.3 - age_mean) / age_sd; // age_z
        treatment[i] = r.4;
    }

    let findings = fit(&features, &treatment, &Config::default())?;
    let labels = ["sex_M", "state_CA", "state_TX", "age_z"];
    println!(
        "fit (iterations = {}, cost = {:.4}):",
        findings.iterations, findings.cost,
    );
    println!("  intercept = {:+.3}", findings.intercept);
    for (i, name) in labels.iter().enumerate() {
        println!("  {name:>10} weight = {:+.3}", findings.weights[i]);
    }

    let scores = findings.predict(&features);
    println!("\npropensity scores:");
    for (i, r) in rows.iter().enumerate() {
        let state = if r.1 > 0.5 {
            "CA"
        } else if r.2 > 0.5 {
            "TX"
        } else {
            "NY"
        };
        let sex = if r.0 > 0.5 { "M" } else { "F" };
        println!(
            "  #{i:>2}: {sex} {state} age {:>2}  →  score = {:.3}  (treatment = {})",
            r.3 as i32,
            scores[i],
            r.4 as i32,
        );
    }

    let auc = findings.auc(&scores, &treatment);
    println!("\nAUC = {auc:.3}   (0.5 = random, 1.0 = perfect)");

    // Quintile bins: subjects with similar scores share a bin. Useful
    // starting point for stratified matching.
    let bins = findings.bin(
        &scores,
        &BinConfig {
            count: 5,
            strategy: BinStrategy::Quantile,
        },
    );
    println!("\nquintile bin per subject: {bins:?}");

    Ok(())
}
