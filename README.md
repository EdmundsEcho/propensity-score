# propensity-score

[![crates.io](https://img.shields.io/crates/v/propensity-score.svg)](https://crates.io/crates/propensity-score)
[![docs.rs](https://docs.rs/propensity-score/badge.svg)](https://docs.rs/propensity-score)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

Small, dependency-light logistic regression with quantile binning. Built for
propensity-score estimation in matching workflows — especially on zero-heavy
data like healthcare prescribing, marketing-touch, or sales-activity datasets
where many subjects have zero activity.

## Why

If you need propensity scores for a matching / causal-inference pipeline
and you don't want to pull in the full [`linfa`](https://crates.io/crates/linfa)
or [`smartcore`](https://crates.io/crates/smartcore) ecosystem just for that,
this crate gives you:

- **Logistic regression** via gradient descent — fits weights + intercept,
  reports cost and iteration count.
- **`predict`** returns a `[0, 1]` propensity vector.
- **`bin`** assigns each subject to a quantile or equal-range bin, with
  **zero-aware quantile binning**: subjects with score `0.0` get bin `0`,
  and the rest are ranked into equal-count bins `1..N`. This matches the
  way matching frameworks usually treat non-writers / non-actives.
- **`auc`** — area under the ROC curve, for sanity-checking the fit.
- One library, three modules (`logistic`, `binning`, `config`), ~650 lines,
  three dependencies (`nalgebra`, `thiserror`, `tracing`).

## Quick start

```toml
# Cargo.toml
[dependencies]
propensity-score = "0.1"
nalgebra = "0.33"
```

```rust
use nalgebra::{DMatrix, DVector};
use propensity_score::{fit, BinConfig, BinStrategy, Config};

// 6 subjects, 1 feature (the +/- pattern is clearly separable here).
let features = DMatrix::from_row_slice(6, 1, &[
    3.0, 2.0, 1.0,
   -1.0, -2.0, -3.0,
]);
let target = DVector::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);

let findings = fit(&features, &target, &Config::default())?;
let scores = findings.predict(&features);

println!("AUC = {:.3}", findings.auc(&scores, &target));

// Decile binning (quantile, count = 10).
let bins = findings.bin(
    &scores,
    &BinConfig { count: 10, strategy: BinStrategy::Quantile },
);
println!("Decile per subject: {:?}", bins);
# Ok::<(), Box<dyn std::error::Error>>(())
```

See `examples/basic.rs` for a runnable version (`cargo run --example basic`).

## Zero-aware binning

Quantile binning treats `score == 0.0` as a special bucket (bin `0`) and
ranks the non-zero subjects into `1..N`. This is a common need in matching:

- **Healthcare**: subjects who wrote 0 prescriptions in a window.
- **Marketing**: customers with 0 prior engagement.
- **A/B testing**: users with 0 actions in the pre-period.

Lumping these into the lowest decile distorts the ranking of *actual* writers.
This crate keeps them separate by default.

If you don't want the zero handling, use `BinStrategy::EqualRange` instead.

## What this is *not*

- Not a full ML library — no L1/L2 regularization, no multinomial, no
  cross-validation helpers. If you need those, look at `linfa-logistic`.
- Not the fastest solver for huge datasets — gradient descent with adaptive
  learning rate. Fine for thousands to ~hundreds of thousands of rows; for
  millions you'll want IRLS or L-BFGS.
- Not a matching engine — just the propensity-score input to one. The matching
  itself (caliper, 1:1, nearest neighbor, etc.) is downstream of this crate.

## Comparison

| Crate | Logistic regression | Quantile binning | Zero-aware | Deps |
|---|---|---|---|---|
| `propensity-score` | ✓ (GD) | ✓ | ✓ | 3 |
| `linfa-logistic`   | ✓ (L-BFGS) | — | — | many |
| `smartcore`        | ✓ | — | — | many |

If you're already using `linfa` or `smartcore` for other models, use their
logistic regression — there's no need to add this crate. This is for the
case where logistic regression + propensity binning is *all* you need.

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual-licensed as above, without any additional terms or
conditions.
