# Changelog

All notable changes to this crate are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-05-11

Initial release.

- `propensity_score::fit` — logistic regression via gradient descent with
  adaptive convergence on cost-delta.
- `Findings::predict` — propensity scores in `[0, 1]`.
- `Findings::bin` — quantile (zero-aware) or equal-range binning of scores.
- `Findings::auc` — area under the ROC curve for sanity checks.
- Configurable via `Config { max_iterations, tolerance, learning_rate }` and
  `BinConfig { count, strategy }`.
- Dependencies: `nalgebra`, `thiserror`, `tracing`.

[Unreleased]: https://github.com/EdmundsEcho/propensity-score/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/EdmundsEcho/propensity-score/releases/tag/v0.1.0
