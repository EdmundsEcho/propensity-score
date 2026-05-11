#![doc = include_str!("../README.md")]
#![deny(missing_docs)]

pub mod binning;
pub mod config;
pub mod logistic;

pub use config::{BinConfig, BinStrategy, Config};
pub use logistic::{fit, Findings};
