//! Vectra - A multi-dimensional array library for Rust
//!
//! This library provides numpy-like functionality for Rust, including:
//! - Multi-dimensional arrays with broadcasting
//! - Mathematical operations (trigonometric, logarithmic, exponential)
//! - Random number generation
//! - Linear algebra operations

#[cfg(feature = "blas")]
extern crate blas_src;

mod core;
mod math;
mod ops;
mod random;

#[cfg(test)]
mod tests;

pub mod prelude {
    pub use crate::core::Array;
    pub use crate::math::MatmulPolicy;
    pub use crate::math::matmul::Matmul;
}
