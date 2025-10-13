//! Vectra - A multi-dimensional array library for Rust
//!
//! This library provides numpy-like functionality for Rust, including:
//! - Multi-dimensional arrays with broadcasting
//! - Mathematical operations (trigonometric, logarithmic, exponential)
//! - Random number generation
//! - Linear algebra operations

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use std::{cmp::Ordering, fmt::Debug, iter::Sum};

use num_traits::{Num, NumAssign, NumCast};

#[cfg(feature = "blas")]
extern crate blas_src;

pub trait CmpExt {
    fn cmp_ext(&self, other: &Self) -> Ordering;
}

pub trait NumExt: Num + NumCast + NumAssign + CmpExt + Copy + Debug + Sum + Default {}
impl<T> NumExt for T where T: Num + NumCast + NumAssign + CmpExt + Copy + Debug + Sum + Default {}

macro_rules! impl_cmp_for_ord {
    ($($ty:ty),*) => {
        $(
            impl CmpExt for $ty {
                fn cmp_ext(&self, other: &Self) -> Ordering {
                    self.cmp(other)
                }
            }
        )*
    };
}

macro_rules! impl_cmp_for_total_cmp {
    ($($ty:ty),*) => {
        $(
            impl CmpExt for $ty {
                fn cmp_ext(&self, other: &Self) -> Ordering {
                    self.total_cmp(other)
                }
            }
        )*
    };
}

impl_cmp_for_ord!(
    i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize
);
impl_cmp_for_total_cmp!(f32, f64);

mod core;
mod math;
mod ops;
mod random;
mod utils;

pub mod prelude {
    pub use crate::core::Array;
    pub use crate::math::MatmulPolicy;
    pub use crate::math::matmul::Matmul;
}
