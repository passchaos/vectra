//! # Vectra - A multi-dimensional array library for Rust
//!
//! Vectra is a high-performance multi-dimensional array library for Rust, inspired by NumPy.
//! It provides efficient array operations, mathematical functions, and linear algebra capabilities
//! with a focus on performance and ergonomics.
//!
//! ## Features
//!
//! - **Multi-dimensional Arrays**: Support for N-dimensional arrays with flexible indexing
//! - **Broadcasting**: NumPy-style broadcasting for element-wise operations
//! - **Mathematical Functions**: Comprehensive set of mathematical operations including:
//!   - Trigonometric functions (sin, cos, tan, etc.)
//!   - Hyperbolic functions (sinh, cosh, tanh, etc.)
//!   - Logarithmic and exponential functions
//!   - Power and root functions
//! - **Linear Algebra**: Matrix multiplication with multiple backend options:
//!   - BLAS integration for high performance
//!   - Faer backend for pure Rust implementation
//!   - Custom optimized implementations
//! - **Random Number Generation**: Built-in support for random array creation
//! - **Memory Efficient**: Zero-copy operations where possible
//! - **Type Safety**: Compile-time dimension checking
//!
//! ## Quick Start
//!
//! ```rust
//! use vectra::prelude::*;
//!
//! // Create arrays
//! let zeros = Array::<_, f64>::zeros([2, 3]);
//! let ones = Array::<_, i32>::ones([3, 3]);
//! let eye = Array::<_, f32>::eye(3); // Identity matrix
//!
//! // Create from vector
//! let data = vec![1, 2, 3, 4, 5, 6];
//! let arr = Array::from_vec(data, [2, 3]);
//!
//! // Array operations
//! let reshaped = arr.reshape([3, 2]);
//! let transposed = arr.transpose();
//!
//! // Mathematical operations
//! let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
//! let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], [2, 2]);
//!
//! let sum = &a + &b;                    // Element-wise addition
//! let product = &a * &b;                // Element-wise multiplication
//! let dot_product = a.matmul(&b);       // Matrix multiplication
//!
//! // Mathematical functions
//! let angles = Array::from_vec(vec![0.0, std::f64::consts::PI/2.0], [2]);
//! let sines = angles.sin();
//! let exponentials = angles.exp();
//!
//! // Random arrays
//! let random_arr = Array::<_, f64>::random([3, 3]);
//! let normal_arr = Array::<_, f64>::randn([2, 4]);
//! ```
//!
//! ## Array Creation
//!
//! Vectra provides multiple ways to create arrays:
//!
//! ```rust
//! use vectra::prelude::*;
//!
//! // Create arrays filled with specific values
//! let zeros = Array::<_, f64>::zeros([2, 3]);
//! let ones = Array::<_, i32>::ones([3, 3]);
//! let filled = Array::full([2, 2], 42);
//!
//! // Create from existing data
//! let data = vec![1, 2, 3, 4, 5, 6];
//! let arr = Array::from_vec(data, [2, 3]);
//!
//! // Create ranges
//! let range1d = Array::arange(0, 10, 1);        // [0, 1, 2, ..., 9]
//! let range_count = Array::arange_c(0, 2, 5);   // 5 elements starting from 0 with step 2
//!
//! // Random arrays
//! let random = Array::<_, f64>::random([3, 3]);           // Uniform [0, 1)
//! let uniform = Array::uniform([2, 2], -1.0, 1.0);        // Uniform [-1, 1)
//! let normal = Array::<_, f64>::randn([2, 3]);             // Standard normal distribution
//!
//! // Identity matrix
//! let identity = Array::<_, f64>::eye(4);
//! ```
//!
//! ## Performance
//!
//! Vectra is designed for high performance with multiple optimization strategies:
//!
//! - **BLAS Integration**: Optional BLAS backend for optimized linear algebra operations
//! - **Faer Backend**: Pure Rust high-performance linear algebra
//! - **SIMD Optimizations**: Vectorized operations where supported
//! - **Memory Layout Control**: Support for both row-major and column-major layouts
//! - **Zero-copy Operations**: Efficient memory usage through view-based operations

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use std::{cmp::Ordering, fmt::Debug, iter::Sum};

use num_traits::{Num, NumAssign, NumCast};

#[cfg(feature = "blas")]
extern crate blas_src;

/// Extension trait for comparison operations that works with both ordered and floating-point types.
///
/// This trait provides a unified interface for comparison operations across different numeric types,
/// handling the special case of floating-point numbers which don't implement `Ord` due to NaN values.
pub trait CmpExt {
    /// Compare two values, returning their ordering.
    ///
    /// For integer types, this uses the standard `cmp` method.
    /// For floating-point types, this uses `total_cmp` to handle NaN values consistently.
    fn cmp_ext(&self, other: &Self) -> Ordering;
}

/// A comprehensive numeric trait that combines all necessary numeric operations for array elements.
///
/// This trait serves as a convenient bound that includes all the numeric traits needed for
/// array operations, including basic arithmetic, casting, assignment operations, comparison,
/// copying, debugging, summation, and default values.
///
/// # Automatically Implemented
///
/// This trait is automatically implemented for any type that satisfies all the constituent traits:
/// - `Num`: Basic arithmetic operations
/// - `NumCast`: Numeric type casting
/// - `NumAssign`: Assignment arithmetic operations
/// - `CmpExt`: Extended comparison operations
/// - `Copy`: Efficient copying
/// - `Debug`: Debug formatting
/// - `Sum`: Summation operations
/// - `Default`: Default value construction
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

/// Core array implementation and fundamental operations.
mod core;
/// Mathematical functions and linear algebra operations.
mod math;
/// Operator overloading for arrays (arithmetic, indexing, etc.).
mod ops;
/// Random number generation and random array creation.
mod random;
/// Array slicing and view operations.
mod slice;
/// Internal utility functions for array operations.
mod utils;

/// Prelude module that re-exports the most commonly used items.
///
/// This module provides a convenient way to import the essential types and traits
/// needed for most array operations. Import this module to get started quickly:
///
/// ```rust
/// use vectra::prelude::*;
///
/// let arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
/// let result = arr.matmul_with_policy(&arr, MatmulPolicy::default());
/// ```
///
/// ## Re-exported Items
///
/// - [`core::Array`]: The main multi-dimensional array type
/// - [`math::MatmulPolicy`]: Policy for choosing matrix multiplication backend
/// - [`math::matmul::Matmul`]: Trait providing matrix multiplication methods
pub mod prelude {
    /// The main multi-dimensional array type.
    pub use crate::core::Array;
    /// Policy enum for selecting matrix multiplication implementation.
    pub use crate::math::MatmulPolicy;
    /// Trait providing matrix multiplication functionality.
    pub use crate::math::matmul::Matmul;
    pub use crate::slice::{SliceArg, SliceArgKind};
}
