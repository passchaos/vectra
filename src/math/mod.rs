use crate::{NumExt, core::Array};
use std::fmt::Debug; // Sub currently unused
pub mod matmul;
pub mod ml;
pub mod stat;

use num_traits::{Float, Pow};

/// Matrix multiplication policy for controlling computation backend.
///
/// This enum allows you to choose different algorithms for matrix multiplication
/// based on performance requirements and available features.
///
/// # Variants
///
/// * `Naive` - Simple triple-loop implementation, good for small matrices
/// * `Faer` - Uses the faer library for optimized linear algebra operations
/// * `Blas` - Uses BLAS (Basic Linear Algebra Subprograms) when available
/// * `LoopReorder` - Optimized loop ordering for better cache performance
/// * `LoopRecorderSimd` - SIMD-optimized version (ARM64 only)
/// * `Blocking(usize)` - Block-based algorithm with specified block size
///
/// # Examples
///
/// ```rust
/// use vectra::math::MatmulPolicy;
/// use vectra::prelude::*;
///
/// let a = Array::ones([100, 50]);
/// let b = Array::ones([50, 80]);
///
/// // Use different policies for matrix multiplication
/// let result_faer = a.matmul_policy(&b, MatmulPolicy::Faer);
/// let result_naive = a.matmul_policy(&b, MatmulPolicy::Naive);
/// let result_blocking = a.matmul_policy(&b, MatmulPolicy::Blocking(32));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MatmulPolicy {
    /// Simple triple-loop matrix multiplication.
    ///
    /// Best for small matrices or when simplicity is preferred over performance.
    Naive,

    /// Use BLAS (Basic Linear Algebra Subprograms) for matrix multiplication.
    ///
    /// Typically provides the best performance when available. This is the
    /// default on macOS where Accelerate framework is used.
    #[cfg(feature = "blas")]
    Blas,

    /// Loop reordering optimization for better cache performance.
    ///
    /// Rearranges the computation order to improve memory access patterns.
    LoopReorder,

    /// SIMD-optimized loop reordering (ARM64 only).
    ///
    /// Combines loop reordering with SIMD instructions for maximum performance
    /// on ARM64 architectures.
    #[cfg(target_arch = "aarch64")]
    LoopRecorderSimd,

    /// Block-based matrix multiplication with specified block size.
    ///
    /// Divides the computation into blocks to improve cache locality.
    /// The parameter specifies the block size.
    Blocking(usize),
}

impl Default for MatmulPolicy {
    fn default() -> Self {
        #[cfg(target_os = "macos")]
        return Self::Blas;

        #[cfg(not(target_os = "macos"))]
        return Self::Faer;
    }
}

impl<const D: usize, T: NumExt> Array<D, T> {
    /// Raise each element to the power of the given exponent.
    ///
    /// This method applies the power operation element-wise to the array.
    /// The exponent can be of a different numeric type than the array elements.
    ///
    /// # Arguments
    ///
    /// * `exponent` - The exponent to raise each element to
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// // Integer powers
    /// let arr = Array::from_vec(vec![2, 3, 4], [3]);
    /// let squared = arr.pow(2); // [4, 9, 16]
    ///
    /// // Floating-point powers
    /// let arr_f = Array::from_vec(vec![4.0, 9.0, 16.0], [3]);
    /// let sqrt_result = arr_f.pow(0.5); // [2.0, 3.0, 4.0]
    ///
    /// // Mixed types
    /// let arr = Array::from_vec(vec![2.0, 3.0], [2]);
    /// let result = arr.pow(3); // [8.0, 27.0]
    /// ```
    pub fn pow<U, O>(&self, exponent: U) -> Array<D, O>
    where
        T: Pow<U, Output = O>,
        U: NumExt,
        O: NumExt,
    {
        self.map(|x| x.pow(exponent.clone()))
    }
}

macro_rules! unary_ops {
    ($($(#[$meta:meta])* fn $id:ident)+) => {
        $($(#[$meta])*
        #[must_use = "method returns a new array and does not mutate the original value"]
        pub fn $id(&self) -> Array<D, T> {
            self.mapv(T::$id)
        })+
    };
}

macro_rules! binary_ops {
    ($($(#[$meta:meta])* fn $id:ident($ty:ty))+) => {
        $($(#[$meta])*
        #[must_use = "method returns a new array and does not mutate the original value"]
        pub fn $id(&self, rhs: $ty) -> Array<D, T> {
            self.mapv(|v| T::$id(v, rhs))
        })+
    };
}

/// Mathematical functions for floating-point arrays.
///
/// This implementation provides element-wise mathematical operations
/// for arrays containing floating-point numbers.
impl<const D: usize, T: NumExt + Float> Array<D, T> {
    unary_ops! {
        /// The largest integer less than or equal to each element.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![1.7, 2.3, -1.2], [3]);
        /// let result = arr.floor(); // [1.0, 2.0, -2.0]
        /// ```
        fn floor

        /// The smallest integer greater than or equal to each element.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![1.2, 2.7, -1.8], [3]);
        /// let result = arr.ceil(); // [2.0, 3.0, -1.0]
        /// ```
        fn ceil

        /// The nearest integer to each element.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![1.4, 2.6, -1.5], [3]);
        /// let result = arr.round(); // [1.0, 3.0, -2.0]
        /// ```
        fn round

        /// The integer part of each element (truncate towards zero).
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![1.7, -2.3], [2]);
        /// let result = arr.trunc(); // [1.0, -2.0]
        /// ```
        fn trunc

        /// The fractional part of each element.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![1.7, -2.3], [2]);
        /// let result = arr.fract(); // [0.7, -0.3]
        /// ```
        fn fract

        /// Absolute value of each element.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![-1.5, 2.3, -4.1], [3]);
        /// let result = arr.abs(); // [1.5, 2.3, 4.1]
        /// ```
        fn abs

        /// Sign of each element.
        ///
        /// Returns:
        /// * `1.0` for positive numbers
        /// * `-1.0` for negative numbers
        /// * `NaN` for `NaN` values
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![-2.5, 0.0, 3.7], [3]);
        /// let result = arr.signum(); // [-1.0, 1.0, 1.0]
        /// ```
        fn signum

        /// Reciprocal (inverse) of each element: `1/x`.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![2.0, 4.0, 0.5], [3]);
        /// let result = arr.recip(); // [0.5, 0.25, 2.0]
        /// ```
        fn recip

        /// Square root of each element.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![4.0, 9.0, 16.0], [3]);
        /// let result = arr.sqrt(); // [2.0, 3.0, 4.0]
        /// ```
        fn sqrt

        /// Exponential function: `e^x` for each element.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![0.0, 1.0, 2.0], [3]);
        /// let result = arr.exp(); // [1.0, e, e²]
        /// ```
        fn exp

        /// Base-2 exponential: `2^x` for each element.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![1.0, 2.0, 3.0], [3]);
        /// let result = arr.exp2(); // [2.0, 4.0, 8.0]
        /// ```
        fn exp2

        /// Exponential minus one: `e^x - 1` for each element.
        ///
        /// More accurate than `exp(x) - 1` for small values of x.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![0.0, 0.1], [2]);
        /// let result = arr.exp_m1();
        /// ```
        fn exp_m1

        /// Natural logarithm of each element.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![1.0, std::f64::consts::E], [2]);
        /// let result = arr.ln(); // [0.0, 1.0]
        /// ```
        fn ln

        /// Base-2 logarithm of each element.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![2.0, 4.0, 8.0], [3]);
        /// let result = arr.log2(); // [1.0, 2.0, 3.0]
        /// ```
        fn log2

        /// Base-10 logarithm of each element.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![10.0, 100.0, 1000.0], [3]);
        /// let result = arr.log10(); // [1.0, 2.0, 3.0]
        /// ```
        fn log10

        /// Natural logarithm of (1 + x) for each element.
        ///
        /// More accurate than `ln(1 + x)` for small values of x.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![0.0, 0.1], [2]);
        /// let result = arr.ln_1p();
        /// ```
        fn ln_1p

        /// Cube root of each element.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use vectra::prelude::*;
        ///
        /// let arr = Array::from_vec(vec![8.0, 27.0, 64.0], [3]);
        /// let result = arr.cbrt(); // [2.0, 3.0, 4.0]
        /// ```
        fn cbrt
        /// Sine of each element (in radians).
        fn sin
        /// Cosine of each element (in radians).
        fn cos
        /// Tangent of each element (in radians).
        fn tan
        /// Arcsine of each element (return in radians).
        fn asin
        /// Arccosine of each element (return in radians).
        fn acos
        /// Arctangent of each element (return in radians).
        fn atan
        /// Hyperbolic sine of each element.
        fn sinh
        /// Hyperbolic cosine of each element.
        fn cosh
        /// Hyperbolic tangent of each element.
        fn tanh
        /// Inverse hyperbolic sine of each element.
        fn asinh
        /// Inverse hyperbolic cosine of each element.
        fn acosh
        /// Inverse hyperbolic tangent of each element.
        fn atanh
        /// Converts radians to degrees for each element.
        fn to_degrees
        /// Converts degrees to radians for each element.
        fn to_radians
    }
    binary_ops! {
        /// Integer power of each element.
        ///
        /// This function is generally faster than using float power.
        fn powi(i32)
        /// Float power of each element.
        fn powf(T)
        /// Logarithm of each element with respect to an arbitrary base.
        fn log(T)
        /// The positive difference between given number and each element.
        fn abs_sub(T)
        /// Length of the hypotenuse of a right-angle triangle of each element
        fn hypot(T)
    }

    pub fn pow2(&self) -> Array<D, T> {
        self.mapv(|v| v * v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let squared = arr.map(|x| x * x);
        assert_eq!(squared[[0, 0]], 1.0);
        assert_eq!(squared[[1, 1]], 16.0);
    }

    #[test]
    fn test_map_type_conversion() {
        let arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
        let result: Array<_, f64> = arr.map(|&x| x as f64 * 1.5);
        assert_eq!(result[[0, 0]], 1.5);
        assert_eq!(result[[1, 1]], 6.0);
    }

    #[test]
    fn test_into_map() {
        let arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
        let result: Array<_, String> = arr.map_into(|x| format!("value_{}", x));
        assert_eq!(result[[0, 0]], "value_1");
        assert_eq!(result[[1, 1]], "value_4");
        assert_eq!(result.shape(), [2, 2]);
    }

    #[test]
    fn test_into_map_type_conversion() {
        let arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
        let result: Array<_, f64> = arr.map_into(|x| x as f64 * 2.5);
        assert_eq!(result[[0, 0]], 2.5);
        assert_eq!(result[[1, 1]], 10.0);
        assert_eq!(result.shape(), [2, 2]);
    }

    #[test]
    fn test_trigonometric_functions() {
        let arr = Array::from_vec(
            vec![
                0.0,
                std::f64::consts::PI / 6.0,
                std::f64::consts::PI / 4.0,
                std::f64::consts::PI / 3.0,
                std::f64::consts::PI / 2.0,
            ],
            [5],
        );

        let sin_result = arr.sin();
        assert!((sin_result[[0]] - 0.0).abs() < 1e-10);
        assert!((sin_result[[1]] - 0.5).abs() < 1e-10);
        assert!((sin_result[[4]] - 1.0).abs() < 1e-10);

        let cos_result = arr.cos();
        assert!((cos_result[[0]] - 1.0).abs() < 1e-10);
        assert!((cos_result[[4]] - 0.0).abs() < 1e-10);

        let _tan_result = arr.tan();

        // Test inverse functions
        let values = Array::from_vec(vec![0.0, 0.5, 1.0], [3]);
        let asin_result = values.asin();
        let acos_result = values.acos();
        let atan_result = values.atan();

        assert!((asin_result[[0]] - 0.0f64).abs() < 1e-10f64);
        assert!((asin_result[[2]] - std::f64::consts::PI / 2.0).abs() < 1e-10);
        assert!((acos_result[[0]] - std::f64::consts::PI / 2.0).abs() < 1e-10);
        assert!((acos_result[[2]] - 0.0).abs() < 1e-10);
        assert!((atan_result[[0]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic_functions() {
        let arr = Array::from_vec(vec![0.0, 1.0, -1.0], [3]);

        let sinh_result = arr.sinh();
        let cosh_result = arr.cosh();
        let tanh_result = arr.tanh();

        assert!((sinh_result[[0]] - 0.0f64).abs() < 1e-10f64);
        assert!((cosh_result[[0]] - 1.0f64).abs() < 1e-10f64);
        assert!((tanh_result[[0]] - 0.0f64).abs() < 1e-10f64);
    }

    #[test]
    fn test_logarithmic_functions() {
        let arr = Array::from_vec(vec![1.0, std::f64::consts::E, 10.0, 2.0], [4]);

        // Test natural logarithm
        let ln_result = arr.ln();
        assert!((ln_result[[0]] - 0.0).abs() < 1e-10); // ln(1) = 0
        assert!((ln_result[[1]] - 1.0).abs() < 1e-10); // ln(e) = 1

        // Test base-10 logarithm
        let log10_result = arr.log10();
        assert!((log10_result[[0]] - 0.0).abs() < 1e-10); // log10(1) = 0
        assert!((log10_result[[2]] - 1.0).abs() < 1e-10); // log10(10) = 1

        // Test base-2 logarithm
        let log2_result = arr.log2();
        assert!((log2_result[[0]] - 0.0).abs() < 1e-10); // log2(1) = 0
        assert!((log2_result[[3]] - 1.0).abs() < 1e-10); // log2(2) = 1

        // Test custom base logarithm
        let values = Array::from_vec(vec![1.0, 3.0, 9.0, 27.0], [4]);
        let log3_result = values.log(3.0);
        assert!((log3_result[[0]] - 0.0f64).abs() < 1e-10f64); // log3(1) = 0
        assert!((log3_result[[1]] - 1.0f64).abs() < 1e-10f64); // log3(3) = 1
        assert!((log3_result[[2]] - 2.0f64).abs() < 1e-10f64); // log3(9) = 2
        assert!((log3_result[[3]] - 3.0f64).abs() < 1e-10f64); // log3(27) = 3

        // Test that exp and ln are inverse operations
        let test_values = Array::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let exp_ln = test_values.clone().ln().exp();
        for i in 0..3 {
            assert!(((exp_ln[[i]] - test_values[[i]]) as f64).abs() < 1e-10);
        }
    }

    #[test]
    fn test_exponential_functions() {
        let arr = Array::from_vec(vec![0.0, 1.0, 2.0], [3]);

        // Test exponential function
        let exp_result = arr.exp();
        assert!((exp_result[[0]] - 1.0f64).abs() < 1e-10f64); // exp(0) = 1
        assert!((exp_result[[1]] - std::f64::consts::E).abs() < 1e-10); // exp(1) = e

        // Test base-2 exponential
        let exp2_result = arr.exp2();
        assert!((exp2_result[[0]] - 1.0).abs() < 1e-10); // 2^0 = 1
        assert!((exp2_result[[1]] - 2.0).abs() < 1e-10); // 2^1 = 2
        assert!((exp2_result[[2]] - 4.0).abs() < 1e-10); // 2^2 = 4

        // Test exp_m1 and ln_1p (inverse operations)
        let small_values = Array::from_vec(vec![0.1, 0.01, 0.001], [3]);
        let exp_m1_result = small_values.clone().exp_m1();
        let ln_1p_result = exp_m1_result.ln_1p();

        for i in 0..3 {
            assert!(((ln_1p_result[[i]] - small_values[[i]]) as f64).abs() < 1e-10);
        }

        // Test that exp_m1(0) = 0 and ln_1p(0) = 0
        let zero_arr = Array::from_vec(vec![0.0], [1]);
        assert!((zero_arr.clone().exp_m1()[[0]] - 0.0f64).abs() < 1e-10f64);
        assert!((zero_arr.ln_1p()[[0]] - 0.0f64).abs() < 1e-10f64);
    }
}
