use crate::{NumExt, core::Array};
use std::fmt::Debug; // Sub currently unused
pub mod matmul;
pub mod stat;

use num_traits::{Float, Pow};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MatmulPolicy {
    Naive,
    Faer,
    #[cfg(feature = "blas")]
    Blas,
    LoopReorder,
    #[cfg(target_arch = "aarch64")]
    LoopRecorderSimd,
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

impl<const D: usize, T: NumExt + Float> Array<D, T> {
    unary_ops! {
        /// The largest integer less than or equal to each element.
        fn floor
        /// The smallest integer less than or equal to each element.
        fn ceil
        /// The nearest integer of each element.
        fn round
        /// The integer part of each element.
        fn trunc
        /// The fractional part of each element.
        fn fract
        /// Absolute of each element.
        fn abs
        /// Sign number of each element.
        ///
        /// + `1.0` for all positive numbers.
        /// + `-1.0` for all negative numbers.
        /// + `NaN` for all `NaN` (not a number).
        fn signum
        /// The reciprocal (inverse) of each element, `1/x`.
        fn recip
        /// Square root of each element.
        fn sqrt
        /// `e^x` of each element (exponential function).
        fn exp
        /// `2^x` of each element.
        fn exp2
        /// `e^x - 1` of each element.
        fn exp_m1
        /// Natural logarithm of each element.
        fn ln
        /// Base 2 logarithm of each element.
        fn log2
        /// Base 10 logarithm of each element.
        fn log10
        /// `ln(1 + x)` of each element.
        fn ln_1p
        /// Cubic root of each element.
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
        let result: Array<_, String> = arr.into_map(|x| format!("value_{}", x));
        assert_eq!(result[[0, 0]], "value_1");
        assert_eq!(result[[1, 1]], "value_4");
        assert_eq!(result.shape(), [2, 2]);
    }

    #[test]
    fn test_into_map_type_conversion() {
        let arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
        let result: Array<_, f64> = arr.into_map(|x| x as f64 * 2.5);
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
