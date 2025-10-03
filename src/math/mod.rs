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

impl<const D: usize, T: NumExt + Float> Array<D, T> {
    // Trigonometric functions
    pub fn sin(&self) -> Array<D, T> {
        self.map(|x| x.sin())
    }

    pub fn cos(&self) -> Array<D, T> {
        self.map(|x| x.cos())
    }

    pub fn tan(&self) -> Array<D, T> {
        self.map(|x| x.tan())
    }

    pub fn asin(&self) -> Array<D, T> {
        self.map(|x| x.asin())
    }

    pub fn acos(&self) -> Array<D, T> {
        self.map(|x| x.acos())
    }

    pub fn atan(&self) -> Array<D, T> {
        self.map(|x| x.atan())
    }

    // Hyperbolic functions
    pub fn sinh(&self) -> Array<D, T> {
        self.map(|x| x.sinh())
    }

    pub fn cosh(&self) -> Array<D, T> {
        self.map(|x| x.cosh())
    }

    pub fn tanh(&self) -> Array<D, T> {
        self.map(|x| x.tanh())
    }

    pub fn sqrt(&self) -> Array<D, T> {
        self.map(|x| x.sqrt())
    }

    pub fn pow2(&self) -> Array<D, T> {
        self.map(|x| x.powi(2))
    }

    pub fn powi(&self, exponent: i32) -> Array<D, T> {
        self.map(|x| x.powi(exponent))
    }

    // Logarithmic functions
    pub fn ln(&self) -> Array<D, T> {
        self.map(|x| x.ln())
    }

    pub fn log10(&self) -> Array<D, T> {
        self.map(|x| x.log10())
    }

    pub fn log2(&self) -> Array<D, T> {
        self.map(|x| x.log2())
    }

    pub fn log(&self, base: T) -> Array<D, T> {
        self.map(|x| x.log(base))
    }

    // Exponential functions
    pub fn exp(&self) -> Array<D, T> {
        self.map(|x| x.exp())
    }

    pub fn exp2(&self) -> Array<D, T> {
        self.map(|x| x.exp2())
    }

    pub fn exp_m1(&self) -> Array<D, T> {
        self.map(|x| x.exp_m1())
    }

    pub fn ln_1p(&self) -> Array<D, T> {
        self.map(|x| x.ln_1p())
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
