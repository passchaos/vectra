use crate::{
    core::{Array, MajorOrder},
    utils::{dyn_dim_to_static, shape_indices_to_flat_idx},
};
use std::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, Div, Index},
}; // Sub currently unused
pub mod matmul;
use itertools::Itertools;
use num_traits::{Float, NumCast, Pow, float::TotalOrder};

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

impl<const D: usize, T> Array<D, T>
where
    T: Clone,
{
    /// Sum all elements
    pub fn sum(&self) -> T
    where
        T: Add<Output = T> + Default,
    {
        self.multi_iter()
            .fold(T::default(), |acc, (_, x)| acc + x.clone())
    }

    /// Sum along specified axis
    pub fn sum_axis(&self, axis: usize) -> Array<D, T>
    where
        T: Add<Output = T> + Default + Clone,
    {
        self.map_axis(axis, |values| {
            values
                .into_iter()
                .fold(T::default(), |acc, x| acc + x.clone())
        })
    }

    /// Calculate mean of all elements
    pub fn mean(&self) -> T
    where
        T: Add<Output = T> + Div<Output = T> + Default,
        u32: Into<T>,
    {
        let sum = self.sum();
        let size = self.size() as u32;
        sum / size.into()
    }

    pub fn mean_axis(&self, axis: usize) -> Array<D, T>
    where
        T: Default + Clone + NumCast + Sum + Div<Output = T>,
    {
        let axis_size = self.shape[axis];

        let axis_size = T::from(axis_size)
            .ok_or(format!(
                "cannot convert {} to type {}",
                std::any::type_name_of_val(&axis_size),
                std::any::type_name::<T>()
            ))
            .unwrap();

        self.map_axis(axis, |values| {
            let sum: T = values.into_iter().cloned().sum();
            sum / axis_size.clone()
        })
    }
}

pub trait TotalOrd {
    type Output;
    fn min(&self) -> Option<Self::Output>;
    fn max(&self) -> Option<Self::Output>;
}

macro_rules! impl_total_ord {
    ($($t:ty),*) => {
        $(
            impl<const D: usize> TotalOrd for Array<D, $t> {
                type Output = $t;
                fn min(&self) -> Option<Self::Output> {
                    self.multi_iter()
                        .min_by(|(_, a), (_, b)| a.cmp(b))
                        .map(|(_, x)| x.clone())
                }
                fn max(&self) -> Option<Self::Output> {
                    self.multi_iter()
                        .max_by(|(_, a), (_, b)| a.cmp(b))
                        .map(|(_, x)| x.clone())
                }
            }
        )*
    };
}

impl_total_ord!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
);

impl<const D: usize, T: TotalOrder + Clone> Array<D, T> {
    pub fn max_axis(&self, axis: usize) -> Self
    where
        T: Default,
    {
        self.map_axis(axis, |v| {
            v.into_iter()
                .max_by(|a, b| a.total_cmp(b))
                .cloned()
                .unwrap()
        })
    }

    pub fn min_axis(&self, axis: usize) -> Self
    where
        T: Default,
    {
        self.map_axis(axis, |v| {
            v.into_iter()
                .min_by(|a, b| a.total_cmp(b))
                .cloned()
                .unwrap()
        })
    }

    /// Find maximum value
    pub fn max(&self) -> Option<T> {
        self.multi_iter()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(_, x)| x.clone())
    }

    /// Find minimum value
    pub fn min(&self) -> Option<T> {
        self.multi_iter()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(_, x)| x.clone())
    }
}

impl<const D: usize, T> Array<D, T> {
    pub fn map<F, U>(&self, f: F) -> Array<D, U>
    where
        F: Fn(&T) -> U,
    {
        Array {
            data: self.data.iter().map(f).collect(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            major_order: self.major_order,
        }
    }

    pub fn map_axis<F, U>(&self, axis: usize, f: F) -> Array<D, U>
    where
        F: Fn(Vec<&T>) -> U,
        U: Default + Clone,
    {
        if axis >= D {
            panic!("Axis out of bounds");
        }

        let mut result_shape = self.shape();
        let axis_len = result_shape[axis];
        result_shape[axis] = 1;

        let result_size = result_shape.iter().product();
        let mut result_data = vec![U::default(); result_size];

        let major_order = MajorOrder::RowMajor;

        for idx in result_shape.iter().map(|&n| 0..n).multi_cartesian_product() {
            let idx = dyn_dim_to_static(&idx);

            let axis_values = (0..axis_len)
                .map(|i| {
                    let mut i_idx = idx.clone();
                    i_idx[axis] = i;

                    self.index(i_idx)
                })
                .collect();

            let value = f(axis_values);

            let flat_idx = shape_indices_to_flat_idx(result_shape, idx, major_order);
            result_data[flat_idx] = value;
        }

        Array::from_vec_major(result_data, result_shape, major_order)
    }

    /// Apply function to each element, consuming self and returning a new Array with different type
    pub fn into_map<F, U>(self, f: F) -> Array<D, U>
    where
        F: Fn(T) -> U,
    {
        let Self {
            data,
            shape,
            strides,
            major_order,
        } = self;

        Array {
            data: data.into_iter().map(f).collect(),
            shape,
            strides,
            major_order,
        }
    }
}

impl<const D: usize, T: Pow<T, Output = T> + Clone> Array<D, T> {
    pub fn pow(&self, exponent: T) -> Array<D, T> {
        self.map(|x| x.clone().pow(exponent.clone()))
    }
}

impl<const D: usize, T: Float> Array<D, T> {
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
    fn test_sum() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);

        let arr1 = arr.sum_axis(0);
        assert_eq!(
            arr1,
            Array::from_vec(vec![8, 10, 12, 14, 16, 18], [1, 3, 2])
        );
        let arr2 = arr.sum_axis(1);
        assert_eq!(arr2, Array::from_vec(vec![9, 12, 27, 30], [2, 1, 2]));
        let arr3 = arr.sum_axis(2);
        assert_eq!(arr3, Array::from_vec(vec![3, 7, 11, 15, 19, 23], [2, 3, 1]));

        let arr4 = arr.sum_axis(2);
        let arr4 = arr4.reshape([2, 3]);
        assert_eq!(arr4, Array::from_vec(vec![3, 7, 11, 15, 19, 23], [2, 3]));
        println!("arr= {arr:?} arr1= {arr1:?} arr2= {arr2:?} arr3= {arr3:?} arr4= {arr4:?}");
    }

    #[test]
    fn test_mean_axis() {
        let arr = Array::<_, f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let mean_axis_0 = arr.mean_axis(0);
        assert_eq!(mean_axis_0, Array::from_vec(vec![2.0, 3.0], [1, 2]));
        let mean_axis_1 = arr.mean_axis(1);
        assert_eq!(mean_axis_1, Array::from_vec(vec![1.5, 3.5], [2, 1]));
    }

    #[test]
    fn test_aggregations() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        assert_eq!(arr.sum(), 10.0);
        assert_eq!(arr.mean(), 2.5);
        assert_eq!(arr.max(), Some(4.0));
        assert_eq!(arr.min(), Some(1.0));
    }

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
