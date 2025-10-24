//! Statistical summary operations for arrays.
//!
//! This module provides functions for computing statistical summaries
//! such as sum, mean, variance, and standard deviation.

use num_traits::Float;

use crate::{NumExt, core::Array};

impl<const D: usize, T: NumExt> Array<D, T> {
    /// Sum all elements in the array.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use vectra::prelude::*;
    /// 
    /// let arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
    /// let sum = arr.sum();
    /// assert_eq!(sum, 10);
    /// ```
    pub fn sum(&self) -> T {
        self.multi_iter()
            .fold(T::zero(), |acc, (_, x)| acc + x.clone())
    }

    /// Sum along the specified axis.
    /// 
    /// # Arguments
    /// 
    /// * `axis` - The axis along which to compute the sum
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use vectra::prelude::*;
    /// 
    /// let arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
    /// let sum_axis0 = arr.sum_axis(0); // Sum along rows
    /// ```
    pub fn sum_axis(&self, axis: isize) -> Array<D, T> {
        self.map_axis(axis, |values| {
            values.into_iter().fold(T::zero(), |acc, x| acc + x.clone())
        })
    }
}

impl<const D: usize, T: NumExt> Array<D, T> {
    /// Calculate the mean of all elements in the array.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use vectra::prelude::*;
    /// 
    /// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    /// let mean: f64 = arr.mean();
    /// assert_eq!(mean, 2.5);
    /// ```
    pub fn mean<U>(&self) -> U
    where
        U: NumExt,
    {
        let sum = self.sum();
        let size = self.size();

        U::from(sum).unwrap() / U::from(size).unwrap()
    }

    /// Calculate the mean along the specified axis.
    /// 
    /// # Arguments
    /// 
    /// * `axis` - The axis along which to compute the mean
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use vectra::prelude::*;
    /// 
    /// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    /// let mean_axis0: Array<2, f64> = arr.mean_axis(0);
    /// ```
    pub fn mean_axis<U>(&self, axis: isize) -> Array<D, U>
    where
        U: NumExt,
    {
        self.map_axis(axis, |values| {
            let len = values.len();
            let sum: T = values.into_iter().cloned().sum();

            U::from(sum).unwrap() / U::from(len).unwrap()
        })
    }
}

impl<const D: usize, T: Float + NumExt> Array<D, T> {
    /// Calculate the variance of all elements in the array.
    /// 
    /// # Arguments
    /// 
    /// * `ddof` - Delta degrees of freedom (divisor is N - ddof)
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use vectra::prelude::*;
    /// 
    /// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    /// let variance = arr.var(0.0);
    /// ```
    pub fn var(&self, ddof: T) -> T {
        let n = T::from(self.size()).unwrap();
        let dof = n - ddof;

        let mut mean = T::zero();
        let mut sum_sq = T::zero();

        let mut i = 0;
        self.multi_iter().for_each(|(_idx, value)| {
            let count = T::from(i + 1).unwrap();

            let value = value.clone();
            let delta = value - mean;
            mean += delta / count;
            sum_sq += (value - mean) * delta;
            i += 1;
        });

        sum_sq / dof
    }

    /// Calculate the standard deviation of all elements in the array.
    /// 
    /// # Arguments
    /// 
    /// * `ddof` - Delta degrees of freedom (divisor is N - ddof)
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use vectra::prelude::*;
    /// 
    /// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    /// let std_dev = arr.std(0.0);
    /// ```
    pub fn std(&self, ddof: T) -> T {
        self.var(ddof).sqrt()
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

        let mean_axis_2 = arr.mean_axis(-1);
        assert_eq!(mean_axis_1, mean_axis_2);
    }

    #[test]
    fn test_aggregations() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        assert_eq!(arr.sum(), 10.0);
        assert_eq!(arr.mean::<f64>(), 2.5);
    }

    #[test]
    fn test_variance() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let var = arr.var(0.0);
        assert_eq!(var, 1.25);
    }

    // #[test]
    // fn test_variance_axis() {
    //     let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    //     let var_axis_0 = arr.variance_axis(0);
    //     assert_eq!(var_axis_0, Array::from_vec(vec![0.25, 0.25], [1, 2]));
    //     let var_axis_1 = arr.variance_axis(1);
    //     assert_eq!(var_axis_1, Array::from_vec(vec![0.25, 0.25], [2, 1]));

    //     let var_axis_2 = arr.variance_axis(-1);
    //     assert_eq!(var_axis_1, var_axis_2);
    // }
}
