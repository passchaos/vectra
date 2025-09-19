use std::ops::{Add, Sub, Mul, Div};
use crate::core::Array;

impl<T> Array<T>
where
    T: Clone,
{
    /// Sum all elements
    pub fn sum(&self) -> T
    where
        T: Add<Output = T> + Default,
    {
        self.data.iter().fold(T::default(), |acc, x| acc + x.clone())
    }

    /// Calculate mean of all elements
    pub fn mean(&self) -> T
    where
        T: Add<Output = T> + Div<Output = T> + Default + Copy,
        f64: Into<T>,
    {
        let sum = self.sum();
        let len: f64 = self.data.len() as f64;
        sum / len.into()
    }

    /// Calculate mean of all elements (integer version)
    pub fn mean_int(&self) -> T
    where
        T: Add<Output = T> + Div<Output = T> + Default + From<usize>,
    {
        let sum = self.sum();
        let len = T::from(self.data.len());
        sum / len
    }

    /// Find maximum value
    pub fn max(&self) -> Option<T>
    where
        T: Clone + PartialOrd,
    {
        self.data.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).cloned()
    }

    /// Find minimum value
    pub fn min(&self) -> Option<T>
    where
        T: Clone + PartialOrd,
    {
        self.data.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).cloned()
    }

    /// Matrix multiplication (dot product)
    pub fn dot(&self, other: &Array<T>) -> Result<Array<T>, String>
    where
        T: Add<Output = T> + Mul<Output = T> + Default,
    {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err("Dot product only supported for 2D arrays".to_string());
        }
        if self.shape[1] != other.shape[0] {
            return Err("Matrix dimensions incompatible for multiplication".to_string());
        }
        
        let result_shape = vec![self.shape[0], other.shape[1]];
        let mut result_data = vec![T::default(); result_shape.iter().product()];
        
        for i in 0..self.shape[0] {
            for j in 0..other.shape[1] {
                let mut sum = T::default();
                for k in 0..self.shape[1] {
                    sum = sum + self.data[i * self.shape[1] + k].clone() * other.data[k * other.shape[1] + j].clone();
                }
                result_data[i * other.shape[1] + j] = sum;
            }
        }
        
        Array::from_vec(result_data, result_shape)
    }

    /// Apply function to each element
    pub fn map<F, U>(&self, f: F) -> Array<U>
    where
        F: Fn(&T) -> U,
        U: Clone + Default,
    {
        Array {
            data: self.data.iter().map(f).collect(),
            shape: self.shape.clone(),
            strides: Array::<U>::compute_strides_for_shape(&self.shape),
        }
    }

    /// Sum along specified axis
    pub fn sum_axis(&self, axis: usize) -> Result<Array<T>, String>
    where
        T: Add<Output = T> + Default,
    {
        if axis >= self.shape.len() {
            return Err("Axis out of bounds".to_string());
        }
        
        let mut result_shape = self.shape.clone();
        result_shape.remove(axis);
        
        if result_shape.is_empty() {
            // Sum over all dimensions, return scalar as 0-d array
            let total_sum = self.sum();
            return Array::from_vec(vec![total_sum], vec![]);
        }
        
        let result_size: usize = result_shape.iter().product();
        let mut result_data = vec![T::default(); result_size];
        
        // For each position in the result array
        for result_idx in 0..result_size {
            // Convert flat index to multi-dimensional index for result
            let mut result_indices = vec![0; result_shape.len()];
            let mut temp = result_idx;
            for i in (0..result_shape.len()).rev() {
                result_indices[i] = temp % result_shape[i];
                temp /= result_shape[i];
            }
            
            // Sum over the specified axis
            let mut sum = T::default();
            for axis_idx in 0..self.shape[axis] {
                // Construct full index for original array
                let mut full_indices = Vec::new();
                let mut result_pos = 0;
                for i in 0..self.shape.len() {
                    if i == axis {
                        full_indices.push(axis_idx);
                    } else {
                        full_indices.push(result_indices[result_pos]);
                        result_pos += 1;
                    }
                }
                
                let flat_idx = self.index_to_flat(&full_indices)?;
                sum = sum + self.data[flat_idx].clone();
            }
            result_data[result_idx] = sum;
        }
        
        Array::from_vec(result_data, result_shape)
    }

    // Trigonometric functions
    pub fn sin(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().sin()))
    }

    pub fn cos(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().cos()))
    }

    pub fn tan(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().tan()))
    }

    pub fn asin(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().asin()))
    }

    pub fn acos(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().acos()))
    }

    pub fn atan(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().atan()))
    }

    pub fn atan2(&self, other: &Array<T>) -> Result<Array<T>, String>
    where
        T: Into<f64> + From<f64> + Clone,
    {
        if self.shape != other.shape {
            return Err("Arrays must have the same shape for atan2".to_string());
        }
        
        let result_data: Vec<T> = self.data.iter()
            .zip(other.data.iter())
            .map(|(y, x)| {
                let y_f64: f64 = y.clone().into();
                let x_f64: f64 = x.clone().into();
                T::from(y_f64.atan2(x_f64))
            })
            .collect();
        
        Ok(Array {
            data: result_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    // Hyperbolic functions
    pub fn sinh(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().sinh()))
    }

    pub fn cosh(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().cosh()))
    }

    pub fn tanh(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().tanh()))
    }

    pub fn asinh(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().asinh()))
    }

    pub fn acosh(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().acosh()))
    }

    pub fn atanh(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().atanh()))
    }

    // Angle conversion
    pub fn to_radians(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().to_radians()))
    }

    pub fn to_degrees(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().to_degrees()))
    }

    // Logarithmic functions
    pub fn ln(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().ln()))
    }

    pub fn log10(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().log10()))
    }

    pub fn log2(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().log2()))
    }

    pub fn log(&self, base: f64) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().log(base)))
    }

    // Exponential functions
    pub fn exp(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().exp()))
    }

    pub fn exp2(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().exp2()))
    }

    pub fn exp_m1(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().exp_m1()))
    }

    pub fn ln_1p(&self) -> Array<T>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        self.map(|x| T::from(x.clone().into().ln_1p()))
    }
}