use crate::core::Array;
use std::{
    fmt::Debug,
    ops::{Add, Div, Mul},
}; // Sub currently unused
pub mod matmul;
use num_traits::Float;

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
        Self::Naive
    }
}

impl<T> Array<T>
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
    pub fn sum_axis(&self, axis: usize) -> Result<Array<T>, String>
    where
        T: Add<Output = T> + Default,
    {
        if axis >= self.shape.len() {
            return Err("Axis out of bounds".to_string());
        }

        // let shape = self.shape.clone();
        // let axis_size = shape[axis];
        // let mut result_shape = shape;
        // result_shape[axis] = 1;

        // let result_size = result_shape.iter().product();
        // let mut result_data = vec![T::default(); result_size];
        // for (idx, value) in self.multi_iter() {
        //     let mut result_idx = idx[axis];
        //     for i in 0..axis {
        //         result_idx += idx[i] * shape[i];
        //     }
        //     result_data[result_idx] += value;
        // }

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

    // pub fn mean_axis(&self, axis: usize) -> Result<Array<T>, String>
    // where
    //     T: Add<Output = T> + Div<Output = T> + Default,
    //     u32: Into<T>,
    // {
    //     let shape = self.shape.clone();
    //     let axis_size = shape[axis];
    //     let mut result_shape = shape;
    //     result_shape[axis] = 1;

    //     let mut result_data = vec![T::default(); result_shape.size()];
    //     let mut result_idx = 0;

    //     for (indices, value) in self.multi_iter() {
    //         let mut full_indices = indices.clone();
    //         full_indices[axis] = 0;

    //         let flat_idx = self.index_to_flat(&full_indices)?;
    //         result_data[result_idx] = result_data[result_idx] + value.clone();
    //     }

    //     Array::from_vec(result_data, result_shape)
    // }

    /// Find maximum value
    pub fn max(&self) -> Option<T>
    where
        T: Clone + PartialOrd,
    {
        self.multi_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, x)| x.clone())
    }

    /// Find minimum value
    pub fn min(&self) -> Option<T>
    where
        T: Clone + PartialOrd,
    {
        self.multi_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, x)| x.clone())
    }

    /// Vector dot product (returns scalar)
    pub fn dot(&self, other: &Array<T>) -> Result<T, String>
    where
        T: Add<Output = T> + Mul<Output = T> + Default,
    {
        // Check if both arrays are 1D vectors
        if self.shape.len() != 1 || other.shape.len() != 1 {
            return Err("Dot product only supported for 1D vectors".to_string());
        }

        // Check if vectors have the same length
        if self.shape[0] != other.shape[0] {
            return Err("Vectors must have the same length for dot product".to_string());
        }

        let mut result = T::default();
        for i in 0..self.shape[0] {
            let self_val = &self.data[i * self.strides[0]];
            let other_val = &other.data[i * other.strides[0]];
            result = result + (self_val.clone() * other_val.clone());
        }

        Ok(result)
    }

    pub fn map<F, U>(&self, f: F) -> Array<U>
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

    /// Apply function to each element, consuming self and returning a new Array with different type
    pub fn into_map<F, U>(self, f: F) -> Array<U>
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

impl<T: Float> Array<T> {
    // Trigonometric functions
    pub fn sin(&self) -> Array<T> {
        self.map(|x| x.sin())
    }

    pub fn cos(&self) -> Array<T> {
        self.map(|x| x.cos())
    }

    pub fn tan(&self) -> Array<T> {
        self.map(|x| x.tan())
    }

    pub fn asin(&self) -> Array<T> {
        self.map(|x| x.asin())
    }

    pub fn acos(&self) -> Array<T> {
        self.map(|x| x.acos())
    }

    pub fn atan(&self) -> Array<T> {
        self.map(|x| x.atan())
    }

    // Hyperbolic functions
    pub fn sinh(&self) -> Array<T> {
        self.map(|x| x.sinh())
    }

    pub fn cosh(&self) -> Array<T> {
        self.map(|x| x.cosh())
    }

    pub fn tanh(&self) -> Array<T> {
        self.map(|x| x.tanh())
    }

    // Logarithmic functions
    pub fn ln(&self) -> Array<T> {
        self.map(|x| x.ln())
    }

    pub fn log10(&self) -> Array<T> {
        self.map(|x| x.log10())
    }

    pub fn log2(&self) -> Array<T> {
        self.map(|x| x.log2())
    }

    pub fn log(&self, base: T) -> Array<T> {
        self.map(|x| x.log(base))
    }

    // Exponential functions
    pub fn exp(&self) -> Array<T> {
        self.map(|x| x.exp())
    }

    pub fn exp2(&self) -> Array<T> {
        self.map(|x| x.exp2())
    }

    pub fn exp_m1(&self) -> Array<T> {
        self.map(|x| x.exp_m1())
    }

    pub fn ln_1p(&self) -> Array<T> {
        self.map(|x| x.ln_1p())
    }
}

impl<T> Array<T>
where
    T: Clone,
{
    /// Singular Value Decomposition (SVD)
    /// Returns (U, singular_values, V_transpose) where A = U * Σ * V^T
    pub fn svd(&self) -> Result<(Array<T>, Array<T>, Array<T>), String>
    where
        T: Into<f64> + From<f64> + Clone + Default + PartialOrd,
    {
        if self.ndim() != 2 {
            return Err("SVD requires a 2D matrix".to_string());
        }

        let m = self.shape[0];
        let n = self.shape[1];

        // Convert to f64 for computation
        let mut a: Vec<Vec<f64>> = Vec::new();
        for i in 0..m {
            let mut row = Vec::new();
            for j in 0..n {
                let idx = i * self.strides[0] + j * self.strides[1];
                row.push(self.data[idx].clone().into());
            }
            a.push(row);
        }

        // Use Jacobi SVD algorithm for small matrices, or power iteration for larger ones
        if m <= 10 && n <= 10 {
            self.jacobi_svd(a)
        } else {
            self.power_iteration_svd(a)
        }
    }

    /// Truncated Singular Value Decomposition (SVD)
    /// Computes only the top k singular values and vectors for faster computation
    /// Returns (U_k, singular_values_k, V_transpose_k) where A ≈ U_k * Σ_k * V_k^T
    pub fn svd_truncated(&self, k: usize) -> Result<(Array<T>, Array<T>, Array<T>), String>
    where
        T: Into<f64> + From<f64> + Clone + Default + PartialOrd,
    {
        if self.ndim() != 2 {
            return Err("SVD requires a 2D matrix".to_string());
        }

        let m = self.shape[0];
        let n = self.shape[1];
        let min_dim = m.min(n);

        if k == 0 {
            return Err("k must be greater than 0".to_string());
        }

        if k > min_dim {
            return Err(format!(
                "k ({}) cannot be larger than min(m, n) ({})",
                k, min_dim
            ));
        }

        // Convert to f64 for computation
        let mut a: Vec<Vec<f64>> = Vec::new();
        for i in 0..m {
            let mut row = Vec::new();
            for j in 0..n {
                let idx = i * self.strides[0] + j * self.strides[1];
                row.push(self.data[idx].clone().into());
            }
            a.push(row);
        }

        self.truncated_power_iteration_svd(a, k)
    }

    /// Jacobi SVD algorithm for small matrices
    fn jacobi_svd(&self, mut a: Vec<Vec<f64>>) -> Result<(Array<T>, Array<T>, Array<T>), String>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        let m = a.len();
        let n = a[0].len();
        let min_dim = m.min(n);

        // Initialize U as identity matrix
        let mut u = vec![vec![0.0; m]; m];
        for i in 0..m {
            u[i][i] = 1.0;
        }

        // Initialize V as identity matrix
        let mut v = vec![vec![0.0; n]; n];
        for i in 0..n {
            v[i][i] = 1.0;
        }

        // Jacobi iterations
        let max_iterations = 100;
        let tolerance = 1e-10;

        for _ in 0..max_iterations {
            let mut converged = true;

            // Sweep through all pairs of columns
            for p in 0..n {
                for q in (p + 1)..n {
                    // Compute 2x2 submatrix A^T * A
                    let mut app = 0.0;
                    let mut aqq = 0.0;
                    let mut apq = 0.0;

                    for i in 0..m {
                        app += a[i][p] * a[i][p];
                        aqq += a[i][q] * a[i][q];
                        apq += a[i][p] * a[i][q];
                    }

                    if apq.abs() > tolerance {
                        converged = false;

                        // Compute rotation angle
                        let tau = (aqq - app) / (2.0 * apq);
                        let t = if tau >= 0.0 {
                            1.0 / (tau + (1.0 + tau * tau).sqrt())
                        } else {
                            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                        };

                        let c = 1.0 / (1.0 + t * t).sqrt();
                        let s = t * c;

                        // Apply rotation to A
                        for i in 0..m {
                            let temp = a[i][p];
                            a[i][p] = c * temp - s * a[i][q];
                            a[i][q] = s * temp + c * a[i][q];
                        }

                        // Apply rotation to V
                        for i in 0..n {
                            let temp = v[i][p];
                            v[i][p] = c * temp - s * v[i][q];
                            v[i][q] = s * temp + c * v[i][q];
                        }
                    }
                }
            }

            if converged {
                break;
            }
        }

        // Extract singular values and sort
        let mut singular_values = Vec::new();
        let mut indices = Vec::new();

        for j in 0..min_dim {
            let mut norm = 0.0;
            for i in 0..m {
                norm += a[i][j] * a[i][j];
            }
            singular_values.push(norm.sqrt());
            indices.push(j);
        }

        // Sort by singular values (descending)
        indices.sort_by(|&i, &j| singular_values[j].partial_cmp(&singular_values[i]).unwrap());

        // Reorder and normalize columns of A to get U
        for j in 0..min_dim {
            let orig_j = indices[j];
            let sigma = singular_values[orig_j];

            if sigma > tolerance {
                for i in 0..m {
                    u[i][j] = a[i][orig_j] / sigma;
                }
            }
        }

        // Create result arrays
        let u_data: Vec<T> = u.into_iter().flatten().map(|x| T::from(x)).collect();
        let u_array = Array::from_vec(u_data, vec![m, m]).unwrap();

        let mut sorted_singular_values = Vec::new();
        for &i in &indices {
            sorted_singular_values.push(singular_values[i]);
        }
        let s_data: Vec<T> = sorted_singular_values
            .into_iter()
            .map(|x| T::from(x))
            .collect();
        let s_array = Array::from_vec(s_data, vec![min_dim]).unwrap();

        // Reorder V columns
        let mut v_reordered = vec![vec![0.0; n]; n];
        for j in 0..min_dim {
            let orig_j = indices[j];
            for i in 0..n {
                v_reordered[i][j] = v[i][orig_j];
            }
        }

        let vt_data: Vec<T> = v_reordered
            .into_iter()
            .flatten()
            .map(|x| T::from(x))
            .collect();
        let mut vt_array = Array::from_vec(vt_data, vec![n, n]).unwrap();
        vt_array.transpose().unwrap();

        Ok((u_array, s_array, vt_array))
    }

    /// Truncated power iteration SVD for computing top k singular values/vectors
    fn truncated_power_iteration_svd(
        &self,
        a: Vec<Vec<f64>>,
        k: usize,
    ) -> Result<(Array<T>, Array<T>, Array<T>), String>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        let m = a.len();
        let n = a[0].len();
        let max_iter = 100;
        let tolerance = 1e-10;

        let mut u_vectors = Vec::new();
        let mut singular_values = Vec::new();
        let mut v_vectors = Vec::new();

        // Create a copy of the matrix for deflation
        let mut a_deflated = a.clone();

        for _ in 0..k {
            // Power iteration to find the largest singular value and vectors
            let mut v = vec![1.0; n];
            let mut norm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
            for x in &mut v {
                *x /= norm;
            }

            let mut prev_sigma = 0.0;
            let mut sigma = 0.0;

            for _ in 0..max_iter {
                // v = A^T * (A * v)
                let mut av = vec![0.0; m];
                for i in 0..m {
                    for j in 0..n {
                        av[i] += a_deflated[i][j] * v[j];
                    }
                }

                let mut atav = vec![0.0; n];
                for j in 0..n {
                    for i in 0..m {
                        atav[j] += a_deflated[i][j] * av[i];
                    }
                }

                // Normalize
                norm = (atav.iter().map(|x| x * x).sum::<f64>()).sqrt();
                if norm < tolerance {
                    break;
                }

                for x in &mut atav {
                    *x /= norm;
                }
                v = atav;

                // Compute singular value: sigma = ||A * v||
                let mut av = vec![0.0; m];
                for i in 0..m {
                    for j in 0..n {
                        av[i] += a_deflated[i][j] * v[j];
                    }
                }
                sigma = (av.iter().map(|x| x * x).sum::<f64>()).sqrt();

                // Check convergence
                if (sigma - prev_sigma).abs() < tolerance {
                    break;
                }
                prev_sigma = sigma;
            }

            if sigma < tolerance {
                break;
            }

            // Compute u = A * v / sigma
            let mut u = vec![0.0; m];
            for i in 0..m {
                for j in 0..n {
                    u[i] += a_deflated[i][j] * v[j];
                }
                u[i] /= sigma;
            }

            // Store the singular triplet
            u_vectors.push(u.clone());
            singular_values.push(sigma);
            v_vectors.push(v.clone());

            // Deflate the matrix: A = A - sigma * u * v^T
            for i in 0..m {
                for j in 0..n {
                    a_deflated[i][j] -= sigma * u[i] * v[j];
                }
            }
        }

        // Construct result matrices
        let actual_k = u_vectors.len();

        // U matrix (m x k)
        let mut u_data = Vec::new();
        for i in 0..m {
            for j in 0..actual_k {
                u_data.push(T::from(u_vectors[j][i]));
            }
        }
        let u_array = Array::from_vec(u_data, vec![m, actual_k]).unwrap();

        // Singular values vector (k,)
        let s_data: Vec<T> = singular_values.into_iter().map(|x| T::from(x)).collect();
        let s_array = Array::from_vec(s_data, vec![actual_k]).unwrap();

        // V^T matrix (k x n)
        let mut vt_data = Vec::new();
        for i in 0..actual_k {
            for j in 0..n {
                vt_data.push(T::from(v_vectors[i][j]));
            }
        }
        let vt_array = Array::from_vec(vt_data, vec![actual_k, n]).unwrap();

        Ok((u_array, s_array, vt_array))
    }

    /// Power iteration SVD for larger matrices (simplified version)
    fn power_iteration_svd(
        &self,
        a: Vec<Vec<f64>>,
    ) -> Result<(Array<T>, Array<T>, Array<T>), String>
    where
        T: Into<f64> + From<f64> + Clone + Default,
    {
        let _m = a.len();
        let _n = a[0].len();

        // For now, fall back to Jacobi for simplicity
        // In a full implementation, this would use power iteration or other methods
        self.jacobi_svd(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let arr =
            Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], vec![2, 3, 2]).unwrap();

        let arr1 = arr.sum_axis(0).unwrap();
        let arr2 = arr.sum_axis(1).unwrap();
        let arr3 = arr.sum_axis(2).unwrap();
        println!("arr= {arr:?} arr1= {arr1:?} arr2= {arr2:?} arr3= {arr3:?}");
    }

    #[test]
    fn test_aggregations() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(arr.sum(), 10.0);
        assert_eq!(arr.mean(), 2.5);
        assert_eq!(arr.max(), Some(4.0));
        assert_eq!(arr.min(), Some(1.0));
    }

    #[test]
    fn test_map() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let squared = arr.map(|x| x * x);
        assert_eq!(squared[[0, 0]], 1.0);
        assert_eq!(squared[[1, 1]], 16.0);
    }

    #[test]
    fn test_map_type_conversion() {
        let arr = Array::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let result: Array<f64> = arr.map(|&x| x as f64 * 1.5);
        assert_eq!(result[[0, 0]], 1.5);
        assert_eq!(result[[1, 1]], 6.0);
    }

    #[test]
    fn test_into_map() {
        let arr = Array::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let result: Array<String> = arr.into_map(|x| format!("value_{}", x));
        assert_eq!(result[[0, 0]], "value_1");
        assert_eq!(result[[1, 1]], "value_4");
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_into_map_type_conversion() {
        let arr = Array::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let result: Array<f64> = arr.into_map(|x| x as f64 * 2.5);
        assert_eq!(result[[0, 0]], 2.5);
        assert_eq!(result[[1, 1]], 10.0);
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_svd() {
        // Test SVD on a simple 2x2 matrix
        let matrix = Array::from_vec(vec![3.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let (u, s, vt) = matrix.svd().unwrap();

        // Check dimensions
        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 2]);

        // Check that singular values are positive and sorted in descending order
        assert!(s[[0]] >= s[[1]]);
        assert!(s[[0]] > 0.0);
        assert!(s[[1]] >= 0.0);
    }

    #[test]
    fn test_svd_reconstruction() {
        use crate::prelude::*;

        // Test that U * S * V^T reconstructs the original matrix (approximately)
        let matrix = Array::from_vec(vec![2.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let (u, s, vt) = matrix.svd().unwrap();

        // Create diagonal matrix from singular values
        let mut sigma = Array::zeros(vec![2, 2]);
        sigma[[0, 0]] = s[[0]];
        sigma[[1, 1]] = s[[1]];

        // Reconstruct: U * Sigma * V^T
        let us = u.matmul(&sigma, crate::math::MatmulPolicy::Naive).unwrap();
        let reconstructed = us.matmul(&vt, crate::math::MatmulPolicy::Naive).unwrap();

        // Check reconstruction accuracy (within tolerance)
        let tolerance = 1e-10f64;
        for i in 0..2 {
            for j in 0..2 {
                let diff = (matrix[[i, j]] - reconstructed[[i, j]]) as f64;
                let diff = diff.abs();
                assert!(diff < tolerance, "Reconstruction error too large: {}", diff);
            }
        }
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
            vec![5],
        )
        .unwrap();

        let sin_result = arr.sin();
        assert!((sin_result[[0]] - 0.0).abs() < 1e-10);
        assert!((sin_result[[1]] - 0.5).abs() < 1e-10);
        assert!((sin_result[[4]] - 1.0).abs() < 1e-10);

        let cos_result = arr.cos();
        assert!((cos_result[[0]] - 1.0).abs() < 1e-10);
        assert!((cos_result[[4]] - 0.0).abs() < 1e-10);

        let _tan_result = arr.tan();

        // Test inverse functions
        let values = Array::from_vec(vec![0.0, 0.5, 1.0], vec![3]).unwrap();
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
        let arr = Array::from_vec(vec![0.0, 1.0, -1.0], vec![3]).unwrap();

        let sinh_result = arr.sinh();
        let cosh_result = arr.cosh();
        let tanh_result = arr.tanh();

        assert!((sinh_result[[0]] - 0.0f64).abs() < 1e-10f64);
        assert!((cosh_result[[0]] - 1.0f64).abs() < 1e-10f64);
        assert!((tanh_result[[0]] - 0.0f64).abs() < 1e-10f64);
    }

    #[test]
    fn test_logarithmic_functions() {
        let arr = Array::from_vec(vec![1.0, std::f64::consts::E, 10.0, 2.0], vec![4]).unwrap();

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
        let values = Array::from_vec(vec![1.0, 3.0, 9.0, 27.0], vec![4]).unwrap();
        let log3_result = values.log(3.0);
        assert!((log3_result[[0]] - 0.0f64).abs() < 1e-10f64); // log3(1) = 0
        assert!((log3_result[[1]] - 1.0f64).abs() < 1e-10f64); // log3(3) = 1
        assert!((log3_result[[2]] - 2.0f64).abs() < 1e-10f64); // log3(9) = 2
        assert!((log3_result[[3]] - 3.0f64).abs() < 1e-10f64); // log3(27) = 3

        // Test that exp and ln are inverse operations
        let test_values = Array::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let exp_ln = test_values.clone().ln().exp();
        for i in 0..3 {
            assert!(((exp_ln[[i]] - test_values[[i]]) as f64).abs() < 1e-10);
        }
    }

    #[test]
    fn test_exponential_functions() {
        let arr = Array::from_vec(vec![0.0, 1.0, 2.0], vec![3]).unwrap();

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
        let small_values = Array::from_vec(vec![0.1, 0.01, 0.001], vec![3]).unwrap();
        let exp_m1_result = small_values.clone().exp_m1();
        let ln_1p_result = exp_m1_result.ln_1p();

        for i in 0..3 {
            assert!(((ln_1p_result[[i]] - small_values[[i]]) as f64).abs() < 1e-10);
        }

        // Test that exp_m1(0) = 0 and ln_1p(0) = 0
        let zero_arr = Array::from_vec(vec![0.0], vec![1]).unwrap();
        assert!((zero_arr.clone().exp_m1()[[0]] - 0.0f64).abs() < 1e-10f64);
        assert!((zero_arr.ln_1p()[[0]] - 0.0f64).abs() < 1e-10f64);
    }
}
