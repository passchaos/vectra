use faer::traits::ComplexField;

use crate::core::{Array, compute_strides_for_shape};
use std::{
    any::TypeId,
    fmt::Debug,
    ops::{Add, AddAssign, Div, Mul},
}; // Sub currently unused

#[derive(Clone, Copy, Debug)]
pub enum MatmulPolicy {
    Naive,
    Faer,
    #[cfg(feature = "blas")]
    Blas,
    LoopReorder,
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
        self.data
            .iter()
            .fold(T::default(), |acc, x| acc + x.clone())
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
        self.data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
    }

    /// Find minimum value
    pub fn min(&self) -> Option<T>
    where
        T: Clone + PartialOrd,
    {
        self.data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
    }

    pub fn matmul_general(&self, other: &Array<T>, policy: MatmulPolicy) -> Result<Array<T>, String>
    where
        T: Add<Output = T> + AddAssign + Mul<Output = T> + Default + 'static,
        Array<T>: 'static,
    {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err("Matrix multiplication only supported for 2D arrays".to_string());
        }
        if self.shape[1] != other.shape[0] {
            return Err("Matrix dimensions incompatible for multiplication".to_string());
        }

        let result_shape = vec![self.shape[0], other.shape[1]];
        let mut result_data = vec![T::default(); result_shape.iter().product()];

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        match policy {
            #[cfg(feature = "blas")]
            MatmulPolicy::Blas => {
                use std::ffi::c_char;

                let type_id = TypeId::of::<T>();

                let m = m as i32;
                let n = n as i32;
                let k = k as i32;

                let a = self.data.as_ptr();
                let b = other.data.as_ptr();
                let c = result_data.as_mut_ptr();

                if type_id == TypeId::of::<f32>() {
                    let alpha = 1.0;
                    let beta = 0.0;

                    unsafe {
                        blas_sys::sgemm_(
                            &(b'N' as c_char),
                            &(b'N' as c_char),
                            &n,
                            &m,
                            &k,
                            &alpha,
                            b.cast(),
                            &n,
                            a.cast(),
                            &k,
                            &beta,
                            c.cast(),
                            &n,
                        );
                    }
                } else if type_id == TypeId::of::<f64>() {
                    let alpha = 1.0;
                    let beta = 0.0;

                    unsafe {
                        blas_sys::dgemm_(
                            &(b'N' as c_char),
                            &(b'N' as c_char),
                            &n,
                            &m,
                            &k,
                            &alpha,
                            b.cast(),
                            &n,
                            a.cast(),
                            &k,
                            &beta,
                            c.cast(),
                            &n,
                        );
                    }
                } else {
                    return Err("Unsupported type for BLAS matrix multiplication".to_string());
                }
            }
            MatmulPolicy::Naive => {
                for i in 0..m {
                    for j in 0..n {
                        for l in 0..k {
                            result_data[i * n + j] +=
                                self.data[i * k + l].clone() * other.data[l * n + j].clone();
                        }
                    }
                }
            }
            MatmulPolicy::LoopReorder => {
                for i in 0..m {
                    for l in 0..k {
                        // 这里实测，直接使用get_unchecked只能减少1%的耗时
                        // let a_v = unsafe { self.data.get_unchecked(i * k + l) };
                        let a_v = &self.data[i * k + l];
                        for j in 0..n {
                            // let data = unsafe { result_data.get_unchecked_mut(i * n + j) };
                            // *data += a_v.clone()
                            //     * unsafe { other.data.get_unchecked(l * n + j) }.clone();
                            result_data[i * n + j] += a_v.clone() * other.data[l * n + j].clone();
                        }
                    }
                }
            }
            MatmulPolicy::Blocking(block_size) => {
                for i in (0..m).step_by(block_size) {
                    let i_end = (i + block_size).min(m);

                    for l in (0..k).step_by(block_size) {
                        let l_end = (l + block_size).min(k);

                        for j in (0..n).step_by(block_size) {
                            let j_end = (j + block_size).min(n);

                            for ii in i..i_end {
                                for ll in l..l_end {
                                    let a_v = &self.data[ii * k + ll];
                                    for jj in j..j_end {
                                        result_data[ii * n + jj] +=
                                            a_v.clone() * other.data[ll * n + jj].clone();
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => unreachable!(),
        }

        Array::from_vec(result_data, result_shape)
    }

    /// Matrix multiplication
    /// 在macos平台上，faer耗时是accelerate的2.35倍，但是比openblas少21%
    /// 在linux平台上，faer耗时比openblas多5%
    pub fn matmul(&self, other: &Array<T>, policy: MatmulPolicy) -> Result<Array<T>, String>
    where
        T: Add<Output = T> + AddAssign + Mul<Output = T> + Default + ComplexField + 'static,
    {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err("Matrix multiplication only supported for 2D arrays".to_string());
        }
        if self.shape[1] != other.shape[0] {
            return Err("Matrix dimensions incompatible for multiplication".to_string());
        }

        match policy {
            MatmulPolicy::Faer => {
                let l = self.as_faer();
                let r = other.as_faer();

                Ok((l * r).into())
            }
            p => self.matmul_general(other, p),
        }
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

    /// Apply function to each element
    pub fn map<F, U>(&self, f: F) -> Array<U>
    where
        F: Fn(&T) -> U,
        U: Clone + Default,
    {
        Array {
            data: self.data.iter().map(f).collect(),
            shape: self.shape.clone(),
            strides: compute_strides_for_shape(&self.shape),
        }
    }

    /// Apply function to each element, consuming self and returning a new Array with different type
    pub fn into_map<F, U>(self, f: F) -> Array<U>
    where
        F: Fn(T) -> U,
        U: Clone + Default,
    {
        let Self {
            data,
            shape,
            strides,
        } = self;

        Array {
            data: data.into_iter().map(f).collect(),
            shape,
            strides,
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

        let result_data: Vec<T> = self
            .data
            .iter()
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
        let vt_array = Array::from_vec(vt_data, vec![n, n])
            .unwrap()
            .transpose()
            .unwrap();

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
    use approx::assert_relative_eq;

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
    fn test_matmul() {
        let shapes: Vec<(usize, usize, usize)> = vec![
            (50, 50, 50),
            (50, 75, 60),
            (100, 110, 120),
            (220, 230, 250),
            (250, 250, 250),
            (300, 300, 300),
            (300, 400, 350),
            (1000, 1000, 1000),
            (1000, 1200, 1100),
        ];

        let inputs_64: Vec<_> = shapes
            .iter()
            .map(|(m, n, k)| {
                (
                    Array::<f64>::randn(vec![*m, *k]),
                    Array::<f64>::randn(vec![*k, *n]),
                )
            })
            .collect();

        let results_64: Vec<_> = inputs_64
            .iter()
            .map(|(l, r)| l.matmul(r, MatmulPolicy::Naive).unwrap())
            .collect();

        let inputs_32: Vec<_> = shapes
            .iter()
            .map(|(m, n, k)| {
                (
                    Array::<f32>::randn(vec![*m, *k]),
                    Array::<f32>::randn(vec![*k, *n]),
                )
            })
            .collect();
        let results_32: Vec<_> = inputs_32
            .iter()
            .map(|(l, r)| l.matmul(r, MatmulPolicy::Naive).unwrap())
            .collect();

        for policy in [
            MatmulPolicy::Blas,
            MatmulPolicy::Faer,
            MatmulPolicy::LoopReorder,
            MatmulPolicy::Blocking(512),
        ] {
            for ((l, r), res) in inputs_64.iter().zip(results_64.iter()) {
                let new_res = l.matmul(r, policy).unwrap();
                assert_relative_eq!(*res, new_res);
            }

            for ((l, r), res) in inputs_32.iter().zip(results_32.iter()) {
                let new_res = l.matmul(r, policy).unwrap();
                assert_relative_eq!(*res, new_res);
            }
        }
    }
}
