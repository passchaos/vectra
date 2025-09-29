use std::collections::HashSet;
use std::fmt::{self, Display};
use std::ops::{Add, IndexMut, Mul, Sub};

use approx::{AbsDiffEq, RelativeEq};
use faer::{Mat, MatRef};
use itertools::Itertools;
use num_traits::{NumCast, One, Zero};

/// Multi-dimensional array structure, similar to numpy's ndarray
pub struct Array<T> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl<T> Array<T> {
    pub fn as_faer(&self) -> MatRef<'_, T> {
        MatRef::from_row_major_slice(self.data.as_slice(), self.shape[0], self.shape[1])
    }
}

impl<T> From<Mat<T>> for Array<T> {
    fn from(mut mat: Mat<T>) -> Self {
        let nrows = mat.nrows();
        let ncols = mat.ncols();

        let col_stride = mat.col_stride() as usize;
        let row_stride = mat.row_stride() as usize;

        // Data may not be contiguous and could have padding, so we can only calculate data length as follows
        let len = nrows * row_stride + ncols * col_stride;

        let data = unsafe { Vec::from_raw_parts(mat.as_ptr_mut(), len, len) };
        // println!("data: {data:?}");
        std::mem::forget(mat);

        let shape = vec![nrows, ncols];
        // let strides = Self::compute_strides(&shape);
        let strides = vec![row_stride as usize, col_stride as usize];

        Self {
            data,
            shape,
            strides,
        }
    }
}

impl<T: PartialEq> PartialEq for Array<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        self.multi_iter()
            .zip(other.multi_iter())
            .all(|((_, a_v), (_, b_v))| a_v == b_v)
    }
}

impl<T: AbsDiffEq> AbsDiffEq for Array<T>
where
    T::Epsilon: Copy + NumCast + Mul<T::Epsilon, Output = T::Epsilon>,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        // <T::Epsilon as NumCast>::from(1e-15).unwrap()
        T::default_epsilon() * <T::Epsilon as NumCast>::from(3e3).unwrap()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        // Strides don't need to be equal as there might be padding
        // if self.strides() != other.strides() {
        //     return false;
        // }
        self.multi_iter()
            .zip(other.multi_iter())
            .all(|((_, a_v), (_, b_v))| {
                a_v.abs_diff_eq(
                    b_v,
                    epsilon, // <<T as AbsDiffEq>::Epsilon as NumCast>::from(epsilon).unwrap(),
                )
            })
    }
}

impl<T: RelativeEq + Copy + Sub<T> + Display> RelativeEq for Array<T>
where
    T: RelativeEq + Sub<T> + Display,
    T::Output: Display,
    T::Epsilon: Copy + Display + NumCast + Mul<T::Epsilon, Output = T::Epsilon>,
{
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        // Strides don't need to be equal as there might be padding
        // if self.strides() != other.strides() {
        //     return false;
        // }
        // let epsilon = <<T as AbsDiffEq>::Epsilon as NumCast>::from(epsilon).unwrap();
        // let max_relative = <<T as AbsDiffEq>::Epsilon as NumCast>::from(max_relative).unwrap();

        self.multi_iter()
            .zip(other.multi_iter())
            .map(|((a_i, a_v), (b_i, b_v))| {
                if !a_v.relative_eq(b_v, epsilon, max_relative) {
                    println!("meet neq: a_i= {a_i:?} a_v= {a_v} b_i= {b_i:?} b_v= {b_v} \nepsilon= \t{epsilon} \nmax_relative= \t{max_relative} \ndif= \t\t{}", *a_v - *b_v);
                }
                ((a_i, a_v), (b_i, b_v))
            })
            .all(|((_, a_v), (_, b_v))| a_v.relative_eq(b_v, epsilon, max_relative))
        // self.data
        //     .iter()
        //     .zip(other.data.iter())
        //     .all(|(a, b)| a.relative_eq(b, epsilon, max_relative))
    }
}

impl<T: Clone> Clone for Array<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<T> Array<T> {
    /// Create array from data and shape, accepts any type that implements Into<Vec<T>>
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Result<Self, String> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(format!(
                "Data length {} does not match shape {:?}, expected length {}",
                data.len(),
                shape,
                expected_size
            ));
        }
        let strides = Self::compute_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Create zero array
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Clone + Zero,
    {
        let size = shape.iter().product();
        let strides = Self::compute_strides(&shape);
        Self {
            data: vec![T::zero(); size],
            shape,
            strides,
        }
    }

    /// Create array like np.arange
    pub fn arange(mut start: T, end: T, step: T) -> Self
    where
        T: PartialOrd + Add<Output = T> + Clone,
    {
        let mut data = Vec::new();

        while start < end {
            data.push(start.clone());

            start = start + step.clone();
        }

        let size = data.len();

        let strides = Self::compute_strides(&[size]);
        Self {
            data,
            shape: vec![size],
            strides,
        }
    }

    /// Create ones array
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: Clone + One,
    {
        let size = shape.iter().product();
        let strides = Self::compute_strides(&shape);
        Self {
            data: vec![T::one(); size],
            shape,
            strides,
        }
    }

    /// Create identity matrix
    pub fn eye(n: usize) -> Self
    where
        T: Clone + Zero + One,
    {
        let mut arr = Self::zeros(vec![n, n]);
        for i in 0..n {
            arr[[i, i]] = T::one();
        }
        arr
    }

    /// Reshape array
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), String> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.data.len() {
            return Err("New shape size does not match array size".to_string());
        }

        let new_strides = Self::compute_strides(&new_shape);

        self.shape = new_shape;
        self.strides = new_strides;

        Ok(())
    }

    pub fn squeeze(&mut self, axes: Vec<usize>) -> Result<(), String> {
        let axes: HashSet<_> = axes.into_iter().collect();

        let mut new_shape = Vec::new();

        let need_remove_one_dim = axes.len() == 0;
        for (idx, &shape) in self.shape.iter().enumerate() {
            if !axes.contains(&idx) {
                if need_remove_one_dim {
                    if shape == 1 {
                        continue;
                    }
                }

                new_shape.push(shape);
                continue;
            }

            if shape != 1 {
                return Err(format!("Cannot squeeze axis {} with size {}", idx, shape));
            }
        }

        let new_strides = Self::compute_strides(&new_shape);
        self.shape = new_shape;
        self.strides = new_strides;

        Ok(())
    }

    pub fn unsqueeze(&mut self, axe: usize) -> Result<(), String> {
        let mut new_shape = self.shape.clone();
        new_shape.insert(axe, 1);

        let new_strides = Self::compute_strides(&new_shape);
        self.shape = new_shape;
        self.strides = new_strides;

        Ok(())
    }

    /// Transpose array (2D only)
    pub fn transpose(self) -> Result<Self, String> {
        self.permute(vec![1, 0])
    }

    /// Permute the dimensions of the array according to the given axes
    /// Similar to PyTorch's permute function
    pub fn permute(self, axes: Vec<usize>) -> Result<Self, String> {
        // Validate axes
        if axes.len() != self.shape.len() {
            return Err(format!(
                "Number of axes {} does not match array dimensions {}",
                axes.len(),
                self.shape.len()
            ));
        }

        // Check if all axes are valid and unique
        let mut sorted_axes = axes.clone();
        sorted_axes.sort();
        for (i, &axis) in sorted_axes.iter().enumerate() {
            if axis != i {
                return Err(format!("Invalid axis {} or duplicate axis found", axis));
            }
        }

        let new_shape = axes.iter().map(|&i| self.shape()[i]).collect();
        let new_strides = axes.iter().map(|&i| self.strides[i]).collect();

        Ok(Self {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
        })
    }

    // must be row-major order
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    fn compute_flat_idx(strides: &[usize], indices: &[usize]) -> usize {
        strides.iter().zip(indices).map(|(&s, &i)| s * i).sum()
    }
}

impl<T> Array<T> {
    /// Convert multi-dimensional index to flat index
    pub fn index_to_flat(&self, indices: &[usize]) -> Result<usize, String> {
        if indices.len() != self.shape.len() {
            return Err("Index dimension mismatch".to_string());
        }

        if indices
            .iter()
            .zip(self.shape().iter())
            .any(|(&i_dim, &s_dim)| i_dim >= s_dim)
        {
            return Err(format!(
                "Index out of bounds: shape= {:?} indices= {:?}",
                self.shape, indices
            ));
        }

        Ok(Self::compute_flat_idx(&self.strides, indices))
    }

    /// Broadcast two shapes to a common shape
    pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>, String> {
        let max_len = shape1.len().max(shape2.len());
        let mut result = vec![1; max_len];

        for i in 0..max_len {
            let dim1 = if i < shape1.len() {
                shape1[shape1.len() - 1 - i]
            } else {
                1
            };
            let dim2 = if i < shape2.len() {
                shape2[shape2.len() - 1 - i]
            } else {
                1
            };

            if dim1 == dim2 {
                result[max_len - 1 - i] = dim1;
            } else if dim1 == 1 {
                result[max_len - 1 - i] = dim2;
            } else if dim2 == 1 {
                result[max_len - 1 - i] = dim1;
            } else {
                return Err(format!(
                    "Cannot broadcast shapes {:?} and {:?}",
                    shape1, shape2
                ));
            }
        }
        Ok(result)
    }

    /// Broadcast array to target shape
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Array<T>, String>
    where
        T: Clone,
    {
        if self.shape == target_shape {
            return Ok(self.clone());
        }

        let target_size: usize = target_shape.iter().product();
        let mut new_data = Vec::with_capacity(target_size);

        for flat_idx in 0..target_size {
            let mut target_indices = vec![0; target_shape.len()];
            let mut temp = flat_idx;
            for i in (0..target_shape.len()).rev() {
                target_indices[i] = temp % target_shape[i];
                temp /= target_shape[i];
            }

            let mut source_indices = vec![0; self.shape.len()];
            let offset = target_shape.len() - self.shape.len();
            for i in 0..self.shape.len() {
                let target_idx = target_indices[offset + i];
                source_indices[i] = if self.shape[i] == 1 { 0 } else { target_idx };
            }

            let source_flat = self.index_to_flat(&source_indices)?;
            new_data.push(self.data[source_flat].clone());
        }

        Ok(Array {
            data: new_data,
            shape: target_shape.to_vec(),
            strides: compute_strides_for_shape(target_shape),
        })
    }

    /// Get shape of the array
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get total number of elements
    pub fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Apply a closure to each element in-place, modifying the current Array
    pub fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T,
    {
        for idx in self
            .shape()
            .into_iter()
            .map(|&n| 0..n)
            .multi_cartesian_product()
        {
            let item = self.index_mut(idx);
            *item = f(item);
        }
    }
}

impl<T> Into<Vec<T>> for Array<T> {
    fn into(self) -> Vec<T> {
        self.data
    }
}

impl<T> Into<Vec<T>> for &Array<T>
where
    T: Clone,
{
    fn into(self) -> Vec<T> {
        self.data.clone()
    }
}

impl<T> From<Vec<T>> for Array<T>
where
    T: Clone + Default,
{
    fn from(data: Vec<T>) -> Self {
        let shape = vec![data.len()];
        let strides = Self::compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }
}

pub fn compute_strides_for_shape(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// Additional From implementations for common array types
impl<T, const N: usize> From<[T; N]> for Array<T>
where
    T: Clone + Default,
{
    fn from(data: [T; N]) -> Self {
        let data = data.to_vec();
        let shape = vec![data.len()];
        let strides = Self::compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }
}

impl<T> From<&[T]> for Array<T>
where
    T: Clone + Default,
{
    fn from(data: &[T]) -> Self {
        let data = data.to_vec();
        let shape = vec![data.len()];
        let strides = Self::compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }
}

impl<T> fmt::Display for Array<T>
where
    T: fmt::Display + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_recursive(f, 0, &[])
    }
}

impl<T> fmt::Debug for Array<T>
where
    T: fmt::Display + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Array {{ data: {}, shape: {:?}, strides: {:?} }}",
            self, self.shape, self.strides
        )
    }
}

impl<T> Array<T>
where
    T: fmt::Display + Clone,
{
    /// Unified recursive formatting method for all dimensions
    fn fmt_recursive(
        &self,
        f: &mut fmt::Formatter<'_>,
        depth: usize,
        indices: &[usize],
    ) -> fmt::Result {
        let ndim = self.shape.len();

        if depth == ndim {
            // Base case: we've reached a scalar element
            let flat_idx = self.indices_to_flat(indices).unwrap_or(0);
            write!(f, "{}", self.data[flat_idx])
        } else if depth == ndim - 1 {
            // Last dimension: format as 1D array
            self.fmt_1d_slice(f, indices)
        } else {
            // Recursive case: format as nested arrays
            self.fmt_nd_slice(f, depth, indices)
        }
    }

    /// Helper method to convert multi-dimensional indices to flat index
    fn indices_to_flat(&self, indices: &[usize]) -> Result<usize, String> {
        if indices.len() != self.shape.len() {
            return Err("Index dimension mismatch".to_string());
        }

        let mut flat_idx = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err("Index out of bounds".to_string());
            }
            flat_idx += idx * self.strides[i];
        }
        Ok(flat_idx)
    }

    /// Format a 1D slice (last dimension)
    fn fmt_1d_slice(&self, f: &mut fmt::Formatter<'_>, base_indices: &[usize]) -> fmt::Result {
        let max_items = if base_indices.is_empty() { 1000 } else { 6 };
        let current_dim_size = self.shape[base_indices.len()];

        let line_size = 18;

        write!(f, "[")?;

        if current_dim_size <= max_items {
            for i in 0..current_dim_size {
                if i > 0 {
                    if i % line_size == 0 {
                        write!(f, "\n")?;
                    } else {
                        write!(f, " ")?;
                    }
                }
                let mut indices = base_indices.to_vec();
                indices.push(i);
                let flat_idx = self.indices_to_flat(&indices).unwrap_or(0);
                write!(f, "{}", self.data[flat_idx])?;
            }
        } else {
            // Show first 3 and last 3 items with ellipsis
            for i in 0..3 {
                if i > 0 {
                    write!(f, " ")?;
                }
                let mut indices = base_indices.to_vec();
                indices.push(i);
                let flat_idx = self.indices_to_flat(&indices).unwrap_or(0);
                write!(f, "{}", self.data[flat_idx])?;
            }
            write!(f, " ... ")?;
            for i in (current_dim_size - 3)..current_dim_size {
                let mut indices = base_indices.to_vec();
                indices.push(i);
                let flat_idx = self.indices_to_flat(&indices).unwrap_or(0);
                write!(f, "{}", self.data[flat_idx])?;
                if i < current_dim_size - 1 {
                    write!(f, " ")?;
                }
            }
        }

        write!(f, "]")
    }

    /// Format an N-dimensional slice (recursive case)
    fn fmt_nd_slice(
        &self,
        f: &mut fmt::Formatter<'_>,
        depth: usize,
        base_indices: &[usize],
    ) -> fmt::Result {
        let current_dim_size = self.shape[depth];
        let max_slices = 3;
        let ndim = self.shape.len();

        write!(f, "[")?;

        let show_all = current_dim_size <= max_slices;
        let slice_indices: Vec<usize> = if show_all {
            (0..current_dim_size).collect()
        } else {
            vec![0, 1, current_dim_size - 1]
        };

        for (idx, &slice_idx) in slice_indices.iter().enumerate() {
            if idx > 0 {
                // Add appropriate spacing based on dimension
                if depth == ndim - 2 {
                    // 2D case: new line with space
                    write!(f, "\n ")?;
                    for _ in 0..depth {
                        write!(f, " ")?;
                    }
                } else {
                    // Higher dimensions: double new line
                    write!(f, "\n\n ")?;
                    for _ in 0..depth {
                        write!(f, " ")?;
                    }
                }
            }

            if !show_all && idx == 2 {
                if depth == ndim - 2 {
                    write!(f, "\n ")?;
                    for _ in 0..depth {
                        write!(f, " ")?;
                    }
                    write!(f, "...\n ")?;
                    for _ in 0..depth {
                        write!(f, " ")?;
                    }
                } else {
                    write!(f, "\n ...\n\n ")?;
                    for _ in 0..depth {
                        write!(f, " ")?;
                    }
                }
            }

            let mut indices = base_indices.to_vec();
            indices.push(slice_idx);
            self.fmt_recursive(f, depth + 1, &indices)?;
        }

        write!(f, "]")
    }

    // Legacy methods for backward compatibility (now unused)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ones_and_eye() {
        let ones: Array<f64> = Array::ones(vec![2, 2]);
        assert_eq!(ones[[0, 0]], 1.0);
        assert_eq!(ones[[1, 1]], 1.0);

        let eye: Array<f64> = Array::eye(3);
        assert_eq!(eye[[0, 0]], 1.0);
        assert_eq!(eye[[1, 1]], 1.0);
        assert_eq!(eye[[0, 1]], 0.0);
    }

    #[test]
    fn test_arithmetic() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Array::from_vec(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]).unwrap();

        let sum = a.clone() + b.clone();
        assert_eq!(sum[[0, 0]], 3.0);
        assert_eq!(sum[[1, 1]], 6.0);

        let product = a * b;
        assert_eq!(product[[0, 0]], 2.0);
        assert_eq!(product[[1, 1]], 8.0);
    }

    #[test]
    fn test_transpose() {
        let arr1 = Array::from_vec(vec![1.0; 10000], vec![10, 10, 100]).unwrap();
        println!("arr1: {arr1}");
        // let arr_t = arr.clone().transpose().unwrap();

        // println!("{:?} {:?}", arr, arr_t);
    }

    #[test]
    fn test_map_inplace() {
        let mut arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        arr.map_inplace(|x| x * x);
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[1, 1]], 16.0);
    }

    #[test]
    fn test_reshape_and_transpose() {
        let mut arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        arr.reshape(vec![3, 2]).unwrap();
        assert_eq!(arr.shape(), &[3, 2]);

        let transposed = arr.transpose().unwrap();
        assert_eq!(transposed.shape(), &[2, 3]);
        assert_eq!(transposed[[0, 0]], 1.0);
        assert_eq!(transposed[[1, 0]], 2.0);
    }

    #[test]
    fn test_squeeze() {
        let mut arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]).unwrap();
        arr.squeeze(vec![1]).unwrap();

        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[1, 1]], 4.0);

        let res = arr.squeeze(vec![1]);
        assert!(res.is_err());

        let mut arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).unwrap();

        let res = arr.squeeze(vec![1]);
        assert!(res.is_err());

        arr.squeeze(vec![0]).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr[[1, 0]], 3.0);

        arr.unsqueeze(0).unwrap();
        assert_eq!(arr.shape(), &[1, 2, 2]);
        assert_eq!(arr[[0, 1, 0]], 3.0);

        let mut arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 1, 2]).unwrap();

        arr.squeeze(vec![]).unwrap();
        arr.squeeze(vec![]).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
    }

    #[test]
    fn test_broadcasting() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let b = Array::from_vec(vec![10.0, 20.0], vec![1, 2]).unwrap();

        let result = a + b;
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result[[0, 0]], 11.0); // 1 + 10
        assert_eq!(result[[0, 1]], 21.0); // 1 + 20
        assert_eq!(result[[1, 0]], 12.0); // 2 + 10
        assert_eq!(result[[1, 1]], 22.0); // 2 + 20
        assert_eq!(result[[2, 0]], 13.0); // 3 + 10
        assert_eq!(result[[2, 1]], 23.0); // 3 + 20

        // Test scalar operations
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let added = arr.add_scalar(5.0);
        assert_eq!(added[[0, 0]], 6.0);
        assert_eq!(added[[1, 1]], 9.0);

        let multiplied = arr.mul_scalar(2.0);
        assert_eq!(multiplied[[0, 0]], 2.0);
        assert_eq!(multiplied[[1, 1]], 8.0);
    }

    #[test]
    fn test_broadcast_shapes() {
        let shape1 = vec![3, 1];
        let shape2 = vec![1, 4];
        let result = Array::<f64>::broadcast_shapes(&shape1, &shape2).unwrap();
        assert_eq!(result, vec![3, 4]);

        let shape3 = vec![2, 3];
        let shape4 = vec![2, 4];
        let result2 = Array::<f64>::broadcast_shapes(&shape3, &shape4);
        assert!(result2.is_err());
    }
}
