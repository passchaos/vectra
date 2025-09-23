use std::fmt;
use std::ops::Add;
// use std::ops::{Index, IndexMut}; // Currently unused
use std::slice::Iter;

use approx::{AbsDiffEq, RelativeEq};
use num_traits::{One, Zero};

/// Multi-dimensional array structure, similar to numpy's ndarray
#[derive(PartialEq)]
pub struct Array<T> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl<T: AbsDiffEq> AbsDiffEq for Array<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }
}

impl<T: RelativeEq> RelativeEq for Array<T>
where
    T::Epsilon: Copy,
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
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b)| a.relative_eq(b, epsilon, max_relative))
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
    pub fn reshape(self, new_shape: Vec<usize>) -> Result<Self, String> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.data.len() {
            return Err("New shape size does not match array size".to_string());
        }

        let new_strides = Self::compute_strides(&new_shape);
        Ok(Self {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
        })
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

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}

impl<T> Array<T> {
    /// Convert multi-dimensional index to flat index
    pub fn index_to_flat(&self, indices: &[usize]) -> Result<usize, String> {
        if indices.len() != self.shape.len() {
            return Err("Index dimension mismatch".to_string());
        }
        let mut flat_index = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(format!(
                    "Index {} out of bounds for dimension {} with size {}",
                    idx, i, self.shape[i]
                ));
            }
            flat_index += idx * self.strides[i];
        }
        Ok(flat_index)
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
        self.data.len()
    }

    /// Get iterator over elements
    pub fn iter(&self) -> Iter<'_, T> {
        self.data.iter()
    }

    /// Get mutable iterator
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Apply a closure to each element in-place, modifying the current Array
    pub fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T,
    {
        for item in self.data.iter_mut() {
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
    fn test_transpose() {
        let arr1 = Array::from_vec(vec![1.0; 10000], vec![10, 10, 100]).unwrap();
        println!("arr1: {arr1}");
        // let arr_t = arr.clone().transpose().unwrap();

        // println!("{:?} {:?}", arr, arr_t);
    }
}
