use std::fmt;
// use std::ops::{Index, IndexMut}; // Currently unused
use std::slice::Iter;

/// Multi-dimensional array structure, similar to numpy's ndarray
#[derive(Clone, PartialEq)]
pub struct Array<T> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl<T> Array<T>
where
    T: Clone + Default,
{
    /// Create a new multi-dimensional array
    pub fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let strides = Self::compute_strides(&shape);
        Self {
            data: vec![T::default(); size],
            shape,
            strides,
        }
    }

    /// Create array from data and shape, accepts any type that implements Into<Vec<T>>
    pub fn from_vec<V: Into<Vec<T>>>(data: V, shape: Vec<usize>) -> Result<Self, String> {
        let data = data.into();
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
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::new(shape)
    }

    /// Create ones array
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: From<u8>,
    {
        let size = shape.iter().product();
        let strides = Self::compute_strides(&shape);
        Self {
            data: vec![T::from(1u8); size],
            shape,
            strides,
        }
    }

    /// Create identity matrix
    pub fn eye(n: usize) -> Self
    where
        T: From<u8>,
    {
        let mut arr = Self::zeros(vec![n, n]);
        for i in 0..n {
            arr[[i, i]] = T::from(1u8);
        }
        arr
    }

    /// Reshape array
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, String> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.data.len() {
            return Err("New shape size does not match array size".to_string());
        }
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape.clone(),
            strides: Self::compute_strides(&new_shape),
        })
    }

    /// Transpose array (2D only)
    pub fn transpose(&self) -> Result<Self, String> {
        if self.shape.len() != 2 {
            return Err("Transpose only supported for 2D arrays".to_string());
        }
        let new_shape = vec![self.shape[1], self.shape[0]];
        let mut new_data = vec![T::default(); self.data.len()];
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                new_data[j * self.shape[0] + i] = self.data[i * self.shape[1] + j].clone();
            }
        }
        Ok(Self {
            data: new_data,
            shape: new_shape.clone(),
            strides: Self::compute_strides(&new_shape),
        })
    }

    /// Permute the dimensions of the array according to the given axes
    /// Similar to PyTorch's permute function
    pub fn permute(&self, axes: Vec<usize>) -> Result<Self, String> {
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
        
        // Create new shape by permuting dimensions
        let new_shape: Vec<usize> = axes.iter().map(|&i| self.shape[i]).collect();
        let new_strides = Self::compute_strides(&new_shape);
        
        // Create new data array
        let mut new_data = vec![T::default(); self.data.len()];
        
        // Generate all possible indices for the new array
        let total_elements = self.data.len();
        for flat_idx in 0..total_elements {
            // Convert flat index to multi-dimensional indices in new array
            let mut new_indices = vec![0; new_shape.len()];
            let mut remaining = flat_idx;
            for i in 0..new_shape.len() {
                new_indices[i] = remaining / new_strides[i];
                remaining %= new_strides[i];
            }
            
            // Map new indices back to original indices using inverse permutation
            let mut orig_indices = vec![0; self.shape.len()];
            for (new_dim, &orig_dim) in axes.iter().enumerate() {
                orig_indices[orig_dim] = new_indices[new_dim];
            }
            
            // Calculate flat index in original array
            let orig_flat_idx = self.index_to_flat(&orig_indices)?;
            new_data[flat_idx] = self.data[orig_flat_idx].clone();
        }
        
        Ok(Self {
            data: new_data,
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
        write!(f, "Array {{ data: {}, shape: {:?}, strides: {:?} }}", 
               self, self.shape, self.strides)
    }
}

impl<T> Array<T>
where
    T: fmt::Display + Clone,
{
    /// Unified recursive formatting method for all dimensions
    fn fmt_recursive(&self, f: &mut fmt::Formatter<'_>, depth: usize, indices: &[usize]) -> fmt::Result {
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
        let max_items = 6;
        let current_dim_size = self.shape[base_indices.len()];
        
        write!(f, "[")?;
        
        if current_dim_size <= max_items {
            for i in 0..current_dim_size {
                if i > 0 {
                    write!(f, " ")?;
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
    fn fmt_nd_slice(&self, f: &mut fmt::Formatter<'_>, depth: usize, base_indices: &[usize]) -> fmt::Result {
        let current_dim_size = self.shape[depth];
        let max_slices = 3;
        let ndim = self.shape.len();
        
        // Special handling for very high dimensions (5D+)
        if ndim >= 5 && depth == 0 {
            return write!(f, "array(shape={:?}, data=[{} {} {} ...])", 
                         self.shape, 
                         self.data.get(0).map_or("?".to_string(), |x| format!("{}", x)),
                         self.data.get(1).map_or("?".to_string(), |x| format!("{}", x)),
                         self.data.get(2).map_or("?".to_string(), |x| format!("{}", x)));
        }
        
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
