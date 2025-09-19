use std::fmt;
// use std::ops::{Index, IndexMut}; // Currently unused
use std::slice::Iter;

/// Multi-dimensional array structure, similar to numpy's ndarray
#[derive(Debug, Clone, PartialEq)]
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
            strides: Self::compute_strides_for_shape(target_shape),
        })
    }

    pub fn compute_strides_for_shape(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
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
        match self.shape.len() {
            1 => {
                write!(f, "[")?;
                for (i, item) in self.data.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            2 => {
                writeln!(f, "[")?;
                for i in 0..self.shape[0] {
                    write!(f, "  [")?;
                    for j in 0..self.shape[1] {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", self.data[i * self.shape[1] + j])?;
                    }
                    if i < self.shape[0] - 1 {
                        writeln!(f, "],")?;
                    } else {
                        writeln!(f, "]")?;
                    }
                }
                write!(f, "]")
            }
            _ => {
                write!(f, "Array with shape {:?}", self.shape)
            }
        }
    }
}
