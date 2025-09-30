use std::fmt::{self, Display};
use std::ops::{Add, IndexMut, Mul, Sub};

use approx::{AbsDiffEq, RelativeEq};
use faer::{Mat, MatRef};
use itertools::Itertools;
use num_traits::{NumCast, One, Zero};

use crate::utils::{compute_strides, dyn_dim_to_static, indices_to_flat_idx};

#[derive(Debug, Default, Clone, Copy)]
pub enum MajorOrder {
    #[default]
    RowMajor,
    ColumnMajor,
}

/// Multi-dimensional array structure, similar to numpy's ndarray
pub struct Array<const D: usize, T> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: [usize; D],
    pub(crate) strides: [usize; D],
    pub(crate) major_order: MajorOrder,
}

impl<T> Array<2, T> {
    pub fn as_faer(&self) -> MatRef<'_, T> {
        let (nrows, ncols) = (self.shape[0], self.shape[1]);
        match self.major_order {
            MajorOrder::RowMajor => {
                MatRef::from_row_major_slice(self.data.as_slice(), nrows, ncols)
            }
            MajorOrder::ColumnMajor => {
                MatRef::from_column_major_slice(self.data.as_slice(), nrows, ncols)
            }
        }
    }

    /// Create identity matrix
    pub fn eye(n: usize) -> Self
    where
        T: Clone + Zero + One,
    {
        let mut arr = Self::zeros([n, n]);
        for i in 0..n {
            arr[[i, i]] = T::one();
        }
        arr
    }

    /// Transpose array (2D only)
    pub fn transpose(self) -> Self {
        self.permute([1, 0])
    }
}

impl<T> From<Mat<T>> for Array<2, T> {
    fn from(mut mat: Mat<T>) -> Self {
        let nrows = mat.nrows();
        let ncols = mat.ncols();

        let col_stride = mat.col_stride() as usize;
        let row_stride = mat.row_stride() as usize;

        // Data may not be contiguous and could have padding, so we can only calculate data length as follows
        let len = nrows * row_stride + ncols * col_stride;

        // zero-copy
        let data = unsafe { Vec::from_raw_parts(mat.as_ptr_mut(), len, len) };
        // println!("data: {data:?}");
        std::mem::forget(mat);

        let shape = [nrows, ncols];
        // let strides = compute_strides(&shape);
        let strides = [row_stride as usize, col_stride as usize];

        Self {
            data,
            shape,
            strides,
            major_order: MajorOrder::ColumnMajor,
        }
    }
}

impl<const D: usize, T: PartialEq> PartialEq for Array<D, T> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        self.multi_iter()
            .zip(other.multi_iter())
            .all(|((_, a_v), (_, b_v))| a_v == b_v)
    }
}

impl<const D: usize, T: AbsDiffEq> AbsDiffEq for Array<D, T>
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

impl<const D: usize, T: RelativeEq + Copy + Sub<T> + Display> RelativeEq for Array<D, T>
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

impl<const D: usize, T: Clone> Clone for Array<D, T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            major_order: self.major_order.clone(),
        }
    }
}

impl<T> Array<1, T> {
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

        let major_order = MajorOrder::RowMajor;

        let shape = [size];
        let strides = compute_strides(shape, major_order);

        Self {
            data,
            shape,
            strides,
            major_order,
        }
    }
}

impl<const D: usize, T> Array<D, T> {
    pub fn from_vec(data: Vec<T>, shape: [usize; D]) -> Self {
        Self::from_vec_major(data, shape, MajorOrder::RowMajor)
    }

    pub fn from_vec_major(data: Vec<T>, shape: [usize; D], major_order: MajorOrder) -> Self {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            panic!(
                "Data length {} does not match shape {:?}, expected length {}",
                data.len(),
                shape,
                expected_size
            );
        }

        let strides = compute_strides(shape, major_order);

        Self {
            data,
            shape,
            strides,
            major_order,
        }
    }

    /// Create zero array
    pub fn zeros(shape: [usize; D]) -> Self
    where
        T: Clone + Zero,
    {
        let size = shape.iter().product();

        let major_order = MajorOrder::RowMajor;
        let strides = compute_strides(shape, major_order);

        Self {
            data: vec![T::zero(); size],
            shape,
            strides,
            major_order,
        }
    }

    /// Create ones array
    pub fn ones(shape: [usize; D]) -> Self
    where
        T: Clone + One,
    {
        let size = shape.iter().product();

        let major_order = MajorOrder::RowMajor;
        let strides = compute_strides(shape, major_order);

        Self {
            data: vec![T::one(); size],
            shape,
            strides,
            major_order,
        }
    }

    /// Reshape array
    pub fn reshape<const D1: usize>(self, new_shape: [usize; D1]) -> Array<D1, T> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.data.len() {
            panic!("New shape size does not match array size");
        }

        let new_strides = compute_strides(new_shape, self.major_order);

        Array {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            major_order: self.major_order,
        }
    }

    /// Permute the dimensions of the array according to the given axes
    /// Similar to PyTorch's permute function
    pub fn permute(self, axes: [usize; D]) -> Self {
        // Check if all axes are valid and unique
        let mut sorted_axes = axes.clone();
        sorted_axes.sort();
        if sorted_axes[D - 1] != D - 1 {
            panic!("invalid permute axes: {axes:?}");
        }

        let mut shape = [0; D];
        let new_shape: Vec<_> = axes.iter().map(|&i| self.shape[i]).collect();
        shape.copy_from_slice(&new_shape);

        let strides = compute_strides(shape, self.major_order);

        Self {
            data: self.data,
            shape,
            strides,
            major_order: self.major_order,
        }
    }
}

impl<const D: usize, T> Array<D, T> {
    /// Convert multi-dimensional index to flat index
    pub fn index_to_flat(&self, indices: [usize; D]) -> usize {
        if indices
            .iter()
            .zip(self.shape().iter())
            .any(|(&i_dim, &s_dim)| i_dim >= s_dim)
        {
            panic!(
                "Index out of bounds: shape= {:?} indices= {:?}",
                self.shape, indices
            );
        }

        indices_to_flat_idx(self.strides, indices)
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

    /// Get shape of the array
    pub fn shape(&self) -> [usize; D] {
        self.shape
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        D
    }

    /// Get total number of elements
    pub fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Apply a closure to each element in-place, modifying the current Array
    pub fn map_inplace<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&T) -> T,
    {
        for idx in self
            .shape()
            .into_iter()
            .map(|n| 0..n)
            .multi_cartesian_product()
        {
            let idx = dyn_dim_to_static(&idx);
            let item = self.index_mut(idx);
            *item = f(item);
        }

        self
    }
}

impl<const D: usize, T> Into<Vec<T>> for Array<D, T> {
    fn into(self) -> Vec<T> {
        self.data
    }
}

impl<T> From<Vec<T>> for Array<1, T>
where
    T: Clone + Default,
{
    fn from(data: Vec<T>) -> Self {
        let shape = [data.len()];

        let major_order = MajorOrder::RowMajor;
        let strides = compute_strides(shape, major_order);
        Self {
            data,
            shape,
            strides,
            major_order,
        }
    }
}

impl<const D: usize, T> fmt::Display for Array<D, T>
where
    T: fmt::Display + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_recursive(f, 0, &[])
    }
}

impl<const D: usize, T> fmt::Debug for Array<D, T>
where
    T: fmt::Display + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Array {{ data: {}, shape: {:?}, strides: {:?} major_order: {:?} }}",
            self, self.shape, self.strides, self.major_order
        )
    }
}

impl<const D: usize, T> Array<D, T>
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
    fn test_from_vec() {
        let arr1 = Array::from_vec(vec![0, 1, 2, 3, 4, 5], [2, 3]);
        let arr2 = Array::from_vec_major(vec![0, 3, 1, 4, 2, 5], [2, 3], MajorOrder::ColumnMajor);

        println!("arr1= {arr1:?} arr2= {arr2:?}");
        println!("arr2: [0, 1]= {}", arr2[[0, 1]]);
        assert_eq!(arr1, arr2);
    }

    #[test]
    fn test_faer() {
        let arr_f = faer::mat![[1, 2, 3], [4, 5, 6]];
        println!(
            "strides: row= {} col= {}",
            arr_f.row_stride(),
            arr_f.col_stride()
        );
        let arr_f_i = Array::from(arr_f);

        println!("aff_f_i: {arr_f_i:?} data= {:?}", arr_f_i.data);

        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6], [2, 3]);
        let arr_f_r = arr.as_faer();
        println!(
            "arr_f_r: shape= {:?} strides= {:?} {:?}",
            arr_f_r.shape(),
            arr_f_r.row_stride(),
            arr_f_r.col_stride()
        );
        let arr_f_r_c = arr_f_r.cloned();
        println!(
            "arr_f_r_c: shape= {:?} strides= {:?} {:?}",
            arr_f_r_c.shape(),
            arr_f_r_c.row_stride(),
            arr_f_r_c.col_stride()
        );

        let arr_f: Array<_, i32> = Array::from(arr_f_r_c);
        assert_eq!(arr, arr_f);
        println!("arr= {:?} arr_f= {:?}", arr, arr_f);
    }

    #[test]
    fn test_ones_and_eye() {
        let ones: Array<_, f64> = Array::ones([2, 2]);
        assert_eq!(ones[[0, 0]], 1.0);
        assert_eq!(ones[[1, 1]], 1.0);

        let eye: Array<_, f64> = Array::eye(3);
        assert_eq!(eye[[0, 0]], 1.0);
        assert_eq!(eye[[1, 1]], 1.0);
        assert_eq!(eye[[0, 1]], 0.0);
    }

    #[test]
    fn test_arithmetic() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let b = Array::from_vec(vec![2.0, 2.0, 2.0, 2.0], [2, 2]);

        let sum = a.clone() + b.clone();
        assert_eq!(sum[[0, 0]], 3.0);
        assert_eq!(sum[[1, 1]], 6.0);

        let product = a * b;
        assert_eq!(product[[0, 0]], 2.0);
        assert_eq!(product[[1, 1]], 8.0);
    }

    #[test]
    fn test_transpose() {
        let arr1 = Array::from_vec(vec![1.0; 10000], [10, 10, 100]);
        println!("arr1: {arr1}");
        // let arr_t = arr.clone().transpose().unwrap();

        // println!("{:?} {:?}", arr, arr_t);
    }

    #[test]
    fn test_map_inplace() {
        let mut arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        arr.map_inplace(|x| x * x);
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[1, 1]], 16.0);
    }

    #[test]
    fn test_reshape_and_transpose() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let arr = arr.reshape([3, 2]);
        assert_eq!(arr.shape(), [3, 2]);

        let arr = arr.transpose();
        assert_eq!(arr.shape(), [2, 3]);
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[1, 0]], 4.0);
    }

    #[test]
    fn test_reshape() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 1, 2]);

        let arr = arr.reshape([2, 2]);
        assert_eq!(arr.shape(), [2, 2]);
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[1, 1]], 4.0);
    }
}
