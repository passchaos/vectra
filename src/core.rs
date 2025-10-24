use std::any::type_name;
use std::fmt::{self, Debug};
use std::ops::{Index, IndexMut};

use approx::{AbsDiffEq, RelativeEq};
use faer::{Mat, MatRef};
use itertools::Itertools;
use num_traits::{Float, NumCast, One, Zero};

use crate::NumExt;
use crate::utils::{
    compute_strides, dyn_dim_to_static, flat_idx_to_indices, indices_to_flat_idx,
    negative_idx_to_positive, negative_indices_to_positive, shape_indices_to_flat_idx,
};

/// Memory layout order for multi-dimensional arrays.
///
/// This enum determines how multi-dimensional array data is stored in memory,
/// which affects performance characteristics for different access patterns.
///
/// # Examples
///
/// ```rust
/// use vectra::core::MajorOrder;
///
/// // Row-major order (C-style): elements in the same row are contiguous
/// let row_major = MajorOrder::RowMajor;
///
/// // Column-major order (Fortran-style): elements in the same column are contiguous
/// let col_major = MajorOrder::ColumnMajor;
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum MajorOrder {
    /// Row-major order (C-style layout).
    ///
    /// In row-major order, elements of each row are stored contiguously in memory.
    /// This is the default layout and is generally more cache-friendly for
    /// row-wise operations.
    #[default]
    RowMajor,
    /// Column-major order (Fortran-style layout).
    ///
    /// In column-major order, elements of each column are stored contiguously in memory.
    /// This layout is more efficient for column-wise operations and is compatible
    /// with many BLAS/LAPACK routines.
    ColumnMajor,
}

/// A multi-dimensional array structure, similar to NumPy's ndarray.
///
/// `Array<D, T>` represents a D-dimensional array containing elements of type T.
/// The array supports various operations including mathematical functions,
/// linear algebra, broadcasting, and efficient indexing.
///
/// # Type Parameters
///
/// * `D` - The number of dimensions (compile-time constant)
/// * `T` - The element type, which must implement [`crate::NumExt`] for most operations
///
/// # Memory Layout
///
/// Arrays store their data in a contiguous `Vec<T>` with configurable memory layout
/// (row-major or column-major). The shape and strides determine how multi-dimensional
/// indices map to linear memory addresses.
///
/// # Examples
///
/// ## Creating Arrays
///
/// ```rust
/// use vectra::prelude::*;
///
/// // Create a 2x3 array of zeros
/// let zeros = Array::<_, f64>::zeros([2, 3]);
///
/// // Create from a vector with specified shape
/// let data = vec![1, 2, 3, 4, 5, 6];
/// let arr = Array::from_vec(data, [2, 3]);
///
/// // Create an identity matrix
/// let eye = Array::<_, f32>::eye(3);
/// ```
///
/// ## Array Operations
///
/// ```rust
/// use vectra::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
/// let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], [2, 2]);
///
/// // Element-wise operations
/// let sum = &a + &b;
/// let product = &a * &b;
///
/// // Matrix multiplication
/// let matmul_result = a.matmul(&b);
///
/// // Mathematical functions
/// let sin_a = a.sin();
/// let exp_a = a.exp();
/// ```
///
/// ## Indexing and Reshaping
///
/// ```rust
/// use vectra::prelude::*;
///
/// let mut arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6], [2, 3]);
///
/// // Access elements
/// let element = arr[[0, 1]];
///
/// // Modify elements
/// arr[[1, 2]] = 42;
///
/// // Reshape (for 2D arrays)
/// let reshaped = arr.reshape([3, 2]);
///
/// // Transpose (for 2D arrays)
/// let transposed = arr.transpose();
/// ```
pub struct Array<const D: usize, T> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: [usize; D],
    pub(crate) strides: [usize; D],
    pub(crate) major_order: MajorOrder,
}

impl<T: NumExt + Debug> Array<2, T> {
    /// Convert the 2D array to a `faer::MatRef` for interoperability with the faer linear algebra library.
    ///
    /// This method provides zero-copy conversion to faer's matrix type, allowing you to use
    /// faer's optimized linear algebra operations on Vectra arrays.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    /// let faer_mat = arr.as_faer();
    /// // Now you can use faer operations on faer_mat
    /// ```
    pub fn as_faer(&self) -> MatRef<'_, T> {
        let (nrows, ncols) = (self.shape[0], self.shape[1]);
        let res = match self.major_order {
            MajorOrder::RowMajor => MatRef::from_row_major_slice_with_stride(
                self.data.as_slice(),
                nrows,
                ncols,
                self.strides[0],
            ),
            MajorOrder::ColumnMajor => MatRef::from_column_major_slice_with_stride(
                self.data.as_slice(),
                nrows,
                ncols,
                self.strides[1],
            ),
        };

        res
    }

    /// Create an n×n identity matrix.
    ///
    /// An identity matrix is a square matrix with ones on the main diagonal
    /// and zeros elsewhere. It acts as the multiplicative identity for matrix
    /// multiplication.
    ///
    /// # Arguments
    ///
    /// * `n` - The size of the square identity matrix
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let eye3 = Array::<_, f64>::eye(3);
    /// // Creates:
    /// // [[1.0, 0.0, 0.0],
    /// //  [0.0, 1.0, 0.0],
    /// //  [0.0, 0.0, 1.0]]
    /// ```
    pub fn eye(n: usize) -> Self {
        let mut arr = Self::zeros([n, n]);
        for i in 0..n as isize {
            arr[[i, i]] = T::one();
        }
        arr
    }
}

impl<T> Array<2, T> {
    /// Transpose the 2D array, swapping rows and columns.
    ///
    /// This operation swaps the dimensions of the array, so that element at position `[i, j]`
    /// in the original array becomes element at position `[j, i]` in the transposed array.
    ///
    /// This is a zero-copy operation that only changes the shape and strides metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6], [2, 3]);
    /// // Original: [[1, 2, 3],
    /// //            [4, 5, 6]]
    ///
    /// let transposed = arr.transpose();
    /// // Transposed: [[1, 4],
    /// //              [2, 5],
    /// //              [3, 6]]
    /// ```
    ///
    /// # Note
    ///
    /// This method is only available for 2D arrays. For higher-dimensional arrays,
    /// use the `permute` method to rearrange dimensions.
    pub fn transpose(self) -> Self {
        let new_shape = [self.shape[1], self.shape[0]];
        let new_stride = [self.strides[1], self.strides[0]];

        let major_order = if new_stride[0] == 1 {
            MajorOrder::ColumnMajor
        } else {
            MajorOrder::RowMajor
        };

        Self {
            data: self.data,
            shape: new_shape,
            strides: new_stride,
            major_order: major_order,
        }
    }
}

impl<T: NumExt> From<Mat<T>> for Array<2, T> {
    fn from(mut mat: Mat<T>) -> Self {
        let nrows = mat.nrows();
        let ncols = mat.ncols();

        let col_stride = mat.col_stride() as usize;
        let row_stride = mat.row_stride() as usize;

        // Data may not be contiguous and could have padding, so we can only calculate data length as follows
        let len = nrows * row_stride + ncols * col_stride;
        // println!(
        //     "data info: nrows= {} ncols= {} col_stride= {} row_stride= {} len= {}",
        //     nrows, ncols, col_stride, row_stride, len
        // );

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
    T: NumExt,
    T::Epsilon: NumExt,
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

impl<const D: usize, T: RelativeEq> RelativeEq for Array<D, T>
where
    T: NumExt,
    T::Epsilon: NumExt,
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
                    println!("meet neq: a_i= {a_i:?} a_v= {a_v:?} b_i= {b_i:?} b_v= {b_v:?} \nepsilon= \t{epsilon:?} \nmax_relative= \t{max_relative:?} \ndif= \t\t{:?}", *a_v - *b_v);
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

impl<T: NumExt> Array<1, T> {
    /// Create a 1D array with evenly spaced values using count and step.
    ///
    /// This function creates an array starting from `start`, incrementing by `step`,
    /// and containing exactly `count` elements.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting value
    /// * `step` - The increment between consecutive values
    /// * `count` - The number of elements to generate
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let arr = Array::arange_c(0, 2, 5);
    /// // Creates: [0, 2, 4, 6, 8]
    ///
    /// let arr_f = Array::arange_c(1.0, 0.5, 4);
    /// // Creates: [1.0, 1.5, 2.0, 2.5]
    /// ```
    pub fn arange_c(mut start: T, step: T, count: usize) -> Self {
        let mut data = Vec::new();

        for _ in 0..count {
            data.push(start);

            start = start + step;
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

    /// Create a 1D array with evenly spaced values in a given range.
    ///
    /// This function creates an array starting from `start`, incrementing by `step`,
    /// and stopping before reaching `end` (exclusive upper bound).
    ///
    /// # Arguments
    ///
    /// * `start` - The starting value (inclusive)
    /// * `end` - The ending value (exclusive)
    /// * `step` - The increment between consecutive values
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let arr = Array::arange(0, 10, 2);
    /// // Creates: [0, 2, 4, 6, 8]
    ///
    /// let arr_f = Array::arange(0.0, 3.0, 0.5);
    /// // Creates: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    /// ```
    ///
    /// # Note
    ///
    /// The `end` value is exclusive, meaning it will not be included in the result
    /// even if it would be exactly reachable by the step increment.
    pub fn arange(mut start: T, end: T, step: T) -> Self {
        let mut data = Vec::new();

        while start.cmp_ext(&end).is_lt() {
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
    /// Create an array from a vector with the specified shape.
    ///
    /// This is the primary constructor for creating arrays from existing data.
    /// The data is interpreted in row-major (C-style) order by default.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector containing the array elements
    /// * `shape` - The desired shape of the array
    ///
    /// # Panics
    ///
    /// Panics if the length of `data` doesn't match the product of dimensions in `shape`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// // Create a 2x3 array
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let arr = Array::from_vec(data, [2, 3]);
    /// // Results in:
    /// // [[1, 2, 3],
    /// //  [4, 5, 6]]
    ///
    /// // Create a 3D array
    /// let data = vec![1.0; 24];
    /// let arr = Array::from_vec(data, [2, 3, 4]);
    /// ```
    pub fn from_vec(data: Vec<T>, shape: [usize; D]) -> Self {
        Self::from_vec_major(data, shape, MajorOrder::RowMajor)
    }

    /// Create an array from a vector with the specified shape and memory layout.
    ///
    /// This constructor allows you to specify the memory layout order (row-major or column-major).
    ///
    /// # Arguments
    ///
    /// * `data` - A vector containing the array elements
    /// * `shape` - The desired shape of the array
    /// * `major_order` - The memory layout order (RowMajor or ColumnMajor)
    ///
    /// # Panics
    ///
    /// Panics if the length of `data` doesn't match the product of dimensions in `shape`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    /// use vectra::core::MajorOrder;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    ///
    /// // Row-major layout (default)
    /// let arr_row = Array::from_vec_major(data.clone(), [2, 3], MajorOrder::RowMajor);
    ///
    /// // Column-major layout (Fortran-style)
    /// let arr_col = Array::from_vec_major(data, [2, 3], MajorOrder::ColumnMajor);
    /// ```
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

    /// Create an array filled with a specific value.
    ///
    /// All elements in the array will be set to the provided value.
    ///
    /// # Arguments
    ///
    /// * `shape` - The desired shape of the array
    /// * `value` - The value to fill the array with
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// // Create a 2x3 array filled with 42
    /// let arr = Array::full([2, 3], 42);
    ///
    /// // Create a 3D array filled with 3.14
    /// let arr_3d = Array::full([2, 2, 2], 3.14);
    /// ```
    pub fn full(shape: [usize; D], value: T) -> Self
    where
        T: Clone,
    {
        let size = shape.iter().product();

        let major_order = MajorOrder::RowMajor;
        let strides = compute_strides(shape, major_order);

        Self {
            data: vec![value; size],
            shape,
            strides,
            major_order,
        }
    }

    /// Reshape the array to a new shape.
    ///
    /// This method changes the shape of the array while preserving the total number of elements.
    /// One dimension can be inferred by using -1, which will be calculated automatically.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - The desired new shape. Use -1 for one dimension to infer its size.
    ///
    /// # Panics
    ///
    /// * Panics if more than one dimension is set to -1
    /// * Panics if the total number of elements doesn't match the original array
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// // Reshape a 1D array to 2D
    /// let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6], [6]);
    /// let reshaped = arr.reshape([2, 3]);
    /// // Results in a 2x3 array
    ///
    /// // Use -1 to infer one dimension
    /// let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8], [8]);
    /// let reshaped = arr.reshape([2, -1]); // Becomes [2, 4]
    ///
    /// // Reshape 2D to 3D
    /// let arr = Array::from_vec(vec![1; 24], [4, 6]);
    /// let reshaped = arr.reshape([2, 3, 4]);
    /// ```
    pub fn reshape<const D1: usize>(self, mut new_shape: [isize; D1]) -> Array<D1, T> {
        let len = self.shape().iter().product::<usize>();

        let mut negative_indices: Vec<_> = new_shape
            .iter()
            .enumerate()
            .filter_map(|(idx, v)| if *v == -1 { Some(idx) } else { None })
            .collect();
        if negative_indices.len() > 1 {
            panic!("Only one dimension can be inferred");
        }

        if let Some(negative_index) = negative_indices.pop() {
            new_shape[negative_index] = len as isize / new_shape.iter().product::<isize>() * -1;
        }

        let new_shape = new_shape.map(|v| v as usize);

        let new_size = new_shape.iter().product::<usize>();
        if new_size != len {
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

    /// Add a new axis of length 1 at the specified position.
    ///
    /// This operation increases the dimensionality of the array by 1.
    ///
    /// # Arguments
    ///
    /// * `axis` - The position where the new axis should be inserted (supports negative indexing)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// // Add axis at the beginning
    /// let arr = Array::from_vec(vec![1, 2, 3], [3]); // Shape: [3]
    /// let unsqueezed = arr.unsqueeze(0); // Shape: [1, 3]
    ///
    /// // Add axis at the end
    /// let arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2]); // Shape: [2, 2]
    /// let unsqueezed = arr.unsqueeze(-1); // Shape: [2, 2, 1]
    /// ```
    pub fn unsqueeze(self, axis: isize) -> Array<{ D + 1 }, T> {
        let axis = negative_idx_to_positive(axis, D + 1);

        let mut shape = self.shape().map(|v| v as isize).to_vec();
        shape.insert(axis, 1);

        let mut new_shape = [0; D + 1];
        new_shape.copy_from_slice(&shape);

        self.reshape(new_shape)
    }

    /// Remove an axis of length 1 at the specified position.
    ///
    /// This operation decreases the dimensionality of the array by 1.
    ///
    /// # Arguments
    ///
    /// * `axis` - The position of the axis to remove (must have size 1)
    ///
    /// # Panics
    ///
    /// Panics if the specified axis doesn't have size 1.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// // Remove axis of size 1
    /// let arr = Array::from_vec(vec![1, 2, 3], [1, 3]); // Shape: [1, 3]
    /// let squeezed = arr.squeeze(0); // Shape: [3]
    ///
    /// // Remove last axis
    /// let arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2, 1]); // Shape: [2, 2, 1]
    /// let squeezed = arr.squeeze(-1); // Shape: [2, 2]
    /// ```
    pub fn squeeze(self, axis: isize) -> Array<{ D - 1 }, T> {
        let axis = negative_idx_to_positive(axis, D);

        let mut shape = self.shape().map(|v| v as isize).to_vec();

        let axis_size = shape[axis];
        if axis_size != 1 {
            panic!("cannot squeeze axis with size {axis_size}");
        }

        shape.remove(axis);

        let mut new_shape = [0; D - 1];
        new_shape.copy_from_slice(&shape);

        self.reshape(new_shape)
    }

    /// Permute the dimensions of the array according to the given axes
    /// Similar to PyTorch's permute function
    /// Permute the axes of the array.
    ///
    /// This operation rearranges the dimensions of the array according to the specified order.
    /// For 2D arrays, this is equivalent to transpose when axes are [1, 0].
    ///
    /// # Arguments
    ///
    /// * `axes` - The new order of axes. Must be a permutation of [0, 1, ..., D-1]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// // Transpose a 2D array
    /// let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6], [2, 3]);
    /// let transposed = arr.permute([1, 0]); // Equivalent to transpose()
    ///
    /// // Permute a 3D array
    /// let arr = Array::from_vec(vec![1; 24], [2, 3, 4]); // Shape: [2, 3, 4]
    /// let permuted = arr.permute([2, 0, 1]); // Shape: [4, 2, 3]
    /// ```
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

    /// Concatenate arrays along the specified axis.
    ///
    /// All arrays must have the same shape except in the concatenation dimension.
    ///
    /// # Arguments
    ///
    /// * `arrs` - Slice of array references to concatenate
    /// * `axis` - The axis along which to concatenate (supports negative indexing)
    ///
    /// # Panics
    ///
    /// Panics if arrays have incompatible shapes for concatenation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// // Concatenate along rows (axis 0)
    /// let arr1 = Array::from_vec(vec![1, 2, 3], [1, 3]);
    /// let arr2 = Array::from_vec(vec![4, 5, 6], [1, 3]);
    /// let result = Array::cat(&[&arr1, &arr2], 0); // Shape: [2, 3]
    ///
    /// // Concatenate along columns (axis 1)
    /// let arr1 = Array::from_vec(vec![1, 2], [2, 1]);
    /// let arr2 = Array::from_vec(vec![3, 4], [2, 1]);
    /// let result = Array::cat(&[&arr1, &arr2], 1); // Shape: [2, 2]
    /// ```
    pub fn cat(arrs: &[&Self], axis: isize) -> Self
    where
        T: Clone + Default,
    {
        assert!(arrs.len() > 0);

        let mut shape = arrs[0].shape();
        let major_order = arrs[0].major_order;
        let axis = negative_idx_to_positive(axis, shape.len());
        let axis_orig_size = shape[axis];

        for arr in arrs {
            assert_eq!(arr.shape(), arrs[0].shape());
            assert_eq!(arr.major_order, major_order);
        }

        shape[axis as usize] = arrs
            .iter()
            .map(|arr| arr.shape()[axis as usize])
            .sum::<usize>();

        let strides = compute_strides(shape, major_order);
        let mut data = vec![T::default(); shape.iter().product()];

        for idx in 0..data.len() {
            let mut indices = flat_idx_to_indices(shape, idx, major_order);

            let axis_idx = indices[axis as usize];
            let arr_outer_idx = axis_idx / axis_orig_size;
            let arr_inner_idx = axis_idx % axis_orig_size;

            indices[axis] = arr_inner_idx;

            let v = arrs[arr_outer_idx][indices.map(|i| i as isize)].clone();

            data[idx] = v;
        }

        Self {
            data,
            shape,
            strides,
            major_order,
        }
    }

    pub fn stack(arrs: Vec<Array<D, T>>, axis: isize) -> Array<{ D + 1 }, T>
    where
        T: Clone + Default,
    {
        let arrs: Vec<_> = arrs.into_iter().map(|a| a.unsqueeze(axis)).collect();

        let arrs: Vec<_> = arrs.iter().collect();
        Array::cat(&arrs, axis)
    }

    /// Convert multi-dimensional index to flat index
    pub fn index_to_flat(&self, indices: [isize; D]) -> usize {
        let indices = negative_indices_to_positive(indices, self.shape);

        self.positive_index_to_flat(indices)
    }

    pub fn positive_index_to_flat(&self, indices: [usize; D]) -> usize {
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

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
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
    /// Apply a function to each element in-place.
    ///
    /// This method modifies the array by applying the given function to each element.
    /// The function receives a reference to each element and returns a new value.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that takes a reference to an element and returns a new value
    ///
    /// # Returns
    ///
    /// Returns a mutable reference to self for method chaining.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let mut arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
    /// arr.map_inplace(|x| x * 2);
    /// // arr is now [[2, 4], [6, 8]]
    ///
    /// let mut arr_f = Array::from_vec(vec![1.0, 2.0, 3.0], [3]);
    /// arr_f.map_inplace(|x| x.sqrt());
    /// ```
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
            let item = self.index_mut(idx.map(|i| i as isize));
            *item = f(item);
        }

        self
    }

    /// Gather elements along an axis using indices.
    ///
    /// This operation selects elements from the array along the specified axis
    /// using the provided indices array.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to gather (supports negative indexing)
    /// * `indices` - Array of indices specifying which elements to gather
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// // Gather from a 2D array along axis 0 (rows)
    /// let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6], [3, 2]);
    /// let indices = Array::from_vec(vec![0, 2, 1], [3, 1]);
    /// let result = arr.gather(0, &indices);
    /// // Selects rows 0, 2, 1 from the original array
    ///
    /// // Gather along axis 1 (columns)
    /// let indices = Array::from_vec(vec![1, 0], [1, 2]);
    /// let result = arr.gather(1, &indices);
    /// // Selects columns 1, 0 from each row
    /// ```
    pub fn gather(&self, axis: isize, indices: &Array<D, isize>) -> Self
    where
        T: Clone + Default,
    {
        let axis = negative_idx_to_positive(axis, D);

        let target_shape = indices.shape();
        let mut result_data = vec![T::default(); target_shape.iter().product()];

        for (idx, i_v) in indices.multi_iter() {
            let mut target_idx = idx;
            target_idx[axis] = *i_v as usize;

            let target_value = self[target_idx.map(|i| i as isize)].clone();
            let flat_idx = indices.positive_index_to_flat(idx);
            result_data[flat_idx] = target_value;
        }

        let major_order = MajorOrder::RowMajor;
        let strides = compute_strides(target_shape, major_order);
        Self {
            data: result_data,
            shape: target_shape,
            strides,
            major_order,
        }
    }

    /// Scatter values into the array along an axis using indices.
    ///
    /// This operation places values from the `values` array into positions
    /// specified by the `indices` array along the given axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to scatter (supports negative indexing)
    /// * `indices` - Array of indices specifying where to place values
    /// * `values` - Array of values to scatter
    ///
    /// # Panics
    ///
    /// Panics if `indices` and `values` don't have the same shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let mut arr = Array::zeros([3, 2]);
    /// let indices = Array::from_vec(vec![0, 2], [1, 2]);
    /// let values = Array::from_vec(vec![10, 20], [1, 2]);
    ///
    /// arr.scatter(0, &indices, &values);
    /// // Places values [10, 20] into rows 0 and 2 respectively
    /// ```
    pub fn scatter(&mut self, axis: isize, indices: &Array<D, isize>, values: &Array<D, T>)
    where
        T: Clone,
    {
        assert_eq!(indices.shape(), values.shape());

        let axis = negative_idx_to_positive(axis, D);

        for ((a_idx, a_v), (_b_idx, b_v)) in indices.multi_iter().zip(values.multi_iter()) {
            let mut target_idx = a_idx;
            target_idx[axis] = negative_idx_to_positive(*a_v, self.shape()[axis]);

            *self.index_mut(target_idx.map(|i| i as isize)) = b_v.clone();
        }
    }

    /// Replace elements where mask is true with corresponding values.
    ///
    /// This method conditionally replaces elements in the array based on a boolean mask.
    /// Where the mask is `true`, elements are replaced with values from the `values` array.
    ///
    /// # Arguments
    ///
    /// * `mark` - Boolean mask array indicating which elements to replace
    /// * `values` - Array of replacement values
    ///
    /// # Panics
    ///
    /// Panics if the shapes of `mark` and `values` don't match the array's shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let mut arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
    /// let mask = Array::from_vec(vec![true, false, false, true], [2, 2]);
    /// let values = Array::from_vec(vec![10, 20, 30, 40], [2, 2]);
    ///
    /// arr.mask_where(&mask, &values);
    /// // arr becomes [[10, 2], [3, 40]]
    /// ```
    pub fn mask_where(&mut self, mark: &Array<D, bool>, values: &Array<D, T>)
    where
        T: Clone,
    {
        assert_eq!(self.shape(), mark.shape());
        assert_eq!(mark.shape(), values.shape());

        self.multi_iter_mut(|idx, val| {
            let idx = idx.map(|i| i as isize);

            if mark[idx] {
                *val = values[idx].clone();
            }
        });
    }

    /// Fill elements where mask is true with a single value.
    ///
    /// This method conditionally fills elements in the array based on a boolean mask.
    /// Where the mask is `true`, elements are replaced with the specified value.
    ///
    /// # Arguments
    ///
    /// * `mark` - Boolean mask array indicating which elements to fill
    /// * `value` - The value to fill with
    ///
    /// # Panics
    ///
    /// Panics if the shape of `mark` doesn't match the array's shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let mut arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
    /// let mask = Array::from_vec(vec![true, false, false, true], [2, 2]);
    ///
    /// arr.mask_fill(&mask, 99);
    /// // arr becomes [[99, 2], [3, 99]]
    /// ```
    pub fn mask_fill(&mut self, mark: &Array<D, bool>, value: T)
    where
        T: Clone,
    {
        assert_eq!(self.shape(), mark.shape());

        self.multi_iter_mut(|idx, val| {
            let idx = idx.map(|i| i as isize);

            if mark[idx] {
                *val = value.clone();
            }
        });
    }

    /// Broadcast array to target shape
    /// Broadcast the array to a new shape.
    ///
    /// Broadcasting allows arrays with different shapes to be used together in operations.
    /// The array is virtually expanded to match the target shape without copying data when possible.
    ///
    /// # Arguments
    ///
    /// * `target_shape` - The desired shape to broadcast to
    ///
    /// # Panics
    ///
    /// Panics if the array cannot be broadcast to the target shape.
    ///
    /// # Broadcasting Rules
    ///
    /// - Dimensions are aligned from the rightmost dimension
    /// - Each dimension must either be 1 or match the target dimension
    /// - Missing dimensions are treated as 1
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// // Broadcast a 1D array to 2D
    /// let arr = Array::from_vec(vec![1, 2, 3], [3]);
    /// let broadcasted = arr.broadcast_to([2, 3]);
    /// // Results in:
    /// // [[1, 2, 3],
    /// //  [1, 2, 3]]
    ///
    /// // Broadcast with dimension of size 1
    /// let arr = Array::from_vec(vec![1, 2], [2, 1]);
    /// let broadcasted = arr.broadcast_to([2, 3]);
    /// // Results in:
    /// // [[1, 1, 1],
    /// //  [2, 2, 2]]
    /// ```
    pub fn broadcast_to(&self, target_shape: [usize; D]) -> Self
    where
        T: Clone,
    {
        if self.shape == target_shape {
            return self.clone();
        }

        let target_size: usize = target_shape.iter().product();
        let mut new_data = Vec::with_capacity(target_size);

        for flat_idx in 0..target_size {
            let mut target_indices = [0; D];
            let mut temp = flat_idx;
            for i in (0..D).rev() {
                target_indices[i] = temp % target_shape[i];
                temp /= target_shape[i];
            }

            let mut source_indices = [0; D];
            for i in 0..D {
                let target_idx = target_indices[i];
                source_indices[i] = if self.shape[i] == 1 { 0 } else { target_idx };
            }

            let source_flat = self.positive_index_to_flat(source_indices);
            new_data.push(self.data[source_flat].clone());
        }

        let major_order = MajorOrder::RowMajor;
        let strides = compute_strides(target_shape, major_order);

        Array {
            data: new_data,
            shape: target_shape,
            strides,
            major_order,
        }
    }

    pub fn map<F, U>(&self, f: F) -> Array<D, U>
    where
        F: FnMut(&T) -> U,
    {
        Array {
            data: self.data.iter().map(f).collect(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            major_order: self.major_order,
        }
    }

    pub fn mapv<F, U>(&self, f: F) -> Array<D, U>
    where
        F: FnMut(T) -> U,
        T: Clone,
    {
        Array {
            data: self.data.iter().cloned().map(f).collect(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            major_order: self.major_order,
        }
    }

    pub fn map_axis<F, U>(&self, axis: isize, f: F) -> Array<D, U>
    where
        U: Default + Clone,
        F: Fn(Vec<&T>) -> U,
    {
        if axis >= (D as isize) || axis < -(D as isize) {
            panic!("Axis out of bounds: rank= {}, axis= {}", D, axis);
        }

        // Adjust negative axis to a positive index
        let axis = if axis < 0 {
            (axis + D as isize) as usize
        } else {
            axis as usize
        };

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

                    self.index(i_idx.map(|i| i as isize))
                })
                .collect();

            let value = f(axis_values);

            let flat_idx = shape_indices_to_flat_idx(result_shape, idx, major_order);
            result_data[flat_idx] = value;
        }

        Array::from_vec_major(result_data, result_shape, major_order)
    }

    /// Apply function to each element, consuming self and returning a new Array with different type
    pub fn map_into<F, U>(self, f: F) -> Array<D, U>
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

    pub fn equal(&self, other: &Self) -> Array<D, bool>
    where
        T: PartialEq,
    {
        self.map(|x| x == &other.data[0])
    }
}

// only for number type
impl<const D: usize, T> Array<D, T> {
    /// Create an array filled with zeros.
    ///
    /// This is a convenience method for creating arrays where all elements are zero.
    /// The element type must implement the `Zero` trait.
    ///
    /// # Arguments
    ///
    /// * `shape` - The desired shape of the array
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// // Create a 3x3 matrix of zeros
    /// let zeros = Array::<_, f64>::zeros([3, 3]);
    ///
    /// // Create a 1D array of zeros
    /// let zeros_1d = Array::<_, i32>::zeros([10]);
    ///
    /// // Create a 3D array of zeros
    /// let zeros_3d = Array::<_, f32>::zeros([2, 4, 3]);
    /// ```
    pub fn zeros(shape: [usize; D]) -> Self
    where
        T: Clone + Zero,
    {
        Self::full(shape, T::zero())
    }

    /// Create an array filled with ones.
    ///
    /// This is a convenience method for creating arrays where all elements are one.
    /// The element type must implement the `One` trait.
    ///
    /// # Arguments
    ///
    /// * `shape` - The desired shape of the array
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// // Create a 2x4 matrix of ones
    /// let ones = Array::<_, f64>::ones([2, 4]);
    ///
    /// // Create a 1D array of ones
    /// let ones_1d = Array::<_, i32>::ones([5]);
    ///
    /// // Create a 3D array of ones
    /// let ones_3d = Array::<_, f32>::ones([2, 2, 2]);
    /// ```
    pub fn ones(shape: [usize; D]) -> Self
    where
        T: Clone + One,
    {
        Self::full(shape, T::one())
    }

    pub fn is_nan(&self) -> Array<D, bool>
    where
        T: Float,
    {
        self.map(|x| x.is_nan())
    }

    pub fn contains_nan(&self) -> bool
    where
        T: Float + NumExt,
    {
        let sum = self.sum();
        sum.is_nan()
    }
}

impl<const D: usize, T> Into<Vec<T>> for Array<D, T> {
    fn into(self) -> Vec<T> {
        self.data
    }
}

impl<T> From<Vec<T>> for Array<1, T> {
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

impl<const D: usize, T: Debug> fmt::Display for Array<D, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_recursive(f, 0, &[])
    }
}

impl<const D: usize, T: Debug> fmt::Debug for Array<D, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Array {{ data: {}, shape: {:?}, strides: {:?} major_order: {:?} }}",
            self, self.shape, self.strides, self.major_order
        )
    }
}

fn pad_show_count<T>() -> usize {
    match type_name::<T>() {
        "i8" | "u8" | "i16" | "u16" | "i32" | "u32" | "i64" | "u64" | "isize" | "usize" => 8,
        "f32" => 5,
        _ => 3,
    }
}

impl<const D: usize, T: Debug> Array<D, T> {
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
            write!(f, "{:?}", self.data[flat_idx])
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
        let pad_show_count = pad_show_count::<T>();

        let max_items = if base_indices.is_empty() {
            1000
        } else {
            2 * pad_show_count
        };
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
                write!(f, "{:?}", self.data[flat_idx])?;
            }
        } else {
            // Show first 3 and last 3 items with ellipsis
            for i in 0..pad_show_count {
                if i > 0 {
                    write!(f, " ")?;
                }
                let mut indices = base_indices.to_vec();
                indices.push(i);
                let flat_idx = self.indices_to_flat(&indices).unwrap_or(0);
                write!(f, "{:?}", self.data[flat_idx])?;
            }
            write!(f, " ... ")?;
            for i in (current_dim_size - pad_show_count)..current_dim_size {
                let mut indices = base_indices.to_vec();
                indices.push(i);
                let flat_idx = self.indices_to_flat(&indices).unwrap_or(0);
                write!(f, "{:?}", self.data[flat_idx])?;
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
        let pad_show_count = pad_show_count::<T>();

        let current_dim_size = self.shape[depth];
        // let max_slices = 3;
        let ndim = self.shape.len();

        write!(f, "[")?;

        let show_all = current_dim_size <= 2 * pad_show_count;
        let slice_indices: Vec<usize> = if show_all {
            (0..current_dim_size).collect()
        } else {
            let mut indices = vec![];
            for i in 0..pad_show_count {
                indices.push(i);
            }

            for i in (current_dim_size - pad_show_count)..current_dim_size {
                indices.push(i);
            }

            indices
        };

        for (idx, &slice_idx) in slice_indices.iter().enumerate() {
            // Add appropriate spacing based on dimension
            if idx > 0 {
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

            if !show_all && idx == pad_show_count {
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
    use approx::assert_relative_eq;

    use crate::{math::MatmulPolicy, prelude::Matmul};

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
    fn test_transpose() {
        for policy in [
            // MatmulPolicy::Blas,
            MatmulPolicy::Faer,
            // MatmulPolicy::LoopReorder,
        ] {
            let l: Vec<f32> = rand::random_iter().take(20).collect();
            let r: Vec<f32> = rand::random_iter().take(20).collect();

            for shape in [[4, 5], [5, 4], [2, 10], [10, 2], [1, 20], [20, 1]] {
                let mut shape_reverse = shape.clone();
                shape_reverse.reverse();

                let test_consistence_with_ndarray = |transpose: bool| {
                    let mut arr_v_l = Array::from_vec(l.clone(), shape);
                    let mut arr_v_r = Array::from_vec(r.clone(), shape_reverse);

                    if transpose {
                        arr_v_l = arr_v_l.transpose();
                        arr_v_r = arr_v_r.transpose();
                    }

                    let arr_v = arr_v_l.matmul_with_policy(&arr_v_r, policy);

                    let mut arr_n_l = ndarray::Array2::from_shape_vec(shape, l.clone()).unwrap();
                    let mut arr_n_r =
                        ndarray::Array::from_shape_vec(shape_reverse, r.clone()).unwrap();

                    if transpose {
                        arr_n_l = arr_n_l.t().to_owned();
                        arr_n_r = arr_n_r.t().to_owned();
                    }

                    let arr_n = arr_n_l.dot(&arr_n_r);

                    // println!("v= {arr_v:?} n= {arr_n:?}");
                    for ((_, v), n) in arr_v.multi_iter().zip(arr_n.iter()) {
                        assert_relative_eq!(v, n, epsilon = 1e-6);
                    }
                };

                test_consistence_with_ndarray(false);
                test_consistence_with_ndarray(true);
            }
        }
        // let arr1 = Array::from_vec(vec![1, 2, 3, 4, 5, 6], [2, 3]);
        // let arr1_t = arr1.clone().transpose();
        // println!("arr1= {arr1:?} arr1_t= {arr1_t:?}");

        // let res = Array::from_vec(vec![1, 4, 2, 5, 3, 6], [3, 2]);
        // assert_eq!(arr1_t, res);
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
    fn test_cat_stack() {
        let a = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
        let b = Array::from_vec(vec![5, 6, 7, 8], [2, 2]);

        let arrs = vec![&a, &b];

        let res = Array::cat(&arrs, 0);
        assert_eq!(res, Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8], [4, 2]));

        let res = Array::cat(&arrs, 1);
        assert_eq!(res, Array::from_vec(vec![1, 2, 5, 6, 3, 4, 7, 8], [2, 4]));

        let arrs = vec![a, b];

        let res = Array::stack(arrs.clone(), 0);
        assert_eq!(
            res,
            Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])
        );

        let res = Array::stack(arrs, -1);
        assert_eq!(
            res,
            Array::from_vec(vec![1, 5, 2, 6, 3, 7, 4, 8], [2, 2, 2])
        );
    }

    #[test]
    fn test_pad() {
        let a = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);

        let res = a.pad((0, 1, 1, 0), 10);
        assert_eq!(
            res,
            Array::from_vec(vec![10, 1, 2, 10, 3, 4, 10, 10, 10], [3, 3])
        );
    }

    #[test]
    fn test_gather_scatter() {
        let a = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
        let indices = Array::from_vec(vec![0, 0, 1, 0], [2, 2]);

        let res = a.gather(1, &indices);
        assert_eq!(res, Array::from_vec(vec![1, 1, 4, 3], [2, 2]));

        let mut target = Array::<_, usize>::zeros([2, 3]);
        let indices = Array::from_vec(vec![0, -1, 1, 0], [2, 2]);
        let values = Array::from_vec(vec![10, 20, 30, 40], [2, 2]);
        target.scatter(-1, &indices, &values);

        assert_eq!(target, Array::from_vec(vec![10, 0, 20, 40, 30, 0], [2, 3]));

        let mut target = Array::<_, usize>::zeros([2, 3]);
        let indices = Array::from_vec(vec![0, -1, 1, 0], [2, 2]);
        let values = Array::from_vec(vec![10, 20, 30, 40], [2, 2]);
        target.scatter(0, &indices, &values);

        assert_eq!(target, Array::from_vec(vec![10, 40, 0, 30, 20, 0], [2, 3]));
    }

    #[test]
    fn test_reshape_and_transpose() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let arr = arr.reshape([3, 2]);
        assert_eq!(arr.shape(), [3, 2]);

        let arr = arr.transpose();
        assert_eq!(arr.shape(), [2, 3]);
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[1, 0]], 2.0);
    }

    #[test]
    fn test_reshape() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 1, 2]);

        let arr = arr.reshape([2, 2]);
        assert_eq!(arr.shape(), [2, 2]);
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[1, 1]], 4.0);
    }

    #[test]
    fn test_fmt() {
        println!("type name: {}", type_name::<String>());
        println!("type info: {}", pad_show_count::<String>());
        let a = Array::arange_c(1.0, 0.1, 100).reshape([-1, 10]);
        println!("a= {a:?}");
        let a = Array::<_, f32>::arange_c(1.0, 0.1, 1000).reshape([10, -1, 1, 10]);
        println!("a= {a:?}");

        let c = Array::arange_c(1, 1, 1000).reshape([20, 50]);
        println!("c= {c:?}");
    }

    #[test]
    fn test_map_axis() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]);
        println!("arr= {arr:?}");
        let arr = arr.map_axis(0, |x| {
            println!("x: {x:?}");
            2.0
        });
        println!("arr= {arr:?}");
        // assert_eq!(arr.shape(), [2, 2]);
        // assert_eq!(arr[[0, 0]], 2.0);
        // assert_eq!(arr[[1, 1]], 8.0);
    }
}
