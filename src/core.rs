use std::any::type_name;
use std::fmt::{self, Debug};
use std::ops::{Index, IndexMut, RangeInclusive};

use approx::{AbsDiffEq, RelativeEq};
use faer::{Mat, MatRef};
use itertools::Itertools;
use num_traits::{Float, NumCast};

use crate::NumExt;
use crate::utils::{
    compute_strides, dyn_dim_to_static, flat_idx_to_indices, indices_to_flat_idx,
    negative_idx_to_positive, negative_indices_to_positive, shape_indices_to_flat_idx,
};

#[derive(Debug, Default, Clone, Copy, PartialEq)]
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

impl<T: NumExt + Debug> Array<2, T> {
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

    /// Create identity matrix
    pub fn eye(n: usize) -> Self {
        let mut arr = Self::zeros([n, n]);
        for i in 0..n as isize {
            arr[[i, i]] = T::one();
        }
        arr
    }
}

impl<T> Array<2, T> {
    /// Transpose array (2D only)
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

    /// Create array like np.arange
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

    /// Reshape array
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

    pub fn unsqueeze(self, axis: isize) -> Array<{ D + 1 }, T> {
        let axis = negative_idx_to_positive(axis, D + 1);

        let mut shape = self.shape().map(|v| v as isize).to_vec();
        shape.insert(axis, 1);

        let mut new_shape = [0; D + 1];
        new_shape.copy_from_slice(&shape);

        self.reshape(new_shape)
    }

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

    pub fn pad(&self, padding: (usize, usize, usize, usize), value: T) -> Self
    where
        T: Clone,
    {
        let (top, bottom, left, right) = padding;

        let mut padded_shape = self.shape();
        padded_shape[D - 2] += top + bottom;
        padded_shape[D - 1] += left + right;

        let ranges = padded_shape
            .iter()
            .enumerate()
            .map(|(i, &dim)| {
                if i == D - 2 {
                    top..=(dim - bottom - 1)
                } else if i == D - 1 {
                    left..=(dim - right - 1)
                } else {
                    0..=(dim - 1)
                }
            })
            .map(|a| {
                let (s, e) = a.into_inner();
                s as isize..=e as isize
            })
            .collect::<Vec<_>>();

        let slices = ranges.try_into().unwrap();

        let mut padded_tensor = Self::full(padded_shape, value);

        padded_tensor.slice_assign(slices, self);
        padded_tensor
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
            let item = self.index_mut(idx.map(|i| i as isize));
            *item = f(item);
        }

        self
    }

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

    pub fn slice_assign(&mut self, slices: [RangeInclusive<isize>; D], values: &Self)
    where
        T: Clone,
    {
        let self_shape = self.shape();
        let value_shape = values.shape();

        let slices = self_shape
            .iter()
            .zip(slices.iter())
            .zip(value_shape.iter())
            .map(|((a, b), c)| {
                let a_i = *a as isize;

                let (&start, &end) = (b.start(), b.end());

                assert!(start >= -a_i && start < a_i);
                assert!(end >= -a_i && end < a_i);

                let start = if start < 0 {
                    (a_i + start) as usize
                } else {
                    start as usize
                };

                let end = if end < 0 {
                    (a_i + end) as usize
                } else {
                    end as usize
                };

                assert!(end >= start);
                assert!(end - start + 1 == *c as usize);

                start..=end
            });

        let slices_begin = slices.clone().map(|r| *r.start());

        let slices_index = slices.multi_cartesian_product();

        for idx in slices_index {
            let value_idx: Vec<_> = idx
                .iter()
                .zip(slices_begin.clone())
                .map(|(a, b)| a - b)
                .collect();

            let self_idx = dyn_dim_to_static::<D>(&idx).map(|i| i as isize);
            let value_idx = dyn_dim_to_static::<D>(&value_idx).map(|i| i as isize);

            self[self_idx] = values[value_idx].clone();
        }
    }

    pub fn slice_fill(&mut self, slices: [RangeInclusive<isize>; D], value: T)
    where
        T: Clone,
    {
        let self_shape = self.shape();

        let slices = self_shape.iter().zip(slices.iter()).map(|(a, b)| {
            let a_i = *a as isize;

            let (&start, &end) = (b.start(), b.end());

            assert!(start >= -a_i && start < a_i);
            assert!(end >= -a_i && end < a_i);

            let start = if start < 0 {
                (a_i + start) as usize
            } else {
                start as usize
            };

            let end = if end < 0 {
                (a_i + end) as usize
            } else {
                end as usize
            };

            assert!(end >= start);

            start..=end
        });

        let slices_index = slices.multi_cartesian_product();

        for idx in slices_index {
            let self_idx = dyn_dim_to_static::<D>(&idx).map(|i| i as isize);

            self[self_idx] = value.clone();
        }
    }

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
impl<const D: usize, T: NumExt> Array<D, T> {
    /// Create zero array
    pub fn zeros(shape: [usize; D]) -> Self {
        Self::full(shape, T::zero())
    }

    /// Create ones array
    pub fn ones(shape: [usize; D]) -> Self {
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
        T: Float,
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
    fn test_slice_assign() {
        let mut arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);

        arr.slice_assign([1..=1, 0..=-1], &Array::from_vec(vec![9.0, 10.0], [1, 2]));

        assert_eq!(arr, Array::from_vec(vec![1.0, 2.0, 9.0, 10.0], [2, 2]));

        arr.slice_assign([0..=-1, 0..=-1], &Array::full([2, 2], 12.0));
        assert_eq!(arr, Array::full([2, 2], 12.0));

        arr.slice_fill([0..=-1, 0..=-1], 13.0);
        assert_eq!(arr, Array::full([2, 2], 13.0));
    }
}
