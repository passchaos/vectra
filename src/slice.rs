use std::{
    fmt::Debug,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

use itertools::Itertools;
use num_traits::Zero;

use crate::{
    core::Array,
    utils::{dyn_dim_to_static, negative_idx_to_positive},
};

/// Trait for types that can be used as slice arguments.
///
/// This trait is implemented for various range types and arrays of indices,
/// allowing flexible slicing operations on multi-dimensional arrays.
pub trait SliceArg {
    /// Convert the slice argument to a vector of indices for the given dimension size.
    fn op_indices(&self, guard: usize) -> Vec<usize>;
}

/// Enum wrapper for different types of slice arguments.
///
/// This enum allows for dynamic dispatch of slice operations when the slice type
/// is not known at compile time.
#[derive(Debug)]
pub enum SliceArgKind {
    Array(Vec<isize>),
    Range(Range<isize>),
    RangeFrom(RangeFrom<isize>),
    RangeFull(RangeFull),
    RangeInclusive(RangeInclusive<isize>),
    RangeTo(RangeTo<isize>),
    RangeToInclusive(RangeToInclusive<isize>),
}

impl SliceArg for SliceArgKind {
    fn op_indices(&self, guard: usize) -> Vec<usize> {
        match self {
            SliceArgKind::Array(indices) => indices.op_indices(guard),
            SliceArgKind::Range(range) => range.op_indices(guard),
            SliceArgKind::RangeFrom(range) => range.op_indices(guard),
            SliceArgKind::RangeFull(range) => range.op_indices(guard),
            SliceArgKind::RangeInclusive(range) => range.op_indices(guard),
            SliceArgKind::RangeTo(range) => range.op_indices(guard),
            SliceArgKind::RangeToInclusive(range) => range.op_indices(guard),
        }
    }
}

impl<const D: usize> From<[isize; D]> for SliceArgKind {
    fn from(value: [isize; D]) -> Self {
        SliceArgKind::Array(value.to_vec())
    }
}

impl From<Vec<isize>> for SliceArgKind {
    fn from(value: Vec<isize>) -> Self {
        SliceArgKind::Array(value)
    }
}

impl From<Range<isize>> for SliceArgKind {
    fn from(value: Range<isize>) -> Self {
        SliceArgKind::Range(value)
    }
}

impl From<RangeFrom<isize>> for SliceArgKind {
    fn from(value: RangeFrom<isize>) -> Self {
        SliceArgKind::RangeFrom(value)
    }
}

impl From<RangeTo<isize>> for SliceArgKind {
    fn from(value: RangeTo<isize>) -> Self {
        SliceArgKind::RangeTo(value)
    }
}

impl From<RangeToInclusive<isize>> for SliceArgKind {
    fn from(value: RangeToInclusive<isize>) -> Self {
        SliceArgKind::RangeToInclusive(value)
    }
}

impl From<RangeInclusive<isize>> for SliceArgKind {
    fn from(value: RangeInclusive<isize>) -> Self {
        SliceArgKind::RangeInclusive(value)
    }
}

impl From<RangeFull> for SliceArgKind {
    fn from(value: RangeFull) -> Self {
        SliceArgKind::RangeFull(value)
    }
}

impl<const D: usize> SliceArg for [isize; D] {
    fn op_indices(&self, guard: usize) -> Vec<usize> {
        let mut indices = Vec::new();

        for &index in self {
            let index = negative_idx_to_positive(index, guard);

            indices.push(index);
        }

        indices
    }
}

impl SliceArg for Vec<isize> {
    fn op_indices(&self, guard: usize) -> Vec<usize> {
        let mut indices = Vec::new();

        for &index in self {
            let index = negative_idx_to_positive(index, guard);

            indices.push(index);
        }

        indices
    }
}

impl SliceArg for RangeFull {
    fn op_indices(&self, guard: usize) -> Vec<usize> {
        (0..guard).collect()
    }
}

impl SliceArg for RangeToInclusive<isize> {
    fn op_indices(&self, guard: usize) -> Vec<usize> {
        let end = self.end;

        let guard = guard as isize;

        if end < -guard || end >= guard {
            panic!("slice index must in {}..{}", -guard, guard);
        }

        let end = if end < 0 {
            (end + guard) as usize
        } else {
            end as usize
        };

        (0..=end).collect()
    }
}

impl SliceArg for RangeTo<isize> {
    fn op_indices(&self, guard: usize) -> Vec<usize> {
        let end = self.end;

        let guard = guard as isize;

        if end <= -guard || end > guard {
            panic!("slice index must in {}..={}", -guard + 1, guard)
        }

        let end = if end < 0 {
            (end + guard) as usize
        } else {
            end as usize
        };

        (0..end).collect()
    }
}

impl SliceArg for RangeFrom<isize> {
    fn op_indices(&self, guard: usize) -> Vec<usize> {
        let start = self.start;

        let guard = guard as isize;

        if start < -guard || start >= guard {
            panic!("slice index must in {}..", guard);
        }

        let start = if start < 0 {
            (start + guard) as usize
        } else {
            start as usize
        };

        (start..guard as usize).collect()
    }
}

impl SliceArg for Range<isize> {
    fn op_indices(&self, guard: usize) -> Vec<usize> {
        let (start, end) = (self.start, self.end);

        let guard = guard as isize;

        if start < -guard || start >= guard || end <= -guard || end > guard {
            panic!(
                "slice start index must be in {}..{} end index must be in {}..{}",
                -guard,
                guard - 1,
                -guard + 1,
                guard
            );
        }

        let start = if start < 0 {
            (start + guard) as usize
        } else {
            start as usize
        };

        let end = if end < 0 {
            (end + guard) as usize
        } else {
            end as usize
        };

        if start >= end {
            return vec![];
        }

        (start..end).collect()
    }
}

impl SliceArg for RangeInclusive<isize> {
    fn op_indices(&self, guard: usize) -> Vec<usize> {
        let (&start, &end) = (self.start(), self.end());

        let guard = guard as isize;

        if start < -guard || start >= guard || end < -guard || end >= guard {
            panic!(
                "slice start index must be in {}..{} end index must be in {}..{}",
                -guard,
                guard - 1,
                -guard,
                guard - 1
            );
        }

        let start = if start < 0 {
            (start + guard) as usize
        } else {
            start as usize
        };

        let end = if end < 0 {
            (end + guard) as usize
        } else {
            end as usize
        };

        if start > end {
            return vec![];
        }

        (start..=end).collect()
    }
}

impl<const D: usize, T> Array<D, T> {
    /// Extract a slice from the array using the specified slice arguments.
    ///
    /// # Arguments
    ///
    /// * `slices` - Array of slice arguments, one for each dimension
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6], [2, 3]);
    /// let slice = arr.slice([0..1, 1..3]);
    /// ```
    pub fn slice<S: SliceArg>(&self, slices: [S; D]) -> Array<D, T>
    where
        T: Clone + Zero,
    {
        let self_shape = self.shape();

        let slices: Vec<_> = self_shape
            .iter()
            .zip(slices.into_iter())
            .map(|(&a, b)| b.op_indices(a))
            .collect();
        let slice_shape: Vec<_> = slices.iter().map(|s| s.len()).collect();

        let mut arr = Self::zeros(dyn_dim_to_static(&slice_shape));

        let slices_index = slices.iter().cloned().multi_cartesian_product();

        for idx in slices_index {
            let value_idx: Vec<_> = idx
                .iter()
                .zip(slices.iter())
                .map(|(idx_bit, slice)| {
                    let bit_idx = slice.iter().position(|x| x == idx_bit).unwrap();
                    bit_idx
                })
                .collect();

            let value_idx = dyn_dim_to_static(&value_idx).map(|i| i as isize);
            let idx = dyn_dim_to_static(&idx).map(|i| i as isize);

            arr[value_idx] = self[idx].clone();
        }

        arr
    }

    /// Assign values to a slice of the array.
    ///
    /// # Arguments
    ///
    /// * `slices` - Array of slice arguments specifying the target region
    /// * `values` - Array containing the values to assign
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let mut arr = Array::zeros([3, 3]);
    /// let values = Array::ones([2, 2]);
    /// arr.slice_assign([0..2, 0..2], &values);
    /// ```
    pub fn slice_assign<S: SliceArg>(&mut self, slices: [S; D], values: &Self)
    where
        T: Clone,
    {
        let self_shape = self.shape();
        let value_shape = values.shape();

        let slices: Vec<_> = self_shape
            .iter()
            .zip(slices.into_iter())
            .zip(value_shape.iter())
            .map(|((&a, b), &c)| {
                let indices = b.op_indices(a);
                assert_eq!(indices.len(), c);

                indices
            })
            .collect();

        let slices_index = slices.clone().into_iter().multi_cartesian_product();

        for idx in slices_index {
            let value_idx: Vec<_> = idx
                .iter()
                .zip(slices.iter())
                .map(|(idx_bit, slice)| {
                    let bit_idx = slice.iter().position(|x| x == idx_bit).unwrap();
                    bit_idx
                })
                .collect();

            let self_idx = dyn_dim_to_static::<D, _>(&idx).map(|i| i as isize);
            let value_idx = dyn_dim_to_static::<D, _>(&value_idx).map(|i| i as isize);

            self[self_idx] = values[value_idx].clone();
        }
    }

    /// Fill a slice of the array with a single value.
    ///
    /// # Arguments
    ///
    /// * `slices` - Array of slice arguments specifying the target region
    /// * `value` - Value to fill the slice with
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let mut arr = Array::zeros([3, 3]);
    /// arr.slice_fill([0..2, 0..2], 42);
    /// ```
    pub fn slice_fill<S: SliceArg + Debug>(&mut self, slices: [S; D], value: T)
    where
        T: Clone,
    {
        let self_shape = self.shape();

        let slices = self_shape
            .iter()
            .zip(slices.iter())
            .map(|(&a, b)| b.op_indices(a));

        let slices_index = slices.multi_cartesian_product();

        for idx in slices_index {
            let self_idx = dyn_dim_to_static::<D, _>(&idx).map(|i| i as isize);

            self[self_idx] = value.clone();
        }
    }

    /// Pad the array with the specified value.
    ///
    /// # Arguments
    ///
    /// * `padding` - Tuple of (top, bottom, left, right) padding amounts
    /// * `value` - Value to use for padding
    ///
    /// # Returns
    ///
    /// A new array with the specified padding applied.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
    /// let padded = arr.pad((1, 1, 1, 1), 0);
    /// ```
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_get() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let slice = arr.slice([1..=1, 0..=-1]);
        assert_eq!(slice, Array::from_vec(vec![3.0, 4.0], [1, 2]));

        let arr3 = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]);
        let slice = arr3.slice::<SliceArgKind>([vec![0, 2].into(), (1..=2).into()]);
        assert_eq!(slice, Array::from_vec(vec![2.0, 3.0, 8.0, 9.0], [2, 2]));
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

        arr.slice_fill([[0], [0]], 20.0);
        assert_eq!(arr, Array::from_vec(vec![20.0, 13.0, 13.0, 13.0], [2, 2]));

        arr.slice_fill([vec![0], vec![0, 1]], 20.0);
        assert_eq!(arr, Array::from_vec(vec![20.0, 20.0, 13.0, 13.0], [2, 2]));

        let arr1 = Array::from_vec(vec![1.1, 1.2, 2.1, 2.2], [2, 2]);
        arr.slice_assign([.., ..], &arr1);
        assert_eq!(arr, arr1);

        arr.slice_fill(
            [
                SliceArgKind::Array(vec![0, -2]),
                SliceArgKind::RangeFull(..),
            ],
            3.1415,
        );
        println!("arr: {arr:?}");
    }
}
