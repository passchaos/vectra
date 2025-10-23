use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use itertools::Itertools;

use crate::{
    core::Array,
    utils::{dyn_dim_to_static, negative_idx_to_positive},
};

pub trait SliceArg {
    fn out_indices(&self, gurad: usize) -> Vec<usize>;
}

impl<const D: usize> SliceArg for [isize; D] {
    fn out_indices(&self, guard: usize) -> Vec<usize> {
        let mut indices = Vec::new();

        for &index in self {
            let index = negative_idx_to_positive(index, guard);

            indices.push(index);
        }

        indices
    }
}

impl SliceArg for &[isize] {
    fn out_indices(&self, guard: usize) -> Vec<usize> {
        let mut indices = Vec::new();

        for &index in *self {
            let index = negative_idx_to_positive(index, guard);

            indices.push(index);
        }

        indices
    }
}

impl SliceArg for Vec<isize> {
    fn out_indices(&self, guard: usize) -> Vec<usize> {
        let mut indices = Vec::new();

        for &index in self {
            let index = negative_idx_to_positive(index, guard);

            indices.push(index);
        }

        indices
    }
}

impl SliceArg for RangeFull {
    fn out_indices(&self, guard: usize) -> Vec<usize> {
        (0..guard).collect()
    }
}

impl SliceArg for RangeToInclusive<isize> {
    fn out_indices(&self, guard: usize) -> Vec<usize> {
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
    fn out_indices(&self, guard: usize) -> Vec<usize> {
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
    fn out_indices(&self, guard: usize) -> Vec<usize> {
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
    fn out_indices(&self, guard: usize) -> Vec<usize> {
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
    fn out_indices(&self, guard: usize) -> Vec<usize> {
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
    pub fn slice_assign<S: SliceArg>(&mut self, slices: [S; D], values: &Self)
    where
        T: Clone,
    {
        let self_shape = self.shape();
        let value_shape = values.shape();

        let slices: Vec<_> = self_shape
            .iter()
            .zip(slices.iter())
            .zip(value_shape.iter())
            .map(|((&a, b), &c)| {
                let indices = b.out_indices(a);
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

            let self_idx = dyn_dim_to_static::<D>(&idx).map(|i| i as isize);
            let value_idx = dyn_dim_to_static::<D>(&value_idx).map(|i| i as isize);

            self[self_idx] = values[value_idx].clone();
        }
    }

    pub fn slice_fill<S: SliceArg>(&mut self, slices: [S; D], value: T)
    where
        T: Clone,
    {
        let self_shape = self.shape();

        let slices = self_shape
            .iter()
            .zip(slices.iter())
            .map(|(&a, b)| b.out_indices(a));

        let slices_index = slices.multi_cartesian_product();

        for idx in slices_index {
            let self_idx = dyn_dim_to_static::<D>(&idx).map(|i| i as isize);

            self[self_idx] = value.clone();
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }
}
