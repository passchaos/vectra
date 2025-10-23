use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use itertools::Itertools;

use crate::{
    core::Array,
    utils::{dyn_dim_to_static, negative_idx_to_positive},
};

pub trait SliceArg {
    fn out_indices(&self, gurad: usize) -> Vec<usize>;
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
            return vec![];
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
            return vec![];
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
            return vec![];
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
            return vec![];
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
            return vec![];
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
    pub fn slice_assign<R>(&mut self, slices: [RangeInclusive<isize>; D], values: &Self)
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
}
