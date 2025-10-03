use crate::{NumExt, core::Array};

impl<const D: usize, T: NumExt> Array<D, T> {
    pub fn min(&self) -> T {
        self.multi_iter()
            .min_by(|(_, a), (_, b)| a.cmp_ext(b))
            .map(|(_, x)| x.clone())
            .unwrap()
    }

    pub fn argmin(&self) -> [usize; D] {
        self.multi_iter()
            .min_by(|(_, a), (_, b)| a.cmp_ext(b))
            .map(|(i, _)| i)
            .unwrap()
    }

    pub fn min_axis(&self, axis: isize) -> Self {
        self.map_axis(axis, |v| {
            v.into_iter().min_by(|a, b| a.cmp_ext(b)).cloned().unwrap()
        })
    }

    pub fn argmin_axis(&self, axis: isize) -> Array<D, usize> {
        self.map_axis(axis, |v| {
            v.into_iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.cmp_ext(b))
                .map(|(i, _)| i)
                .unwrap()
        })
    }

    pub fn max(&self) -> T {
        self.multi_iter()
            .max_by(|(_, a), (_, b)| a.cmp_ext(b))
            .map(|(_, x)| x.clone())
            .unwrap()
    }

    pub fn argmax(&self) -> [usize; D] {
        self.multi_iter()
            .max_by(|(_, a), (_, b)| a.cmp_ext(b))
            .map(|(i, _)| i)
            .unwrap()
    }

    pub fn max_axis(&self, axis: isize) -> Self {
        self.map_axis(axis, |v| {
            v.into_iter().max_by(|a, b| a.cmp_ext(b)).cloned().unwrap()
        })
    }

    pub fn argmax_axis(&self, axis: isize) -> Array<D, usize> {
        self.map_axis(axis, |v| {
            v.into_iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.cmp_ext(b))
                .map(|(i, _)| i)
                .unwrap()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        assert_eq!(arr.max(), 4.0);
        assert_eq!(arr.min(), 1.0);

        let arr_min_a = arr.min_axis(-1);
        let arr_max_a = arr.max_axis(0);
        assert_eq!(arr_min_a, Array::from_vec(vec![1.0, 3.0], [2, 1]));
        assert_eq!(arr_max_a, Array::from_vec(vec![3.0, 4.0], [1, 2]));
    }

    #[test]
    fn test_int() {
        let arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
        assert_eq!(arr.max(), 4);
        assert_eq!(arr.min(), 1);

        let arr_min_a = arr.min_axis(-1);
        let arr_max_a = arr.max_axis(0);
        assert_eq!(arr_min_a, Array::from_vec(vec![1, 3], [2, 1]));
        assert_eq!(arr_max_a, Array::from_vec(vec![3, 4], [1, 2]));
    }
}
