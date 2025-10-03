use crate::{NumExt, core::Array};

impl<const D: usize, T: NumExt> Array<D, T> {
    /// Sum all elements
    pub fn sum(&self) -> T {
        self.multi_iter()
            .fold(T::zero(), |acc, (_, x)| acc + x.clone())
    }

    /// Sum along specified axis
    pub fn sum_axis(&self, axis: isize) -> Array<D, T> {
        self.map_axis(axis, |values| {
            values.into_iter().fold(T::zero(), |acc, x| acc + x.clone())
        })
    }
}

impl<const D: usize, T: NumExt> Array<D, T> {
    /// Calculate mean of all elements
    pub fn mean<U>(&self) -> U
    where
        U: NumExt,
    {
        let sum = self.sum();
        let size = self.size();

        U::from(sum).unwrap() / U::from(size).unwrap()
    }

    pub fn mean_axis<U>(&self, axis: isize) -> Array<D, U>
    where
        U: NumExt,
    {
        self.map_axis(axis, |values| {
            let len = values.len();
            let sum: T = values.into_iter().cloned().sum();

            U::from(sum).unwrap() / U::from(len).unwrap()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);

        let arr1 = arr.sum_axis(0);
        assert_eq!(
            arr1,
            Array::from_vec(vec![8, 10, 12, 14, 16, 18], [1, 3, 2])
        );
        let arr2 = arr.sum_axis(1);
        assert_eq!(arr2, Array::from_vec(vec![9, 12, 27, 30], [2, 1, 2]));
        let arr3 = arr.sum_axis(2);
        assert_eq!(arr3, Array::from_vec(vec![3, 7, 11, 15, 19, 23], [2, 3, 1]));

        let arr4 = arr.sum_axis(2);
        let arr4 = arr4.reshape([2, 3]);
        assert_eq!(arr4, Array::from_vec(vec![3, 7, 11, 15, 19, 23], [2, 3]));
        println!("arr= {arr:?} arr1= {arr1:?} arr2= {arr2:?} arr3= {arr3:?} arr4= {arr4:?}");
    }

    #[test]
    fn test_mean_axis() {
        let arr = Array::<_, f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let mean_axis_0 = arr.mean_axis(0);
        assert_eq!(mean_axis_0, Array::from_vec(vec![2.0, 3.0], [1, 2]));
        let mean_axis_1 = arr.mean_axis(1);
        assert_eq!(mean_axis_1, Array::from_vec(vec![1.5, 3.5], [2, 1]));

        let mean_axis_2 = arr.mean_axis(-1);
        assert_eq!(mean_axis_1, mean_axis_2);
    }

    #[test]
    fn test_aggregations() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        assert_eq!(arr.sum(), 10.0);
        assert_eq!(arr.mean::<f64>(), 2.5);
    }
}
