use itertools::Itertools;
use num_traits::cast;

use crate::{
    NumExt,
    core::Array,
    utils::{broadcast_shapes, compute_strides, dyn_dim_to_static, negative_idx_to_positive},
};
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Neg, Not, Sub, SubAssign};

impl<T> Index<isize> for Array<1, T> {
    type Output = T;

    fn index(&self, index: isize) -> &Self::Output {
        let flat_index = self.index_to_flat([index]);
        &self.data[flat_index]
    }
}

impl<T> IndexMut<isize> for Array<1, T> {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        let flat_index = self.index_to_flat([index]);
        &mut self.data[flat_index]
    }
}

// macro_rules! impl_index {
//     ($($dim:literal),*) => {
//         $(
//             impl<T> Index<isize> for Array<$dim, T> {
//                 type Output = T;

//                 fn index(&self, index: isize) -> &Self::Output {
//                     let flat_index = self.index_to_flat([index]);
//                     &self.data[flat_index]
//                 }
//             }

//             impl<T> IndexMut<isize> for Array<$dim, T> {
//                 fn index_mut(&mut self, index: isize) -> &mut Self::Output {
//                     let flat_index = self.index_to_flat([index]);
//                     &mut self.data[flat_index]
//                 }
//             }
//         )*
//     };
// }

// impl_index!(2, 3, 4, 5, 6, 7, 8, 9, 10);

// impl<const D: usize, T> Index<isize> for Array<D, T>
// where
//     [(); D - 2]:,
// {
//     type Output = T;

//     fn index(&self, index: isize) -> &Self::Output {
//         let flat_index = self.index_to_flat([index]);
//         &self.data[flat_index]
//     }
// }

impl<const D: usize, T> Index<[isize; D]> for Array<D, T> {
    type Output = T;

    fn index(&self, indices: [isize; D]) -> &Self::Output {
        let flat_index = self.index_to_flat(indices);
        &self.data[flat_index]
    }
}

impl<const D: usize, T> IndexMut<[isize; D]> for Array<D, T> {
    fn index_mut(&mut self, indices: [isize; D]) -> &mut Self::Output {
        let flat_index = self.index_to_flat(indices);
        &mut self.data[flat_index]
    }
}

impl<const D: usize, T: Neg<Output = T> + Clone> Neg for Array<D, T> {
    type Output = Array<D, T>;

    fn neg(self) -> Self::Output {
        self.map_into(|x| -x)
    }
}

impl<const D: usize, T: Neg<Output = T> + Clone> Neg for &Array<D, T> {
    type Output = Array<D, T>;

    fn neg(self) -> Self::Output {
        self.map(|x| -x.clone())
    }
}

impl<const D: usize, T: Not<Output = T> + Clone> Not for Array<D, T> {
    type Output = Array<D, T>;

    fn not(self) -> Self::Output {
        self.map_into(|x| !x)
    }
}

// Arithmetic operations between arrays
impl<const D: usize, T: NumExt> AddAssign for Array<D, T> {
    fn add_assign(&mut self, rhs: Self) {
        let result = add_impl(self, &rhs);
        *self = result;
    }
}

impl<const D: usize, T: NumExt> AddAssign<&Array<D, T>> for Array<D, T> {
    fn add_assign(&mut self, rhs: &Array<D, T>) {
        let result = add_impl(self, rhs);
        *self = result;
    }
}

impl<const D: usize, T: NumExt> Add for &Array<D, T> {
    type Output = Array<D, T>;

    fn add(self, rhs: Self) -> Self::Output {
        add_impl(self, rhs)
    }
}

fn add_impl<const D: usize, T: NumExt>(left: &Array<D, T>, right: &Array<D, T>) -> Array<D, T> {
    let target_shape =
        broadcast_shapes(left.shape, right.shape).expect("Cannot broadcast arrays for add");

    let left = left.broadcast_to(target_shape);
    let right = right.broadcast_to(target_shape);

    let result_data = left
        .multi_iter()
        .zip(right.multi_iter())
        .map(|((_, a), (_, b))| a.clone() + b.clone())
        .collect();

    let major_order = crate::core::MajorOrder::RowMajor;
    let strides = compute_strides(left.shape, major_order);

    Array {
        data: result_data,
        shape: left.shape.clone(),
        strides,
        major_order,
    }
}

impl<const D: usize, T: NumExt> SubAssign for Array<D, T> {
    fn sub_assign(&mut self, rhs: Self) {
        let result = sub_impl(self, &rhs);
        *self = result;
    }
}

impl<const D: usize, T: NumExt> Sub for &Array<D, T> {
    type Output = Array<D, T>;

    fn sub(self, rhs: Self) -> Self::Output {
        sub_impl(self, rhs)
    }
}

fn sub_impl<const D: usize, T: NumExt>(left: &Array<D, T>, right: &Array<D, T>) -> Array<D, T> {
    let target_shape =
        broadcast_shapes(left.shape, right.shape).expect("Cannot broadcast arrays for sub");

    let left = left.broadcast_to(target_shape);
    let right = right.broadcast_to(target_shape);

    let result_data = left
        .multi_iter()
        .zip(right.multi_iter())
        .map(|((_, a), (_, b))| a.clone() - b.clone())
        .collect();

    let major_order = crate::core::MajorOrder::RowMajor;
    let strides = compute_strides(left.shape, major_order);

    Array {
        data: result_data,
        shape: left.shape.clone(),
        strides,
        major_order,
    }
}

impl<const D: usize, T: NumExt> MulAssign<&Array<D, T>> for Array<D, T> {
    fn mul_assign(&mut self, rhs: &Array<D, T>) {
        *self = mul_impl(&*self, rhs);
    }
}

impl<const D: usize, T: NumExt> Mul for &Array<D, T> {
    type Output = Array<D, T>;

    fn mul(self, rhs: Self) -> Self::Output {
        mul_impl(self, &rhs)
    }
}

fn mul_impl<const D: usize, T: NumExt>(left: &Array<D, T>, right: &Array<D, T>) -> Array<D, T> {
    let target_shape =
        broadcast_shapes(left.shape, right.shape).expect("Cannot broadcast arrays for mul");

    let left = left.broadcast_to(target_shape);
    let right = right.broadcast_to(target_shape);

    let result_data = left
        .multi_iter()
        .zip(right.multi_iter())
        .map(|((_, a), (_, b))| a.clone() * b.clone())
        .collect();

    let major_order = crate::core::MajorOrder::RowMajor;
    let strides = compute_strides(left.shape, major_order);

    Array {
        data: result_data,
        shape: left.shape.clone(),
        strides,
        major_order,
    }
}

impl<const D: usize, T: NumExt> Div for &Array<D, T> {
    type Output = Array<D, T>;

    fn div(self, rhs: Self) -> Self::Output {
        div_impl(self, rhs)
    }
}

fn div_impl<const D: usize, T: NumExt>(left: &Array<D, T>, right: &Array<D, T>) -> Array<D, T>
where
    T: Div<Output = T> + Clone,
{
    let target_shape =
        broadcast_shapes(left.shape, right.shape).expect("Cannot broadcast arrays for div");

    let left = left.broadcast_to(target_shape);
    let right = right.broadcast_to(target_shape);

    let result_data = left
        .multi_iter()
        .zip(right.multi_iter())
        .map(|((_, a), (_, b))| a.clone() / b.clone())
        .collect();

    let major_order = crate::core::MajorOrder::RowMajor;
    let strides = compute_strides(left.shape, major_order);

    Array {
        data: result_data,
        shape: left.shape.clone(),
        strides,
        major_order,
    }
}

// Scalar operations
impl<const D: usize, T: NumExt> Array<D, T> {
    /// Add scalar to all elements
    pub fn add_scalar(mut self, scalar: T) -> Self {
        self.data.iter_mut().for_each(|x| *x += scalar.clone());
        self
    }

    /// Subtract scalar from all elements
    pub fn sub_scalar(mut self, scalar: T) -> Self {
        self.data.iter_mut().for_each(|x| *x -= scalar.clone());
        self
    }

    /// Multiply all elements by scalar
    pub fn mul_scalar(mut self, scalar: T) -> Self {
        self.data.iter_mut().for_each(|x| *x *= scalar.clone());
        self
    }

    /// Divide all elements by scalar
    pub fn div_scalar(mut self, scalar: T) -> Self {
        self.data.iter_mut().for_each(|x| *x /= scalar.clone());
        self
    }
}

impl<const D: usize, T: NumExt> Array<D, T> {
    pub fn one_hot<F: NumExt>(&self, num_classes: usize) -> Array<{ D + 1 }, F> {
        self.one_hot_fill(num_classes, F::one(), F::zero(), -1)
    }

    pub fn one_hot_fill<F>(
        &self,
        num_classes: usize,
        on_value: F,
        off_value: F,
        axis: isize,
    ) -> Array<{ D + 1 }, F>
    where
        F: NumExt,
    {
        let indices = self
            .map(|v| {
                let value = cast::<_, isize>(v.clone()).expect("Failed to cast T to usize");

                if value < -(num_classes as isize) || value >= num_classes as isize {
                    panic!("self value as index meet failure, contains values greater than or equal to num_classes");
                }

                negative_idx_to_positive(value, num_classes)
            })
            .unsqueeze(axis);

        let axis = negative_idx_to_positive(axis, D + 1);

        let mut shape = self.shape().to_vec();
        shape.insert(axis, num_classes);
        let mut target_shape = [0; D + 1];
        target_shape.copy_from_slice(&shape);

        let mut result = Array::full(target_shape, off_value);

        for (mut idx, value) in indices.multi_iter() {
            idx[axis] = *value;

            let raw_idx = result.positive_index_to_flat(idx);
            result.data[raw_idx] = on_value.clone();
        }

        result
    }
}

impl<const D: usize, T> Array<D, T> {
    pub fn select(&self, axis: isize, indices: &[usize]) -> Array<D, T>
    where
        T: Clone,
    {
        let axis = negative_idx_to_positive(axis, D);
        let axis_size = self.shape()[axis];

        let mut non_axis_shape = self.shape().to_vec();
        non_axis_shape.remove(axis);

        let indices: Vec<_> = indices
            .into_iter()
            .filter(|&idx| idx < &axis_size)
            .cloned()
            .collect();
        let mut products: Vec<_> = non_axis_shape
            .iter()
            .map(|&n| 0..n)
            .map(|v| v.into_iter().collect::<Vec<_>>())
            .collect();
        // let indices: Vec<_> = indices.into_iter().sorted().cloned().collect();
        non_axis_shape.insert(axis, indices.len());

        products.insert(axis, indices);

        let data: Vec<_> = products
            .into_iter()
            .multi_cartesian_product()
            .map(|idx| {
                let idx = dyn_dim_to_static::<D>(&idx).map(|a| a as isize);
                self[idx].clone()
            })
            .collect();

        let mut shape = [0; D];
        shape.copy_from_slice(&non_axis_shape);
        Array::from_vec(data, shape)
    }
    pub fn multi_iter(&self) -> impl Iterator<Item = ([usize; D], &T)> {
        self.shape
            .into_iter()
            .map(|n| 0..n)
            .multi_cartesian_product()
            .map(|idx| {
                let idx = dyn_dim_to_static::<D>(&idx);

                let data = self.index(idx.map(|i| i as isize));
                (idx, data)
            })
    }

    pub fn multi_iter_mut<F>(&mut self, mut f: F)
    where
        F: FnMut([usize; D], &mut T),
    {
        for idx in self
            .shape()
            .into_iter()
            .map(|n| 0..n)
            .multi_cartesian_product()
        {
            let idx = dyn_dim_to_static(&idx);
            let data = self.index_mut(idx.map(|i| i as isize));

            f(idx, data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_iter_mut() {
        let arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
        let mut iter = arr.multi_iter();
        assert_eq!(iter.next(), Some(([0, 0], &1)));
        assert_eq!(iter.next(), Some(([0, 1], &2)));
        assert_eq!(iter.next(), Some(([1, 0], &3)));
        assert_eq!(iter.next(), Some(([1, 1], &4)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_arithmetic() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let b = Array::from_vec(vec![2.0, 2.0, 2.0, 2.0], [2, 2]);

        let sum = &a + &b;
        assert_eq!(sum[[0, 0]], 3.0);
        assert_eq!(sum[[1, 1]], 6.0);
        assert_eq!(sum[[-1, -1]], 6.0);

        let product = &a * &b;
        assert_eq!(product[[0, 0]], 2.0);
        assert_eq!(product[[1, 1]], 8.0);

        let c = Array::from_vec(vec![1.0, 2.0], [1, 2]);
        let d = &a + &c;
        println!("d= {d}");
    }

    #[test]
    fn test_one_hot() {
        // let a = cast::<_, usize>(2.2);
        // println!("a: {a:?}");
        let arr = Array::from_vec(vec![1, 2, 2, 4], [4]);
        let one_hot = arr.one_hot::<f32>(5);

        println!("arr= {arr:?} one_hot= {one_hot:?}");

        assert_eq!(
            one_hot,
            Array::from_vec(
                vec![
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0
                ],
                [4, 5]
            )
        );

        let indices = Array::from_vec(vec![0, 2, 1, -1], [2, 2]);
        let res = indices.one_hot_fill(3, 5.0, 0.0, -1);
        println!("res: {res:?}");
        assert_eq!(
            res,
            Array::from_vec(
                vec![5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0],
                [2, 2, 3]
            )
        );
    }

    #[test]
    fn test_select() {
        let a = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
        assert_eq!(a.select(0, &[0]), Array::from_vec(vec![1, 2], [1, 2]));
        assert_eq!(a.select(-1, &[0]), Array::from_vec(vec![1, 3], [2, 1]));
        assert_eq!(a.select(-1, &[0, 10]), Array::from_vec(vec![1, 3], [2, 1]));
    }
}
