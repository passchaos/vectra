use itertools::Itertools;
use num_traits::{One, PrimInt, Zero, cast};

use crate::{
    core::Array,
    utils::{broadcast_shapes, compute_strides, dyn_dim_to_static, negative_idx_to_positive},
};
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

// // Indexing implementations
// impl<const D: usize, T> Index<&[usize]> for Array<D, T> {
//     type Output = T;

//     fn index(&self, indices: &[usize]) -> &Self::Output {
//         let flat_index = self.index_to_flat(indices).expect("Invalid index");
//         &self.data[flat_index]
//     }
// }

// impl<const D: usize, T> IndexMut<&[usize]> for Array<D, T> {
//     fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
//         let flat_index = self.index_to_flat(indices).expect("Invalid index");
//         &mut self.data[flat_index]
//     }
// }

// impl<const D: usize, T> Index<[usize; D]> for Array<D, T> {
//     type Output = T;

//     fn index(&self, indices: [usize; D]) -> &Self::Output {
//         let flat_index = self.index_to_flat(indices);
//         &self.data[flat_index]
//     }
// }

// impl<const D: usize, T> IndexMut<[usize; D]> for Array<D, T> {
//     fn index_mut(&mut self, indices: [usize; D]) -> &mut Self::Output {
//         let flat_index = self.index_to_flat(indices);
//         &mut self.data[flat_index]
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

// Arithmetic operations between arrays
impl<const D: usize, T> AddAssign for Array<D, T>
where
    T: Add<Output = T> + Clone,
{
    fn add_assign(&mut self, rhs: Self) {
        let result = add_impl(self, &rhs);
        *self = result;
    }
}

impl<const D: usize, T> AddAssign<&Array<D, T>> for Array<D, T>
where
    T: Add<Output = T> + Clone,
{
    fn add_assign(&mut self, rhs: &Array<D, T>) {
        let result = add_impl(self, rhs);
        *self = result;
    }
}

impl<const D: usize, T> Add for &Array<D, T>
where
    T: Add<Output = T> + Clone,
{
    type Output = Array<D, T>;

    fn add(self, rhs: Self) -> Self::Output {
        add_impl(self, rhs)
    }
}

fn add_impl<const D: usize, T>(left: &Array<D, T>, right: &Array<D, T>) -> Array<D, T>
where
    T: Add<Output = T> + Clone,
{
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

impl<const D: usize, T> SubAssign for Array<D, T>
where
    T: Sub<Output = T> + Clone,
{
    fn sub_assign(&mut self, rhs: Self) {
        let result = sub_impl(self, &rhs);
        *self = result;
    }
}

impl<const D: usize, T> Sub for &Array<D, T>
where
    T: Sub<Output = T> + Clone,
{
    type Output = Array<D, T>;

    fn sub(self, rhs: Self) -> Self::Output {
        sub_impl(self, rhs)
    }
}

fn sub_impl<const D: usize, T>(left: &Array<D, T>, right: &Array<D, T>) -> Array<D, T>
where
    T: Sub<Output = T> + Clone,
{
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

impl<const D: usize, T> MulAssign<&Array<D, T>> for Array<D, T>
where
    T: Mul<Output = T> + Clone,
{
    fn mul_assign(&mut self, rhs: &Array<D, T>) {
        *self = mul_impl(&*self, rhs);
    }
}

impl<const D: usize, T> Mul for &Array<D, T>
where
    T: Mul<Output = T> + Clone,
{
    type Output = Array<D, T>;

    fn mul(self, rhs: Self) -> Self::Output {
        mul_impl(self, &rhs)
    }
}

fn mul_impl<const D: usize, T>(left: &Array<D, T>, right: &Array<D, T>) -> Array<D, T>
where
    T: Mul<Output = T> + Clone,
{
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

impl<const D: usize, T> Div for &Array<D, T>
where
    T: Div<Output = T> + Clone,
{
    type Output = Array<D, T>;

    fn div(self, rhs: Self) -> Self::Output {
        div_impl(self, rhs)
    }
}

fn div_impl<const D: usize, T>(left: &Array<D, T>, right: &Array<D, T>) -> Array<D, T>
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
impl<const D: usize, T> Array<D, T>
where
    T: Clone,
{
    /// Add scalar to all elements
    pub fn add_scalar(mut self, scalar: T) -> Self
    where
        T: AddAssign,
    {
        self.data.iter_mut().for_each(|x| *x += scalar.clone());
        self
    }

    /// Subtract scalar from all elements
    pub fn sub_scalar(mut self, scalar: T) -> Self
    where
        T: SubAssign,
    {
        self.data.iter_mut().for_each(|x| *x -= scalar.clone());
        self
    }

    /// Multiply all elements by scalar
    pub fn mul_scalar(mut self, scalar: T) -> Self
    where
        T: MulAssign,
    {
        self.data.iter_mut().for_each(|x| *x *= scalar.clone());
        self
    }

    /// Divide all elements by scalar
    pub fn div_scalar(mut self, scalar: T) -> Self
    where
        T: DivAssign,
    {
        self.data.iter_mut().for_each(|x| *x /= scalar.clone());
        self
    }
}

impl<const D: usize, T: PrimInt> Array<D, T> {
    pub fn one_hot<F>(&self, num_classes: usize) -> Array<{ D + 1 }, F>
    where
        F: Clone + Zero + One,
        T: Debug,
    {
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
        F: Clone,
        T: Debug,
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
}
