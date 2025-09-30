use itertools::Itertools;

use crate::{core::Array, utils::compute_strides};
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

// Indexing implementations
impl<T> Index<&[usize]> for Array<T> {
    type Output = T;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        let flat_index = self.index_to_flat(indices).expect("Invalid index");
        &self.data[flat_index]
    }
}

impl<T> IndexMut<&[usize]> for Array<T> {
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        let flat_index = self.index_to_flat(indices).expect("Invalid index");
        &mut self.data[flat_index]
    }
}

impl<T, const N: usize> Index<[usize; N]> for Array<T> {
    type Output = T;

    fn index(&self, indices: [usize; N]) -> &Self::Output {
        &self[&indices[..]]
    }
}

impl<T, const N: usize> IndexMut<[usize; N]> for Array<T> {
    fn index_mut(&mut self, indices: [usize; N]) -> &mut Self::Output {
        &mut self[&indices[..]]
    }
}

impl<T> Index<Vec<usize>> for Array<T> {
    type Output = T;
    fn index(&self, indices: Vec<usize>) -> &Self::Output {
        let flat_index = self.index_to_flat(&indices).expect("Invalid index");
        &self.data[flat_index]
    }
}

impl<T> IndexMut<Vec<usize>> for Array<T> {
    fn index_mut(&mut self, indices: Vec<usize>) -> &mut Self::Output {
        let flat_index = self.index_to_flat(&indices).expect("Invalid index");
        &mut self.data[flat_index]
    }
}

// Arithmetic operations between arrays

impl<T> Add for Array<T>
where
    T: Add<Output = T> + Clone,
{
    type Output = Array<T>;

    fn add(self, rhs: Self) -> Self::Output {
        add_impl(&self, &rhs)
    }
}

impl<T> AddAssign for Array<T>
where
    T: Add<Output = T> + Clone,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = add_impl(self, &rhs);
    }
}

impl<T> Add for &Array<T>
where
    T: Add<Output = T> + Clone,
{
    type Output = Array<T>;

    fn add(self, rhs: Self) -> Self::Output {
        add_impl(self, rhs)
    }
}

fn add_impl<T>(left: &Array<T>, right: &Array<T>) -> Array<T>
where
    T: Add<Output = T> + Clone,
{
    let broadcast_shape = Array::<T>::broadcast_shapes(&left.shape, &right.shape)
        .expect("Cannot broadcast arrays for addition");

    let self_broadcast = left
        .broadcast_to(&broadcast_shape)
        .expect("Failed to broadcast first array");
    let other_broadcast = right
        .broadcast_to(&broadcast_shape)
        .expect("Failed to broadcast second array");

    let result_data = self_broadcast
        .multi_iter()
        .zip(other_broadcast.multi_iter())
        .map(|((_, a), (_, b))| a.clone() + b.clone())
        .collect();

    let major_order = crate::core::MajorOrder::RowMajor;
    let strides = compute_strides(&broadcast_shape, major_order);

    Array {
        data: result_data,
        shape: broadcast_shape.clone(),
        strides,
        major_order,
    }
}

impl<T> Sub for Array<T>
where
    T: Sub<Output = T> + Clone,
{
    type Output = Array<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        sub_impl(&self, &rhs)
    }
}

impl<T> Sub for &Array<T>
where
    T: Sub<Output = T> + Clone,
{
    type Output = Array<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        sub_impl(self, rhs)
    }
}

fn sub_impl<T>(left: &Array<T>, right: &Array<T>) -> Array<T>
where
    T: Sub<Output = T> + Clone,
{
    let broadcast_shape = Array::<T>::broadcast_shapes(&left.shape, &right.shape)
        .expect("Cannot broadcast arrays for subtraction");

    let self_broadcast = left
        .broadcast_to(&broadcast_shape)
        .expect("Failed to broadcast first array");
    let other_broadcast = right
        .broadcast_to(&broadcast_shape)
        .expect("Failed to broadcast second array");

    let result_data = self_broadcast
        .multi_iter()
        .zip(other_broadcast.multi_iter())
        .map(|((_, a), (_, b))| a.clone() - b.clone())
        .collect();

    let major_order = crate::core::MajorOrder::RowMajor;
    let strides = compute_strides(&broadcast_shape, major_order);

    Array {
        data: result_data,
        shape: broadcast_shape.clone(),
        strides,
        major_order,
    }
}

impl<T> Mul for Array<T>
where
    T: Mul<Output = T> + Clone,
{
    type Output = Array<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        mul_impl(&self, &rhs)
    }
}

impl<T> Mul for &Array<T>
where
    T: Mul<Output = T> + Clone,
{
    type Output = Array<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        mul_impl(self, &rhs)
    }
}

fn mul_impl<T>(left: &Array<T>, right: &Array<T>) -> Array<T>
where
    T: Mul<Output = T> + Clone,
{
    let broadcast_shape = Array::<T>::broadcast_shapes(&left.shape, &right.shape)
        .expect("Cannot broadcast arrays for multiplication");

    let self_broadcast = left
        .broadcast_to(&broadcast_shape)
        .expect("Failed to broadcast first array");
    let other_broadcast = right
        .broadcast_to(&broadcast_shape)
        .expect("Failed to broadcast second array");

    let result_data = self_broadcast
        .multi_iter()
        .zip(other_broadcast.multi_iter())
        .map(|((_, a), (_, b))| a.clone() * b.clone())
        .collect();

    let major_order = crate::core::MajorOrder::RowMajor;
    let strides = compute_strides(&broadcast_shape, major_order);

    Array {
        data: result_data,
        shape: broadcast_shape.clone(),
        strides,
        major_order,
    }
}

impl<T> Div for Array<T>
where
    T: Div<Output = T> + Clone,
{
    type Output = Array<T>;

    fn div(self, rhs: Self) -> Self::Output {
        div_impl(&self, &rhs)
    }
}

impl<T> Div for &Array<T>
where
    T: Div<Output = T> + Clone,
{
    type Output = Array<T>;

    fn div(self, rhs: Self) -> Self::Output {
        div_impl(self, rhs)
    }
}

fn div_impl<T>(left: &Array<T>, right: &Array<T>) -> Array<T>
where
    T: Div<Output = T> + Clone,
{
    let broadcast_shape = Array::<T>::broadcast_shapes(&left.shape, &right.shape)
        .expect("Cannot broadcast arrays for division");

    let self_broadcast = left
        .broadcast_to(&broadcast_shape)
        .expect("Failed to broadcast first array");
    let other_broadcast = right
        .broadcast_to(&broadcast_shape)
        .expect("Failed to broadcast second array");

    let result_data = self_broadcast
        .multi_iter()
        .zip(other_broadcast.multi_iter())
        .map(|((_, a), (_, b))| a.clone() / b.clone())
        .collect();

    let major_order = crate::core::MajorOrder::RowMajor;
    let strides = compute_strides(&broadcast_shape, major_order);

    Array {
        data: result_data,
        shape: broadcast_shape.clone(),
        strides,
        major_order,
    }
}

// Scalar operations
impl<T> Array<T>
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

impl<T> Array<T> {
    pub fn multi_iter(&self) -> impl Iterator<Item = (Vec<usize>, &T)> {
        self.shape()
            .into_iter()
            .map(|&n| 0..n)
            .multi_cartesian_product()
            .map(|idx| (idx.clone(), self.index(idx.as_slice())))
    }

    pub fn multi_iter_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(Vec<usize>, &mut T),
    {
        for idx in self
            .shape()
            .into_iter()
            .map(|&n| 0..n)
            .multi_cartesian_product()
        {
            let data = self.index_mut(idx.as_slice());
            f(idx, data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_iter_mut() {
        let arr = Array::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let mut iter = arr.multi_iter();
        assert_eq!(iter.next(), Some((vec![0, 0], &1)));
        assert_eq!(iter.next(), Some((vec![0, 1], &2)));
        assert_eq!(iter.next(), Some((vec![1, 0], &3)));
        assert_eq!(iter.next(), Some((vec![1, 1], &4)));
        assert_eq!(iter.next(), None);
    }
}
