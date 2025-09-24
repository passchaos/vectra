use itertools::Itertools;

use crate::core::{Array, compute_strides_for_shape};
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

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

// Arithmetic operations between arrays
impl<T> Add for Array<T>
where
    T: Add<Output = T> + Clone,
{
    type Output = Array<T>;

    fn add(self, other: Self) -> Self::Output {
        let broadcast_shape = Array::<T>::broadcast_shapes(&self.shape, &other.shape)
            .expect("Cannot broadcast arrays for addition");

        let self_broadcast = self
            .broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast first array");
        let other_broadcast = other
            .broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast second array");

        let result_data: Vec<T> = self_broadcast
            .data
            .iter()
            .zip(other_broadcast.data.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Array {
            data: result_data,
            shape: broadcast_shape.clone(),
            strides: compute_strides_for_shape(&broadcast_shape),
        }
    }
}

impl<T> Sub for Array<T>
where
    T: Sub<Output = T> + Clone,
{
    type Output = Array<T>;

    fn sub(self, other: Self) -> Self::Output {
        let broadcast_shape = Array::<T>::broadcast_shapes(&self.shape, &other.shape)
            .expect("Cannot broadcast arrays for subtraction");

        let self_broadcast = self
            .broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast first array");
        let other_broadcast = other
            .broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast second array");

        let result_data: Vec<T> = self_broadcast
            .data
            .iter()
            .zip(other_broadcast.data.iter())
            .map(|(a, b)| a.clone() - b.clone())
            .collect();

        Array {
            data: result_data,
            shape: broadcast_shape.clone(),
            strides: compute_strides_for_shape(&broadcast_shape),
        }
    }
}

impl<T> Mul for Array<T>
where
    T: Mul<Output = T> + Clone,
{
    type Output = Array<T>;

    fn mul(self, other: Self) -> Self::Output {
        let broadcast_shape = Array::<T>::broadcast_shapes(&self.shape, &other.shape)
            .expect("Cannot broadcast arrays for multiplication");

        let self_broadcast = self
            .broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast first array");
        let other_broadcast = other
            .broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast second array");

        let result_data: Vec<T> = self_broadcast
            .data
            .iter()
            .zip(other_broadcast.data.iter())
            .map(|(a, b)| a.clone() * b.clone())
            .collect();

        Array {
            data: result_data,
            shape: broadcast_shape.clone(),
            strides: compute_strides_for_shape(&broadcast_shape),
        }
    }
}

impl<T> Div for Array<T>
where
    T: Div<Output = T> + Clone,
{
    type Output = Array<T>;

    fn div(self, other: Self) -> Self::Output {
        let broadcast_shape = Array::<T>::broadcast_shapes(&self.shape, &other.shape)
            .expect("Cannot broadcast arrays for division");

        let self_broadcast = self
            .broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast first array");
        let other_broadcast = other
            .broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast second array");

        let result_data: Vec<T> = self_broadcast
            .data
            .iter()
            .zip(other_broadcast.data.iter())
            .map(|(a, b)| a.clone() / b.clone())
            .collect();

        Array {
            data: result_data,
            shape: broadcast_shape.clone(),
            strides: compute_strides_for_shape(&broadcast_shape),
        }
    }
}

// Scalar operations
impl<T> Array<T>
where
    T: Clone,
{
    /// Add scalar to all elements
    pub fn add_scalar(&self, scalar: T) -> Array<T>
    where
        T: Add<Output = T>,
    {
        Array {
            data: self
                .data
                .iter()
                .map(|x| x.clone() + scalar.clone())
                .collect(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Subtract scalar from all elements
    pub fn sub_scalar(&self, scalar: T) -> Array<T>
    where
        T: Sub<Output = T>,
    {
        Array {
            data: self
                .data
                .iter()
                .map(|x| x.clone() - scalar.clone())
                .collect(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Multiply all elements by scalar
    pub fn mul_scalar(&self, scalar: T) -> Array<T>
    where
        T: Mul<Output = T>,
    {
        Array {
            data: self
                .data
                .iter()
                .map(|x| x.clone() * scalar.clone())
                .collect(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Divide all elements by scalar
    pub fn div_scalar(&self, scalar: T) -> Array<T>
    where
        T: Div<Output = T>,
    {
        Array {
            data: self
                .data
                .iter()
                .map(|x| x.clone() / scalar.clone())
                .collect(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<T> Array<T> {
    pub fn multi_iter(&self) -> impl Iterator<Item = (Vec<usize>, &T)> {
        self.shape()
            .iter()
            .map(|&n| 0..n)
            .multi_cartesian_product()
            // .map(|idx| (idx.clone(), Index::index(self, idx.as_slice())))
            .map(|idx| (idx.clone(), self.index(&idx[..])))
    }
}
