use std::ops::{Index, IndexMut, Add, Sub, Mul, Div};
use crate::core::Array;

// Indexing implementations
impl<T> Index<&[usize]> for Array<T>
where
    T: Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        let flat_index = self.index_to_flat(indices).expect("Invalid index");
        &self.data[flat_index]
    }
}

impl<T> IndexMut<&[usize]> for Array<T>
where
    T: Clone,
{
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        let flat_index = self.index_to_flat(indices).expect("Invalid index");
        &mut self.data[flat_index]
    }
}

impl<T> Index<[usize; 1]> for Array<T>
where
    T: Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 1]) -> &Self::Output {
        &self[&indices[..]]
    }
}

impl<T> IndexMut<[usize; 1]> for Array<T>
where
    T: Clone,
{
    fn index_mut(&mut self, indices: [usize; 1]) -> &mut Self::Output {
        &mut self[&indices[..]]
    }
}

impl<T> Index<[usize; 2]> for Array<T>
where
    T: Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 2]) -> &Self::Output {
        &self[&indices[..]]
    }
}

impl<T> IndexMut<[usize; 2]> for Array<T>
where
    T: Clone,
{
    fn index_mut(&mut self, indices: [usize; 2]) -> &mut Self::Output {
        &mut self[&indices[..]]
    }
}

impl<T> Index<[usize; 3]> for Array<T>
where
    T: Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 3]) -> &Self::Output {
        &self[&indices[..]]
    }
}

impl<T> IndexMut<[usize; 3]> for Array<T>
where
    T: Clone,
{
    fn index_mut(&mut self, indices: [usize; 3]) -> &mut Self::Output {
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
        
        let self_broadcast = self.broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast first array");
        let other_broadcast = other.broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast second array");
        
        let result_data: Vec<T> = self_broadcast.data.iter()
            .zip(other_broadcast.data.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        
        Array {
            data: result_data,
            shape: broadcast_shape.clone(),
            strides: Array::<T>::compute_strides_for_shape(&broadcast_shape),
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
        
        let self_broadcast = self.broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast first array");
        let other_broadcast = other.broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast second array");
        
        let result_data: Vec<T> = self_broadcast.data.iter()
            .zip(other_broadcast.data.iter())
            .map(|(a, b)| a.clone() - b.clone())
            .collect();
        
        Array {
            data: result_data,
            shape: broadcast_shape.clone(),
            strides: Array::<T>::compute_strides_for_shape(&broadcast_shape),
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
        
        let self_broadcast = self.broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast first array");
        let other_broadcast = other.broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast second array");
        
        let result_data: Vec<T> = self_broadcast.data.iter()
            .zip(other_broadcast.data.iter())
            .map(|(a, b)| a.clone() * b.clone())
            .collect();
        
        Array {
            data: result_data,
            shape: broadcast_shape.clone(),
            strides: Array::<T>::compute_strides_for_shape(&broadcast_shape),
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
        
        let self_broadcast = self.broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast first array");
        let other_broadcast = other.broadcast_to(&broadcast_shape)
            .expect("Failed to broadcast second array");
        
        let result_data: Vec<T> = self_broadcast.data.iter()
            .zip(other_broadcast.data.iter())
            .map(|(a, b)| a.clone() / b.clone())
            .collect();
        
        Array {
            data: result_data,
            shape: broadcast_shape.clone(),
            strides: Array::<T>::compute_strides_for_shape(&broadcast_shape),
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
            data: self.data.iter().map(|x| x.clone() + scalar.clone()).collect(),
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
            data: self.data.iter().map(|x| x.clone() - scalar.clone()).collect(),
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
            data: self.data.iter().map(|x| x.clone() * scalar.clone()).collect(),
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
            data: self.data.iter().map(|x| x.clone() / scalar.clone()).collect(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}