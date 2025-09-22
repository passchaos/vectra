use crate::core::{Array, compute_strides_for_shape};
use num_traits::{One, Zero};
use rand::distr::{Distribution, Uniform, uniform::SampleUniform};
use rand_distr::StandardNormal;

impl<T> Array<T> {
    /// Create array with random values between 0 and 1
    pub fn random(shape: Vec<usize>) -> Self
    where
        T: SampleUniform + One + Zero,
    {
        Self::uniform(shape, T::zero(), T::one())
    }

    pub fn randn(shape: Vec<usize>) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let size = shape.iter().product();
        let strides = compute_strides_for_shape(&shape);
        let mut rng = rand::rng();

        let form = StandardNormal;

        let data = form.sample_iter(&mut rng).take(size).collect();

        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create array with uniformly distributed random values in range [low, high)
    pub fn uniform(shape: Vec<usize>, low: T, high: T) -> Self
    where
        T: SampleUniform,
    {
        let size = shape.iter().product();

        let form = Uniform::new(low, high).unwrap();
        let mut rng = rand::rng();

        let data = form.sample_iter(&mut rng).take(size).collect();
        // let data: Vec<_> = (0..size).map(|_| rng.random_range(0.0..1.0)).collect();
        let strides = compute_strides_for_shape(&shape);

        Self {
            data,
            shape,
            strides,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand() {
        let arr: Array<i32> = Array::random(vec![2, 3]);
        println!("{arr}");
        assert_eq!(arr.shape(), vec![2, 3]);

        let arr1 = Array::<f32>::randn(vec![5, 6]);
        println!("{arr1}");
    }
}
