use crate::{core::Array, utils::compute_strides};
use num_traits::{One, Zero};
use rand::distr::{Distribution, Uniform, uniform::SampleUniform};
use rand_distr::StandardNormal;

impl<const D: usize, T> Array<D, T> {
    /// Create array with random values between 0 and 1
    pub fn random(shape: [usize; D]) -> Self
    where
        T: SampleUniform + One + Zero,
    {
        Self::uniform(shape, T::zero(), T::one())
    }

    pub fn randn(shape: [usize; D]) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let size = shape.iter().product();

        let major_order = crate::core::MajorOrder::RowMajor;
        let strides = compute_strides(shape, major_order);

        let mut rng = rand::rng();

        let form = StandardNormal;

        let data = form.sample_iter(&mut rng).take(size).collect();

        Self {
            data,
            shape,
            strides,
            major_order,
        }
    }

    /// Create array with uniformly distributed random values in range [low, high)
    pub fn uniform(shape: [usize; D], low: T, high: T) -> Self
    where
        T: SampleUniform,
    {
        let size = shape.iter().product();

        let form = Uniform::new(low, high).unwrap();
        let mut rng = rand::rng();

        let data = form.sample_iter(&mut rng).take(size).collect();
        // let data: Vec<_> = (0..size).map(|_| rng.random_range(0.0..1.0)).collect();
        //
        let major_order = crate::core::MajorOrder::RowMajor;
        let strides = compute_strides(shape, major_order);

        Self {
            data,
            shape,
            strides,
            major_order,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand() {
        let arr: Array<2, i32> = Array::random([2, 3]);
        println!("{arr}");
        assert_eq!(arr.shape(), [2, 3]);

        let arr1 = Array::<_, f32>::randn([5, 6]);
        println!("{arr1}");
    }
}
