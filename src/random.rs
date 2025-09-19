use rand::Rng;
use crate::core::Array;

impl<T> Array<T>
where
    T: Clone + Default,
{
    /// Create array with random values between 0 and 1
    pub fn random(shape: Vec<usize>) -> Self
    where
        T: From<f64> + Clone + Default,
    {
        let size = shape.iter().product();
        let mut rng = rand::rng();
        let data: Vec<T> = (0..size)
            .map(|_| T::from(rng.random_range(0.0..1.0)))
            .collect();
        let strides = Array::<T>::compute_strides_for_shape(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create array with random integers in range [low, high)
    pub fn randint(shape: Vec<usize>, low: i32, high: i32) -> Self
    where
        T: From<i32> + Clone + Default,
    {
        let size = shape.iter().product();
        let mut rng = rand::rng();
        let data: Vec<T> = (0..size)
            .map(|_| T::from(rng.random_range(low..high)))
            .collect();
        let strides = Array::<T>::compute_strides_for_shape(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create array with normally distributed random values (Box-Muller transform)
    pub fn randn(shape: Vec<usize>) -> Self
    where
        T: From<f64> + Clone + Default,
    {
        let size = shape.iter().product();
        let strides = Array::<T>::compute_strides_for_shape(&shape);
        let mut rng = rand::rng();
        
        let data: Vec<T> = (0..size).map(|_| {
            // Box-Muller transform
            let u1: f64 = rng.random_range(f64::EPSILON..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            T::from(z0)
        }).collect();
        
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create array with uniformly distributed random values in range [low, high)
    pub fn uniform(shape: Vec<usize>, low: f64, high: f64) -> Self
    where
        T: From<f64> + Clone + Default,
    {
        let size = shape.iter().product();
        let mut rng = rand::rng();
        let data: Vec<T> = (0..size)
            .map(|_| T::from(rng.random_range(low..high)))
            .collect();
        let strides = Array::<T>::compute_strides_for_shape(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }
}