use num_traits::{Float, Zero};

use crate::{NumExt, core::Array};

impl<const D: usize, T: Zero + Ord + Clone> Array<D, T> {
    pub fn relu(&self) -> Self {
        self.map(|x| x.max(&T::zero()).clone())
    }
}

impl<const D: usize> Array<D, f32> {
    pub fn gelu(&self) -> Self {
        self.map(|x| 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh()))
    }
}

impl<const D: usize> Array<D, f64> {
    pub fn gelu(&self) -> Self {
        self.map(|x| 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh()))
    }
}

impl<const D: usize, T: NumExt + Float> Array<D, T> {
    pub fn sigmoid(&self) -> Self {
        (-self).exp().add_scalar(T::one()).recip()
    }

    pub fn softmax(&self) -> Array<D, T> {
        let a = self.max_axis((D - 1) as isize);
        let a = (self - &a).exp();
        let a_t = a.sum_axis((D - 1) as isize);

        &a / &a_t
    }
}
