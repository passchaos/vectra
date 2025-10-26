//! Machine learning activation functions and operations.
//!
//! This module provides common activation functions used in neural networks
//! and machine learning applications.

use std::cmp::Ordering;

use num_traits::{Float, Zero};

use crate::{CmpExt, NumExt, core::Array};

impl<const D: usize, T: Zero + Clone + CmpExt> Array<D, T> {
    /// Apply the ReLU (Rectified Linear Unit) activation function element-wise.
    ///
    /// ReLU(x) = max(0, x)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let arr = Array::from_vec(vec![-1, 0, 1, 2], [2, 2]);
    /// let result = arr.relu();
    /// ```
    pub fn relu(&self) -> Self {
        self.map(|x| {
            let zero = T::zero();

            match x.cmp_ext(&zero) {
                Ordering::Greater => x.clone(),
                _ => zero,
            }
        })
    }
}

impl<const D: usize> Array<D, f32> {
    /// Apply the GELU (Gaussian Error Linear Unit) activation function element-wise.
    ///
    /// GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let arr = Array::from_vec(vec![-1.0, 0.0, 1.0, 2.0], [2, 2]);
    /// let result = arr.gelu();
    /// ```
    pub fn gelu(&self) -> Self {
        self.map(|x| 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh()))
    }
}

impl<const D: usize> Array<D, f64> {
    /// Apply the GELU (Gaussian Error Linear Unit) activation function element-wise.
    ///
    /// GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let arr = Array::from_vec(vec![-1.0, 0.0, 1.0, 2.0], [2, 2]);
    /// let result = arr.gelu();
    /// ```
    pub fn gelu(&self) -> Self {
        self.map(|x| 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh()))
    }
}

impl<const D: usize, T: NumExt + Float> Array<D, T> {
    /// Apply the sigmoid activation function element-wise.
    ///
    /// sigmoid(x) = 1 / (1 + e^(-x))
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let arr = Array::from_vec(vec![-1.0, 0.0, 1.0, 2.0], [2, 2]);
    /// let result = arr.sigmoid();
    /// ```
    pub fn sigmoid(&self) -> Self {
        (-self).exp().add_scalar(T::one()).recip()
    }

    /// Apply the softmax function along the last dimension.
    ///
    /// softmax(x_i) = e^(x_i - max(x)) / Σ(e^(x_j - max(x)))
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    /// let result = arr.softmax();
    /// ```
    pub fn softmax(&self) -> Array<D, T> {
        let a = self.max_axis((D - 1) as isize);
        let a = (self - &a).exp();
        let a_t = a.sum_axis((D - 1) as isize);

        &a / &a_t
    }

    /// Apply RMS (Root Mean Square) normalization along the last dimension.
    ///
    /// RMS_norm(x) = x / sqrt(mean(x²) + eps)
    ///
    /// # Arguments
    ///
    /// * `eps` - Small epsilon value to prevent division by zero
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectra::prelude::*;
    ///
    /// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    /// let result = arr.rms_norm(1e-8);
    /// ```
    pub fn rms_norm(&self, eps: T) -> Self {
        let a = self.map(|x| x.powi(2));
        let a_t = a.sum_axis((D - 1) as isize);

        &a / &a_t.sqrt().add_scalar(eps)
    }
}
