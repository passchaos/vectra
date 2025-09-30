use std::fmt::Debug;

use num_traits::NumAssign;

use crate::{core::Array, math::MatmulPolicy};
mod float_mat;

pub trait Matmul: Sized {
    fn matmul(&self, rhs: &Self, policy: MatmulPolicy) -> Self;
}

macro_rules! impl_matmul_for_type {
    ($($ty:ty),*) => {
        $(
        impl Matmul for Array<2, $ty> {
            fn matmul(&self, rhs: &Self, policy: MatmulPolicy) -> Self {
                matmul_general(self, rhs, policy)
            }
        }
        )*
    };
}
impl_matmul_for_type!(
    u8, i8, u16, i16, u32, i32, u64, i64, usize, isize, u128, i128
);

fn matmul_general<T>(lhs: &Array<2, T>, rhs: &Array<2, T>, policy: MatmulPolicy) -> Array<2, T>
where
    T: NumAssign + Clone + Debug,
{
    let result_shape = [lhs.shape[0], rhs.shape[1]];
    let mut result_data = vec![T::zero(); result_shape.iter().product()];

    let m = lhs.shape[0];
    let k = lhs.shape[1];
    let n = rhs.shape[1];

    let l_s_c = lhs.strides[0];
    let r_s_c = rhs.strides[0];

    match policy {
        MatmulPolicy::Naive => {
            for i in 0..m {
                for j in 0..n {
                    for l in 0..k {
                        result_data[i * n + j] +=
                            lhs.data[i * l_s_c + l].clone() * rhs.data[l * r_s_c + j].clone();
                    }
                }
            }
        }
        MatmulPolicy::Blocking(block_size) => {
            for i in (0..m).step_by(block_size) {
                let i_end = (i + block_size).min(m);

                for l in (0..k).step_by(block_size) {
                    let l_end = (l + block_size).min(k);

                    for j in (0..n).step_by(block_size) {
                        let j_end = (j + block_size).min(n);

                        for ii in i..i_end {
                            for ll in l..l_end {
                                let a_v = &lhs.data[ii * l_s_c + ll];
                                for jj in j..j_end {
                                    result_data[ii * n + jj] +=
                                        a_v.clone() * rhs.data[ll * r_s_c + jj].clone();
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => {
            for i in 0..m {
                for l in 0..k {
                    // In practice, using get_unchecked directly only reduces runtime by 1%
                    // let a_v = unsafe { self.data.get_unchecked(i * k + l) };
                    let a_v = &lhs.data[i * l_s_c + l];
                    for j in 0..n {
                        // let data = unsafe { result_data.get_unchecked_mut(i * n + j) };
                        // *data += a_v.clone()
                        //     * unsafe { other.data.get_unchecked(l * n + j) }.clone();
                        result_data[i * n + j] += a_v.clone() * rhs.data[l * r_s_c + j].clone();
                    }
                }
            }
        }
    }

    Array::from_vec(result_data, result_shape)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_matmul_naive() {
        for policy in [
            MatmulPolicy::Naive,
            MatmulPolicy::Blocking(64),
            MatmulPolicy::Blas,
            MatmulPolicy::Faer,
            MatmulPolicy::LoopReorder,
            #[cfg(target_arch = "aarch64")]
            MatmulPolicy::LoopRecorderSimd,
        ] {
            let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
            let b = Array::from_vec(vec![4.0, 5.0, 6.0, 7.0], [2, 2]);
            println!("begin matmul f64: {policy:?}");
            let expected = Array::from_vec(vec![16.0, 19.0, 36.0, 43.0], [2, 2]);
            let result = a.matmul(&b, policy);
            assert_relative_eq!(result, expected);

            let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
            let b = Array::from_vec(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 2]);
            println!("begin matmul f32: {policy:?}");
            let expected = Array::from_vec(vec![40.0, 46.0, 94.0, 109.0], [2, 2]);
            let result = a.matmul(&b, policy);
            assert_relative_eq!(result, expected);
        }
    }

    #[test]
    fn test_matmul() {
        let shapes: Vec<(usize, usize, usize)> = vec![
            (50, 50, 50),
            (50, 75, 60),
            (100, 110, 120),
            // // (220, 230, 250),
            // (250, 250, 250),
            // (300, 300, 300),
            (300, 400, 350),
            (1000, 1000, 1000),
            // (1000, 1200, 1100),
        ];

        let inputs_64: Vec<_> = shapes
            .iter()
            .map(|(m, n, k)| {
                (
                    Array::<_, f64>::randn([*m, *k]),
                    Array::<_, f64>::randn([*k, *n]),
                )
            })
            .collect();

        let results_64: Vec<_> = inputs_64
            .iter()
            .map(|(l, r)| l.matmul(r, MatmulPolicy::Naive))
            .collect();

        let inputs_32: Vec<_> = shapes
            .iter()
            .map(|(m, n, k)| {
                (
                    Array::<_, f32>::randn([*m, *k]),
                    Array::<_, f32>::randn([*k, *n]),
                )
            })
            .collect();
        let results_32: Vec<_> = inputs_32
            .iter()
            .map(|(l, r)| l.matmul(r, MatmulPolicy::Naive))
            .collect();

        for policy in [
            MatmulPolicy::Blas,
            MatmulPolicy::Faer,
            MatmulPolicy::LoopReorder,
            #[cfg(target_arch = "aarch64")]
            MatmulPolicy::LoopRecorderSimd,
            MatmulPolicy::Blocking(512),
        ] {
            println!("begin matmul check: {policy:?}");
            for ((l, r), res) in inputs_64.iter().zip(results_64.iter()) {
                let new_res = l.matmul(r, policy);
                assert_relative_eq!(*res, new_res);
            }

            for ((l, r), res) in inputs_32.iter().zip(results_32.iter()) {
                let new_res = l.matmul(r, policy);
                assert_relative_eq!(*res, new_res);
            }
        }
    }
}
