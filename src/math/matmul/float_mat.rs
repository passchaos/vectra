use crate::{
    core::Array,
    math::{MatmulPolicy, matmul::Matmul},
};

use std::ffi::c_char;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vfmaq_f32, vfmaq_f64, vld1q_dup_f32, vld1q_dup_f64, vld1q_f32, vld1q_f64, vst1q_f32, vst1q_f64,
};

impl Matmul for Array<f32> {
    fn matmul(&self, rhs: &Self, policy: MatmulPolicy) -> Result<Self, String> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 {
            return Err("Matrix multiplication only supported for 2D arrays".to_string());
        }
        if self.shape[1] != rhs.shape[0] {
            return Err("Matrix dimensions incompatible for multiplication".to_string());
        }

        match policy {
            MatmulPolicy::Faer => {
                let l = self.as_faer();
                let r = rhs.as_faer();

                Ok((l * r).into())
            }
            _ => {
                let m = self.shape[0];
                let k = self.shape[1];
                let n = rhs.shape[1];
                let l_s_c = self.strides[0];
                let r_s_c = rhs.strides[0];

                let result_shape = vec![m, n];
                let mut result_data = vec![0.0; result_shape.iter().product()];

                match policy {
                    #[cfg(target_arch = "aarch64")]
                    MatmulPolicy::LoopRecorderSimd => {
                        for i in 0..m {
                            for l in 0..k {
                                // In practice, using get_unchecked directly only reduces runtime by 1%
                                // let a_v = unsafe { self.data.get_unchecked(i * k + l) };
                                // let a_v = self.data[i * l_s_c + l];
                                let a_v = unsafe { vld1q_dup_f32(&self.data[i * l_s_c + l]) };

                                let mut j = 0;
                                while j + 3 < n {
                                    unsafe {
                                        let b_vec = vld1q_f32(&rhs.data[l * r_s_c + j]);
                                        let c_vec = vld1q_f32(&result_data[i * n + j]);
                                        let res_vec = vfmaq_f32(c_vec, a_v, b_vec);
                                        vst1q_f32(&mut result_data[i * n + j], res_vec);
                                    }
                                    j += 4;
                                }

                                for j in j..n {
                                    result_data[i * n + j] +=
                                        self.data[i * l_s_c + l] * rhs.data[l * r_s_c + j].clone();
                                }
                            }
                        }
                    }
                    MatmulPolicy::Blas => {
                        let a = self.data.as_ptr();
                        let b = rhs.data.as_ptr();
                        let c = result_data.as_mut_ptr();

                        let alpha = 1.0;
                        let beta = 0.0;

                        let m = m as i32;
                        let n = n as i32;
                        let k = k as i32;

                        unsafe {
                            blas_sys::sgemm_(
                                &(b'N' as c_char),
                                &(b'N' as c_char),
                                &n,
                                &m,
                                &k,
                                &alpha,
                                b.cast(),
                                &n,
                                a.cast(),
                                &k,
                                &beta,
                                c.cast(),
                                &n,
                            );
                        }
                    }
                    _ => return super::matmul_general(self, rhs, policy),
                }

                Array::from_vec(result_data, result_shape)
            }
        }
    }
}

impl Matmul for Array<f64> {
    fn matmul(&self, rhs: &Self, policy: MatmulPolicy) -> Result<Self, String> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 {
            return Err("Matrix multiplication only supported for 2D arrays".to_string());
        }
        if self.shape[1] != rhs.shape[0] {
            return Err("Matrix dimensions incompatible for multiplication".to_string());
        }

        match policy {
            MatmulPolicy::Faer => {
                let l = self.as_faer();
                let r = rhs.as_faer();

                Ok((l * r).into())
            }
            _ => {
                let m = self.shape[0];
                let k = self.shape[1];
                let n = rhs.shape[1];
                let l_s_c = self.strides[0];
                let r_s_c = rhs.strides[0];

                let result_shape = vec![m, n];
                let mut result_data = vec![0.0; result_shape.iter().product()];

                match policy {
                    #[cfg(target_arch = "aarch64")]
                    MatmulPolicy::LoopRecorderSimd => {
                        for i in 0..m {
                            for l in 0..k {
                                // In practice, using get_unchecked directly only reduces runtime by 1%
                                // let a_v = unsafe { self.data.get_unchecked(i * k + l) };
                                // let a_v = self.data[i * l_s_c + l];

                                // 因为一次读取2个位置数据，所以每行的结尾需要避免越界
                                let mut j = 0;

                                let a_v = unsafe { vld1q_dup_f64(&self.data[i * l_s_c + l]) };

                                while j + 1 < n {
                                    unsafe {
                                        let b_vec = vld1q_f64(&rhs.data[l * r_s_c + j]);
                                        let c_vec = vld1q_f64(&result_data[i * n + j]);

                                        let res_vec = vfmaq_f64(c_vec, a_v, b_vec);
                                        vst1q_f64(&mut result_data[i * n + j], res_vec);
                                    }
                                    j += 2;
                                }

                                for j in j..n {
                                    result_data[i * n + j] +=
                                        self.data[i * l_s_c + l] * rhs.data[l * r_s_c + j].clone();
                                }
                            }
                        }
                    }
                    MatmulPolicy::Blas => {
                        let m = m as i32;
                        let k = k as i32;
                        let n = n as i32;

                        let a = self.data.as_ptr();
                        let b = rhs.data.as_ptr();
                        let c = result_data.as_mut_ptr();

                        let alpha = 1.0;
                        let beta = 0.0;

                        unsafe {
                            blas_sys::dgemm_(
                                &(b'N' as c_char),
                                &(b'N' as c_char),
                                &n,
                                &m,
                                &k,
                                &alpha,
                                b.cast(),
                                &n,
                                a.cast(),
                                &k,
                                &beta,
                                c.cast(),
                                &n,
                            );
                        }
                    }
                    _ => return super::matmul_general(self, rhs, policy),
                }

                Array::from_vec(result_data, result_shape)
            }
        }
    }
}
