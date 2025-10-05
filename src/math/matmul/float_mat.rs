use crate::{
    core::{Array, MajorOrder},
    math::{MatmulPolicy, matmul::Matmul},
};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vfmaq_f32, vfmaq_f64, vld1q_dup_f32, vld1q_dup_f64, vld1q_f32, vld1q_f64, vst1q_f32, vst1q_f64,
};

impl Matmul for Array<2, f32> {
    fn matmul(&self, rhs: &Self, policy: MatmulPolicy) -> Self {
        assert_eq!(self.shape()[1], rhs.shape()[0]);

        match policy {
            MatmulPolicy::Faer => {
                let l = self.as_faer();
                let r = rhs.as_faer();

                (l * r).into()
            }
            _ => {
                let m = self.shape[0];
                let k = self.shape[1];
                let n = rhs.shape[1];

                let l_s_r = self.strides[0];
                let l_s_c = self.strides[1];
                let r_s_r = rhs.strides[0];
                let r_s_c = rhs.strides[1];

                let result_shape = [m, n];
                let mut result_data = vec![0.0; result_shape.iter().product()];

                match policy {
                    #[cfg(target_arch = "aarch64")]
                    MatmulPolicy::LoopRecorderSimd => {
                        for i in 0..m {
                            for l in 0..k {
                                // In practice, using get_unchecked directly only reduces runtime by 1%
                                // let a_v = unsafe { self.data.get_unchecked(i * k + l) };
                                // let a_v = self.data[i * l_s_c + l];
                                let a_v =
                                    unsafe { vld1q_dup_f32(&self.data[i * l_s_r + l * l_s_c]) };

                                let mut j = 0;
                                while j + 3 < n {
                                    unsafe {
                                        let b_vec = vld1q_f32(&rhs.data[l * r_s_r + j * r_s_c]);
                                        let c_vec = vld1q_f32(&result_data[i * n + j]);
                                        let res_vec = vfmaq_f32(c_vec, a_v, b_vec);
                                        vst1q_f32(&mut result_data[i * n + j], res_vec);
                                    }
                                    j += 4;
                                }

                                for j in j..n {
                                    result_data[i * n + j] += self.data[i * l_s_r + l * l_s_c]
                                        * rhs.data[l * r_s_r + j * r_s_c].clone();
                                }
                            }
                        }
                    }
                    MatmulPolicy::Blas => {
                        let (m, n, a, b) = match self.major_order {
                            MajorOrder::RowMajor => {
                                let m = n as i32;
                                let n = m as i32;

                                let a = rhs.data.as_ptr();
                                let b = self.data.as_ptr();

                                (m, n, a, b)
                            }
                            MajorOrder::ColumnMajor => {
                                let m = m as i32;
                                let n = n as i32;

                                let a = self.data.as_ptr();
                                let b = rhs.data.as_ptr();

                                (m, n, a, b)
                            }
                        };

                        let k = k as i32;
                        let c = result_data.as_mut_ptr();

                        let alpha = 1.0;
                        let beta = 0.0;

                        unsafe {
                            cblas_sys::cblas_sgemm(
                                cblas_sys::CBLAS_LAYOUT::CblasRowMajor,
                                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                                m,
                                n,
                                k,
                                alpha,
                                a.cast(),
                                m,
                                b.cast(),
                                k,
                                beta,
                                c.cast(),
                                n,
                            );
                        }
                    }
                    _ => return super::matmul_general(self, rhs, policy),
                }

                Array::from_vec_major(result_data, result_shape, self.major_order)
            }
        }
    }
}

impl Matmul for Array<2, f64> {
    fn matmul(&self, rhs: &Self, policy: MatmulPolicy) -> Self {
        assert_eq!(self.shape()[1], rhs.shape()[0]);

        match policy {
            MatmulPolicy::Faer => {
                let l = self.as_faer();
                let r = rhs.as_faer();

                (l * r).into()
            }
            _ => {
                let m = self.shape[0];
                let k = self.shape[1];
                let n = rhs.shape[1];

                let l_s_r = self.strides[0];
                let l_s_c = self.strides[1];
                let r_s_r = rhs.strides[0];
                let r_s_c = rhs.strides[1];

                let result_shape = [m, n];
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

                                let a_v =
                                    unsafe { vld1q_dup_f64(&self.data[i * l_s_r + l * l_s_c]) };

                                while j + 1 < n {
                                    unsafe {
                                        let b_vec = vld1q_f64(&rhs.data[l * r_s_r + j * r_s_c]);
                                        let c_vec = vld1q_f64(&result_data[i * n + j]);

                                        let res_vec = vfmaq_f64(c_vec, a_v, b_vec);
                                        vst1q_f64(&mut result_data[i * n + j], res_vec);
                                    }
                                    j += 2;
                                }

                                for j in j..n {
                                    result_data[i * n + j] += self.data[i * l_s_r + l * l_s_c]
                                        * rhs.data[l * r_s_r + j * r_s_c].clone();
                                }
                            }
                        }
                    }
                    MatmulPolicy::Blas => {
                        let (row_major, transa, transb, m, k, n, a, lda, b, ldb) =
                            match (self.major_order, rhs.major_order) {
                                (MajorOrder::RowMajor, MajorOrder::RowMajor) => (
                                    cblas_sys::CBLAS_LAYOUT::CblasRowMajor,
                                    cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                                    cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                                    m as i32,
                                    k as i32,
                                    n as i32,
                                    self.data.as_ptr(),
                                    k as i32,
                                    rhs.data.as_ptr(),
                                    n as i32,
                                ),
                                (MajorOrder::RowMajor, MajorOrder::ColumnMajor) => (
                                    cblas_sys::CBLAS_LAYOUT::CblasRowMajor,
                                    cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                                    cblas_sys::CBLAS_TRANSPOSE::CblasTrans,
                                    m as i32,
                                    k as i32,
                                    n as i32,
                                    self.data.as_ptr(),
                                    k as i32,
                                    rhs.data.as_ptr(),
                                    k as i32,
                                ),
                                (MajorOrder::ColumnMajor, MajorOrder::RowMajor) => (
                                    cblas_sys::CBLAS_LAYOUT::CblasRowMajor,
                                    cblas_sys::CBLAS_TRANSPOSE::CblasTrans,
                                    cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                                    m as i32,
                                    k as i32,
                                    n as i32,
                                    self.data.as_ptr(),
                                    m as i32,
                                    rhs.data.as_ptr(),
                                    n as i32,
                                ),
                                (MajorOrder::ColumnMajor, MajorOrder::ColumnMajor) => (
                                    cblas_sys::CBLAS_LAYOUT::CblasRowMajor,
                                    cblas_sys::CBLAS_TRANSPOSE::CblasTrans,
                                    cblas_sys::CBLAS_TRANSPOSE::CblasTrans,
                                    m as i32,
                                    k as i32,
                                    n as i32,
                                    self.data.as_ptr(),
                                    m as i32,
                                    rhs.data.as_ptr(),
                                    k as i32,
                                ),
                            };

                        let c = result_data.as_mut_ptr();

                        let alpha = 1.0;
                        let beta = 0.0;

                        unsafe {
                            cblas_sys::cblas_dgemm(
                                row_major,
                                transa,
                                transb,
                                m,
                                n,
                                k,
                                alpha,
                                a.cast(),
                                lda,
                                b.cast(),
                                ldb,
                                beta,
                                c.cast(),
                                n,
                            );
                        }
                    }
                    _ => return super::matmul_general(self, rhs, policy),
                }

                Array::from_vec_major(result_data, result_shape, MajorOrder::RowMajor)
            }
        }
    }
}
