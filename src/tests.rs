#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array;

    #[test]
    fn test_basic_creation() {
        let arr: Array<f64> = Array::new(vec![2, 3]);
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.size(), 6);
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let arr = Array::from_vec(data, vec![2, 2]).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
    }

    #[test]
    fn test_ones_and_eye() {
        let ones: Array<f64> = Array::ones(vec![2, 2]);
        assert_eq!(ones[[0, 0]], 1.0);
        assert_eq!(ones[[1, 1]], 1.0);
        
        let eye: Array<f64> = Array::eye(3);
        assert_eq!(eye[[0, 0]], 1.0);
        assert_eq!(eye[[1, 1]], 1.0);
        assert_eq!(eye[[0, 1]], 0.0);
    }

    #[test]
    fn test_arithmetic() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Array::from_vec(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]).unwrap();
        
        let sum = a.clone() + b.clone();
        assert_eq!(sum[[0, 0]], 3.0);
        assert_eq!(sum[[1, 1]], 6.0);
        
        let product = a * b;
        assert_eq!(product[[0, 0]], 2.0);
        assert_eq!(product[[1, 1]], 8.0);
    }

    #[test]
    fn test_reshape_and_transpose() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let reshaped = arr.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        
        let transposed = arr.transpose().unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed[[0, 0]], 1.0);
        assert_eq!(transposed[[1, 0]], 2.0);
    }

    #[test]
    fn test_aggregations() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(arr.sum(), 10.0);
        assert_eq!(arr.mean(), 2.5);
        assert_eq!(arr.max(), Some(4.0));
        assert_eq!(arr.min(), Some(1.0));
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Array::from_vec(vec![2.0, 0.0, 1.0, 2.0], vec![2, 2]).unwrap();
        
        let result = a.dot(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 4.0); // 1*2 + 2*1 = 4
        assert_eq!(result[[0, 1]], 4.0); // 1*0 + 2*2 = 4
    }

    #[test]
    fn test_map() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let squared = arr.map(|x| x * x);
        assert_eq!(squared[[0, 0]], 1.0);
        assert_eq!(squared[[1, 1]], 16.0);
    }

    #[test]
    fn test_random_arrays() {
        let random_arr: Array<f64> = Array::random(vec![3, 3]);
        assert_eq!(random_arr.shape(), &[3, 3]);
        assert_eq!(random_arr.size(), 9);
        
        // Check that values are in [0, 1) range
        for &val in random_arr.iter() {
            assert!(val >= 0.0 && val < 1.0);
        }
        
        let randint_arr: Array<i32> = Array::randint(vec![2, 2], 1, 10);
        assert_eq!(randint_arr.shape(), &[2, 2]);
        
        // Check that values are in [1, 10) range
        for &val in randint_arr.iter() {
            assert!(val >= 1 && val < 10);
        }
        
        let randn_arr: Array<f64> = Array::randn(vec![5, 5]);
        assert_eq!(randn_arr.shape(), &[5, 5]);
        assert_eq!(randn_arr.size(), 25);
        
        let uniform_arr: Array<f64> = Array::uniform(vec![3, 3], -1.0, 1.0);
        assert_eq!(uniform_arr.shape(), &[3, 3]);
        
        // Check that values are in [-1, 1) range
        for &val in uniform_arr.iter() {
            assert!(val >= -1.0 && val < 1.0);
        }
    }

    #[test]
    fn test_broadcasting() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let b = Array::from_vec(vec![10.0, 20.0], vec![1, 2]).unwrap();
        
        let result = a + b;
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result[[0, 0]], 11.0); // 1 + 10
        assert_eq!(result[[0, 1]], 21.0); // 1 + 20
        assert_eq!(result[[1, 0]], 12.0); // 2 + 10
        assert_eq!(result[[1, 1]], 22.0); // 2 + 20
        assert_eq!(result[[2, 0]], 13.0); // 3 + 10
        assert_eq!(result[[2, 1]], 23.0); // 3 + 20
        
        // Test scalar operations
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let added = arr.add_scalar(5.0);
        assert_eq!(added[[0, 0]], 6.0);
        assert_eq!(added[[1, 1]], 9.0);
        
        let multiplied = arr.mul_scalar(2.0);
        assert_eq!(multiplied[[0, 0]], 2.0);
        assert_eq!(multiplied[[1, 1]], 8.0);
    }

    #[test]
    fn test_broadcast_shapes() {
        let shape1 = vec![3, 1];
        let shape2 = vec![1, 4];
        let result = Array::<f64>::broadcast_shapes(&shape1, &shape2).unwrap();
        assert_eq!(result, vec![3, 4]);
        
        let shape3 = vec![2, 3];
        let shape4 = vec![2, 4];
        let result2 = Array::<f64>::broadcast_shapes(&shape3, &shape4);
        assert!(result2.is_err());
    }

    #[test]
    fn test_trigonometric_functions() {
        let arr = Array::from_vec(vec![0.0, std::f64::consts::PI / 6.0, std::f64::consts::PI / 4.0, std::f64::consts::PI / 3.0, std::f64::consts::PI / 2.0], vec![5]).unwrap();
        
        let sin_result = arr.sin();
        assert!((sin_result[[0]] - 0.0).abs() < 1e-10);
        assert!((sin_result[[1]] - 0.5).abs() < 1e-10);
        assert!((sin_result[[4]] - 1.0).abs() < 1e-10);
        
        let cos_result = arr.cos();
        assert!((cos_result[[0]] - 1.0).abs() < 1e-10);
        assert!((cos_result[[4]] - 0.0).abs() < 1e-10);
        
        let _tan_result = arr.tan();
        
        // Test inverse functions
        let values = Array::from_vec(vec![0.0, 0.5, 1.0], vec![3]).unwrap();
        let asin_result = values.asin();
        let acos_result = values.acos();
        let atan_result = values.atan();
        
        assert!((asin_result[[0]] - 0.0f64).abs() < 1e-10f64);
        assert!((asin_result[[2]] - std::f64::consts::PI / 2.0).abs() < 1e-10);
        assert!((acos_result[[0]] - std::f64::consts::PI / 2.0).abs() < 1e-10);
        assert!((acos_result[[2]] - 0.0).abs() < 1e-10);
        assert!((atan_result[[0]] - 0.0).abs() < 1e-10);
        
        // Test atan2
        let y = Array::from_vec(vec![1.0, 1.0, -1.0, -1.0], vec![4]).unwrap();
        let x = Array::from_vec(vec![1.0, -1.0, 1.0, -1.0], vec![4]).unwrap();
        let atan2_result = y.atan2(&x).unwrap();
        
        assert!((atan2_result[[0]] - std::f64::consts::PI / 4.0).abs() < 1e-10);
        assert!((atan2_result[[1]] - 3.0 * std::f64::consts::PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic_functions() {
        let arr = Array::from_vec(vec![0.0, 1.0, -1.0], vec![3]).unwrap();
        
        let sinh_result = arr.sinh();
        let cosh_result = arr.cosh();
        let tanh_result = arr.tanh();
        
        assert!((sinh_result[[0]] - 0.0f64).abs() < 1e-10f64);
        assert!((cosh_result[[0]] - 1.0f64).abs() < 1e-10f64);
        assert!((tanh_result[[0]] - 0.0f64).abs() < 1e-10f64);
        
        // Test inverse hyperbolic functions
        let values = Array::from_vec(vec![0.0, 1.0, 2.0], vec![3]).unwrap();
        let asinh_result = values.asinh();
        let acosh_result = Array::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap().acosh();
        let atanh_result = Array::from_vec(vec![0.0, 0.5], vec![2]).unwrap().atanh();
        
        assert!((asinh_result[[0]] - 0.0f64).abs() < 1e-10f64);
        assert!((acosh_result[[0]] - 0.0f64).abs() < 1e-10f64);
        assert!((atanh_result[[0]] - 0.0f64).abs() < 1e-10f64);
    }

    #[test]
    fn test_angle_conversion() {
        let degrees = Array::from_vec(vec![0.0, 90.0, 180.0, 270.0, 360.0], vec![5]).unwrap();
        let radians = degrees.to_radians();
        
        assert!((radians[[0]] - 0.0f64).abs() < 1e-10f64);
        assert!((radians[[1]] - std::f64::consts::PI / 2.0).abs() < 1e-10);
        assert!((radians[[2]] - std::f64::consts::PI).abs() < 1e-10);
        assert!((radians[[3]] - 3.0 * std::f64::consts::PI / 2.0).abs() < 1e-10);
        assert!((radians[[4]] - 2.0 * std::f64::consts::PI).abs() < 1e-10);
        
        let back_to_degrees = radians.to_degrees();
        assert!((back_to_degrees[[0]] - 0.0).abs() < 1e-10);
        assert!((back_to_degrees[[1]] - 90.0).abs() < 1e-10);
        assert!((back_to_degrees[[2]] - 180.0).abs() < 1e-10);
        assert!((back_to_degrees[[3]] - 270.0).abs() < 1e-10);
        assert!((back_to_degrees[[4]] - 360.0).abs() < 1e-10);
        
        // Test with common angles
        let common_degrees = Array::from_vec(vec![30.0, 45.0, 60.0], vec![3]).unwrap();
        let common_radians = common_degrees.to_radians();
        
        assert!((common_radians[[0]] - std::f64::consts::PI / 6.0).abs() < 1e-10);
        assert!((common_radians[[1]] - std::f64::consts::PI / 4.0).abs() < 1e-10);
        assert!((common_radians[[2]] - std::f64::consts::PI / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_logarithmic_functions() {
        let arr = Array::from_vec(vec![1.0, std::f64::consts::E, 10.0, 2.0], vec![4]).unwrap();
        
        // Test natural logarithm
        let ln_result = arr.ln();
        assert!((ln_result[[0]] - 0.0).abs() < 1e-10); // ln(1) = 0
        assert!((ln_result[[1]] - 1.0).abs() < 1e-10); // ln(e) = 1
        
        // Test base-10 logarithm
        let log10_result = arr.log10();
        assert!((log10_result[[0]] - 0.0).abs() < 1e-10); // log10(1) = 0
        assert!((log10_result[[2]] - 1.0).abs() < 1e-10); // log10(10) = 1
        
        // Test base-2 logarithm
        let log2_result = arr.log2();
        assert!((log2_result[[0]] - 0.0).abs() < 1e-10); // log2(1) = 0
        assert!((log2_result[[3]] - 1.0).abs() < 1e-10); // log2(2) = 1
        
        // Test custom base logarithm
        let values = Array::from_vec(vec![1.0, 3.0, 9.0, 27.0], vec![4]).unwrap();
        let log3_result = values.log(3.0);
        assert!((log3_result[[0]] - 0.0f64).abs() < 1e-10f64); // log3(1) = 0
        assert!((log3_result[[1]] - 1.0f64).abs() < 1e-10f64); // log3(3) = 1
        assert!((log3_result[[2]] - 2.0f64).abs() < 1e-10f64); // log3(9) = 2
        assert!((log3_result[[3]] - 3.0f64).abs() < 1e-10f64); // log3(27) = 3
        
        // Test that exp and ln are inverse operations
        let test_values = Array::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let exp_ln = test_values.clone().ln().exp();
        for i in 0..3 {
            assert!(((exp_ln[[i]] - test_values[[i]]) as f64).abs() < 1e-10);
        }
    }

    #[test]
    fn test_exponential_functions() {
        let arr = Array::from_vec(vec![0.0, 1.0, 2.0], vec![3]).unwrap();
        
        // Test exponential function
        let exp_result = arr.exp();
        assert!((exp_result[[0]] - 1.0f64).abs() < 1e-10f64); // exp(0) = 1
        assert!((exp_result[[1]] - std::f64::consts::E).abs() < 1e-10); // exp(1) = e
        
        // Test base-2 exponential
        let exp2_result = arr.exp2();
        assert!((exp2_result[[0]] - 1.0).abs() < 1e-10); // 2^0 = 1
        assert!((exp2_result[[1]] - 2.0).abs() < 1e-10); // 2^1 = 2
        assert!((exp2_result[[2]] - 4.0).abs() < 1e-10); // 2^2 = 4
        
        // Test exp_m1 and ln_1p (inverse operations)
        let small_values = Array::from_vec(vec![0.1, 0.01, 0.001], vec![3]).unwrap();
        let exp_m1_result = small_values.clone().exp_m1();
        let ln_1p_result = exp_m1_result.ln_1p();
        
        for i in 0..3 {
            assert!(((ln_1p_result[[i]] - small_values[[i]]) as f64).abs() < 1e-10);
        }
        
        // Test that exp_m1(0) = 0 and ln_1p(0) = 0
        let zero_arr = Array::from_vec(vec![0.0], vec![1]).unwrap();
        assert!((zero_arr.clone().exp_m1()[[0]] - 0.0f64).abs() < 1e-10f64);
        assert!((zero_arr.ln_1p()[[0]] - 0.0f64).abs() < 1e-10f64);
    }
}