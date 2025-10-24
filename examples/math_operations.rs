//! Mathematical Operations Examples
//!
//! This example demonstrates various mathematical operations in Vectra,
//! including element-wise operations, matrix multiplication, and scalar operations.

use vectra::prelude::*;

fn main() {
    println!("=== Mathematical Operations Examples ===");

    // Create sample matrices
    let a = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
    let b = Array::from_vec(vec![5, 6, 7, 8], [2, 2]);

    println!("Matrix A (2x2):\n{}", a);
    println!("\nMatrix B (2x2):\n{}", b);

    // 1. Element-wise operations
    println!("\n1. Element-wise operations:");

    let sum = &a + &b;
    println!("A + B (element-wise addition):\n{}", sum);

    let diff = &a - &b;
    println!("\nA - B (element-wise subtraction):\n{}", diff);

    let product = &a * &b;
    println!("\nA * B (element-wise multiplication):\n{}", product);

    let quotient = &a / &b;
    println!("\nA / B (element-wise division):\n{}", quotient);

    // 2. Matrix multiplication
    println!("\n2. Matrix multiplication:");

    let matmul_result = a.matmul(&b);
    println!("A · B (matrix multiplication):\n{}", matmul_result);

    // Different matrix multiplication policies
    let matmul_naive = a.matmul_with_policy(&b, MatmulPolicy::Naive);
    println!("\nA · B (naive policy):\n{}", matmul_naive);

    let matmul_blas = a.matmul_with_policy(&b, MatmulPolicy::Blas);
    println!("\nA · B (BLAS policy):\n{}", matmul_blas);

    // 3. Scalar operations
    println!("\n3. Scalar operations:");

    let scalar_mult = a.clone().mul_scalar(3);
    println!("A * 3 (scalar multiplication):\n{}", scalar_mult);

    let scalar_add = a.clone().add_scalar(10);
    println!("\nA + 10 (scalar addition):\n{}", scalar_add);

    let scalar_sub = a.clone().sub_scalar(1);
    println!("\nA - 1 (scalar subtraction):\n{}", scalar_sub);

    let scalar_div = a.clone().div_scalar(2);
    println!("\nA / 2 (scalar division):\n{}", scalar_div);

    // 4. Power operations
    println!("\n4. Power operations:");

    let float_a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    println!("\nFloat matrix A for power operations:\n{}", float_a);

    let squared = float_a.pow2();
    println!("\nA² (element-wise square):\n{}", squared);

    let cubed = float_a.powi(3);
    println!("\nA³ (element-wise cube):\n{}", cubed);

    let float_arr = Array::from_vec(vec![1.0, 4.0, 9.0, 16.0], [2, 2]);
    println!("\nFloat array for roots:\n{}", float_arr);

    let sqrt_result = float_arr.sqrt();
    println!("\nSquare root:\n{}", sqrt_result);

    let cbrt_result = float_arr.cbrt();
    println!("\nCube root:\n{}", cbrt_result);

    // 5. Absolute value and sign operations
    println!("\n5. Absolute value and sign operations:");

    let mixed_arr = Array::from_vec(vec![-3.0, -1.0, 0.0, 2.0, 5.0], [5]);
    println!("Mixed sign array:\n{}", mixed_arr);

    let abs_result = mixed_arr.abs();
    println!("\nAbsolute values:\n{}", abs_result);

    let sign_result = mixed_arr.signum();
    println!("\nSign values:\n{}", sign_result);

    // 6. Rounding operations
    println!("\n6. Rounding operations:");

    let float_vals = Array::from_vec(vec![1.2, 2.7, -1.8, -2.3, 3.5], [5]);
    println!("Float values:\n{}", float_vals);

    let floor_result = float_vals.floor();
    println!("\nFloor:\n{}", floor_result);

    let ceil_result = float_vals.ceil();
    println!("\nCeiling:\n{}", ceil_result);

    let round_result = float_vals.round();
    println!("\nRounded:\n{}", round_result);

    let trunc_result = float_vals.trunc();
    println!("\nTruncated:\n{}", trunc_result);

    let fract_result = float_vals.fract();
    println!("\nFractional part:\n{}", fract_result);

    // 7. Reciprocal
    println!("\n7. Reciprocal operations:");

    let recip_arr = Array::from_vec(vec![1.0, 2.0, 4.0, 8.0], [2, 2]);
    println!("Array for reciprocal:\n{}", recip_arr);

    let recip_result = recip_arr.recip();
    println!("\nReciprocal (1/x):\n{}", recip_result);

    // 8. Comparison operations
    println!("\n8. Comparison operations:");

    let arr1 = Array::from_vec(vec![1, 3, 5, 7], [2, 2]);
    let arr2 = Array::from_vec(vec![2, 3, 4, 8], [2, 2]);

    println!("Array 1:\n{}", arr1);
    println!("\nArray 2:\n{}", arr2);

    let equal_result = arr1.equal(&arr2);
    println!("\nElement-wise equality:\n{}", equal_result);

    // 9. Large matrix operations
    println!("\n9. Large matrix operations:");

    let large_a = Array::from_vec((1..=16).collect::<Vec<i32>>(), [4, 4]);
    let large_b = Array::from_vec((16..=31).collect::<Vec<i32>>(), [4, 4]);

    println!("Large matrix A (4x4):\n{}", large_a);
    println!("\nLarge matrix B (4x4):\n{}", large_b);

    let large_sum = &large_a + &large_b;
    println!("\nA + B (4x4):\n{}", large_sum);

    let large_matmul = large_a.matmul(&large_b);
    println!("\nA · B (4x4 matrix multiplication):\n{}", large_matmul);

    println!("\n=== Mathematical Operations Examples Complete ===");
}
