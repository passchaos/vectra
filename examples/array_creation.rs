//! Array Creation Examples
//!
//! This example demonstrates various ways to create arrays in Vectra,
//! including zeros, ones, identity matrices, and arrays from vectors.

use vectra::prelude::*;

fn main() {
    println!("=== Array Creation Examples ===");

    // 1. Zero arrays
    println!("\n1. Zero arrays:");
    let zeros_2d = Array::<_, f64>::zeros([2, 3]);
    println!("Zero array (2x3):\n{}", zeros_2d);

    let zeros_3d = Array::<_, i32>::zeros([2, 2, 2]);
    println!("\nZero array (2x2x2):\n{}", zeros_3d);

    // 2. Ones arrays
    println!("\n2. Ones arrays:");
    let ones_2d = Array::<_, i32>::ones([3, 3]);
    println!("Ones array (3x3):\n{}", ones_2d);

    let ones_1d = Array::<_, f32>::ones([5]);
    println!("\nOnes array (5,):\n{}", ones_1d);

    // 3. Identity matrices
    println!("\n3. Identity matrices:");
    let eye_3x3 = Array::<_, i32>::eye(3);
    println!("Identity matrix (3x3):\n{}", eye_3x3);

    let eye_5x5 = Array::<_, f64>::eye(5);
    println!("\nIdentity matrix (5x5):\n{}", eye_5x5);

    // 4. Arrays from vectors
    println!("\n4. Arrays from vectors:");
    let data = vec![1, 2, 3, 4, 5, 6];
    let arr_2x3 = Array::from_vec(data.clone(), [2, 3]);
    println!("Array from vector (2x3):\n{}", arr_2x3);

    let arr_3x2 = Array::from_vec(data, [3, 2]);
    println!("\nSame data as (3x2):\n{}", arr_3x2);

    // 5. Arrays with specific values
    println!("\n5. Arrays with specific values:");
    let full_array = Array::<_, f64>::full([2, 4], 3.14);
    println!("Array filled with π (2x4):\n{}", full_array);

    // 6. Range arrays
    println!("\n6. Range arrays:");
    let range_array = Array::<_, i32>::arange(0, 10, 1);
    println!("Range array [0, 10) step 1:\n{}", range_array);

    let range_array_step2 = Array::<_, i32>::arange(0, 20, 2);
    println!("\nRange array [0, 20) step 2:\n{}", range_array_step2);

    let range_float = Array::<_, f64>::arange(0.0, 5.0, 0.5);
    println!("\nFloat range array [0.0, 5.0) step 0.5:\n{}", range_float);

    // 7. Large arrays
    println!("\n7. Large array example:");
    let large_data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
    let large_arr = Array::from_vec(large_data, [4, 5]);
    println!("4x5 array from range 1-20:\n{}", large_arr);

    // 8. Different data types
    println!("\n8. Different data types:");
    let int_array = Array::<_, i64>::ones([2, 2]);
    println!("i64 array:\n{}", int_array);

    let float_array = Array::<_, f32>::zeros([2, 2]);
    println!("\nf32 array:\n{}", float_array);

    let usize_array = Array::<_, usize>::full([3], 42);
    println!("\nusize array:\n{}", usize_array);

    println!("\n=== Array Creation Examples Complete ===");
}
