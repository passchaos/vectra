//! Broadcasting Examples
//!
//! This example demonstrates NumPy-style broadcasting operations in Vectra,
//! showing how arrays with different shapes can be combined in element-wise operations.

use vectra::prelude::*;

fn main() {
    println!("=== Broadcasting Examples ===");

    // 1. Scalar-like broadcasting
    println!("\n1. Scalar-like broadcasting:");
    
    let matrix = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
    let scalar_like = Array::from_vec(vec![10, 10, 10, 10], [2, 2]);
    
    println!("Matrix (2x2):\n{}", matrix);
    println!("\nScalar-like array (2x2):\n{}", scalar_like);
    
    let broadcast_add = &matrix + &scalar_like;
    println!("\nMatrix + Scalar-like:\n{}", broadcast_add);
    
    let broadcast_mult = &matrix * &scalar_like;
    println!("\nMatrix * Scalar-like:\n{}", broadcast_mult);

    // 2. Vector with matrix broadcasting
    println!("\n2. Vector with matrix broadcasting:");
    
    let matrix_3x3 = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
    let row_vector = Array::from_vec(vec![10, 20, 30], [1, 3]);
    let col_vector = Array::from_vec(vec![100, 200, 300], [3, 1]);
    
    println!("Matrix (3x3):\n{}", matrix_3x3);
    println!("\nRow vector (1x3):\n{}", row_vector);
    println!("\nColumn vector (3x1):\n{}", col_vector);
    
    // Broadcasting with row vector
    let row_broadcast = &matrix_3x3 + &row_vector;
    println!("\nMatrix + Row vector:\n{}", row_broadcast);
    
    // Broadcasting with column vector
    let col_broadcast = &matrix_3x3 + &col_vector;
    println!("\nMatrix + Column vector:\n{}", col_broadcast);

    // 3. Different broadcasting scenarios
    println!("\n3. Different broadcasting scenarios:");
    
    // 2D array with 1D array
    let array_2d = Array::from_vec(vec![1, 2, 3, 4, 5, 6], [2, 3]);
    let array_1d = Array::from_vec(vec![10, 20, 30], [3]);
    
    println!("2D array (2x3):\n{}", array_2d);
    println!("\n1D array (3,):\n{}", array_1d);
    
    // Reshape 1D to broadcast properly
    let reshaped_1d = array_1d.clone().reshape([1, 3]);
    let broadcast_2d_1d = &array_2d + &reshaped_1d;
    println!("\n2D + 1D (reshaped to 1x3):\n{}", broadcast_2d_1d);

    // 4. Broadcasting with different operations
    println!("\n4. Broadcasting with different operations:");
    
    let base_matrix = Array::from_vec(vec![2, 4, 6, 8], [2, 2]);
    let multiplier = Array::from_vec(vec![3, 3, 3, 3], [2, 2]);
    
    println!("Base matrix (2x2):\n{}", base_matrix);
    println!("\nMultiplier (2x2):\n{}", multiplier);
    
    let add_result = &base_matrix + &multiplier;
    println!("\nAddition:\n{}", add_result);
    
    let sub_result = &base_matrix - &multiplier;
    println!("\nSubtraction:\n{}", sub_result);
    
    let mul_result = &base_matrix * &multiplier;
    println!("\nMultiplication:\n{}", mul_result);
    
    let div_result = &base_matrix / &multiplier;
    println!("\nDivision:\n{}", div_result);

    // 5. Broadcasting with larger arrays
    println!("\n5. Broadcasting with larger arrays:");
    
    let large_matrix = Array::from_vec((1..=12).collect::<Vec<i32>>(), [3, 4]);
    let broadcast_vector = Array::from_vec(vec![1, 2, 3, 4], [1, 4]);
    
    println!("Large matrix (3x4):\n{}", large_matrix);
    println!("\nBroadcast vector (1x4):\n{}", broadcast_vector);
    
    let large_broadcast = &large_matrix * &broadcast_vector;
    println!("\nLarge matrix * broadcast vector:\n{}", large_broadcast);

    // 6. Broadcasting with floating point arrays
    println!("\n6. Broadcasting with floating point arrays:");
    
    let float_matrix = Array::from_vec(vec![1.5, 2.5, 3.5, 4.5], [2, 2]);
    let float_scalar = Array::from_vec(vec![0.5, 0.5, 0.5, 0.5], [2, 2]);
    
    println!("Float matrix (2x2):\n{}", float_matrix);
    println!("\nFloat scalar-like (2x2):\n{}", float_scalar);
    
    let float_add = &float_matrix + &float_scalar;
    println!("\nFloat addition:\n{}", float_add);
    
    let float_div = &float_matrix / &float_scalar;
    println!("\nFloat division:\n{}", float_div);

    // 7. Complex broadcasting patterns
    println!("\n7. Complex broadcasting patterns:");
    
    // Broadcasting between different shaped arrays
    let arr_2x1 = Array::from_vec(vec![10, 20], [2, 1]);
    let arr_1x3 = Array::from_vec(vec![1, 2, 3], [1, 3]);
    
    println!("Array (2x1):\n{}", arr_2x1);
    println!("\nArray (1x3):\n{}", arr_1x3);
    
    let complex_broadcast = &arr_2x1 + &arr_1x3;
    println!("\n(2x1) + (1x3) = (2x3):\n{}", complex_broadcast);

    // 8. Broadcasting with mathematical functions
    println!("\n8. Broadcasting with mathematical functions:");
    
    let base_values = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    let scale_factor = Array::from_vec(vec![2.0, 2.0, 2.0, 2.0], [2, 2]);
    
    println!("Base values (2x2):\n{}", base_values);
    println!("\nScale factor (2x2):\n{}", scale_factor);
    
    // Apply scaling and then mathematical functions
    let scaled = &base_values * &scale_factor;
    println!("\nScaled values:\n{}", scaled);
    
    let squared = scaled.map(|x| x * x);
    println!("\nSquared scaled values:\n{}", squared);

    // 9. Broadcasting validation examples
    println!("\n9. Broadcasting shape compatibility:");
    
    // Show compatible shapes
    let shape_4x1 = Array::from_vec(vec![1, 2, 3, 4], [4, 1]);
    let shape_1x3 = Array::from_vec(vec![10, 20, 30], [1, 3]);
    
    println!("Shape (4x1): {:?}", shape_4x1.shape());
    println!("Shape (1x3): {:?}", shape_1x3.shape());
    
    let compatible_broadcast = &shape_4x1 + &shape_1x3;
    println!("\nBroadcast result shape: {:?}", compatible_broadcast.shape());
    println!("Result (4x3):\n{}", compatible_broadcast);

    // 10. Real-world broadcasting example
    println!("\n10. Real-world example - Image processing simulation:");
    
    // Simulate a small "image" (height x width x channels)
    let image_data = (1..=24).collect::<Vec<i32>>();
    let image = Array::from_vec(image_data, [4, 6]); // 4x6 "image"
    
    // Brightness adjustment (add constant to all pixels)
    let brightness = Array::from_vec(vec![50; 24], [4, 6]);
    let brightened = &image + &brightness;
    
    println!("Original 'image' (4x6):\n{}", image);
    println!("\nBrightness adjustment (+50):\n{}", brightened);
    
    // Contrast adjustment (multiply by factor)
    let contrast = Array::from_vec(vec![2; 24], [4, 6]);
    let contrasted = &image * &contrast;
    
    println!("\nContrast adjustment (×2):\n{}", contrasted);

    println!("\n=== Broadcasting Examples Complete ===");
}
