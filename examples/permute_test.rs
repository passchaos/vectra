use vectra::prelude::Array;

fn main() {
    // Test 2D array permutation (transpose)
    println!("=== 2D Array Permutation Test ===");
    let mut arr_2d = Array::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    println!("Original 2D array (2x3): {}", arr_2d);
    println!("Shape: {:?}", arr_2d.shape());

    // Permute dimensions [0, 1] -> [1, 0] (transpose)
    match arr_2d.clone().permute(vec![1, 0]) {
        Ok(permuted) => {
            println!("Permuted [1, 0] (3x2): {}", permuted);
            println!("Shape: {:?}", permuted.shape());
        }
        Err(e) => println!("Error: {}", e),
    }

    // Test 3D array permutation
    println!("\n=== 3D Array Permutation Test ===");
    let mut arr_3d =
        Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], vec![2, 2, 3]).unwrap();
    println!("Original 3D array (2x2x3): {:?}", arr_3d);
    println!("Shape: {:?}", arr_3d.shape());

    // Permute dimensions [0, 1, 2] -> [2, 0, 1]
    match arr_3d.clone().permute(vec![2, 0, 1]) {
        Ok(permuted) => {
            println!("Permuted [2, 0, 1] (3x2x2): {:?}", permuted);
            println!("Shape: {:?}", permuted.shape());
        }
        Err(e) => println!("Error: {}", e),
    }

    // Permute dimensions [0, 1, 2] -> [1, 2, 0]
    match arr_3d.permute(vec![1, 2, 0]) {
        Ok(permuted) => {
            println!("Permuted [1, 2, 0] (2x3x2): {:?}", permuted);
            println!("Shape: {:?}", permuted.shape());
        }
        Err(e) => println!("Error: {}", e),
    }

    // Test error cases
    println!("\n=== Error Cases Test ===");

    // Wrong number of axes
    match arr_2d.clone().permute(vec![0]) {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Expected error (wrong number of axes): {}", e),
    }

    // Invalid axis
    match arr_2d.clone().permute(vec![0, 2]) {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Expected error (invalid axis): {}", e),
    }

    // Duplicate axis
    match arr_2d.permute(vec![0, 0]) {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Expected error (duplicate axis): {}", e),
    }

    // Test with floating point numbers
    println!("\n=== Float Array Permutation Test ===");
    let mut arr_float = Array::from_vec(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![3, 2]).unwrap();
    println!("Original float array (3x2): {}", arr_float);

    match arr_float.permute(vec![1, 0]) {
        Ok(permuted) => {
            println!("Permuted float array (2x3): {}", permuted);
        }
        Err(e) => println!("Error: {}", e),
    }
}
