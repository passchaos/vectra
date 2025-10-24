//! Array Operations Examples
//!
//! This example demonstrates various array operations in Vectra,
//! including reshaping, transposing, indexing, and aggregation functions.

use vectra::prelude::*;

fn main() {
    println!("=== Array Operations Examples ===");

    // Create a sample array for operations
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let arr = Array::from_vec(data, [3, 4]);
    println!("Original array (3x4):\n{}", arr);

    // 1. Reshaping
    println!("\n1. Reshaping operations:");
    let reshaped_2x6 = arr.clone().reshape([2, 6]);
    println!("Reshaped to (2x6):\n{}", reshaped_2x6);

    let reshaped_4x3 = arr.clone().reshape([4, 3]);
    println!("\nReshaped to (4x3):\n{}", reshaped_4x3);

    let reshaped_1d = arr.clone().reshape([12]);
    println!("\nReshaped to 1D (12,):\n{}", reshaped_1d);

    // 2. Transposing
    println!("\n2. Transposing:");
    let transposed = arr.clone().transpose();
    println!("Transposed (4x3):\n{}", transposed);

    // 3. Index access
    println!("\n3. Index access:");
    println!("arr[0, 0] = {}", arr[[0, 0]]);
    println!("arr[1, 2] = {}", arr[[1, 2]]);
    println!("arr[2, 3] = {}", arr[[2, 3]]);

    // 4. Aggregation functions
    println!("\n4. Aggregation functions:");
    let arr_for_agg = arr.clone();
    println!("Sum of all elements: {}", arr_for_agg.sum());
    println!("Maximum value: {:?}", arr_for_agg.max());
    println!("Minimum value: {:?}", arr_for_agg.min());
    println!("Mean value: {:.2}", arr_for_agg.mean::<u64>());

    // 5. Sum along axes
    println!("\n5. Sum along axes:");
    let sum_axis0 = arr.sum_axis(0);
    println!("Sum along axis 0 (columns): {}", sum_axis0);

    let sum_axis1 = arr.sum_axis(1);
    println!("Sum along axis 1 (rows): {}", sum_axis1);

    // 6. Function mapping
    println!("\n6. Function mapping:");
    let squared = arr.clone().map(|x| x * x);
    println!("Array squared:\n{}", squared);

    let doubled = arr.clone().map(|x| x * 2);
    println!("\nArray doubled:\n{}", doubled);

    let conditional = arr.clone().map(|x| if *x > 6 { *x * 10 } else { *x });
    println!("\nConditional mapping (>6 → ×10):\n{}", conditional);

    // 7. Element-wise transformations
    println!("\n7. Element-wise transformations:");
    let transformed_arr = arr.clone().map(|x| *x + 100);
    println!("After adding 100 to each element:\n{}", transformed_arr);

    // 8. Array slicing and manipulation
    println!("\n8. Array manipulation:");

    // Unsqueeze (add dimension)
    let unsqueezed = arr.clone().unsqueeze(0);
    println!("Unsqueezed at axis 0 (1x3x4):\n{}", unsqueezed);

    let unsqueezed_end = arr.clone().unsqueeze(2);
    println!("\nUnsqueezed at axis 2 (3x4x1):\n{}", unsqueezed_end);

    // Squeeze (remove dimension of size 1)
    let squeezed = unsqueezed.squeeze(0);
    println!("\nSqueezed back to original (3x4):\n{}", squeezed);

    // 9. Array concatenation
    println!("\n9. Array concatenation:");
    let arr1 = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
    let arr2 = Array::from_vec(vec![5, 6, 7, 8], [2, 2]);

    println!("Array 1 (2x2):\n{}", arr1);
    println!("Array 2 (2x2):\n{}", arr2);

    let concat_axis0 = Array::cat(&[&arr1, &arr2], 0);
    println!("\nConcatenated along axis 0:\n{}", concat_axis0);

    let concat_axis1 = Array::cat(&[&arr1, &arr2], 1);
    println!("\nConcatenated along axis 1 (2x4):\n{}", concat_axis1);

    // 10. Array stacking
    println!("\n10. Array stacking:");
    let stack_arr1 = Array::from_vec(vec![1, 2, 3], [3]);
    let stack_arr2 = Array::from_vec(vec![4, 5, 6], [3]);
    let stack_arr3 = Array::from_vec(vec![7, 8, 9], [3]);

    let stacked = Array::stack(vec![stack_arr1, stack_arr2, stack_arr3], 0);
    println!("Stacked arrays (3x3):\n{}", stacked);

    // 11. Permutation (axis reordering)
    println!("\n11. Axis permutation:");
    let arr_3d = Array::from_vec((1..=24).collect::<Vec<i32>>(), [2, 3, 4]);
    println!("Original 3D array (2x3x4):\n{}", arr_3d);

    let permuted = arr_3d.permute([2, 0, 1]);
    println!("\nPermuted to (4x2x3):\n{}", permuted);

    println!("\n=== Array Operations Examples Complete ===");
}
