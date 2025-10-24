//! Random Arrays Examples
//!
//! This example demonstrates various ways to generate random arrays in Vectra,
//! including uniform, normal, and integer random distributions.

use vectra::prelude::*;

fn main() {
    println!("=== Random Arrays Examples ===");

    // 1. Basic random arrays (0-1 uniform distribution)
    println!("\n1. Basic random arrays (0-1 uniform):");
    
    let random_1d = Array::<_, f64>::random([5]);
    println!("Random 1D array (5 elements):\n{}", random_1d);
    
    let random_2d = Array::<_, f64>::random([2, 3]);
    println!("\nRandom 2D array (2x3):\n{}", random_2d);
    
    let random_3d = Array::<_, f32>::random([2, 2, 2]);
    println!("\nRandom 3D array (2x2x2):\n{}", random_3d);

    // 2. Uniform distribution with custom range
    println!("\n2. Uniform distribution with custom range:");
    
    let uniform_float = Array::<_, f64>::uniform([3, 3], -2.0, 2.0);
    println!("Uniform float array [-2.0, 2.0):\n{}", uniform_float);
    
    let uniform_large = Array::<_, f64>::uniform([2, 4], 10.0, 100.0);
    println!("\nUniform float array [10.0, 100.0):\n{}", uniform_large);
    
    // Integer uniform distribution
    let uniform_int = Array::<_, i32>::uniform([2, 3], 1, 10);
    println!("\nUniform integer array [1, 10):\n{}", uniform_int);
    
    let uniform_large_int = Array::<_, i32>::uniform([3, 2], -50, 50);
    println!("\nUniform integer array [-50, 50):\n{}", uniform_large_int);

    // 3. Normal (Gaussian) distribution
    println!("\n3. Normal (Gaussian) distribution:");
    
    let normal_2d = Array::<_, f64>::randn([3, 3]);
    println!("Normal distribution array (mean=0, std=1):\n{}", normal_2d);
    
    let normal_1d = Array::<_, f32>::randn([8]);
    println!("\nNormal distribution 1D array:\n{}", normal_1d);
    
    let normal_large = Array::<_, f64>::randn([4, 4]);
    println!("\nLarge normal distribution array (4x4):\n{}", normal_large);

    // 4. Random array statistics
    println!("\n4. Random array statistics:");
    
    let stats_array = Array::<_, f64>::random([100]);
    println!("Statistics for 100 random numbers [0, 1):");
    println!("Mean: {:.4}", stats_array.mean::<u64>());
    println!("Max: {:.4}", stats_array.max());
    println!("Min: {:.4}", stats_array.min());
    println!("Sum: {:.4}", stats_array.sum());
    
    let normal_stats = Array::<_, f64>::randn([1000]);
    println!("\nStatistics for 1000 normal random numbers:");
    println!("Mean: {:.4}", normal_stats.mean::<u64>());
    println!("Max: {:.4}", normal_stats.max());
    println!("Min: {:.4}", normal_stats.min());

    // 5. Different data types
    println!("\n5. Random arrays with different data types:");
    
    let f32_random = Array::<_, f32>::random([2, 2]);
    println!("f32 random array:\n{}", f32_random);
    
    let f64_random = Array::<_, f64>::random([2, 2]);
    println!("\nf64 random array:\n{}", f64_random);
    
    let i32_random = Array::<_, i32>::uniform([2, 2], 0, 100);
    println!("\ni32 random array [0, 100):\n{}", i32_random);
    
    let i64_random = Array::<_, i64>::uniform([2, 2], -1000, 1000);
    println!("\ni64 random array [-1000, 1000):\n{}", i64_random);

    // 6. Large random arrays for performance testing
    println!("\n6. Large random arrays:");
    
    let large_uniform = Array::<_, f64>::uniform([10, 10], 0.0, 1.0);
    println!("Large uniform array (10x10) - showing shape: {:?}", large_uniform.shape());
    println!("Mean of large array: {:.4}", large_uniform.mean::<u64>());
    
    let large_normal = Array::<_, f32>::randn([5, 5, 4]);
    println!("\nLarge normal array (5x5x4) - showing shape: {:?}", large_normal.shape());
    println!("Sum of large normal array: {:.4}", large_normal.sum());

    // 7. Random arrays for mathematical operations
    println!("\n7. Random arrays for mathematical operations:");
    
    let rand_a = Array::<_, f64>::uniform([3, 3], 1.0, 5.0);
    let rand_b = Array::<_, f64>::uniform([3, 3], 1.0, 5.0);
    
    println!("Random matrix A:\n{}", rand_a);
    println!("\nRandom matrix B:\n{}", rand_b);
    
    let rand_sum = &rand_a + &rand_b;
    println!("\nA + B:\n{}", rand_sum);
    
    let rand_product = rand_a.matmul(&rand_b);
    println!("\nA · B (matrix multiplication):\n{}", rand_product);

    // 8. Seeded random generation (for reproducibility)
    println!("\n8. Multiple random generations (different each time):");
    
    for i in 1..=3 {
        let random_sample = Array::<_, f64>::random([2, 2]);
        println!("Random sample {}:\n{}", i, random_sample);
    }

    // 9. Random arrays with specific ranges for different use cases
    println!("\n9. Specialized random arrays:");
    
    // Probability-like values [0, 1]
    let probabilities = Array::<_, f64>::random([3, 3]);
    println!("Probability-like values [0, 1):\n{}", probabilities);
    
    // Percentage-like values [0, 100]
    let percentages = Array::<_, f64>::uniform([2, 3], 0.0, 100.0);
    println!("\nPercentage-like values [0, 100):\n{}", percentages);
    
    // Angle-like values [0, 2π]
    let angles = Array::<_, f64>::uniform([2, 2], 0.0, 2.0 * std::f64::consts::PI);
    println!("\nAngle-like values [0, 2π):\n{}", angles);
    
    // Temperature-like values [-10, 40] (Celsius)
    let temperatures = Array::<_, f64>::uniform([3, 2], -10.0, 40.0);
    println!("\nTemperature-like values [-10, 40) °C:\n{}", temperatures);

    println!("\n=== Random Arrays Examples Complete ===");
}
