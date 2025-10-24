//! Statistical Operations Example
//!
//! This example demonstrates various statistical functions available in vectra,
//! including descriptive statistics, variance, standard deviation, and aggregations.

use vectra::prelude::*;

fn main() {
    println!("=== Statistical Operations Example ===");

    // Create sample data arrays
    let data_1d = Array::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        [10],
    );
    let data_2d = Array::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        [3, 4],
    );

    println!("\n1D Data: {}", data_1d);
    println!("\n2D Data:\n{}", data_2d);

    // Basic aggregation functions
    println!("\n=== Basic Aggregations ===");

    // Sum operations
    let sum_1d = data_1d.sum();
    println!("Sum of 1D data: {}", sum_1d);

    let sum_2d = data_2d.sum();
    println!("Sum of 2D data: {}", sum_2d);

    // Sum along specific axes
    let sum_axis_0 = data_2d.sum_axis(0);
    println!("Sum along axis 0 (columns): {}", sum_axis_0);

    let sum_axis_1 = data_2d.sum_axis(1);
    println!("Sum along axis 1 (rows): {}", sum_axis_1);

    // Mean operations
    println!("\n=== Mean Operations ===");

    let mean_1d = data_1d.mean::<f64>();
    println!("Mean of 1D data: {}", mean_1d);

    let mean_2d = data_2d.mean::<f64>();
    println!("Mean of 2D data: {}", mean_2d);

    // Mean along specific axes
    let mean_axis_0 = data_2d.mean_axis::<f64>(0);
    println!("Mean along axis 0 (columns): {}", mean_axis_0);

    let mean_axis_1 = data_2d.mean_axis::<f64>(1);
    println!("Mean along axis 1 (rows): {}", mean_axis_1);

    // Variance and Standard Deviation
    println!("\n=== Variance and Standard Deviation ===");

    let var_1d = data_1d.var(0.0);
    println!("Variance of 1D data: {}", var_1d);

    let std_1d = data_1d.std(0.0);
    println!("Standard deviation of 1D data: {}", std_1d);

    let var_2d = data_2d.var(0.0);
    println!("Variance of 2D data: {}", var_2d);

    let std_2d = data_2d.std(0.0);
    println!("Standard deviation of 2D data: {}", std_2d);

    // Min and Max operations
    println!("\n=== Min and Max Operations ===");

    let min_1d = data_1d.min();
    println!("Minimum of 1D data: {}", min_1d);

    let max_1d = data_1d.max();
    println!("Maximum of 1D data: {}", max_1d);

    let min_2d = data_2d.min();
    println!("Minimum of 2D data: {}", min_2d);

    let max_2d = data_2d.max();
    println!("Maximum of 2D data: {}", max_2d);

    // Statistical analysis on different data distributions
    println!("\n=== Different Data Distributions ===");

    // Normal-like distribution
    let normal_data = Array::from_vec(vec![2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9], [10]);
    println!("\nNormal-like data: {}", normal_data);
    println!(
        "Mean: {:.3}, Std: {:.3}",
        normal_data.mean::<f64>(),
        normal_data.std(0.0)
    );

    // Skewed distribution
    let skewed_data = Array::from_vec(
        vec![1.0, 1.1, 1.2, 1.3, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0],
        [10],
    );
    println!("\nSkewed data: {}", skewed_data);
    println!(
        "Mean: {:.3}, Std: {:.3}",
        skewed_data.mean::<f64>(),
        skewed_data.std(0.0)
    );
    println!(
        "Min: {:.3}, Max: {:.3}",
        skewed_data.min(),
        skewed_data.max()
    );

    // Uniform distribution
    let uniform_data = Array::from_vec(vec![5.0; 10], [10]);
    println!("\nUniform data: {}", uniform_data);
    println!(
        "Mean: {:.3}, Std: {:.3}",
        uniform_data.mean::<f64>(),
        uniform_data.std(0.0)
    );

    // Large dataset statistics
    println!("\n=== Large Dataset Statistics ===");

    let large_data: Vec<f64> = (1..=1000).map(|x| x as f64).collect();
    let large_array = Array::from_vec(large_data, [1000]);

    println!("Large dataset size: {}", large_array.shape()[0]);
    println!("Sum: {:.0}", large_array.sum());
    println!("Mean: {:.3}", large_array.mean::<f64>());
    println!("Std: {:.3}", large_array.std(0.0));
    println!(
        "Min: {:.0}, Max: {:.0}",
        large_array.min(),
        large_array.max()
    );

    // Multi-dimensional statistical analysis
    println!("\n=== Multi-dimensional Analysis ===");

    let matrix_3x3 = Array::from_vec(vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0], [3, 3]);

    println!("\n3x3 Matrix:\n{}", matrix_3x3);
    println!("Overall mean: {:.3}", matrix_3x3.mean::<f64>());
    println!("Overall std: {:.3}", matrix_3x3.std(0.0));

    println!("\nColumn means: {}", matrix_3x3.mean_axis::<f64>(0));
    println!("Row means: {}", matrix_3x3.mean_axis::<f64>(1));

    println!("\nColumn sums: {}", matrix_3x3.sum_axis(0));
    println!("Row sums: {}", matrix_3x3.sum_axis(1));

    println!("\n=== Statistical Operations Complete ===");
}
