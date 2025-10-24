//! Machine Learning Operations Example
//!
//! This example demonstrates various machine learning functions available in vectra,
//! including activation functions, normalization, and neural network operations.

use vectra::prelude::*;

fn main() {
    println!("=== Machine Learning Operations Example ===");
    
    // Create sample data for ML operations
    let input_data = Array::from_vec(vec![-2, -1, 0, 1, 2, 3], [6]);
    let matrix_data = Array::from_vec(vec![
        -1, 0, 1, 2,
        -2, 0, 1, 2,
        -1, 1, 2, 3
    ], [3, 4]);
    
    println!("\nInput data: {}", input_data);
    println!("\nMatrix data (3x4):\n{}", matrix_data);
    
    // 1. ReLU Activation Function
    println!("\n=== ReLU Activation Function ===");
    
    let relu_result = input_data.relu();
    println!("ReLU(input): {}", relu_result);
    
    let relu_matrix = matrix_data.relu();
    println!("\nReLU applied to matrix:\n{}", relu_matrix);
    
    // 2. GELU Activation Function
    println!("\n=== GELU Activation Function ===");
    
    let input_data_f64 = Array::from_vec(vec![-2.0f64, -1.0, 0.0, 1.0, 2.0, 3.0], [6]);
    let gelu_result = input_data_f64.gelu();
    println!("GELU(input): {}", gelu_result);
    
    let matrix_data_f64 = Array::from_vec(vec![
        -1.5f64, -0.5, 0.5, 1.5,
        -2.0, 0.0, 1.0, 2.0,
        -1.0, 0.5, 1.5, 2.5
    ], [3, 4]);
    let gelu_matrix = matrix_data_f64.gelu();
    println!("\nGELU applied to matrix:\n{}", gelu_matrix);
    
    // 3. Sigmoid Activation Function
    println!("\n=== Sigmoid Activation Function ===");
    
    let sigmoid_result = input_data_f64.sigmoid();
    println!("Sigmoid(input): {}", sigmoid_result);
    
    let sigmoid_matrix = matrix_data_f64.sigmoid();
    println!("\nSigmoid applied to matrix:\n{}", sigmoid_matrix);
    
    // 4. Softmax Function
    println!("\n=== Softmax Function ===");
    
    // Softmax for probability distribution
    let logits = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 1.0], [5]);
    println!("Logits: {}", logits);
    
    let softmax_result = logits.softmax();
    println!("Softmax(logits): {}", softmax_result);
    println!("Sum of softmax (should be ~1.0): {:.6}", softmax_result.sum());
    
    // Softmax on matrix rows
    let logits_matrix = Array::from_vec(vec![
        1.0f64, 2.0, 3.0,
        4.0, 5.0, 6.0,
        0.5, 1.5, 2.5
    ], [3, 3]);
    println!("\nLogits matrix (3x3):\n{}", logits_matrix);
    
    let softmax_matrix = logits_matrix.softmax();
    println!("\nSoftmax applied to matrix:\n{}", softmax_matrix);
    
    // Verify each row sums to 1
    let row_sums = softmax_matrix.sum_axis(1);
    println!("Row sums (should be ~1.0): {}", row_sums);
    
    // 5. RMS Normalization
    println!("\n=== RMS Normalization ===");
    
    let rms_input = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0], [5]);
    println!("Input for RMS norm: {}", rms_input);
    
    let rms_result = rms_input.rms_norm(1e-8);
    println!("RMS normalized: {}", rms_result);
    
    let rms_matrix = matrix_data_f64.rms_norm(1e-8);
    println!("\nRMS normalized matrix:\n{}", rms_matrix);
    
    // 6. Comparison of Activation Functions
    println!("\n=== Activation Function Comparison ===");
    
    let test_range = Array::from_vec(vec![-3.0f64, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [7]);
    println!("Test range: {}", test_range);
    
    let test_range_int = Array::from_vec(vec![-3, -2, -1, 0, 1, 2, 3], [7]);
    let relu_comp = test_range_int.relu();
    let gelu_comp = test_range.gelu();
    let sigmoid_comp = test_range.sigmoid();
    
    println!("\nReLU:    {}", relu_comp);
    println!("GELU:    {}", gelu_comp);
    println!("Sigmoid: {}", sigmoid_comp);
    
    // 7. Neural Network Layer Simulation
    println!("\n=== Neural Network Layer Simulation ===");
    
    // Simulate a simple neural network layer: input -> linear -> activation
    let nn_input = Array::from_vec(vec![0.5f64, -0.3, 1.2, -0.8], [1, 4]);
    let weights = Array::from_vec(vec![
        0.1f64, 0.2, -0.1, 0.3,
        -0.2, 0.4, 0.1, -0.1,
        0.3, -0.1, 0.2, 0.4
    ], [4, 3]);
    let bias = Array::from_vec(vec![0.1f64, -0.05, 0.2], [1, 3]);
    
    println!("Input (1x4): {}", nn_input);
    println!("\nWeights (4x3):\n{}", weights);
    println!("\nBias (1x3): {}", bias);
    
    // Linear transformation: input * weights + bias
    let matmul_result = nn_input.matmul(&weights);
    let linear_output = &matmul_result + &bias;
    println!("\nLinear output (1x3): {}", linear_output);
    
    // Apply different activations
    // Convert to integer for ReLU
    let linear_int: Array<2, i32> = linear_output.map(|x| (*x * 100.0) as i32);
    let relu_output = linear_int.relu();
    let gelu_output = linear_output.gelu();
    let sigmoid_output = linear_output.sigmoid();
    let softmax_output = linear_output.softmax();
    
    println!("\nAfter ReLU: {}", relu_output);
    println!("After GELU: {}", gelu_output);
    println!("After Sigmoid: {}", sigmoid_output);
    println!("After Softmax: {}", softmax_output);
    
    // 8. Batch Processing Example
    println!("\n=== Batch Processing Example ===");
    
    // Simulate a batch of inputs
    let batch_input = Array::from_vec(vec![
        0.5f64, -0.3, 1.2,
        -0.8, 0.9, -0.1,
        0.2, 0.7, -0.5,
        1.1, -0.4, 0.3
    ], [4, 3]); // 4 samples, 3 features each
    
    println!("Batch input (4x3):\n{}", batch_input);
    
    // Apply activations to the entire batch
    let batch_input_int = Array::from_vec(vec![
        1, 0, 1,
        -1, 1, 0,
        0, 1, -1,
        1, 0, 0
    ], [4, 3]); // 4 samples, 3 features each
    let batch_relu = batch_input_int.relu();
    let batch_sigmoid = batch_input.sigmoid();
    let batch_softmax = batch_input.softmax();
    
    println!("\nBatch after ReLU:\n{}", batch_relu);
    println!("\nBatch after Sigmoid:\n{}", batch_sigmoid);
    println!("\nBatch after Softmax:\n{}", batch_softmax);
    
    // Verify softmax properties for batch
    let batch_row_sums = batch_softmax.sum_axis(1);
    println!("\nSoftmax row sums (should be ~1.0): {}", batch_row_sums);
    
    // 9. Gradient-like Operations
    println!("\n=== Gradient-like Operations ===");
    
    // Simulate derivative of ReLU (step function)
    let grad_input = Array::from_vec(vec![-1.0f64, 0.0, 1.0, 2.0, -0.5], [5]);
    println!("Input: {}", grad_input);
    
    // ReLU derivative: 1 if x > 0, 0 otherwise
    let relu_grad = grad_input.map(|x| if *x > 0.0 { 1.0 } else { 0.0 });
    println!("ReLU gradient: {}", relu_grad);
    
    // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
    let sigmoid_vals = grad_input.sigmoid();
    let sigmoid_grad = &sigmoid_vals * &sigmoid_vals.map(|x| 1.0 - x);
    println!("Sigmoid values: {}", sigmoid_vals);
    println!("Sigmoid gradient: {}", sigmoid_grad);
    
    println!("\n=== Machine Learning Operations Complete ===");
}
