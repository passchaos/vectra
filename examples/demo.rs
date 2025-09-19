use vectra::Array;

fn main() {
    println!("=== Vectra Multi-dimensional Array Library Demo ===");
    
    // 1. Array Creation
    println!("\n1. Array Creation:");
    let zeros = Array::<f64>::zeros(vec![2, 3]);
    println!("Zero array (2x3):\n{}", zeros);
    
    let ones = Array::<i32>::ones(vec![3, 3]);
    println!("\nOnes array (3x3):\n{}", ones);
    
    let eye = Array::<i32>::eye(3);
    println!("\nIdentity matrix (3x3):\n{}", eye);
    
    // 2. Create array from vector
    println!("\n2. Create array from vector:");
    let data = vec![1, 2, 3, 4, 5, 6];
    let arr = Array::from_vec(data, vec![2, 3]).unwrap();
    println!("Array created from vector (2x3):\n{}", arr);
    
    // 3. Array operations
    println!("\n3. Array operations:");
    let reshaped = arr.reshape(vec![3, 2]).unwrap();
    println!("Reshaped to (3x2):\n{}", reshaped);
    
    let transposed = arr.transpose().unwrap();
    println!("\nTransposed (3x2):\n{}", transposed);
    
    // 4. Mathematical operations
    println!("\n4. Mathematical operations:");
    let a = Array::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let b = Array::from_vec(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
    
    println!("Matrix A:\n{}", a);
    println!("\nMatrix B:\n{}", b);
    
    let sum = a.clone() + b.clone();
    println!("\nA + B:\n{}", sum);
    
    let product = a.clone() * b.clone();
    println!("\nA * B (element-wise multiplication):\n{}", product);
    
    let dot_product = a.dot(&b).unwrap();
    println!("\nA · B (matrix multiplication):\n{}", dot_product);
    
    // 5. Scalar operations
    println!("\n5. Scalar operations:");
    let scalar_mult = a.mul_scalar(3);
    println!("A * 3:\n{}", scalar_mult);
    
    let scalar_add = a.add_scalar(10);
    println!("\nA + 10:\n{}", scalar_add);
    
    // 6. Aggregation functions
    println!("\n6. Aggregation functions:");
    println!("Sum of A: {}", a.sum());
    println!("Mean of A: {}", a.mean_int());
    println!("Max of A: {:?}", a.max());
    println!("Min of A: {:?}", a.min());
    
    // 7. Sum along axis
    println!("\n7. Sum along axis:");
    let sum_axis0 = a.sum_axis(0).unwrap();
    println!("Sum along axis 0: {}", sum_axis0);
    
    let sum_axis1 = a.sum_axis(1).unwrap();
    println!("Sum along axis 1: {}", sum_axis1);
    
    // 8. Function mapping
    println!("\n8. Function mapping:");
    let squared = a.map(|x| x * x);
    println!("A squared:\n{}", squared);
    
    let doubled = a.map(|x| x * 2);
    println!("\nA doubled:\n{}", doubled);
    
    // 9. Index access
    println!("\n9. Index access:");
    println!("A[0, 0] = {}", a[[0, 0]]);
    println!("A[1, 1] = {}", a[[1, 1]]);
    
    // 10. Large array example
    println!("\n10. Large array example:");
    let large_data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
    let large_arr = Array::from_vec(large_data, vec![4, 5]).unwrap();
    println!("4x5 array:\n{}", large_arr);
    
    let large_sum = large_arr.sum();
    println!("\nSum of large array: {}", large_sum);
    println!("Mean of large array: {:.2}", large_arr.mean());
    
    // 11. Random number generation
    println!("\n11. Random number generation:");
    
    // Generate random numbers between 0-1
    let random_arr = Array::<f64>::random(vec![2, 3]);
    println!("Random array (0-1):\n{}", random_arr);
    
    // Generate random integers in specified range
    let randint_arr = Array::<i32>::randint(vec![2, 3], 1, 10);
    println!("\nRandom integer array (1-9):\n{}", randint_arr);
    
    // Generate normal distribution random numbers
    let randn_arr = Array::<f64>::randn(vec![2, 2]);
    println!("\nNormal distribution random array:\n{}", randn_arr);
    
    // Generate uniform distribution random numbers in specified range
    let uniform_arr = Array::<f64>::uniform(vec![2, 2], -2.0, 2.0);
    println!("\nUniform distribution random array (-2.0 to 2.0):\n{}", uniform_arr);
    
    // Random array statistics
    println!("\nRandom array statistics:");
    println!("Random array mean: {:.4}", random_arr.mean());
    println!("Random array max: {:.4}", random_arr.max().unwrap());
    println!("Random array min: {:.4}", random_arr.min().unwrap());
    
    // 12. Broadcasting examples
    println!("\n12. Broadcasting examples:");
    
    // Example 1: Scalar-like array with matrix
    let matrix = Array::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let scalar = Array::from_vec(vec![10], vec![1, 1]).unwrap();
    let broadcast_result1 = matrix.clone() + scalar.clone();
    println!("Matrix (2x2):\n{}", matrix);
    println!("Scalar-like (1x1):\n{}", scalar);
    println!("Matrix + Scalar:\n{}", broadcast_result1);
    
    // Example 2: Vector with matrix
    let vector = Array::from_vec(vec![1, 2], vec![2]).unwrap();
    let column = Array::from_vec(vec![10, 20], vec![2, 1]).unwrap();
    let broadcast_result2 = vector.clone() + column.clone();
    println!("\nVector (2,):\n{}", vector);
    println!("Column (2x1):\n{}", column);
    println!("Vector + Column:\n{}", broadcast_result2);
    
    // Example 3: Broadcasting with multiplication
    let row = Array::from_vec(vec![2, 3], vec![1, 2]).unwrap();
    let col = Array::from_vec(vec![4, 5, 6], vec![3, 1]).unwrap();
    let broadcast_result3 = row.clone() * col.clone();
    println!("\nRow (1x2):\n{}", row);
    println!("Column (3x1):\n{}", col);
    println!("Row * Column:\n{}", broadcast_result3);
    
    // 13. Trigonometric functions
    println!("\n13. Trigonometric functions:");
    
    use std::f64::consts::PI;
    let angles = Array::from_vec(vec![0.0, PI/6.0, PI/4.0, PI/3.0, PI/2.0], vec![5]).unwrap();
    println!("Angles (radians):\n{}", angles);
    
    let sin_values = angles.sin();
    let cos_values = angles.cos();
    let tan_values = angles.tan();
    
    println!("\nSine values:\n{}", sin_values);
    println!("Cosine values:\n{}", cos_values);
    println!("Tangent values:\n{}", tan_values);
    
    // Inverse trigonometric functions
    let values = Array::from_vec(vec![0.0, 0.5, 0.866, 1.0], vec![4]).unwrap();
    let asin_values = values.asin();
    let acos_values = values.acos();
    
    println!("\nValues for inverse functions:\n{}", values);
    println!("Arcsine values:\n{}", asin_values);
    println!("Arccosine values:\n{}", acos_values);
    
    // Hyperbolic functions
    let hyp_values = Array::from_vec(vec![0.0, 1.0, 2.0], vec![3]).unwrap();
    let sinh_values = hyp_values.sinh();
    let cosh_values = hyp_values.cosh();
    let tanh_values = hyp_values.tanh();
    
    println!("\nHyperbolic functions for [0, 1, 2]:");
    println!("Sinh values:\n{}", sinh_values);
    println!("Cosh values:\n{}", cosh_values);
    println!("Tanh values:\n{}", tanh_values);
    
    // Angle conversion
    let degrees = Array::from_vec(vec![0.0, 30.0, 45.0, 60.0, 90.0], vec![5]).unwrap();
    let radians_converted = degrees.to_radians();
    let back_to_degrees = radians_converted.to_degrees();
    
    println!("\nAngle conversion:");
    println!("Degrees:\n{}", degrees);
    println!("Converted to radians:\n{}", radians_converted);
    println!("Back to degrees:\n{}", back_to_degrees);
    
    // 14. Logarithmic and exponential functions
    println!("\n14. Logarithmic and exponential functions:");
    
    // Natural logarithm and exponential
    let exp_values = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0], vec![4]).unwrap();
    println!("Values for exp/ln:");
    println!("{}", exp_values);
    
    let exp_results = exp_values.exp();
    println!("Exponential (e^x):");
    println!("{}", exp_results);
    
    let ln_results = exp_results.ln();
    println!("Natural log (should match original):");
    println!("{}", ln_results);
    
    // Base-10 and base-2 logarithms
    let powers_of_10 = Array::from_vec(vec![1.0, 10.0, 100.0, 1000.0], vec![4]).unwrap();
    println!("\nPowers of 10:");
    println!("{}", powers_of_10);
    
    let log10_results = powers_of_10.log10();
    println!("Base-10 logarithm:");
    println!("{}", log10_results);
    
    let powers_of_2 = Array::from_vec(vec![1.0, 2.0, 4.0, 8.0, 16.0], vec![5]).unwrap();
    println!("\nPowers of 2:");
    println!("{}", powers_of_2);
    
    let log2_results = powers_of_2.log2();
    println!("Base-2 logarithm:");
    println!("{}", log2_results);
    
    let exp2_results = log2_results.exp2();
    println!("2^x (should match original):");
    println!("{}", exp2_results);
    
    // Custom base logarithm
    let base_3_values = Array::from_vec(vec![1.0, 3.0, 9.0, 27.0], vec![4]).unwrap();
    println!("\nPowers of 3:");
    println!("{}", base_3_values);
    
    let log3_results = base_3_values.log(3.0);
    println!("Base-3 logarithm:");
    println!("{}", log3_results);
    
    // High precision functions for small values
    let small_values = Array::from_vec(vec![0.0, 0.01, 0.1, 0.5], vec![4]).unwrap();
    println!("\nSmall values:");
    println!("{}", small_values);
    
    let exp_m1_results = small_values.exp_m1();
    println!("exp(x) - 1 (more accurate for small x):");
    println!("{}", exp_m1_results);
    
    let ln_1p_results = small_values.ln_1p();
    println!("ln(1 + x) (more accurate for small x):");
    println!("{}", ln_1p_results);

    println!("\n=== Demo Complete ===");
}