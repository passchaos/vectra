//! Trigonometric Functions Examples
//!
//! This example demonstrates trigonometric and hyperbolic functions in Vectra,
//! including sine, cosine, tangent, and their inverse functions.

use vectra::prelude::*;
use std::f64::consts::PI;

fn main() {
    println!("=== Trigonometric Functions Examples ===");

    // 1. Basic trigonometric functions
    println!("\n1. Basic trigonometric functions:");
    
    let angles = Array::from_vec(vec![0.0, PI/6.0, PI/4.0, PI/3.0, PI/2.0], [5]);
    println!("Angles (radians): [0, π/6, π/4, π/3, π/2]\n{}", angles);
    
    let sin_values = angles.sin();
    println!("\nSine values:\n{}", sin_values);
    
    let cos_values = angles.cos();
    println!("\nCosine values:\n{}", cos_values);
    
    let tan_values = angles.tan();
    println!("\nTangent values:\n{}", tan_values);

    // 2. Full circle trigonometry
    println!("\n2. Full circle trigonometry:");
    
    let full_circle = Array::from_vec(
        vec![0.0, PI/2.0, PI, 3.0*PI/2.0, 2.0*PI], 
        [5]
    );
    println!("Full circle angles: [0, π/2, π, 3π/2, 2π]\n{}", full_circle);
    
    let full_sin = full_circle.sin();
    println!("\nSine over full circle:\n{}", full_sin);
    
    let full_cos = full_circle.cos();
    println!("\nCosine over full circle:\n{}", full_cos);

    // 3. Inverse trigonometric functions
    println!("\n3. Inverse trigonometric functions:");
    
    let values = Array::from_vec(vec![0.0, 0.5, 0.707, 0.866, 1.0], [5]);
    println!("Values for inverse functions: [0, 0.5, √2/2, √3/2, 1]\n{}", values);
    
    let asin_values = values.asin();
    println!("\nArcsine values (radians):\n{}", asin_values);
    
    let acos_values = values.acos();
    println!("\nArccosine values (radians):\n{}", acos_values);
    
    // Arctangent with extended range
    let tan_inputs = Array::from_vec(vec![-1.0, 0.0, 1.0, 1.732, f64::INFINITY], [5]);
    println!("\nTangent inputs: [-1, 0, 1, √3, ∞]\n{}", tan_inputs);
    
    let atan_values = tan_inputs.atan();
    println!("\nArctangent values (radians):\n{}", atan_values);

    // 4. Degree conversions
    println!("\n4. Degree conversions:");
    
    let degrees = Array::from_vec(vec![0.0, 30.0, 45.0, 60.0, 90.0, 180.0, 360.0], [7]);
    println!("Angles in degrees:\n{}", degrees);
    
    // Convert to radians
    let radians = degrees.map(|x| x * PI / 180.0);
    println!("\nConverted to radians:\n{}", radians);
    
    let sin_degrees = radians.sin();
    println!("\nSine of degree angles:\n{}", sin_degrees);
    
    let cos_degrees = radians.cos();
    println!("\nCosine of degree angles:\n{}", cos_degrees);

    // 5. Hyperbolic functions
    println!("\n5. Hyperbolic functions:");
    
    let hyp_values = Array::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0], [5]);
    println!("Values for hyperbolic functions:\n{}", hyp_values);
    
    let sinh_values = hyp_values.sinh();
    println!("\nHyperbolic sine (sinh):\n{}", sinh_values);
    
    let cosh_values = hyp_values.cosh();
    println!("\nHyperbolic cosine (cosh):\n{}", cosh_values);
    
    let tanh_values = hyp_values.tanh();
    println!("\nHyperbolic tangent (tanh):\n{}", tanh_values);

    // 6. Inverse hyperbolic functions
    println!("\n6. Inverse hyperbolic functions:");
    
    let asinh_inputs = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], [5]);
    println!("Inputs for asinh:\n{}", asinh_inputs);
    
    let asinh_values = asinh_inputs.asinh();
    println!("\nInverse hyperbolic sine (asinh):\n{}", asinh_values);
    
    let acosh_inputs = Array::from_vec(vec![1.0, 1.5, 2.0, 3.0, 5.0], [5]);
    println!("\nInputs for acosh (≥1):\n{}", acosh_inputs);
    
    let acosh_values = acosh_inputs.acosh();
    println!("\nInverse hyperbolic cosine (acosh):\n{}", acosh_values);
    
    let atanh_inputs = Array::from_vec(vec![-0.9, -0.5, 0.0, 0.5, 0.9], [5]);
    println!("\nInputs for atanh (-1 < x < 1):\n{}", atanh_inputs);
    
    let atanh_values = atanh_inputs.atanh();
    println!("\nInverse hyperbolic tangent (atanh):\n{}", atanh_values);

    // 7. Trigonometric identities verification
    println!("\n7. Trigonometric identities verification:");
    
    let test_angles = Array::from_vec(vec![PI/6.0, PI/4.0, PI/3.0], [3]);
    println!("Test angles: [π/6, π/4, π/3]\n{}", test_angles);
    
    let sin_vals = test_angles.sin();
    let cos_vals = test_angles.cos();
    
    // sin²(x) + cos²(x) = 1
    let sin_squared = sin_vals.map(|x| x * x);
    let cos_squared = cos_vals.map(|x| x * x);
    let identity_check = &sin_squared + &cos_squared;
    
    println!("\nsin²(x):\n{}", sin_squared);
    println!("\ncos²(x):\n{}", cos_squared);
    println!("\nsin²(x) + cos²(x) (should be ≈ 1):\n{}", identity_check);

    // 8. Wave generation
    println!("\n8. Wave generation:");
    
    // Generate a sine wave
    let time_points: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
    let time_array = Array::from_vec(time_points, [20]);
    
    println!("Time points (0 to 1.9, step 0.1):");
    println!("Shape: {:?}, First 5 values: [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}]", 
             time_array.shape(), time_array[[0]], time_array[[1]], 
             time_array[[2]], time_array[[3]], time_array[[4]]);
    
    // Sine wave with frequency 2π (1 Hz)
    let sine_wave = time_array.map(|t| (2.0 * PI * t).sin());
    println!("\nSine wave (1 Hz):");
    println!("First 10 values: {:?}", 
             (0..10).map(|i| format!("{:.3}", sine_wave[[i]])).collect::<Vec<_>>());
    
    // Cosine wave with frequency 4π (2 Hz)
    let cosine_wave = time_array.map(|t| (4.0 * PI * t).cos());
    println!("\nCosine wave (2 Hz):");
    println!("First 10 values: {:?}", 
             (0..10).map(|i| format!("{:.3}", cosine_wave[[i]])).collect::<Vec<_>>());

    // 9. Complex trigonometric calculations
    println!("\n9. Complex trigonometric calculations:");
    
    let complex_angles = Array::from_vec(
        vec![PI/12.0, PI/8.0, PI/6.0, PI/4.0, PI/3.0, 5.0*PI/12.0], 
        [2, 3]
    );
    println!("Complex angle matrix (2x3):\n{}", complex_angles);
    
    let sin_matrix = complex_angles.sin();
    let cos_matrix = complex_angles.cos();
    
    println!("\nSine matrix:\n{}", sin_matrix);
    println!("\nCosine matrix:\n{}", cos_matrix);
    
    // Calculate sin(x) * cos(x) = sin(2x)/2
    let sin_cos_product = &sin_matrix * &cos_matrix;
    println!("\nsin(x) * cos(x):\n{}", sin_cos_product);

    // 10. Practical applications
    println!("\n10. Practical applications:");
    
    // Calculate positions on a unit circle
    let circle_angles = Array::from_vec(
        (0..8).map(|i| i as f64 * PI / 4.0).collect(), 
        [8]
    );
    println!("Circle angles (8 points): [0, π/4, π/2, ..., 7π/4]\n{}", circle_angles);
    
    let x_positions = circle_angles.cos();
    let y_positions = circle_angles.sin();
    
    println!("\nX positions on unit circle:\n{}", x_positions);
    println!("\nY positions on unit circle:\n{}", y_positions);
    
    // Verify they're on the unit circle: x² + y² = 1
    let radius_squared = &x_positions.map(|x| x * x) + &y_positions.map(|y| y * y);
    println!("\nRadius squared (should be ≈ 1):\n{}", radius_squared);

    // Pendulum motion simulation
    println!("\n11. Pendulum motion simulation:");
    
    let time_steps: Vec<f64> = (0..10).map(|i| i as f64 * 0.2).collect();
    let time_sim = Array::from_vec(time_steps, [10]);
    
    // Simple harmonic motion: θ(t) = A * cos(ωt + φ)
    let amplitude = 0.5; // 0.5 radians
    let frequency = 2.0; // rad/s
    let phase = 0.0;
    
    let pendulum_angle = time_sim.map(|t| amplitude * (frequency * t + phase).cos());
    println!("Pendulum angle over time:\n{}", pendulum_angle);
    
    let pendulum_velocity = time_sim.map(|t| -amplitude * frequency * (frequency * t + phase).sin());
    println!("\nPendulum velocity over time:\n{}", pendulum_velocity);

    println!("\n=== Trigonometric Functions Examples Complete ===");
}
