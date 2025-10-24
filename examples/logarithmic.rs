//! Logarithmic and Exponential Functions Examples
//!
//! This example demonstrates logarithmic and exponential functions in Vectra,
//! including natural log, base-10, base-2, and custom base logarithms.

use std::f64::consts::{E, LN_2, LN_10};
use vectra::prelude::*;

fn main() {
    println!("=== Logarithmic and Exponential Functions Examples ===");

    // 1. Natural exponential and logarithm
    println!("\n1. Natural exponential and logarithm:");

    let exp_inputs = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0], [5]);
    println!("Input values:\n{}", exp_inputs);

    let exp_results = exp_inputs.exp();
    println!("\nExponential (e^x):\n{}", exp_results);

    let ln_results = exp_results.ln();
    println!("\nNatural log of e^x (should match input):\n{}", ln_results);

    // Verify e^ln(x) = x
    let test_values = Array::from_vec(vec![1.0, 2.0, 5.0, 10.0, 100.0], [5]);
    let ln_then_exp = test_values.ln().exp();
    println!("\nOriginal values:\n{}", test_values);
    println!("\ne^(ln(x)) (should match original):\n{}", ln_then_exp);

    // 2. Base-10 logarithm and powers
    println!("\n2. Base-10 logarithm and powers:");

    let powers_of_10 = Array::from_vec(vec![0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], [6]);
    println!("Powers of 10:\n{}", powers_of_10);

    let log10_results = powers_of_10.log10();
    println!("\nBase-10 logarithm:\n{}", log10_results);

    // Verify 10^log10(x) = x
    let exp10_results = log10_results.map(|x| 10.0_f64.powf(*x));
    println!(
        "\n10^(log10(x)) (should match original):\n{}",
        exp10_results
    );

    // 3. Base-2 logarithm and powers
    println!("\n3. Base-2 logarithm and powers:");

    let powers_of_2 = Array::from_vec(vec![0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], [8]);
    println!("Powers of 2:\n{}", powers_of_2);

    let log2_results = powers_of_2.log2();
    println!("\nBase-2 logarithm:\n{}", log2_results);

    let exp2_results = log2_results.exp2();
    println!("\n2^(log2(x)) (should match original):\n{}", exp2_results);

    // 4. Custom base logarithms
    println!("\n4. Custom base logarithms:");

    // Base-3 logarithm
    let powers_of_3 = Array::from_vec(vec![1.0, 3.0, 9.0, 27.0, 81.0], [5]);
    println!("Powers of 3:\n{}", powers_of_3);

    let log3_results = powers_of_3.log(3.0);
    println!("\nBase-3 logarithm:\n{}", log3_results);

    // Base-5 logarithm
    let powers_of_5 = Array::from_vec(vec![1.0, 5.0, 25.0, 125.0], [4]);
    println!("\nPowers of 5:\n{}", powers_of_5);

    let log5_results = powers_of_5.log(5.0);
    println!("\nBase-5 logarithm:\n{}", log5_results);

    // 5. High precision functions for small values
    println!("\n5. High precision functions for small values:");

    let small_values = Array::from_vec(vec![0.0, 0.001, 0.01, 0.1, 0.5], [5]);
    println!("Small values:\n{}", small_values);

    let exp_m1_results = small_values.exp_m1();
    println!(
        "\nexp(x) - 1 (more accurate for small x):\n{}",
        exp_m1_results
    );

    let ln_1p_results = small_values.ln_1p();
    println!(
        "\nln(1 + x) (more accurate for small x):\n{}",
        ln_1p_results
    );

    // Compare with regular functions
    let regular_exp = small_values.exp().map(|x| x - 1.0);
    let regular_ln = small_values.map(|x| (1.0_f64 + *x).ln());

    println!("\nRegular exp(x) - 1:\n{}", regular_exp);
    println!("\nRegular ln(1 + x):\n{}", regular_ln);

    // 6. Exponential growth and decay
    println!("\n6. Exponential growth and decay:");

    // Population growth model: P(t) = P₀ * e^(rt)
    let time_points = Array::from_vec(vec![0.0_f64, 1.0, 2.0, 5.0, 10.0], [5]);
    let initial_population = 1000.0_f64;
    let growth_rate = 0.05_f64; // 5% per time unit

    println!("Time points:\n{}", time_points);

    let population = time_points.map(|t| initial_population * (growth_rate * *t).exp());
    println!("\nPopulation growth (5% rate):\n{}", population);

    // Radioactive decay: N(t) = N₀ * e^(-λt)
    let decay_constant = 0.1_f64;
    let initial_amount = 1000.0_f64;

    let remaining_amount = time_points.map(|t| initial_amount * (-decay_constant * *t).exp());
    println!("\nRadioactive decay (λ=0.1):\n{}", remaining_amount);

    // 7. Logarithmic scales and data transformation
    println!("\n7. Logarithmic scales and data transformation:");

    let data_values = Array::from_vec(vec![1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0], [6]);
    println!("Original data (wide range):\n{}", data_values);

    let log_transformed = data_values.log10();
    println!("\nLog10 transformed data:\n{}", log_transformed);

    let ln_transformed = data_values.ln();
    println!("\nNatural log transformed data:\n{}", ln_transformed);

    // 8. Compound interest calculations
    println!("\n8. Compound interest calculations:");

    let years = Array::from_vec(vec![1.0_f64, 5.0, 10.0, 20.0, 30.0], [5]);
    let principal = 1000.0_f64;
    let annual_rate = 0.07_f64; // 7% annual interest

    println!("Years:\n{}", years);

    // Continuous compounding: A = P * e^(rt)
    let continuous_compound = years.map(|t| principal * (annual_rate * *t).exp());
    println!(
        "\nContinuous compounding (7% annual):\n{}",
        continuous_compound
    );

    // Annual compounding: A = P * (1 + r)^t
    let annual_compound = years.map(|t| principal * (1.0_f64 + annual_rate).powf(*t));
    println!("\nAnnual compounding (7% annual):\n{}", annual_compound);

    // 9. Scientific logarithmic scales
    println!("\n9. Scientific logarithmic scales:");

    // pH scale: pH = -log10([H+])
    let hydrogen_concentrations = Array::from_vec(vec![1e-1_f64, 1e-3, 1e-7, 1e-10, 1e-14], [5]);
    println!(
        "Hydrogen ion concentrations [H+]:\n{}",
        hydrogen_concentrations
    );

    let ph_values = hydrogen_concentrations.map(|h| -(*h).log10());
    println!("\npH values:\n{}", ph_values);

    // Decibel scale: dB = 10 * log10(P/P₀)
    let power_ratios = Array::from_vec(vec![1.0_f64, 10.0, 100.0, 1000.0, 10000.0], [5]);
    println!("\nPower ratios (P/P₀):\n{}", power_ratios);

    let decibels = power_ratios.map(|p| 10.0_f64 * (*p).log10());
    println!("\nDecibel values:\n{}", decibels);

    // 10. Mathematical constants and relationships
    println!("\n10. Mathematical constants and relationships:");

    println!("Mathematical constants:");
    println!("e ≈ {:.6}", E);
    println!("ln(2) ≈ {:.6}", LN_2);
    println!("ln(10) ≈ {:.6}", LN_10);

    // Verify some mathematical relationships
    let test_vals = Array::from_vec(vec![E, E.powi(2), E.powi(3)], [3]);
    println!("\nPowers of e: [e, e², e³]\n{}", test_vals);

    let ln_of_powers = test_vals.ln();
    println!("\nln of powers (should be [1, 2, 3]):\n{}", ln_of_powers);

    // Change of base formula: log_a(x) = ln(x) / ln(a)
    let test_numbers = Array::from_vec(vec![8.0, 16.0, 32.0], [3]);
    let log2_direct = test_numbers.log2();
    let log2_change_base = test_numbers.ln().map(|x| *x / LN_2);

    println!("\nTest numbers:\n{}", test_numbers);
    println!("\nlog2 (direct):\n{}", log2_direct);
    println!("\nlog2 (change of base):\n{}", log2_change_base);

    // 11. Numerical stability examples
    println!("\n11. Numerical stability examples:");

    // Very small numbers
    let tiny_values = Array::from_vec(vec![1e-15, 1e-10, 1e-5, 1e-3], [4]);
    println!("Tiny values:\n{}", tiny_values);

    let ln_1p_tiny = tiny_values.ln_1p();
    let regular_ln_tiny = tiny_values.map(|x| (1.0_f64 + *x).ln());

    println!("\nln_1p (high precision):\n{}", ln_1p_tiny);
    println!("\nRegular ln(1+x):\n{}", regular_ln_tiny);

    // Very large numbers
    let large_values = Array::from_vec(vec![1e10, 1e20, 1e50, 1e100], [4]);
    println!("\nLarge values:\n{}", large_values);

    let ln_large = large_values.ln();
    println!("\nNatural log of large values:\n{}", ln_large);

    println!("\n=== Logarithmic and Exponential Functions Examples Complete ===");
}
