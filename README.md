# Vectra - Multi-dimensional Array Library for Rust

[![Crates.io](https://img.shields.io/crates/v/vectra.svg)](https://crates.io/crates/vectra)
[![Documentation](https://docs.rs/vectra/badge.svg)](https://docs.rs/vectra)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Vectra is a high-performance multi-dimensional array library for Rust, inspired by NumPy and influenced by Rust's ndarray ecosystem. It provides efficient array operations, mathematical functions, and linear algebra capabilities with a focus on performance and ergonomics.

## Features

- **Multi-dimensional Arrays**: Support for N-dimensional arrays with flexible indexing
- **Static Dimensions**: Compile-time dimensionality specification (not shape) for better code clarity and development experience. The number of dimensions (1D, 2D, 3D, etc.) is known at compile time, while the actual sizes can still be dynamic. This design choice prioritizes ease of use and code readability - when you see a 2D array type, you immediately know you're working with a matrix, making code both easier to write and understand
- **Broadcasting**: NumPy-style broadcasting for element-wise operations
- **Mathematical Functions**: Comprehensive set of mathematical operations including:
  - Trigonometric functions (sin, cos, tan, etc.)
  - Hyperbolic functions (sinh, cosh, tanh, etc.)
  - Logarithmic and exponential functions
  - Power and root functions
- **Linear Algebra**: Matrix multiplication with multiple backend options:
  - BLAS integration for high performance
  - Faer backend for pure Rust implementation
  - Custom optimized implementations
- **Random Number Generation**: Built-in support for random array creation
- **Memory Efficient**: Zero-copy operations where possible
- **Type Safety**: Compile-time dimension checking
- **Familiar API**: Inspired by NumPy and influenced by Rust's ndarray library for intuitive usage

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
vectra = "0.2.0"
```

For BLAS support (recommended for better performance):

```toml
[dependencies]
vectra = { version = "0.2.0", features = ["blas"] }
```

## Quick Start

```rust
use vectra::prelude::*;

fn main() {
    // Create arrays
    let zeros = Array::<_, f64>::zeros([2, 3]);
    let ones = Array::<_, i32>::ones([3, 3]);
    let eye = Array::<_, f32>::eye(3); // Identity matrix
    
    // Create from vector
    let data = vec![1, 2, 3, 4, 5, 6];
    let arr = Array::from_vec(data, [2, 3]);
    
    // Array operations
    let reshaped = arr.reshape([3, 2]);
    let transposed = arr.transpose();
    
    // Mathematical operations
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], [2, 2]);
    
    let sum = &a + &b;                    // Element-wise addition
    let product = &a * &b;                // Element-wise multiplication
    let dot_product = a.matmul(&b);       // Matrix multiplication
    
    // Mathematical functions
    let angles = Array::from_vec(vec![0.0, std::f64::consts::PI/2.0], [2]);
    let sines = angles.sin();
    let exponentials = angles.exp();
    
    // Random arrays
    let random_arr = Array::<_, f64>::random([3, 3]);
    let normal_arr = Array::<_, f64>::randn([2, 4]);
}
```

## Array Creation

```rust
use vectra::prelude::*;

// Create arrays filled with specific values
let zeros = Array::<_, f64>::zeros([2, 3]);
let ones = Array::<_, i32>::ones([3, 3]);
let filled = Array::full([2, 2], 42);

// Create from existing data
let data = vec![1, 2, 3, 4, 5, 6];
let arr = Array::from_vec(data, [2, 3]);

// Create ranges
let range1d = Array::arange(0, 10, 1);        // [0, 1, 2, ..., 9]
let range_count = Array::arange_c(0, 2, 5);   // 5 elements starting from 0 with step 2

// Random arrays
let random = Array::<_, f64>::random([3, 3]);           // Uniform [0, 1)
let uniform = Array::uniform([2, 2], -1.0, 1.0);        // Uniform [-1, 1)
let normal = Array::<_, f64>::randn([2, 3]);             // Standard normal distribution

// Identity matrix
let identity = Array::<_, f64>::eye(4);
```

## Array Operations

```rust
use vectra::prelude::*;

let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6], [2, 3]);

// Reshaping and transposition
let reshaped = arr.reshape([3, 2]);
let transposed = arr.transpose();  // 2D arrays only

// Indexing
let element = arr[[0, 1]];  // Access single element
let mut arr_mut = arr.clone();
arr_mut[[1, 2]] = 42;       // Modify element

// Broadcasting operations
let a = Array::from_vec(vec![1, 2, 3], [3, 1]);
let b = Array::from_vec(vec![4, 5], [1, 2]);
let broadcasted = &a + &b;  // Results in [3, 2] array

// Scalar operations
let scaled = arr.mul_scalar(2);
let shifted = arr.add_scalar(10);
```

## Mathematical Functions

```rust
use vectra::prelude::*;

let arr = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0], [2, 2]);

// Trigonometric functions
let sines = arr.sin();
let cosines = arr.cos();
let tangents = arr.tan();

// Inverse trigonometric functions
let arcsines = arr.asin();
let arccosines = arr.acos();
let arctangents = arr.atan();

// Hyperbolic functions
let sinh_vals = arr.sinh();
let cosh_vals = arr.cosh();
let tanh_vals = arr.tanh();

// Exponential and logarithmic functions
let exponentials = arr.exp();
let logarithms = arr.ln();
let log10_vals = arr.log10();
let log2_vals = arr.log2();

// Power and root functions
let squares = arr.pow2();
let cubes = arr.powi(3);
let square_roots = arr.sqrt();
let cube_roots = arr.cbrt();

// Other mathematical functions
let absolute = arr.abs();
let rounded = arr.round();
let floored = arr.floor();
let ceiled = arr.ceil();
```

## Linear Algebra

```rust
use vectra::prelude::*;

let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], [2, 2]);

// Matrix multiplication with default policy
let result = a.matmul(&b);

// Matrix multiplication with specific backend
let result_blas = a.matmul_with_policy(&b, MatmulPolicy::Blas);
let result_faer = a.matmul_with_policy(&b, MatmulPolicy::Faer);
let result_naive = a.matmul_with_policy(&b, MatmulPolicy::Naive);

// Identity matrix
let identity = Array::<_, f64>::eye(3);

// Transpose
let transposed = a.transpose();
```

## Performance

Vectra is designed for high performance with multiple optimization strategies:

- **BLAS Integration**: Optional BLAS backend for optimized linear algebra operations
- **Faer Backend**: Pure Rust high-performance linear algebra
- **SIMD Optimizations**: Vectorized operations where supported
- **Memory Layout Control**: Support for both row-major and column-major layouts
- **Zero-copy Operations**: Efficient memory usage through view-based operations

## Examples

Check out the `examples/` directory for more comprehensive examples:

- `demo.rs`: Basic usage and array operations
- `matmul.rs`: Matrix multiplication performance comparison

Run examples with:

```bash
cargo run --example demo
cargo run --example matmul
```

## Features

- `default`: Includes BLAS support
- `blas`: Enable BLAS backend for linear algebra operations

## Requirements

- Rust 2024 edition
- For BLAS support: appropriate BLAS library (OpenBLAS on Linux, Accelerate on macOS)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

### Generating Documentation

Vectra includes comprehensive API documentation that can be generated using Rust's built-in documentation system:

```bash
# Generate documentation for the library
cargo doc

# Generate and open documentation in your browser
cargo doc --open

# Generate documentation with private items (for development)
cargo doc --document-private-items

# Generate documentation for all dependencies
cargo doc --no-deps
```

### Documentation Features

The generated documentation includes:

- **Complete API Reference**: Detailed documentation for all public types, functions, and methods
- **Code Examples**: Practical examples showing how to use each feature
- **Mathematical Explanations**: Clear explanations of algorithms and mathematical operations
- **Performance Notes**: Information about computational complexity and optimization strategies
- **Cross-References**: Links between related functions and types

### Key Documentation Sections

#### Core Array Operations
- Array creation and initialization methods
- Shape manipulation (reshape, transpose, permute)
- Element access and indexing
- Broadcasting and dimension handling

#### Mathematical Functions
- Element-wise operations (trigonometric, logarithmic, exponential)
- Linear algebra operations (matrix multiplication, decompositions)
- Statistical functions (mean, variance, etc.)
- Random number generation

#### Performance and Optimization
- Matrix multiplication policies (BLAS, Faer, custom implementations)
- Memory layout considerations (row-major vs column-major)
- SIMD optimizations and platform-specific features

### Documentation Examples

The documentation includes extensive examples for common use cases:

```rust
// Array creation examples
let arr = Array::zeros([3, 3]);           // Create 3x3 zero matrix
let arr = Array::ones([2, 4]);            // Create 2x4 ones matrix
let arr = Array::arange(0, 10, 2);        // Create [0, 2, 4, 6, 8]

// Mathematical operations examples
let result = arr.sin();                   // Element-wise sine
let result = arr.matmul(&other);          // Matrix multiplication
let result = arr.broadcast_to([4, 3]);    // Broadcasting

// Advanced operations examples
let gathered = arr.gather(0, &indices);   // Gather along axis
let reshaped = arr.reshape([2, -1]);      // Reshape with inference
```

### Building Documentation Locally

For development and contribution:

```bash
# Install documentation dependencies
cargo install mdbook  # For additional documentation

# Generate docs with all features enabled
cargo doc --all-features

# Generate docs for specific features
cargo doc --features "blas"

# Check documentation for warnings
cargo doc 2>&1 | grep warning
```

### Documentation Standards

Vectra follows Rust documentation best practices:

- **Comprehensive Examples**: Every public function includes usage examples
- **Mathematical Notation**: Clear mathematical descriptions where applicable
- **Performance Notes**: Computational complexity and optimization details
- **Error Conditions**: Documentation of panic conditions and error cases
- **Cross-Platform Notes**: Platform-specific behavior and optimizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Documentation Contributions

When contributing to Vectra, please ensure:

- All public APIs are documented with examples
- Mathematical operations include clear explanations
- Performance characteristics are documented
- Examples compile and run correctly
- Documentation follows the established style

## Acknowledgments

- Inspired by NumPy's design and functionality
- Built on top of excellent Rust crates like `faer`, `rand`, and `num-traits`
- BLAS integration for high-performance linear algebra
