# Vectra - Rust Multi-dimensional Array Library

A feature-rich multi-dimensional array library, similar to Python's NumPy, designed specifically for the Rust language.

## Design Philosophy

Vectra is designed with **simplicity and ease of use** as the primary goals. Unlike other Rust libraries that prioritize maximum performance or compile-time safety, Vectra focuses on providing an intuitive and accessible API that makes multi-dimensional array operations straightforward for developers of all skill levels.

**Key Design Principles:**
- 🎯 **Simplicity First**: Clean, intuitive API that's easy to learn and use
- 🚀 **Developer Experience**: Familiar NumPy-like syntax for smooth transition
- 📚 **Accessibility**: Comprehensive examples and clear documentation
- 🔧 **Practicality**: Focus on common use cases rather than edge case optimization

While performance and safety are important, they are secondary to creating a library that developers actually enjoy using for rapid prototyping, data analysis, and scientific computing tasks.

## Features

- 📊 **Multi-dimensional Arrays**: Support for arbitrary dimensional array operations
- 🧮 **Mathematical Operations**: Complete mathematical operation support (addition, subtraction, multiplication, division, matrix multiplication, etc.)
- 📐 **Trigonometric Functions**: Complete set of trigonometric and hyperbolic functions
- 📊 **Logarithmic & Exponential Functions**: Natural log, base-10/2 log, exponential functions
- 📡 **Broadcasting**: Automatic shape alignment for operations between arrays of different shapes
- 📈 **Aggregation Functions**: Sum, mean, max, min, etc.
- 🔄 **Shape Operations**: Reshape, transpose and other array transformations
- 🛠️ **Ease of Use**: Intuitive API design, similar to NumPy experience

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
vectra = "0.1.0"
```

### Basic Usage

```rust
use vectra::Array;

fn main() {
    // Create arrays
    let zeros = Array::<f64>::zeros(vec![2, 3]);
    let ones = Array::<i32>::ones(vec![3, 3]);
    let eye = Array::<i32>::eye(3);
    
    // Create array from vector
    let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    println!("Array:\n{}", arr);
    
    // Mathematical operations
    let a = Array::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let b = Array::from_vec(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
    
    let sum = a.clone() + b.clone();
    let product = a.dot(&b).unwrap(); // Matrix multiplication
    
    // Aggregation functions
    println!("Sum: {}", a.sum());
    println!("Mean: {}", a.mean_int());
    println!("Max: {:?}", a.max());
}
```

## Core Features

### 1. Array Creation

```rust
// Create zero array
let zeros = Array::<f64>::zeros(vec![3, 4]);

// Create ones array
let ones = Array::<i32>::ones(vec![2, 2]);

// Create identity matrix
let identity = Array::<f64>::eye(3);

// Create from vector
let arr = Array::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();

// Random array creation
let random = Array::<f64>::random(vec![2, 3]);        // Random numbers between 0-1
let randint = Array::<i32>::randint(vec![2, 3], 1, 10); // Random integers between 1-9
let randn = Array::<f64>::randn(vec![2, 2]);          // Normal distribution random numbers
let uniform = Array::<f64>::uniform(vec![2, 2], -1.0, 1.0); // Uniform distribution random numbers
```

### 2. Array Operations

```rust
let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();

// Reshape
let reshaped = arr.reshape(vec![3, 2]).unwrap();

// Transpose (2D array)
let transposed = arr.transpose().unwrap();

// Index access
let element = arr[[0, 1]];
```

### 3. Mathematical Operations

```rust
let a = Array::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
let b = Array::from_vec(vec![5, 6, 7, 8], vec![2, 2]).unwrap();

// Element-wise operations
let sum = a.clone() + b.clone();
let diff = a.clone() - b.clone();
let product = a.clone() * b.clone();
let quotient = a.clone() / b.clone();

// Scalar operations
let scaled = a.mul_scalar(2);
let shifted = a.add_scalar(10);

// Matrix multiplication
let dot_product = a.dot(&b).unwrap();

// Broadcasting examples
let matrix = Array::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
let scalar = Array::from_vec(vec![10], vec![1, 1]).unwrap();
let broadcast_result = matrix + scalar; // Broadcasts scalar to matrix shape
println!("Broadcasting result: {:?}", broadcast_result);

// Vector and matrix broadcasting
let vector = Array::from_vec(vec![1, 2], vec![2]).unwrap();
let column = Array::from_vec(vec![10, 20], vec![2, 1]).unwrap();
let result = vector + column; // Results in 2x2 matrix
println!("Vector + Column: {:?}", result);

// Trigonometric functions
use std::f64::consts::PI;
let angles = Array::from_vec(vec![0.0, PI/4.0, PI/2.0], vec![3]).unwrap();
let sin_values = angles.sin();
let cos_values = angles.cos();
let tan_values = angles.tan();
println!("Sin: {:?}", sin_values);
println!("Cos: {:?}", cos_values);

// Hyperbolic functions
let values = Array::from_vec(vec![0.0, 1.0, 2.0], vec![3]).unwrap();
let sinh_values = values.sinh();
let tanh_values = values.tanh();
println!("Sinh: {:?}", sinh_values);

// Angle conversion
let degrees = Array::from_vec(vec![0.0, 90.0, 180.0], vec![3]).unwrap();
let radians = degrees.to_radians();
println!("Radians: {:?}", radians);

// Logarithmic and exponential functions
use std::f64::consts::E;
let values = Array::from_vec(vec![1.0, E, E*E], vec![3]).unwrap();
let ln_values = values.ln();
println!("Natural log: {:?}", ln_values); // [0, 1, 2]

// Base-specific logarithms
let powers_of_10 = Array::from_vec(vec![1.0, 10.0, 100.0], vec![3]).unwrap();
let log10_values = powers_of_10.log10();
println!("Log10: {:?}", log10_values); // [0, 1, 2]

let powers_of_2 = Array::from_vec(vec![1.0, 2.0, 4.0], vec![3]).unwrap();
let log2_values = powers_of_2.log2();
println!("Log2: {:?}", log2_values); // [0, 1, 2]

// Exponential functions
let exp_input = Array::from_vec(vec![0.0, 1.0, 2.0], vec![3]).unwrap();
let exp_values = exp_input.exp();
println!("Exp: {:?}", exp_values); // [1, e, e²]

let exp2_values = exp_input.exp2();
println!("Exp2: {:?}", exp2_values); // [1, 2, 4]
```

### 4. Aggregation Functions

```rust
let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();

// Global aggregation
let sum = arr.sum();
let mean = arr.mean_int(); // Integer mean
let max_val = arr.max();
let min_val = arr.min();

// Axis aggregation
let sum_axis0 = arr.sum_axis(0).unwrap();
let sum_axis1 = arr.sum_axis(1).unwrap();
```

### 5. Function Mapping

```rust
let arr = Array::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();

// Apply function to each element
let squared = arr.map(|x| x * x);
let doubled = arr.map(|x| x * 2);
```

## Array Properties

```rust
let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();

// Get shape
let shape = arr.shape(); // &[2, 3]

// Get number of dimensions
let ndim = arr.ndim(); // 2

// Get total number of elements
let size = arr.size(); // 6
```

## Indexing Operations

```rust
let mut arr = Array::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();

// Read element
let element = arr[[0, 1]];

// Modify element
arr[[1, 0]] = 10;

// Support multiple indexing methods
let val1 = arr[[0, 1]];           // 2D indexing
let val2 = arr[&[0, 1][..]];      // Slice indexing
```

## Error Handling

Most operations in the library return `Result` type, providing clear error messages:

```rust
// Shape mismatch error
let result = Array::from_vec(vec![1, 2, 3], vec![2, 2]);
match result {
    Ok(arr) => println!("Creation successful"),
    Err(e) => println!("Error: {}", e),
}

// Matrix multiplication dimension mismatch
let a = Array::from_vec(vec![1, 2], vec![1, 2]).unwrap();
let b = Array::from_vec(vec![1, 2, 3], vec![1, 3]).unwrap();
let result = a.dot(&b); // Returns error
```

## Performance Features

- **Zero-copy operations**: Most view operations don't copy data
- **Memory layout optimization**: Uses row-major storage, cache-friendly
- **Compile-time optimization**: Rust compiler optimization, near C language performance
- **Type specialization**: Optimized implementations for different numeric types

## Running Examples

```bash
# Run demo program
cargo run --example demo

# Run tests
cargo test
```

## Comparison with NumPy

| Feature | NumPy | Vectra |
|---------|-------|--------|
| Array creation | `np.zeros()`, `np.ones()` | `Array::zeros()`, `Array::ones()` |
| Shape operations | `arr.reshape()`, `arr.T` | `arr.reshape()`, `arr.transpose()` |
| Mathematical operations | `+`, `-`, `*`, `@` | `+`, `-`, `*`, `dot()` |
| Aggregation functions | `arr.sum()`, `arr.mean()` | `arr.sum()`, `arr.mean_int()` |
| Indexing | `arr[0, 1]` | `arr[[0, 1]]` |
