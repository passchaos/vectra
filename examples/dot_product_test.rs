use vectra::prelude::Array;

fn main() {
    // Create two 1D vectors
    let a = Array::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let b = Array::from_vec(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
    
    println!("Vector a: {}", a);
    println!("Vector b: {}", b);
    
    // Calculate dot product
    match a.dot(&b) {
        Ok(result) => {
            println!("Dot product: {}", result);
            println!("Expected: {} (1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32)", 1.0*4.0 + 2.0*5.0 + 3.0*6.0);
        }
        Err(e) => println!("Error: {}", e),
    }
    
    // Test with integer vectors
    let c = Array::from_vec(vec![1, 2, 3], vec![3]).unwrap();
    let d = Array::from_vec(vec![4, 5, 6], vec![3]).unwrap();
    
    println!("\nInteger vector c: {}", c);
    println!("Integer vector d: {}", d);
    
    match c.dot(&d) {
        Ok(result) => {
            println!("Integer dot product: {}", result);
            println!("Expected: {} (1*4 + 2*5 + 3*6 = 32)", 1*4 + 2*5 + 3*6);
        }
        Err(e) => println!("Error: {}", e),
    }
    
    // Test error cases
    let e = Array::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
    let f = Array::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    
    println!("\nTesting different length vectors:");
    match e.dot(&f) {
        Ok(result) => println!("Unexpected result: {}", result),
        Err(e) => println!("Expected error: {}", e),
    }
    
    // Test 2D array error
    let g = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let h = Array::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
    
    println!("\nTesting 2D array with 1D vector:");
    match g.dot(&h) {
        Ok(result) => println!("Unexpected result: {}", result),
        Err(e) => println!("Expected error: {}", e),
    }
}