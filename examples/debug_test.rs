use vectra::prelude::*;

fn main() {
    // Test Debug output for small arrays
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    println!("Debug format: {:?}", a);
    println!("Display format: {}", a);
    
    // Test 1D array
    let b = Array::from_vec(vec![1, 2, 3], vec![3]).unwrap();
    println!("\n1D Debug format: {:?}", b);
    println!("1D Display format: {}", b);
    
    // Test 3D array
    let c = Array::from_vec((1..=8).collect::<Vec<i32>>(), vec![2, 2, 2]).unwrap();
    println!("\n3D Debug format: {:?}", c);
    println!("3D Display format: {}", c);
}