use vectra::prelude::*;

fn main() {
    println!("=== Display Format Test ===");
    
    // Test 1D arrays
    println!("\n1. Small 1D array:");
    let small_1d = Array::from_vec(vec![1, 2, 3, 4, 5], vec![5]).unwrap();
    println!("{}", small_1d);
    
    println!("\n2. Large 1D array (should show ellipsis):");
    let large_1d = Array::from_vec((1..=20).collect::<Vec<_>>(), vec![20]).unwrap();
    println!("{}", large_1d);
    
    // Test 2D arrays
    println!("\n3. Small 2D array:");
    let small_2d = Array::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    println!("{}", small_2d);
    
    println!("\n4. Medium 2D array:");
    let medium_2d = Array::from_vec((1..=20).collect::<Vec<_>>(), vec![4, 5]).unwrap();
    println!("{}", medium_2d);
    
    println!("\n5. Large 2D array (should show ellipsis):");
    let large_2d = Array::from_vec((1..=100).collect::<Vec<_>>(), vec![10, 10]).unwrap();
    println!("{}", large_2d);
    
    // Test with different number types
    println!("\n6. Float array with alignment:");
    let float_array = Array::from_vec(vec![1.0, 22.5, 333.75, 4444.125], vec![2, 2]).unwrap();
    println!("{}", float_array);
    
    // Test 3D array
    println!("\n7. Small 3D array (2x3x4):");
    let array_3d = Array::from_vec((1..=24).collect::<Vec<_>>(), vec![2, 3, 4]).unwrap();
    println!("{}", array_3d);
    
    println!("\n8. Large 3D array (5x4x3, should show ellipsis):");
    let large_3d = Array::from_vec((1..=60).collect::<Vec<_>>(), vec![5, 4, 3]).unwrap();
    println!("{}", large_3d);
    
    println!("\n9. Small 4D array (2x3x4x2):");
    let array_4d = Array::from_vec((1..=48).collect::<Vec<_>>(), vec![2, 3, 4, 2]).unwrap();
    println!("{}", array_4d);
    
    println!("\n10. Large 4D array (5x4x3x4, should show ellipsis):");
    let large_4d = Array::from_vec((1..=240).collect::<Vec<_>>(), vec![5, 4, 3, 4]).unwrap();
    println!("{}", large_4d);
    
    println!("\n11. 5D array (should show shape and sample data):");
    let array_5d = Array::from_vec((1..=120).collect::<Vec<_>>(), vec![2, 3, 4, 5, 1]).unwrap();
    println!("{}", array_5d);
    
    println!("\n=== Test Complete ===");
}