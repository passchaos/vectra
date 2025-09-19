use vectra::prelude::Array;

fn main() {
    println!("Testing Truncated SVD functionality...");
    
    // Test 1: 4x3 matrix with k=2
    println!("\n=== Test 1: 4x3 matrix with k=2 ===");
    let data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ];
    let matrix = Array::from_vec(data, vec![4, 3]).unwrap();
    
    match matrix.svd_truncated(2) {
        Ok((u, s, vt)) => {
            println!("Original matrix shape: {:?}", matrix.shape());
            println!("U shape: {:?}", u.shape());
            println!("S shape: {:?}", s.shape());
            println!("VT shape: {:?}", vt.shape());
            println!("Singular values: {:?}", s.iter().collect::<Vec<_>>());
        }
        Err(e) => println!("Error: {}", e),
    }
    
    // Test 2: 3x3 matrix with k=1 (top singular value only)
    println!("\n=== Test 2: 3x3 matrix with k=1 ===");
    let data2 = vec![
        2.0, 1.0, 0.0,
        1.0, 2.0, 1.0,
        0.0, 1.0, 2.0,
    ];
    let matrix2 = Array::from_vec(data2, vec![3, 3]).unwrap();
    
    match matrix2.svd_truncated(1) {
        Ok((u, s, vt)) => {
            println!("Original matrix shape: {:?}", matrix2.shape());
            println!("U shape: {:?}", u.shape());
            println!("S shape: {:?}", s.shape());
            println!("VT shape: {:?}", vt.shape());
            println!("Top singular value: {:?}", s.iter().collect::<Vec<_>>());
        }
        Err(e) => println!("Error: {}", e),
    }
    
    // Test 3: Compare with full SVD (first k values should match)
    println!("\n=== Test 3: Compare with full SVD ===");
    let data3 = vec![
        3.0, 2.0, 2.0,
        2.0, 3.0, -2.0,
    ];
    let matrix3 = Array::from_vec(data3, vec![2, 3]).unwrap();
    
    // Full SVD
    match matrix3.svd() {
        Ok((_, s_full, _)) => {
            println!("Full SVD singular values: {:?}", s_full.iter().collect::<Vec<_>>());
            
            // Truncated SVD with k=1
            match matrix3.svd_truncated(1) {
                Ok((_, s_trunc, _)) => {
                    println!("Truncated SVD (k=1) singular values: {:?}", s_trunc.iter().collect::<Vec<_>>());
                    let s_full_vec: Vec<_> = s_full.iter().collect();
                    let s_trunc_vec: Vec<_> = s_trunc.iter().collect();
                    let diff = *s_full_vec[0] - *s_trunc_vec[0];
                    println!("First singular value matches: {}", 
                        (diff as f64).abs() < 1e-10);
                }
                Err(e) => println!("Truncated SVD error: {}", e),
            }
        }
        Err(e) => println!("Full SVD error: {}", e),
    }
    
    // Test 4: Error cases
    println!("\n=== Test 4: Error cases ===");
    
    // k = 0
    match matrix.svd_truncated(0) {
        Ok(_) => println!("ERROR: k=0 should fail"),
        Err(e) => println!("k=0 correctly failed: {}", e),
    }
    
    // k > min(m, n)
    match matrix.svd_truncated(5) {
        Ok(_) => println!("ERROR: k=5 should fail for 4x3 matrix"),
        Err(e) => println!("k=5 correctly failed: {}", e),
    }
    
    // 1D array
    let vec1d = Array::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    match vec1d.svd_truncated(1) {
        Ok(_) => println!("ERROR: 1D array should fail"),
        Err(e) => println!("1D array correctly failed: {}", e),
    }
    
    println!("\nTruncated SVD tests completed!");
}