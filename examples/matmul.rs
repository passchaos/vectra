use std::time::Instant;

use vectra::prelude::*;

fn main() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let b = Array::from_vec(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
    let c = a.matmul(&b).unwrap();

    println!("first c: {}", c);

    let ins1 = Instant::now();
    let a = Array::random(vec![2048, 1024]);
    let b = Array::random(vec![1024, 2048]);

    println!("a= {} b= {} elapsed= {:?}", a, b, ins1.elapsed());

    let ins2 = Instant::now();

    let c = a.matmul(&b).unwrap();

    println!("second c: {:?} elapsed= {:?}", c, ins2.elapsed());
}
