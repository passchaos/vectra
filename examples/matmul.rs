use std::time::Instant;

use plotiron::prelude::*;
use vectra::prelude::*;

fn main() {
    let mut fig = figure();

    let ax = fig.add_subplot();

    let mut sizes = vec![];
    let mut costs = vec![];

    for i in (250..=2000).step_by(250) {
        let a = Array::<f32>::random(vec![i, i]);
        let b = Array::<f32>::random(vec![i, i]);

        let begin = Instant::now();
        let c = a.matmul(&b).unwrap();

        sizes.push(i as f64);

        let cost_ms = begin.elapsed().as_secs_f64() * 1000.0;
        costs.push(cost_ms);

        println!("get matmul result: mnk= {i} cost_ms= {cost_ms}");
    }

    ax.add_plot(Plot::line(sizes, costs)).legend(true);
    fig.show();
}
