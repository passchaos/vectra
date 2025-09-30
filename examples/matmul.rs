use std::time::Instant;

use plotiron::prelude::*;
use vectra::prelude::*;

fn main() {
    let mut fig = figure();

    let ax = fig.add_subplot();

    for policy in [
        // MatmulPolicy::LoopReorder,
        // MatmulPolicy::LoopRecorderSimd,
        MatmulPolicy::Blas,
        MatmulPolicy::Faer,
        // MatmulPolicy::Blocking(64),
        // MatmulPolicy::Blocking(96),
        // MatmulPolicy::Blocking(128),
        // MatmulPolicy::Blocking(256),
        // MatmulPolicy::Blocking(512),
    ] {
        let mut sizes = vec![];
        let mut costs = vec![];

        for i in (250..=3000).step_by(250) {
            let a = Array::<_, f32>::random([i, i]);
            let b = Array::<_, f32>::random([i, i]);

            // warmup
            let _c = a.matmul(&b, policy);

            let begin = Instant::now();
            let _c = a.matmul(&b, policy);

            sizes.push(i as f64);

            let cost_ms = begin.elapsed().as_secs_f64() * 1000.0;
            costs.push(cost_ms);

            println!("get matmul result: policy={policy:?} mnk= {i} cost_ms= {cost_ms}");
        }

        ax.add_plot(Plot::line(sizes, costs).label(&format!("{policy:?}")))
            .legend(true);
    }

    // let svg_content = fig.to_svg();
    // std::fs::write("matmul.svg", svg_content).unwrap();
    fig.show();
}
