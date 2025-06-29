pub mod enum_based;
pub mod traits_based;
use traits_based::optimizers::Optimizer;

pub fn run_optimization<O: Optimizer>(
    optimizer: &mut O,
    weights: &mut [f64],
    grad_fn: impl Fn(&[f64]) -> Vec<f64>,
    num_steps: usize,
) {
    for _ in 0..num_steps {
        let grads = grad_fn(weights);
        optimizer.step(weights, &grads);
    }
}
