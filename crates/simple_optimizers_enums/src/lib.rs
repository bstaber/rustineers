pub mod optimizers;

use optimizers::Optimizer;

/// Runs an optimization loop over multiple steps.
///
/// # Arguments
/// - `optimizer`: The optimizer to use (e.g., GradientDescent, Momentum).
/// - `weights`: Mutable reference to the weights vector to be optimized.
/// - `grad_fn`: A closure that computes the gradient given the current weights.
/// - `num_steps`: Number of iterations to run.
pub fn run_optimization(
    optimizer: &mut Optimizer,
    weights: &mut [f64],
    grad_fn: impl Fn(&[f64]) -> Vec<f64>,
    num_steps: usize,
) {
    for _ in 0..num_steps {
        let grads = grad_fn(weights);
        optimizer.step(weights, &grads);
    }
}
