// ANCHOR: entry_point
pub mod optimizers;
use optimizers::Optimizer;

pub fn run_optimization<Opt: Optimizer>(
    optimizer: &mut Opt,
    weights: &mut [f64],
    grad_fn: impl Fn(&[f64]) -> Vec<f64>,
    num_steps: usize,
) {
    for _ in 0..num_steps {
        let grads = grad_fn(weights);
        optimizer.step(weights, &grads);
    }
}
// ANCHOR_END: entry_point

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use super::*;

    fn check_optimizer_converges<O: Optimizer>(mut optimizer: O, init_weight: f64) {
        let grad_fn = |w: &[f64]| vec![2.0 * (w[0] - 3.0)];
        let mut weights = vec![init_weight];

        run_optimization(&mut optimizer, &mut weights, grad_fn, 100);

        assert!(
            (weights[0] - 3.0).abs() < 1e-2,
            "Expected weight close to 3.0, got {}",
            weights[0]
        );
    }

    #[test]
    fn test_gradient_descent_converges() {
        let optimizer = optimizers::GradientDescent::new(0.1);
        check_optimizer_converges(optimizer, 0.0);
    }

    #[test]
    fn test_momentum_converges() {
        let optimizer = optimizers::Momentum::new(0.1, 0.9, 1);
        check_optimizer_converges(optimizer, 0.0);
    }
}
// ANCHOR_END: tests
