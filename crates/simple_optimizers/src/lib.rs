pub mod traits_based;
use traits_based::optimizers::Optimizer;

pub fn train(optimizer: &mut dyn Optimizer, weights: &mut [f64], grads: &[f64]) {
    optimizer.update(weights, grads);
}
