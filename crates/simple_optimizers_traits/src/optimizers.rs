/// A trait representing an optimization algorithm that can update weights using gradients.
///
/// Optimizers must implement the `step` method, which modifies weights in place.
// ANCHOR: optimizer_trait
pub trait Optimizer {
    /// Performs a single optimization step.
    ///
    /// # Arguments
    /// - `weights`: Mutable slice of parameters to be updated.
    /// - `grads`: Slice of gradients corresponding to the weights.
    fn step(&mut self, weights: &mut [f64], grads: &[f64]);
}
// ANCHOR_END: optimizer_trait

/// Basic gradient descent optimizer.
///
/// Updates each weight by subtracting the gradient scaled by a learning rate.
// ANCHOR: gd_struct
pub struct GradientDescent {
    pub learning_rate: f64,
}
// ANCHOR_END: gd_struct

// ANCHOR: impl_optimizer_gd
impl GradientDescent {
    /// Creates a new gradient descent optimizer.
    ///
    /// # Arguments
    /// - `learning_rate`: Step size used to update weights.
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}
// ANCHOR_END: impl_optimizer_gd

// ANCHOR: impl_optimizer_gd_step
impl Optimizer for GradientDescent {
    /// Applies the gradient descent step to each weight.
    ///
    /// Each weight is updated as: `w ← w - learning_rate * grad`
    fn step(&mut self, weights: &mut [f64], grads: &[f64]) {
        for (w, g) in weights.iter_mut().zip(grads.iter()) {
            *w -= self.learning_rate * g;
        }
    }
}
// ANCHOR_END: impl_optimizer_gd_step

/// Momentum-based gradient descent optimizer.
///
/// Combines current gradients with a velocity term to smooth updates.
// ANCHOR: momentum_struct
pub struct Momentum {
    pub learning_rate: f64,
    pub momentum: f64,
    pub velocity: Vec<f64>,
}
// ANCHOR_END: momentum_struct

// ANCHOR: impl_optimizer_momentum
impl Momentum {
    /// Creates a new momentum optimizer.
    ///
    /// # Arguments
    /// - `learning_rate`: Step size used to update weights.
    /// - `momentum`: Momentum coefficient (typically between 0.8 and 0.99).
    /// - `dim`: Dimension of the parameter vector, used to initialize velocity.
    pub fn new(learning_rate: f64, momentum: f64, dim: usize) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: vec![0.0; dim],
        }
    }
}
// ANCHOR_END: impl_optimizer_momentum

// ANCHOR: impl_optimizer_momentum_step
impl Optimizer for Momentum {
    /// Applies the momentum update step.
    ///
    /// Each step uses the update rule:
    /// ```text
    /// v ← momentum * v + learning_rate * grad
    /// w ← w - v
    /// ```
    fn step(&mut self, weights: &mut [f64], grads: &[f64]) {
        for ((w, g), v) in weights
            .iter_mut()
            .zip(grads.iter())
            .zip(self.velocity.iter_mut())
        {
            *v = self.momentum * *v + self.learning_rate * *g;
            *w -= *v;
        }
    }
}
// ANCHOR_END: impl_optimizer_momentum_step

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_descent_constructor() {
        let optimizer = GradientDescent::new(1e-3);
        assert_eq!(1e-3, optimizer.learning_rate);
    }

    #[test]
    fn test_step_gradient_descent() {
        let mut opt = GradientDescent::new(0.1);
        let mut weights = vec![1.0, 2.0, 3.0];
        let grads = vec![0.5, 0.5, 0.5];

        opt.step(&mut weights, &grads);

        assert_eq!(weights, vec![0.95, 1.95, 2.95])
    }

    #[test]
    fn test_momentum_constructor() {
        let opt = Momentum::new(0.01, 0.9, 10);
        match opt {
            Momentum {
                learning_rate,
                momentum,
                velocity,
            } => {
                assert_eq!(learning_rate, 0.01);
                assert_eq!(momentum, 0.9);
                assert_eq!(velocity.len(), 10);
            }
        }
    }

    #[test]
    fn test_step_momentum() {
        let mut opt = Momentum::new(0.1, 0.9, 3);
        let mut weights = vec![1.0, 2.0, 3.0];
        let grads = vec![0.5, 0.5, 0.5];

        opt.step(&mut weights, &grads);
        assert_eq!(weights, vec![0.95, 1.95, 2.95]);

        opt.step(&mut weights, &grads);
        assert!(
            weights
                .iter()
                .zip(vec![0.855, 1.855, 2.855])
                .all(|(a, b)| (*a - b).abs() < 1e-6)
        );
    }
}
// ANCHOR_END: tests