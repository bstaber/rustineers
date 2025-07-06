// ANCHOR: enum_definition
/// An enum representing different optimizers with built-in state and update rules.
///
/// Supports both gradient descent and momentum-based methods.
#[derive(Debug, Clone)]
pub enum Optimizer {
    /// Gradient Descent optimizer with a fixed learning rate.
    GradientDescent { learning_rate: f64 },
    /// Momentum-based optimizer with velocity tracking.
    Momentum {
        learning_rate: f64,
        momentum: f64,
        velocity: Vec<f64>,
    },
}
// ANCHOR_END: enum_definition

// ANCHOR: constructors
impl Optimizer {
    /// Creates a new Gradient Descent optimizer.
    ///
    /// # Arguments
    /// - `learning_rate`: Step size for the parameter updates.
    pub fn gradient_descent(learning_rate: f64) -> Self {
        Self::GradientDescent { learning_rate }
    }

    /// Creates a new Momentum optimizer.
    ///
    /// # Arguments
    /// - `learning_rate`: Step size for the updates.
    /// - `momentum`: Momentum coefficient.
    /// - `dim`: Number of parameters (used to initialize velocity vector).
    pub fn momentum(learning_rate: f64, momentum: f64, dim: usize) -> Self {
        Self::Momentum {
            learning_rate,
            momentum,
            velocity: vec![0.0; dim],
        }
    }
}
// ANCHOR_END: constructors

// ANCHOR: step
impl Optimizer {
    /// Applies a single optimization step, depending on the variant.
    ///
    /// # Arguments
    /// - `weights`: Mutable slice of model parameters to be updated.
    /// - `grads`: Gradient slice with same shape as `weights`.
    pub fn step(&mut self, weights: &mut [f64], grads: &[f64]) {
        match self {
            Optimizer::GradientDescent { learning_rate } => {
                for (w, g) in weights.iter_mut().zip(grads.iter()) {
                    *w -= *learning_rate * *g;
                }
            }
            Optimizer::Momentum {
                learning_rate,
                momentum,
                velocity,
            } => {
                for ((w, g), v) in weights
                    .iter_mut()
                    .zip(grads.iter())
                    .zip(velocity.iter_mut())
                {
                    *v = *momentum * *v + *learning_rate * *g;
                    *w -= *v;
                }
            }
        }
    }
}
// ANCHOR_END: step

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_descent_constructor() {
        let opt = Optimizer::gradient_descent(0.01);
        match opt {
            Optimizer::GradientDescent { learning_rate } => {
                assert_eq!(learning_rate, 0.01);
            }
            _ => panic!("Expected GradientDescent optimizer"),
        }
    }

    #[test]
    fn test_momentum_constructor() {
        let opt = Optimizer::momentum(0.01, 0.9, 10);
        match opt {
            Optimizer::Momentum {
                learning_rate,
                momentum,
                velocity,
            } => {
                assert_eq!(learning_rate, 0.01);
                assert_eq!(momentum, 0.9);
                assert_eq!(velocity.len(), 10);
            }
            _ => panic!("Expected Momentum optimizer"),
        }
    }

    #[test]
    fn test_step_gradient_descent() {
        let mut opt = Optimizer::gradient_descent(0.1);
        let mut weights = vec![1.0, 2.0, 3.0];
        let grads = vec![0.5, 0.5, 0.5];

        opt.step(&mut weights, &grads);

        assert_eq!(weights, vec![0.95, 1.95, 2.95])
    }

    #[test]
    fn test_step_momentum() {
        let mut opt = Optimizer::momentum(0.1, 0.9, 3);
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
