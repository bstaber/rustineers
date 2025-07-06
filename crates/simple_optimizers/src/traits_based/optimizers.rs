// ANCHOR: optimizer_trait
/// A trait representing an optimization algorithm that can update weights using gradients.
///
/// Optimizers must implement the `step` method, which modifies weights in place.
pub trait Optimizer {
    /// Performs a single optimization step.
    ///
    /// # Arguments
    /// - `weights`: Mutable slice of parameters to be updated.
    /// - `grads`: Slice of gradients corresponding to the weights.
    fn step(&mut self, weights: &mut [f64], grads: &[f64]);
}
// ANCHOR_END: optimizer_trait

// ANCHOR: gd_struct
/// Basic gradient descent optimizer.
///
/// Updates each weight by subtracting the gradient scaled by a learning rate.
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

// ANCHOR: momentum_struct
/// Momentum-based gradient descent optimizer.
///
/// Combines current gradients with a velocity term to smooth updates.
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
