use ndarray::Array;
use ndarray::Array1;
use ndarray::Zip;

/// Trait for optimizers that update parameters using gradients.
///
/// Implementors must define a `run` method that takes mutable weights,
/// a gradient function, and the number of iterations to run.
// ANCHOR: trait
pub trait Optimizer {
    fn run(
        &mut self,
        weights: &mut Array1<f64>,
        grad_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
        n_steps: usize,
    );
}
// ANCHOR_END: trait

/// Basic Gradient Descent (GD) optimizer.
///
/// Updates parameters in the direction of the negative gradient scaled
/// by a fixed step size.
// ANCHOR: struct_gd
pub struct GD {
    step_size: f64,
}
// ANCHOR_END: struct_gd

/// Create a new gradient descent optimizer.
///
/// # Arguments
/// - `step_size`: the learning rate.
// ANCHOR: impl_gd_new
impl GD {
    pub fn new(step_size: f64) -> Self {
        Self { step_size }
    }
}
// ANCHOR_END: impl_gd_new

/// Run the gradient descent optimizer.
///
/// For each step: `w ← w - step_size * grad(w)`
// ANCHOR: impl_gd_run
impl Optimizer for GD {
    fn run(
        &mut self,
        weights: &mut Array1<f64>,
        grad_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
        n_steps: usize,
    ) {
        for _ in 0..n_steps {
            let grads = grad_fn(weights);
            weights.zip_mut_with(&grads, |w, g| {
                *w -= self.step_size * g;
            });
        }
    }
}
// ANCHOR_END: impl_gd_run

/// Accelerated Gradient Descent (AGD) with classical momentum.
///
/// Combines the previous velocity with the current gradient
/// to speed up convergence in convex problems.
// ANCHOR: struct_agd
pub struct AGD {
    step_size: f64,
    momentum: f64,
}
// ANCHOR_END: struct_agd

/// Create a new AGD optimizer.
///
/// # Arguments
/// - `step_size`: the learning rate.
/// - `momentum`: the momentum coefficient (typically between 0.8 and 0.99).
// ANCHOR: impl_agd_new
impl AGD {
    pub fn new(step_size: f64, momentum: f64) -> Self {
        Self {
            step_size,
            momentum,
        }
    }
}
// ANCHOR_END: impl_agd_new

/// Run AGD with momentum.
///
/// For each step:
/// ```text
/// v ← momentum * v - step_size * grad(w)
/// w ← w + v
/// ```
// ANCHOR: impl_agd_run
impl Optimizer for AGD {
    fn run(
        &mut self,
        weights: &mut Array1<f64>,
        grad_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
        n_steps: usize,
    ) {
        let n: usize = weights.len();
        let mut velocity: Array1<f64> = Array::zeros(n);

        for _ in 0..n_steps {
            let grads = grad_fn(weights);
            for ((w, g), v) in weights
                .iter_mut()
                .zip(grads.iter())
                .zip(velocity.iter_mut())
            {
                *v = self.momentum * *v - self.step_size * g;
                *w += *v;
            }
        }
    }
}
// ANCHOR_END: impl_agd_run

/// Adaptive Accelerated Gradient Descent using Nesterov's method.
///
/// This optimizer implements the variant from smooth convex optimization literature,
/// where extrapolation is based on the difference between consecutive y iterates.
///
/// References:
/// - Beck & Teboulle (2009), FISTA (but without proximal operator)
/// - Nesterov's accelerated gradient (original formulation)
// ANCHOR: AdaptiveAGD_struct
pub struct AdaptiveAGD {
    step_size: f64,
}
// ANCHOR_END: AdaptiveAGD_struct

// ANCHOR: AdaptiveAGD_impl_new
impl AdaptiveAGD {
    /// Create a new instance of AdaptiveAGD with a given step size.
    ///
    /// The step size should be 1 / L, where L is the Lipschitz constant
    /// of the gradient of the objective function.
    pub fn new(step_size: f64) -> Self {
        Self { step_size }
    }
}
// ANCHOR_END: AdaptiveAGD_impl_new

// ANCHOR: AdaptiveAGD_impl_run
impl Optimizer for AdaptiveAGD {
    /// Run the optimizer for `n_steps` iterations.
    ///
    /// # Arguments
    /// - `weights`: mutable reference to the parameter vector (x₀), will be updated in-place.
    /// - `grad_fn`: a function that computes ∇f(x) for a given x.
    /// - `n_steps`: number of optimization steps to perform.
    ///
    /// This implementation follows:
    ///
    /// ```
    /// y_{k+1} = x_k - α ∇f(x_k)
    /// t_{k+1} = (1 + sqrt(1 + 4 t_k²)) / 2
    /// x_{k+1} = y_{k+1} + ((t_k - 1)/t_{k+1}) * (y_{k+1} - y_k)
    /// ```
    fn run(
        &mut self,
        weights: &mut Array1<f64>,
        grad_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
        n_steps: usize,
    ) {
        let mut t_k: f64 = 1.0;
        let mut y_k = weights.clone();

        for _ in 0..n_steps {
            let grad = grad_fn(weights);
            let mut y_next = weights.clone();
            Zip::from(&mut y_next).and(&grad).for_each(|y, &g| {
                *y -= self.step_size * g;
            });

            let t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_k * t_k).sqrt());

            Zip::from(&mut *weights)
                .and(&y_next)
                .and(&y_k)
                .for_each(|x, &y1, &y0| {
                    *x = y1 + ((t_k - 1.0) / t_next) * (y1 - y0);
                });

            y_k = y_next;
            t_k = t_next;
        }
    }
}
// ANCHOR_END: AdaptiveAGD_impl_run
