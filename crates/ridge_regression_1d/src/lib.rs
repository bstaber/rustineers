// ANCHOR: exports_and_fit
/// This module exposes the main components of the Ridge Regression 1D crate.
///
/// It re-exports internal modules and defines high-level helper functions
/// for training (`fit`) and predicting (`predict`) using Ridge regression.
pub mod grad_functions;
pub mod loss_functions;
pub mod optimizer;
pub mod utils;

pub use grad_functions::grad_loss_function_inline;
pub use optimizer::gradient_descent;

/// Fits a Ridge regression model using gradient descent.
///
/// # Arguments
///
/// * `x` - Input features (`&[f64]`)
/// * `y` - Target values (`&[f64]`)
/// * `lambda2` - Regularization strength
/// * `lr` - Learning rate
/// * `n_iters` - Number of gradient descent iterations
/// * `init_beta` - Initial value of the coefficient
///
/// # Returns
///
/// The optimized coefficient `beta` as `f64`.
pub fn fit(x: &[f64], y: &[f64], lambda2: f64, lr: f64, n_iters: usize, init_beta: f64) -> f64 {
    gradient_descent(
        grad_loss_function_inline,
        x,
        y,
        lambda2,
        lr,
        n_iters,
        init_beta,
    )
}

/// Predicts output values using a trained Ridge regression coefficient.
///
/// # Arguments
///
/// * `x` - Input features (`&[f64]`)
/// * `beta` - Trained coefficient
///
/// # Returns
///
/// A `Vec<f64>` with predicted values.
pub fn predict(x: &[f64], beta: f64) -> Vec<f64> {
    x.iter().map(|xi| xi * beta).collect()
}
// ANCHOR_END: exports_and_fit
