use crate::utils::{mul_scalar_vec, subtract_vectors};

/// Computes the loss function for Ridge regression (naive version).
///
/// It implements it in a simple fashion by computing the mean squared error in multiple steps.
///
/// # Arguments
///
/// * `x` - The array of input observations
/// * `y` - The array of output observations
/// * `beta` - The coefficients of the linear regression
/// * `lambda2` - The regularization parameter
///
/// # Returns
///
/// The value of the loss function
/// Computes the Ridge regression loss function.
///
/// This function calculates the following expression:
///
/// $$
/// \mathcal{L}(\beta) = \frac{1}{2n} \sum_i (y_i - \beta x_i)^2 + \lambda \beta^2
/// $$
///
/// where:
/// - `x` and `y` are the input/output observations,
/// - `beta` is the linear coefficient,
/// - `lambda2` is the regularization strength.
///
/// # Arguments
///
/// * `x` - Input features as a slice (`&[f64]`)
/// * `y` - Target values as a slice (`&[f64]`)
/// * `beta` - Coefficient of the regression model
/// * `lambda2` - L2 regularization strength
///
/// # Returns
///
/// The Ridge regression loss value as `f64`.
///
/// # Panics
///
/// Panics if `x` and `y` do not have the same length.
// ANCHOR: loss_function_naive
pub fn loss_function_naive(x: &[f64], y: &[f64], beta: f64, lambda2: f64) -> f64 {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");

    let n: usize = x.len();
    let y_hat: Vec<f64> = mul_scalar_vec(beta, x);
    let residuals: Vec<f64> = subtract_vectors(y, &y_hat);
    let mse: f64 = residuals.iter().map(|x| x * x).sum::<f64>() / (n as f64);
    mse + lambda2 * beta * beta
}
// ANCHOR_END: loss_function_naive

/// Computes the loss function for Ridge regression (inlined version).
///
/// It implements it as a one-liner by computing the mean squared error in a single expression.
///
/// # Arguments
///
/// * `x` - The array of input observations
/// * `y` - The array of output observations
/// * `beta` - The coefficients of the linear regression
/// * `lambda2` - The regularization parameter
///
/// # Returns
///
/// The value of the loss function
// ANCHOR: loss_function_line
pub fn loss_function_inline(x: &[f64], y: &[f64], beta: f64, lambda2: f64) -> f64 {
    let n: usize = y.len();
    let factor = n as f64;
    let mean_squared_error = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| {
            let residual = yi - beta * xi;
            residual * residual
        })
        .sum::<f64>()
        / factor;
    mean_squared_error + lambda2 * beta * beta
}
// ANCHOR_END: loss_function_line
