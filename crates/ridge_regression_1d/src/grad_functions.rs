use crate::utils::dot;

/// Computes the gradient of the Ridge regression loss function (naive version).
///
/// This implementation first explicitly computes the residuals and then performs
/// a dot product between the residuals and the inputs.
///
/// # Arguments
///
/// * `x` - Slice of input features
/// * `y` - Slice of target outputs
/// * `beta` - Coefficient of the regression model
/// * `lambda2` - L2 regularization strength
///
/// # Returns
///
/// The gradient of the loss with respect to `beta`.
///
/// # Panics
///
/// Panics if `x` and `y` do not have the same length.
// ANCHOR: grad_loss_function_naive
pub fn grad_loss_function_naive(x: &[f64], y: &[f64], beta: f64, lambda2: f64) -> f64 {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");

    let n: usize = x.len();
    let residuals: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| yi - beta * xi)
        .collect();

    -dot(&residuals, x) / (n as f64) + 2.0 * lambda2 * beta
}
// ANCHOR_END: grad_loss_function_naive

/// Computes the gradient of the Ridge regression loss function (inlined version).
///
/// This version fuses the residual and gradient computation into a single pass
/// using iterators, minimizing allocations and improving efficiency.
///
/// # Arguments
///
/// * `x` - Slice of input features
/// * `y` - Slice of target outputs
/// * `beta` - Coefficient of the regression model
/// * `lambda2` - L2 regularization strength
///
/// # Returns
///
/// The gradient of the loss with respect to `beta`.
///
/// # Panics
///
/// Panics if `x` and `y` do not have the same length.
// ANCHOR: grad_loss_function_inline
pub fn grad_loss_function_inline(x: &[f64], y: &[f64], beta: f64, lambda2: f64) -> f64 {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");

    let n: usize = x.len();
    let grad_mse: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (yi - beta * xi) * xi)
        .sum::<f64>()
        / (n as f64);

    -grad_mse + 2.0 * lambda2 * beta
}
// ANCHOR_END: grad_loss_function_inline
