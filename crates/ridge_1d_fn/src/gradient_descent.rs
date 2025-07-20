/// Dot product between two vectors.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector
///
/// # Returns
///
/// The float value of the dot product.
///
/// # Panics
///
/// Panics if `a` and `b` do have the same length.
// ANCHOR: dot
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    a.iter().zip(b.iter()).map(|(xi, yi)| xi * yi).sum()
}

// ANCHOR_END: dot
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
    let residuals_dot_x = dot(&residuals, x);

    -2.0 * residuals_dot_x / (n as f64) + 2.0 * lambda2 * beta
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
        .map(|(xi, yi)| 2.0 * (yi - beta * xi) * xi)
        .sum::<f64>()
        / (n as f64);

    -grad_mse + 2.0 * lambda2 * beta
}
// ANCHOR_END: grad_loss_function_inline

/// Performs gradient descent to minimize the Ridge regression loss function.
///
/// # Arguments
///
/// * `grad_fn` - A function that computes the gradient of the Ridge loss
/// * `x` - Input features as a slice (`&[f64]`)
/// * `y` - Target values as a slice (`&[f64]`)
/// * `lambda2` - Regularization parameter
/// * `lr` - Learning rate
/// * `n_iters` - Number of gradient descent iterations
/// * `init_beta` - Initial value of the regression coefficient
///
/// # Returns
///
/// The optimized regression coefficient `beta` after `n_iters` updates
// ANCHOR: gradient_descent_estimator
pub fn ridge_estimator(
    grad_fn: impl Fn(&[f64], &[f64], f64, f64) -> f64,
    x: &[f64],
    y: &[f64],
    lambda2: f64,
    lr: f64,
    n_iters: usize,
    init_beta: f64,
) -> f64 {
    let mut beta = init_beta;

    for _ in 0..n_iters {
        let grad = grad_fn(x, y, beta, lambda2);
        beta -= lr * grad;
    }

    beta
}
// ANCHOR_END: gradient_descent_estimator

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_naive() {
        let x: Vec<f64> = vec![1.0, 2.0];
        let y: Vec<f64> = vec![0.1, 0.2];
        let beta: f64 = 0.1;
        let lambda2: f64 = 1.0;

        let grad = grad_loss_function_naive(&x, &y, beta, lambda2);
        let expected_grad = 0.2;
        let tol = 1e-6;
        assert!(
            (grad - expected_grad).abs() < tol,
            "Expected {}, got {}",
            expected_grad,
            grad
        );
    }

    #[test]
    fn test_grad_inline() {
        let x: Vec<f64> = vec![1.0, 2.0];
        let y: Vec<f64> = vec![0.1, 0.2];
        let beta: f64 = 0.1;
        let lambda2: f64 = 1.0;

        let grad = grad_loss_function_inline(&x, &y, beta, lambda2);
        let expected_grad = 0.2;
        let tol = 1e-6;
        assert!(
            (grad - expected_grad).abs() < tol,
            "Expected {}, got {}",
            expected_grad,
            grad
        );
    }

    #[test]
    fn test_naive_vs_inline() {
        let x: Vec<f64> = vec![1.0, 2.0];
        let y: Vec<f64> = vec![0.1, 0.2];
        let beta: f64 = 0.1;
        let lambda2: f64 = 1.0;

        let grad1 = grad_loss_function_inline(&x, &y, beta, lambda2);
        let grad2 = grad_loss_function_naive(&x, &y, beta, lambda2);
        assert_eq!(grad1, grad2);
    }
}
// ANCHOR_END: tests
