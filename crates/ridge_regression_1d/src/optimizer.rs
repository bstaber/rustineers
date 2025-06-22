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
// ANCHOR: gradient_descent
pub fn gradient_descent(
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
// ANCHOR_END: gradient_descent
