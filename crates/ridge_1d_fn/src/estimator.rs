/// Computes the one-dimensional Ridge regression estimator using centered data.
///
/// This version centers the input data `x` and `y` before applying the closed-form formula.
///
/// # Arguments
///
/// * `x` - A slice of input features.
/// * `y` - A slice of target values (same length as `x`).
/// * `lambda2` - The regularization parameter.
///
/// # Returns
///
/// * `f64` - The estimated Ridge regression coefficient.
///
/// # Panics
///
/// Panics if `x` and `y` do not have the same length.
// ANCHOR: ridge_estimator
pub fn ridge_estimator(x: &[f64], y: &[f64], lambda2: f64) -> f64 {
    let n: usize = x.len();
    assert_eq!(n, y.len(), "x and y must have the same length");

    let x_mean: f64 = x.iter().sum::<f64>() / n as f64;
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

    let num: f64 = x
        .iter()
        .zip(y)
        .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
        .sum::<f64>();

    let denom: f64 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum::<f64>() + lambda2 * (n as f64);

    num / denom
}
// ANCHOR_END: ridge_estimator

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_estimator() {
        let x: Vec<f64> = vec![1.0, 2.0];
        let y: Vec<f64> = vec![0.1, 0.2];
        let true_beta: f64 = 0.1;
        let lambda2: f64 = 0.0;

        let beta_estimate: f64 = ridge_estimator(&x, &y, lambda2);
        assert!(
            (true_beta - beta_estimate).abs() < 1e-6,
            "Estimate {} not close enough to true solution {}",
            beta_estimate,
            true_beta
        );
    }
}
// ANCHOR_END: tests
