/// Multiplies a vector by a scalar.
///
/// # Arguments
///
/// * `scalar` - A scalar multiplier
/// * `vector` - A slice of f64 values
///
/// # Returns
///
/// A new vector containing the result of element-wise multiplication
///
/// # Why `&[f64]` instead of `Vec<f64]`?
///
/// We use a slice (`&[f64]`) because:
/// - It's more general: works with both arrays and vectors
/// - It avoids unnecessary ownership
/// - It's idiomatic and Clippy-compliant
// ANCHOR: mul_scalar_vec
pub fn mul_scalar_vec(scalar: f64, vector: &[f64]) -> Vec<f64> {
    vector.iter().map(|x| x * scalar).collect()
}
// ANCHOR_END: mul_scalar_vec

/// Subtracts two vectors element-wise.
///
/// # Arguments
///
/// * `a` - First input slice
/// * `b` - Second input slice
///
/// # Returns
///
/// A new `Vec<f64>` containing the element-wise difference `a[i] - b[i]`.
///
/// # Panics
///
/// Panics if `a` and `b` do not have the same length.
// ANCHOR: subtract_vectors
pub fn subtract_vectors(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}
// ANCHOR_END: subtract_vectors

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

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_function_naive() {
        let x: Vec<f64> = vec![1.0, 2.0];
        let y: Vec<f64> = vec![0.1, 0.2];
        let beta: f64 = 0.1;
        let lambda2: f64 = 1.0;

        let val: f64 = loss_function_naive(&x, &y, beta, lambda2);
        assert!(val > 0.0);
    }

    #[test]
    fn test_loss_function_line() {
        let x: Vec<f64> = vec![1.0, 2.0];
        let y: Vec<f64> = vec![0.1, 0.2];
        let beta: f64 = 0.1;
        let lambda2: f64 = 1.0;

        let val: f64 = loss_function_inline(&x, &y, beta, lambda2);
        assert!(val > 0.0);
    }

    #[test]
    fn test_naive_vs_inline() {
        let x: Vec<f64> = vec![1.0, 2.0];
        let y: Vec<f64> = vec![0.1, 0.2];
        let beta: f64 = 0.1;
        let lambda2: f64 = 1.0;

        let val1 = loss_function_naive(&x, &y, beta, lambda2);
        let val2 = loss_function_inline(&x, &y, beta, lambda2);
        assert_eq!(val1, val2);
    }
}
// ANCHOR_END: tests
