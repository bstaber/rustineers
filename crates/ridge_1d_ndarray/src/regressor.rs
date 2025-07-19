use ndarray::Array1;

/// A Ridge regression estimator using `ndarray` for vectorized operations.
///
/// This version supports fitting and predicting using `Array1<f64>` arrays.
/// The coefficient `beta` is stored as an `Option<f64>`, allowing the model
/// to represent both fitted and unfitted states.
// ANCHOR: struct
#[derive(Debug, Clone, Default)]
pub struct RidgeEstimator {
    pub beta: Option<f64>,
}
// ANCHOR_END: struct

// ANCHOR: ridge_estimator_impl_new_fit
impl RidgeEstimator {
    /// Creates a new, unfitted Ridge estimator.
    ///
    /// # Returns
    /// A `RidgeEstimator` with `beta` set to `None`.
    pub fn new() -> Self {
        Self { beta: None }
    }

    /// Fits the Ridge regression model using 1D input and output arrays.
    ///
    /// This function computes the coefficient `beta` using the closed-form
    /// solution with L2 regularization.
    ///
    /// # Arguments
    /// - `x`: Input features as a 1D `Array1<f64>`.
    /// - `y`: Target values as a 1D `Array1<f64>`.
    /// - `lambda2`: The regularization strength (λ²).
    pub fn fit(&mut self, x: &Array1<f64>, y: &Array1<f64>, lambda2: f64) {
        let n: usize = x.len();
        assert!(n > 0);
        assert_eq!(x.len(), y.len(), "x and y must have the same length");

        // mean returns None if the array is empty, so we need to unwrap it
        let x_mean: f64 = x.mean().unwrap();
        let y_mean: f64 = y.mean().unwrap();

        let num: f64 = (x - x_mean).dot(&(y - y_mean));
        let denom: f64 = (x - x_mean).mapv(|z| z.powi(2)).sum() + lambda2 * (n as f64);

        self.beta = Some(num / denom);
    }
}
// ANCHOR_END: ridge_estimator_impl_new_fit

// ANCHOR: ridge_estimator_impl_predict
impl RidgeEstimator {
    /// Predicts target values given input features.
    ///
    /// # Arguments
    /// - `x`: Input features as a 1D array.
    ///
    /// # Returns
    /// A `Result` containing the predicted values, or an error if the model
    /// has not been fitted.
    pub fn predict(&self, x: &Array1<f64>) -> Result<Array1<f64>, String> {
        match &self.beta {
            Some(beta) => Ok(*beta * x),
            None => Err("Model not fitted".to_string()),
        }
    }
}
// ANCHOR_END: ridge_estimator_impl_predict

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ridge_estimator_constructor() {
        let model = RidgeEstimator::new();
        assert_eq!(model.beta, None, "beta is expected to be None");
    }

    #[test]
    fn test_unfitted_estimator() {
        let model = RidgeEstimator::new();
        let x: Array1<f64> = array![1.0, 2.0];
        let result: Result<Array1<f64>, String> = model.predict(&x);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Model not fitted");
    }

    #[test]
    fn test_ridge_estimator_solution() {
        let x: Array1<f64> = array![1.0, 2.0];
        let y: Array1<f64> = array![0.1, 0.2];
        let true_beta: f64 = 0.1;
        let lambda2: f64 = 0.0;

        let mut model = RidgeEstimator::new();
        model.fit(&x, &y, lambda2);

        assert!(model.beta.is_some(), "beta is expected to be Some(f64)");

        assert!(
            (true_beta - model.beta.unwrap()).abs() < 1e-6,
            "Estimate {} not close enough to true solution {}",
            true_beta,
            model.beta.unwrap()
        );
    }
}
// ANCHOR_END: tests
