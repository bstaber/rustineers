use ndarray::Array1;
use std::error::Error;

/// A Ridge regression estimator using `ndarray` for vectorized operations.
///
/// This version supports fitting and predicting using `Array1<f64>` arrays.
/// The coefficient `beta` is stored as an `Option<f64>`, allowing the model
/// to represent both fitted and unfitted states.
// ANCHOR: struct
#[derive(Debug, Clone, Default)]
pub struct RidgeEstimator {
    beta: Option<f64>,
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
    pub fn predict(&self, x: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
        match &self.beta {
            Some(beta) => Ok(*beta * x),
            None => Err("Model not fitted".into()),
        }
    }
}
// ANCHOR_END: ridge_estimator_impl_predict
