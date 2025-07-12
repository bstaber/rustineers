use num_traits::Float;
use std::iter::Sum;

/// A trait representing a generic Ridge regression model.
///
/// The model must support fitting to training data and predicting new outputs.
// ANCHOR: ridge_model_trait
pub trait RidgeModel<F: Float + Sum> {
    /// Fits the model to the given data using Ridge regression.
    fn fit(&mut self, x: &[F], y: &[F], lambda2: F);

    /// Predicts output values for a slice of new input features.
    fn predict(&self, x: &[F]) -> Vec<F>;
}
// ANCHOR_END: ridge_model_trait

/// A generic Ridge regression estimator using a single coefficient `beta`.
///
/// This implementation assumes a linear relationship between `x` and `y`
/// and performs scalar Ridge regression (1D).
// ANCHOR: gen_ridge_estimator
pub struct GenRidgeEstimator<F: Float + Sum> {
    pub beta: F,
}
// ANCHOR_END: gen_ridge_estimator

// ANCHOR: gen_ridge_estimator_impl
impl<F: Float + Sum> GenRidgeEstimator<F> {
    /// Creates a new estimator with the given initial beta coefficient.
    pub fn new(init_beta: F) -> Self {
        Self { beta: init_beta }
    }
}
// ANCHOR_END: gen_ridge_estimator_impl

// ANCHOR: gen_ridge_estimator_trait_impl
impl<F: Float + Sum> RidgeModel<F> for GenRidgeEstimator<F> {
    /// Fits the Ridge regression model to 1D data using closed-form solution.
    ///
    /// This method computes the regression coefficient `beta` by minimizing
    /// the Ridge-regularized least squares loss.
    ///
    /// # Arguments
    /// - `x`: Input features.
    /// - `y`: Target values.
    /// - `lambda2`: The regularization parameter (λ²).
    fn fit(&mut self, x: &[F], y: &[F], lambda2: F) {
        let n: usize = x.len();
        let n_f: F = F::from(n).unwrap();
        assert_eq!(x.len(), y.len(), "x and y must have the same length");

        let x_mean: F = x.iter().copied().sum::<F>() / n_f;
        let y_mean: F = y.iter().copied().sum::<F>() / n_f;

        let num: F = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (*xi - x_mean) * (*yi - y_mean))
            .sum::<F>();

        let denom: F = x.iter().map(|xi| (*xi - x_mean).powi(2)).sum::<F>() + lambda2 * n_f;

        self.beta = num / denom;
    }

    /// Applies the trained model to input features to generate predictions.
    ///
    /// # Arguments
    /// - `x`: Input features to predict from.
    ///
    /// # Returns
    /// A vector of predicted values, one for each input in `x`.
    fn predict(&self, x: &[F]) -> Vec<F> {
        x.iter().map(|xi| *xi * self.beta).collect()
    }
}
// ANCHOR_END: gen_ridge_estimator_trait_impl

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_estimator_f64() {
        let x: Vec<f64> = vec![1.0, 2.0];
        let y: Vec<f64> = vec![0.1, 0.2];
        let true_beta: f64 = 0.1;
        let lambda2: f64 = 0.0;

        let mut model = GenRidgeEstimator::new(0.0);
        model.fit(&x, &y, lambda2);

        assert!(
            (true_beta - model.beta).abs() < 1e-6,
            "Estimate {} not close enough to true solution {}",
            true_beta,
            model.beta
        );
    }

    #[test]
    fn test_ridge_estimator_f32() {
        let x: Vec<f32> = vec![1.0, 2.0];
        let y: Vec<f32> = vec![0.1, 0.2];
        let true_beta: f32 = 0.1;
        let lambda2: f32 = 0.0;

        let mut model = GenRidgeEstimator::new(0.0);
        model.fit(&x, &y, lambda2);

        assert!(
            (true_beta - model.beta).abs() < 1e-6,
            "Estimate {} not close enough to true solution {}",
            true_beta,
            model.beta
        );
    }
}
// ANCHOR_END: tests
