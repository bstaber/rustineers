/// A trait for Ridge regression models.
///
/// This trait defines a common interface for fitting a model and making predictions
/// from slices of `f64` values.
// ANCHOR: ridge_model_trait
pub trait RidgeModel {
    /// Fits the model using input features `x`, targets `y`, and regularization `lambda2`.
    fn fit(&mut self, x: &[f64], y: &[f64], lambda2: f64);

    /// Predicts outputs for a slice of input values.
    fn predict(&self, x: &[f64]) -> Vec<f64>;
}
// ANCHOR_END: ridge_model_trait

/// A Ridge regression model trained with batch gradient descent.
///
/// Stores the current value of `beta`, number of iterations, and learning rate.
/// This model updates `beta` iteratively using a gradient-based method.
// ANCHOR: struct_ridge_gd
pub struct RidgeGradientDescent {
    beta: f64,
    n_iters: usize,
    lr: f64,
}
// ANCHOR_END: struct_ridge_gd

/// A closed-form Ridge regression estimator.
///
/// Computes `beta` directly from the input data using the analytical solution.
// ANCHOR: struct_ridge_closedform
pub struct RidgeEstimator {
    pub beta: f64,
}
// ANCHOR_END: struct_ridge_closedform

/// Predicts output values from input features and a coefficient `beta`.
///
/// Applies a simple linear transformation: `y_i = beta * x_i`.
// ANCHOR: predict_from_beta
fn predict_from_beta(beta: f64, x: &[f64]) -> Vec<f64> {
    x.iter().map(|xi| beta * xi).collect()
}
// ANCHOR_END: predict_from_beta

// ANCHOR: ridge_gradient_descent_impl
impl RidgeGradientDescent {
    /// Creates a new gradient descent-based Ridge estimator.
    pub fn new(n_iters: usize, lr: f64, init_beta: f64) -> Self {
        Self {
            beta: init_beta,
            n_iters,
            lr,
        }
    }

    /// Computes the gradient of the Ridge regression loss function.
    ///
    /// This includes both the mean squared error gradient and the L2 regularization term.
    // ANCHOR: grad_function
    fn grad_function(&self, x: &[f64], y: &[f64], lambda2: f64) -> f64 {
        assert_eq!(x.len(), y.len(), "x and y must have the same length");
        let n: usize = x.len();
        let grad_mse: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| {
                let error = yi - self.beta * xi;
                2.0 * error * xi
            })
            .sum::<f64>()
            / (n as f64);

        -grad_mse + 2.0 * lambda2 * self.beta
    }
}
// ANCHOR_END: ridge_gradient_descent_impl

// ANCHOR: impl_gd_model
impl RidgeModel for RidgeGradientDescent {
    /// Fits the model using input features `x`, targets `y`, and regularization `lambda2`.
    fn fit(&mut self, x: &[f64], y: &[f64], lambda2: f64) {
        for _ in 0..self.n_iters {
            let grad = self.grad_function(x, y, lambda2);
            self.beta -= self.lr * grad;
        }
    }

    /// Predicts outputs for a slice of input values.
    fn predict(&self, x: &[f64]) -> Vec<f64> {
        predict_from_beta(self.beta, x)
    }
}
// ANCHOR_END: impl_gd_model

// ANCHOR: ridge_estimator_impl
impl RidgeEstimator {
    /// Creates a new closed-form Ridge estimator with the given initial value.
    pub fn new(init_beta: f64) -> Self {
        Self { beta: init_beta }
    }
}
// ANCHOR_END: ridge_estimator_impl

// ANCHOR: impl_closedform_model
impl RidgeModel for RidgeEstimator {
    /// Fits the model using input features `x`, targets `y`, and regularization `lambda2`.
    fn fit(&mut self, x: &[f64], y: &[f64], lambda2: f64) {
        let n: usize = x.len();
        assert_eq!(n, y.len(), "x and y must have the same length");

        let x_mean: f64 = x.iter().sum::<f64>() / n as f64;
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

        let num: f64 = x
            .iter()
            .zip(y)
            .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
            .sum::<f64>();

        let denom: f64 =
            x.iter().map(|xi| (xi - x_mean).powi(2)).sum::<f64>() + lambda2 * (n as f64);

        self.beta = num / denom;
    }

    /// Predicts outputs for a slice of input values.
    fn predict(&self, x: &[f64]) -> Vec<f64> {
        predict_from_beta(self.beta, x)
    }
}
// ANCHOR_END: impl_closedform_model

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

        let mut model = RidgeEstimator::new(0.0);
        model.fit(&x, &y, lambda2);

        assert!(
            (true_beta - model.beta).abs() < 1e-6,
            "Estimate {} not close enough to true solution {}",
            model.beta,
            true_beta
        );
    }

    #[test]
    fn test_ridge_gd() {
        let x: Vec<f64> = vec![1.0, 2.0];
        let y: Vec<f64> = vec![0.1, 0.2];
        let true_beta: f64 = 0.1;
        let lambda2: f64 = 0.0;

        let mut model = RidgeGradientDescent::new(100, 0.1, 0.0);
        model.fit(&x, &y, lambda2);

        assert!(
            (true_beta - model.beta).abs() < 1e-6,
            "Estimate {} not close enough to true solution {}",
            model.beta,
            true_beta
        );
    }

    #[test]
    fn test_ridge_estimatir_vs_gd() {
        let x: Vec<f64> = vec![1.0, 2.0];
        let y: Vec<f64> = vec![0.1, 0.2];
        let lambda2: f64 = 0.0;

        let mut model1 = RidgeEstimator::new(0.0);
        model1.fit(&x, &y, lambda2);

        let mut model2 = RidgeGradientDescent::new(100, 0.1, 0.0);
        model2.fit(&x, &y, lambda2);

        assert!(
            (model1.beta - model2.beta).abs() < 1e-6,
            "Estimates {} and {} are not close enough to each other",
            model1.beta,
            model2.beta
        );
    }
}
// ANCHOR_END: tests
