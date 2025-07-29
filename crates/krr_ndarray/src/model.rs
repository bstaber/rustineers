use crate::errors::{KRRFitError, KRRPredictError};
use crate::kernel::Kernel;
use ndarray::{Array, Array1, Array2};
use ndarray_linalg::Solve;

pub struct KRRModel<K: Kernel> {
    pub kernel: K,
    pub lambda: f64,
    x_train: Option<Array2<f64>>,
    alpha: Option<Array1<f64>>,
}

impl<K: Kernel> KRRModel<K> {
    pub fn new(kernel: K, lambda: f64) -> Self {
        Self {
            kernel,
            lambda,
            x_train: None,
            alpha: None,
        }
    }

    fn _fit(&mut self, x_train: Array2<f64>, y_train: Array1<f64>) -> Result<(), KRRFitError> {
        let n: usize = y_train.len();
        let mut k_train: Array2<f64> = Array::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let kxy = self.kernel.compute(x_train.row(i), x_train.row(j));
                k_train[(i, j)] = kxy;
                k_train[(j, i)] = kxy;
            }
        }

        let identity_n = Array2::eye(n);
        let a: Array2<f64> = k_train + self.lambda * identity_n;
        let alpha = a
            .solve_into(y_train)
            .map_err(|e| KRRFitError::LinAlgError(e.to_string()))?;

        self.x_train = Some(x_train);
        self.alpha = Some(alpha);

        Ok(())
    }

    pub fn fit(&mut self, x_train: Array2<f64>, y_train: Array1<f64>) -> Result<(), KRRFitError> {
        let n: usize = x_train.nrows();
        let m: usize = y_train.len();

        if n != m {
            eprintln!("[KRR::fit] Shape mismatch: x_train has {n} rows, y_train has {m} elments");
            return Err(KRRFitError::ShapeMismatch { x_n: n, y_n: m });
        }

        match self._fit(x_train, y_train) {
            Ok(_) => {
                eprintln!("[KRR::fit] Model successfully fitted.");
                Ok(())
            }
            Err(e) => {
                eprintln!("[KRR::fit] Fitting failed: {e}");
                Err(e)
            }
        }
    }

    pub fn predict(&self, x_test: &Array2<f64>) -> Result<Array1<f64>, KRRPredictError> {
        let alpha = self.alpha.as_ref().ok_or(KRRPredictError::NotFitted)?;
        let x_train = self.x_train.as_ref().ok_or(KRRPredictError::NotFitted)?;

        let n_train: usize = x_train.nrows();
        let n_test: usize = x_test.nrows();
        let mut y_pred: Array1<f64> = Array::zeros(n_test);

        for i in 0..n_test {
            for j in 0..n_train {
                let k_val = self.kernel.compute(x_train.row(j), x_test.row(i));
                y_pred[i] += alpha[j] * k_val;
            }
        }
        Ok(y_pred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::RBFKernel;
    use ndarray::array;

    #[test]
    fn test_krr_constructor() {
        let kernel = RBFKernel::new(1.0);
        let model: KRRModel<RBFKernel> = KRRModel::new(kernel, 1.0);

        assert_eq!(
            model.lambda, 1.0,
            "Expected lambda equal to 1.0, got {}",
            model.lambda
        );

        assert_eq!(
            model.kernel.lengthscale, 1.0,
            "Expected kernel lengthscale to be 1.0, got {}",
            model.kernel.lengthscale
        );
    }

    #[test]
    fn test_ok_fit_and_predict() {
        let kernel = RBFKernel::new(1.0);
        let mut model: KRRModel<RBFKernel> = KRRModel::new(kernel, 1.0);
        let x_train: Array2<f64> = array![[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]];
        let y_train: Array1<f64> = array![0.9, 0.6];

        let res = model.fit(x_train, y_train);
        assert!(res.is_ok());

        let x_test: Array2<f64> = array![[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]];
        let y_pred = model.predict(&x_test);
        assert!(y_pred.is_ok());
    }

    #[test]
    fn test_dim_mismatch() {
        let kernel = RBFKernel::new(1.0);
        let mut model: KRRModel<RBFKernel> = KRRModel::new(kernel, 1.0);
        let x_train: Array2<f64> = array![[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]];
        let y_train: Array1<f64> = array![0.9, 0.6, 0.9];

        let res = model.fit(x_train, y_train);
        assert!(res.is_err());
    }

    #[test]
    fn test_unfitted_predict_error_type() {
        use crate::errors::KRRPredictError;

        let kernel = RBFKernel::new(1.0);
        let model: KRRModel<RBFKernel> = KRRModel::new(kernel, 1.0);
        let x_test: Array2<f64> = array![[1.0, 2.0, 3.0]];

        let result = model.predict(&x_test);
        assert!(matches!(result, Err(KRRPredictError::NotFitted)));
    }
}
