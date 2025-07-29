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

    pub fn fit(&mut self, x_train: Array2<f64>, y_train: Array1<f64>) {
        let n: usize = x_train.nrows();
        assert_eq!(
            n,
            y_train.len(),
            "x_train and y_train must have the same lengths, got {} and {}",
            n,
            y_train.len()
        );

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
        let alpha = a.solve_into(y_train).unwrap();

        self.x_train = Some(x_train);
        self.alpha = Some(alpha);
    }

    pub fn predict(&self, x_test: &Array2<f64>) -> Result<Array1<f64>, String> {
        match &self.alpha {
            Some(alpha) => {
                let n_train: usize = self.x_train.as_ref().unwrap().nrows();
                let n_test: usize = x_test.nrows();
                let mut y_pred: Array1<f64> = Array::zeros(n_test);
                for i in 0..n_test {
                    for j in 0..n_train {
                        let k_val = self
                            .kernel
                            .compute(self.x_train.as_ref().unwrap().row(j), x_test.row(i));
                        y_pred[i] += alpha[j] * k_val;
                    }
                }
                Ok(y_pred)
            }
            None => Err("Model not fitted".to_string()),
        }
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
    fn test_fit_and_predict() {
        let kernel = RBFKernel::new(1.0);
        let mut model: KRRModel<RBFKernel> = KRRModel::new(kernel, 1.0);
        let x_train: Array2<f64> = array![[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]];
        let y_train: Array1<f64> = array![0.9, 0.6];

        model.fit(x_train, y_train);
        assert!(
            model.alpha.is_some(),
            "alpha should not be None if the model has been fitted"
        );
        assert!(
            model.x_train.is_some(),
            "x_train should not be None if the model has been fitted"
        );

        let x_test: Array2<f64> = array![[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]];
        let y_pred = model.predict(&x_test).unwrap();

        assert_eq!(
            y_pred.len(),
            x_test.nrows(),
            "The length of y_pred must match the number of rows of x_test, got {} and {}",
            y_pred.len(),
            x_test.nrows()
        );
    }

    #[test]
    fn test_unfitted_and_predict() {
        let kernel = RBFKernel::new(1.0);
        let model: KRRModel<RBFKernel> = KRRModel::new(kernel, 1.0);

        assert!(
            model.alpha.is_none(),
            "alpha should be None if the model is unfitted"
        );
        assert!(
            model.x_train.is_none(),
            "x_train should be None if the model is unfitted"
        );

        let x_test: Array2<f64> = array![[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]];
        let y_pred = model.predict(&x_test);

        assert!(y_pred.is_err());
    }
}
