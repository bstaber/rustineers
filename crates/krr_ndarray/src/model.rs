use crate::kernel::Kernel;
use ndarray::{Array1, Array2};

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

    pub fn fit(&self, x_train: &Array2<f64>, y_train: &Array1<f64>) {}

    pub fn predict(&self, x_test: &Array2<f64>) -> Array1<f64> {}
}
