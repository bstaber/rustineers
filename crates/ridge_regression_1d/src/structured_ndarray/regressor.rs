use ndarray::Array1;
use std::error::Error;

#[derive(Debug, Clone, Default)]
pub struct RidgeEstimator {
    beta: Option<f64>,
}

impl RidgeEstimator {
    pub fn new() -> Self {
        Self { beta: None }
    }

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

impl RidgeEstimator {
    pub fn predict(&self, x: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
        match &self.beta {
            Some(beta) => Ok(*beta * x),
            None => Err("Model not fitted".into()),
        }
    }
}
