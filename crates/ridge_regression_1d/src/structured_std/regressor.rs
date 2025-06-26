pub trait RidgeModel {
    fn fit(&mut self, x: &[f64], y: &[f64], lambda2: f64);
    fn predict(&self, x: &[f64]) -> Vec<f64>;
}

pub struct RidgeGradientDescent {
    beta: f64,
    n_iters: usize,
    lr: f64,
}

pub struct RidgeEstimator {
    beta: f64,
}

fn predict_from_beta(beta: f64, x: &[f64]) -> Vec<f64> {
    x.iter().map(|xi| beta * xi).collect()
}

impl RidgeGradientDescent {
    pub fn new(n_iters: usize, lr: f64, init_beta: f64) -> Self {
        Self {
            beta: init_beta,
            n_iters,
            lr,
        }
    }

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

impl RidgeModel for RidgeGradientDescent {
    fn fit(&mut self, x: &[f64], y: &[f64], lambda2: f64) {
        for _ in 0..self.n_iters {
            let grad = self.grad_function(x, y, lambda2);
            self.beta -= self.lr * grad;
        }
    }

    fn predict(&self, x: &[f64]) -> Vec<f64> {
        predict_from_beta(self.beta, x)
    }
}

impl RidgeEstimator {
    pub fn new(init_beta: f64) -> Self {
        Self { beta: init_beta }
    }
}

impl RidgeModel for RidgeEstimator {
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

    fn predict(&self, x: &[f64]) -> Vec<f64> {
        predict_from_beta(self.beta, x)
    }
}
