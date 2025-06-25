pub struct RidgeRegressor {
    beta: f64,
}

impl RidgeRegressor {
    fn loss_function(&self, x: &[f64], y: &[f64], lambda2: f64) -> f64 {
        let n: usize = y.len();
        let factor = 2.0 * n as f64;
        let mean_squared_error = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| {
                let residual = yi - self.beta * xi;
                residual * residual
            })
            .sum::<f64>()
            / factor;
        mean_squared_error + lambda2 * self.beta * self.beta
    }

    fn grad_function(&self, x: &[f64], y: &[f64], lambda2: f64) -> f64 {
        assert_eq!(x.len(), y.len(), "x and y must have the same length");
        let n: usize = x.len();
        let grad_mse: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| {
                let error = yi - self.beta * xi;
                error * xi
            })
            .sum::<f64>()
            / (n as f64);

        -grad_mse + 2.0 * lambda2 * self.beta
    }

    pub fn fit(
        &mut self,
        x: &[f64],
        y: &[f64],
        lambda2: f64,
        init_beta: f64,
        n_iters: i32,
        lr: f64,
    ) -> f64 {
        self.beta = init_beta;

        for _ in 0..n_iters {
            let grad = self.grad_function(x, y, lambda2);
            self.beta -= lr * grad;
        }

        self.loss_function(x, y, lambda2)
    }

    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|xi| self.beta * xi).collect()
    }
}
