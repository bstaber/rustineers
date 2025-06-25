# Gradient descent

```rust
pub struct RidgeGradientDescent {
    beta: f64,
    n_iters: usize,
    lr: f64,
}
```

```rust
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

    fn fit(&mut self, x: &[f64], y: &[f64], lambda2: f64) {
        for _ in 0..self.n_iters {
            let grad = self.grad_function(x, y, lambda2);
            self.beta -= self.lr * grad;
        }
    }

    fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|xi| self.beta * xi).collect()
    }
}
```