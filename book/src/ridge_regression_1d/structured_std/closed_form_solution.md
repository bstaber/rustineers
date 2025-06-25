# Closed-form solution

```rust
pub struct RidgeEstimator {
    beta: f64,
}
```

```rust
impl RidgeEstimator {
    pub fn new(init_beta: f64) -> Self {
        Self { beta: init_beta }
    }

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
        x.iter().map(|xi| self.beta * xi).collect()
    }
}
```