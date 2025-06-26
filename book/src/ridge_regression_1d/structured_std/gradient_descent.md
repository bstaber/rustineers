# Gradient descent

As an another illustration of `struct` and `impl`, let's tackle the gradient descent method for the Ridge regression again. We use the following structure:

```rust
pub struct RidgeGradientDescent {
    beta: f64,
    n_iters: usize,
    lr: f64,
}
```

This struct stores the current coefficient $\beta$, the number of iterations to run, and the learning rate. We can subsequently implement the constructor and all the methods required to perform gradient descent.

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

## Example of usage

Here is how we can use our new Ridge estimator:

```rust
fn main() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![2.1, 4.1, 6.2, 8.3];

    let mut model = RidgeGradientDescent::new(1000, 0.01, 0.0);
    model.fit(&x, &y, 0.1);

    let predictions = model.predict(&x);
    println!("Predictions: {:?}", predictions);
}
```