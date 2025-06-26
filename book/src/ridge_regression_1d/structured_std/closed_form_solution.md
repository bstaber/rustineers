# Closed-form solution

We now present a structured implementation of the 1D Ridge estimator using a dedicated `RidgeEstimator` struct. We implement the same methods, i.e., 

* fit: the method to compute the optimal $\beta$ from data and the regularization parameter $\lambda$,
* predict: the method to compute predictions from new data,

but rely on Rust's `struct` and `impl` to define a new type. We also an additional method `new`, a constructor to initialize the estimator with an initial value of $\beta$.

## Struct definition

This simple struct stores the estimated coefficient $\beta$ as a field.

```rust
pub struct RidgeEstimator {
    beta: f64,
}
```

## Constructor and methods

Once the `struct` is defined, we can implement the constructor `new`, and the methods `fit` and `predict`.

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

Note that we can decompose the implementation into as many blocks as we want:

```rust
impl RidgeEstimator {
    pub fn new(init_beta: f64) -> Self {
        Self { beta: init_beta }
    }
}

impl RidgeEstimator {
    fn fit(&mut self, x: &[f64], y: &[f64], lambda2: f64) {
        ...
    }
}

impl RidgeEstimator {
    fn predict(&self, x: &[f64]) -> Vec<f64> {
        ...
    }
}
```

This can be useful when dealing with complex methods.

## Example of usage

Here is how we can use our new Ridge estimator:

```rust
fn main() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![2.1, 4.1, 6.2, 8.3];
    let lambda = 0.1;

    let mut model = RidgeEstimator::new(0.0);
    model.fit(&x, &y, lambda);

    let predictions = model.predict(&x);
    println!("Predictions: {:?}", predictions);
}
```