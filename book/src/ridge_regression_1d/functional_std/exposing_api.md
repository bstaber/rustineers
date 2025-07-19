# Exposing a clean API

Until now, we've manually chained together the loss, gradient, and optimization steps. This is great for learning, but in real projects, we often want a simplified and reusable API.

Rust gives us a clean way to do this by leveraging the `lib.rs` file as the public interface to our crate.

## `lib.rs` as a public API

In your crate, `lib.rs` is responsible for organizing and exposing the components we want users to interact with.

We can re-export key functions and define top-level utilities like `fit` and `predict`. The complete `lib.rs` file now looks like this:

```rust
pub mod estimator;
pub mod gradient_descent;
pub mod loss_functions;

pub use estimator::ridge_estimator;

/// Fits a Ridge regression model.
///
/// # Arguments
///
/// * `x` - Input features (`&[f64]`)
/// * `y` - Target values (`&[f64]`)
/// * `lambda2` - Regularization strength
///
/// # Returns
///
/// The optimized coefficient `beta` as `f64`.
pub fn fit(x: &[f64], y: &[f64], lambda2: f64) -> f64 {
    ridge_estimator(x, y, lambda2)
}

/// Predicts output values using a trained Ridge regression coefficient.
///
/// # Arguments
///
/// * `x` - Input features (`&[f64]`)
/// * `beta` - Trained coefficient
///
/// # Returns
///
/// A `Vec<f64>` with predicted values.
pub fn predict(x: &[f64], beta: f64) -> Vec<f64> {
    x.iter().map(|xi| xi * beta).collect()
}
```

Everything declared `pub` is available to the user. For simplicity, we decided to only expose the closed-form Ridge estimator.

## Example of usage

You can update your binary entry point to try out the public API.

```rust
use ridge_1d_fn::{fit, predict};

fn main() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];

    let beta = fit(&x, &y, 0.1, 0.01, 1000, 0.0);
    let preds = predict(&x, beta);

    println!("Learned beta: {}", beta);
    println!("Predictions: {:?}", preds);
}
```