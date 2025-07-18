# Optional model state and `#[derive(...)]` for common traits

In addition of using `ndarray` instead of `Vec<f64>`, we also slightly modify the model `struct`.

To make our `RidgeEstimator` struct more ergonomic, we derive a few useful traits: `Debug`, `Clone`, and `Default`:

- The `Debug` trait allows us to print the struct for inspection using `println!("{:?}", ...)`, which is helpful during development. 
- `Clone` lets us duplicate the struct, which is often needed in data processing pipelines. 
- `Default` enables us to create a default value using `RidgeEstimator::default()`, which internally calls the `new()` method we define.

```rust
{{#include ../../../../crates/ridge_regression_1d/src/structured_ndarray/regressor.rs:struct}}
```

The line

```rust
beta: Option<f64>
```

means `beta` can be either:

- `Some(value)`: if the model is trained
- `None`: if the model has not been fitted yet

This way, we can explicitly model the fact that the estimator may not be fitted yet (i.e., no coefficients computed). When we initialize the model, we set `beta` to `None` as follows:

```rust
impl RidgeEstimator {
    /// Creates a new, unfitted Ridge estimator.
    ///
    /// # Returns
    /// A `RidgeEstimator` with `beta` set to `None`.
    pub fn new() -> Self {
        Self { beta: None }
    }
}
```
