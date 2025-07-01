# Closed-form solution with `ndarray`, `Option`, and error handling

This section introduces several important features of the language:

- Using the `ndarray` crate for numerical arrays
- Representing optional values with `Option`, `Some`, and `None`
- Using pattern matching with `match`
- Handling errors using `Box<dyn Error>` and `.into()`
- Automatically deriving common trait implementations using #[derive(...)]

## Motivation

In previous sections, we worked with `Vec<f64>` and returned plain values. In practice, we might need:

- Efficient linear algebra tools, provided by external crates such as `ndarray` and `nalgebra`
- A way to represent "fitted" or "not fitted" states, using `Option<f64>`
- A way to return errors when something goes wrong, using `Result<_, Box<dyn Error>>`
- Automatically implementing traits like `Debug`, `Clone`, and `Default` to simplify testing, debugging, and construction

We combine these in the implementation of the analytical `RidgeEstimator`.

## The full code

```rust
{{#include ../../../../crates/ridge_regression_1d/src/structured_ndarray/regressor.rs}}
```

### 1. `ndarray` instead of `Vec<f64>`

Rust’s standard library does not include built-in numerical computing tools. The `ndarray` crate provides efficient n-dimensional arrays and vectorized operations.

This example uses:

- `Array1<f64>` for 1D arrays
- `.mean()`, `.dot()`, and `.mapv()` for basic mathematical operations
- Broadcasting (`x * beta`) for scalar–array multiplication

### 2. Representing model state with `Option<f64>`

The model's coefficient `beta` is only available after fitting. To represent this, we use:

```rust
beta: Option<f64>
```

This means `beta` can be either:

- `Some(value)`: if the model is trained
- `None`: if the model has not been fitted yet

This eliminates the possibility of using an uninitialized value.

### 3. Pattern matching with `match`

To check whether the model has been fitted, we use pattern matching:

```rust
match self.beta {
    Some(beta) => Ok(x * beta),
    None => Err("Model not fitted".into()),
}
```

Pattern matching ensures that all possible cases of the `Option` type are handled explicitly. In this case, the prediction will only be computed if `beta` is not None, and an error is thrown otherwise.

The error handling is explain hereafter.

### 4. Error handling with `Box<dyn Error>` and `.into()`

Rust requires functions to return a single concrete error type. In practice, this can be achieved in several ways. Here we use a trait object:

```rust
Result<Array1<f64>, Box<dyn Error>>
```

If the function succeeds, it must return a `Array1<f64>`. 

If it doesn't succeed, we allow the function to return any error type that implements the `Error` trait. The `.into()` method converts a string literal into a `Box<dyn Error>`. Internally, Rust converts:

```rust
"Model not fitted"
```

into:

```rust
Box::new(String::from("Model not fitted"))
```

It is worth emphasizing that `Box<dyn Error>` means that the error is heap-allocated.

### 5. Using `#[derive(...)]` for common traits

Rust allows us to automatically implement certain traits using the `#[derive(...)]` attribute. In this example, we write:

```rust
#[derive(Debug, Clone, Default)]
pub struct RidgeEstimator {
    beta: Option<f64>,
}
```

This provides the following implementations:

- `Debug`: Enables printing the struct with `{:?}`, useful for debugging.
- `Clone`: Allows duplicating the struct with `.clone()`.
- `Default`: Provides a default constructor (`RidgeEstimator::default()`), which sets `beta` to `None`.

By deriving these traits, we avoid writing repetitive code and ensure that the model is compatible with common Rust conventions, such as default initialization and copy-on-write semantics.

````admonish tip title="Advanced error handling"
We could have gone even further by defining a custom `ModelError` type as follows.

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Model is not fitted yet")]
    NotFitted,
    #[error("Dimension mismatch")]
    DimensionMismatch,
}
```

This approach uses the `thiserror` crate to simplify the implementation of the standard `Error` trait.

By deriving `#[derive(Debug, Error)]` and annotating each variant with `#[error("...")]`, we define error messages rightaway.

The predict function would be rewritten as:

```rust
pub fn predict(&self, x: &Array1<f64>) -> Result<f64, ModelError> {
    match &self.beta {
        Some(beta) => {
            if beta.len() != x.len() {
                return Err(ModelError::DimensionMismatch);
            }
            Ok(beta.dot(x))
        }
        None => Err(ModelError::NotFitted),
    }
}
```
````