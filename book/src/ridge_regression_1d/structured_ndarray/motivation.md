# Using `ndarray`, `Option`, and error handling

This section introduces several important features of the language:

- Using the `ndarray` crate for numerical arrays
- Representing optional values with `Option`, `Some`, and `None`
- Using pattern matching with `match`
- Handling errors using `Box<dyn Error>` and `.into()`
- Automatically deriving common trait implementations using `#[derive(...)]`

## Motivation

In previous sections, we worked with `Vec<f64>` and returned plain values. In practice, we might need:

- Efficient linear algebra tools, provided by external crates such as `ndarray` and `nalgebra`
- A way to represent "fitted" or "not fitted" states, using `Option<f64>`
- A way to return errors when something goes wrong, using `Result<_, _>>`
- Automatically implementing traits like `Debug`, `Clone`, and `Default` to simplify testing, debugging, and construction

We combine these in the implementation of the analytical `RidgeEstimator`. You can have a look to the full code below before we go through the main features step by step.

<details>
<summary>The full code : <b>regressor.rs</b></summary>

```rust
{{#include ../../../../crates/ridge_regression_1d/src/structured_ndarray/regressor.rs}}
```
</details>

# What we're building here

The aim of this chapter is to build a small crate with the following layout:

```text
crates/ridge_1d_ndarray/
├── Cargo.toml
└── src
    ├── regressor.rs    # Closed-form solution of the Ridge estimator
    └── lib.rs          # Main entry point for the library
```

Again, the module `regressor.rs` implements a `RidgeEstimator` type. We end up with the following user interface:

```rust
use ndarray::array;
use regressor::RidgeEstimator;

let mut model = RidgeEstimator::new();

let x = array![1.0, 2.0];
let y = array![0.1, 0.2];
let lambda2 = 0.001;

model.fit(&x, &y, lambda2);
let preds = model.predict(&x);

match model.beta {
    Some(beta) => println!("Learned beta: {beta}, true solution: 0.1!"),
    None => println!("Model not fitted!"),
}
```