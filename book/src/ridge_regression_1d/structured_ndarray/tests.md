# Adding tests

Tests can be included in the same file as the code using the #[cfg(test)] module. Each test function is annotated with #[test]. Inside a test, you can use assert_eq!, assert!, or similar macros to validate expected behavior.

The full test module can be seen below. We go through each of them in the sequel of this section.

<details>
<summary>Full test module</summary>

```rust
{{#include ../../../../crates/ridge_regression_1d/src/structured_ndarray/regressor.rs:tests}}
```
</details>

Recall that tests can be executed by running `cargo test`.

## Testing the constructor

As a first simple test, we check that `beta` of a new `RidgeEstimator` is `None`.

```rust
#[test]
fn test_ridge_estimator_constructor() {
    let model = RidgeEstimator::new();
    assert_eq!(model.beta, None, "beta is expected to be None");
}
```

## Testing an unfitted model

As a second test, we check that the predict function returns error if the model is unfitted.

```rust
#[test]
fn test_unfitted_estimator() {
    let model = RidgeEstimator::new();
    let x: Array1<f64> = array![1.0, 2.0];
    let result: Result<Array1<f64>, String> = model.predict(&x);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Model not fitted");
}
```

## Testing a fitted model

Finally, we check that a fitted model returns a `Some(f64)` and that the solution is close to the known value.

```rust
#[test]
fn test_ridge_estimator_solution() {
    let x: Array1<f64> = array![1.0, 2.0];
    let y: Array1<f64> = array![0.1, 0.2];
    let true_beta: f64 = 0.1;
    let lambda2: f64 = 0.0;

    let mut model = RidgeEstimator::new();
    model.fit(&x, &y, lambda2);

    assert!(model.beta.is_some(), "beta is expected to be Some(f64)");

    assert!(
        (true_beta - model.beta.unwrap()).abs() < 1e-6,
        "Estimate {} not close enough to true solution {}",
        true_beta,
        model.beta.unwrap()
    );
}
```