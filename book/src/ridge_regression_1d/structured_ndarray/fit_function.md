# Switching to `ndarray`

Rust’s standard library does not include built-in numerical computing tools. The `ndarray` crate provides efficient n-dimensional arrays and vectorized operations.

This example uses:

- `Array1<f64>` for 1D arrays
- `.mean()`, `.dot()`, and `.mapv()` for basic mathematical operations
- Broadcasting (`x * beta`) for scalar–array multiplication

## Fit function

The `impl` of the fit function is shown below. As you can see, we essentially replaced `Vec<f64>` by `Array1<f64>` here and there. This allows us to rely on `.mean`, `dot`, or `mapv` to perform basic linear algebra. Given that `self.beta` is defined as an `Option<f64>`, we return `Some(num / denom)`, from which Rust can infer the type `Some(f64)`.


```rust
{{#include ../../../../crates/ridge_1d_ndarray/src/regressor.rs:ridge_estimator_impl_new_fit}}
```

```admonish
`ndarray` can also handle higher-dimensional vectors and matrices with types like Array2<f64> for 2D arrays. This makes it a powerful choice for implementing linear algebra operations and machine learning models in Rust, where you may generalize from 1D to multi-dimensional data.
```