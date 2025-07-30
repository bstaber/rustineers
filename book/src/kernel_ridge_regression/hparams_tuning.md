
# Hyperparameter tuning with LOO-CV

This section focuses on hyperparameter selection for tuning the kernel lengthscale using Leave-One-Out Cross-Validation (LOO-CV). We implement two key functions in the `model_selection.rs` module:

- `loo_cv_error`: computes the LOO-CV error for a given model and training dataset.
- `tune_lengthscale`: evaluates multiple candidate lengthscales and returns the one with the lowest LOO-CV error.

## Leave-One-Out Cross-Validation (LOO-CV)

LOO-CV is a classical cross-validation strategy in which each data point is left out once as a test sample while the model is trained on the remaining samples. The final error is the average squared difference between the predicted and true values.

In KRR, thanks to the closed-form solution, the LOO-CV error can be computed efficiently without retraining the model $n$ times. The formula is:

$$
	e_{\mathrm{LOO}} = \left( \frac{y_i - (K \alpha)_i}{1 - H_{ii}} \right)^2
$$

Where:

- $H = K (K + \lambda I)^{-1}$ is the hat matrix,
- $(K \alpha)_i$ is the prediction on the training data,
- $H_{ii}$ is the i-th diagonal element of the hat matrix.

The `loo_cv_error` function implements this logic:

```rust
{{#include ../../../crates/krr_ndarray/src/model_selection.rs:loo_cv_error}}
```

It returns the mean squared error over the training set based on the LOO-CV formula. 

Note that we got a bit lazy here:

* We use the `.expect()` method to raise an exception if the inversion fails. This will make the code crash instead of making the function return an error, like we did with our `KRRFitError` and `KRRPredictError` enums.
* We re-compute the Gram matrix whereas it could be stored within the `KRRModel` like we did for `alpha` and `x_train`.

## Tuning the lengthscale

We now want to search for the optimal lengthscale of the RBF kernel that minimizes the LOO-CV error. The function:

```rust
{{#include ../../../crates/krr_ndarray/src/model_selection.rs:tune_lengthscale}}
```

takes in a list of candidate lengthscales, fits a model for each, and selects the one with the lowest LOO-CV error. Internally, it uses:

1. The `RBFKernel` struct to instantiate kernels with varying lengthscales.
2. The `KRRModel::fit` method to train each model.
3. The `loo_cv_error` function to evaluate them.

## Unit test

The `test_tune_lengthscale` test verifies that the tuning function works correctly:

```rust
#[test]
fn test_tune_lengthscale() {
    let x_train = array![[0.0], [1.0], [2.0]];
    let y_train = array![0.0, 1.0, 2.0];
    let candidates = vec![0.01, 0.1, 1.0, 10.0];

    let best = tune_lengthscale(x_train, y_train, &candidates);
    assert!(candidates.contains(&best));
}
```

This test confirms that the selected best value is indeed one of the candidate lengthscales.
