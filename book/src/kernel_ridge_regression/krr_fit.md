# Fit function

This section describes the implementation of the `fit` function, which prepares the model for prediction by solving the kernel ridge regression problem.

The full code is given below and we break it down in the sequel of this section.

```rust
{{#include ../../../crates/krr_ndarray/src/model.rs:fit_function}}
```

## `_fit` and `fit` methods

The `_fit` method performs the main computation:

1. Computes the Gram matrix using the kernel.
2. Adds regularization to the diagonal.
3. Solves the linear system for the dual coefficients.

The public `fit` method wraps `_fit` and performs input validation. It checks that the dimensions of `x_train` and `y_train` match, and logs messages about success or failure.

The signature of the `fit` function is given by:

```rust
pub fn fit(&mut self, x_train: Array2<f64>, y_train: Array1<f64>) -> Result<(), KRRFitError>
```

Before looking at `fit` and `_fit`, we need to define the enum `KRRFitError`.

## KRRFitError enum

The `KRRFitError` enum defines two error types:

- `ShapeMismatch`: occurs when the number of samples in `x_train` and `y_train` do not match.
- `LinAlgError`: returned if solving the linear system fails.

This enum is used to cleanly propagate and format error messages via `Result`. This enum is implement thanks to `thiserror` as follows:

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KRRFitError {
    #[error("Shape mismatch: x has {x_n} rows but y has {y_n} elements")]
    ShapeMismatch { x_n: usize, y_n: usize },

    #[error("Solving the linear system failed")]
    LinAlgError(String),
}
```

## The `_fit` function

The `_fit` function computes the dual coefficients $\alpha$ by solving the linear system:

$$
(K(X, X) + \lambda I_n) \alpha = y
$$

Here, $X$ and $y$ represent the training data, $K(X, X)$ is the Gram matrix computed from the kernel, $\lambda$ is the regularization parameter, and $I_n$ is the identity matrix of size $n$.

The function proceeds as follows:

* It first computes the symmetric Gram matrix and stores it in the variable k_train.
* It constructs the left-hand side matrix $a := K(X, X) + \lambda I_n$.
* It solves the resulting linear system for $\alpha$ using the `solve_into()` method.
* Finally, it stores `x_train` and the computed alpha inside the model. Keeping `x_train` is essential for future predictions on new inputs.

The full function is shown below. You can try to spot interesting stuff that we haven't mentioned yet. We make a few additional comments afterwards.

```rust
fn _fit(&mut self, x_train: Array2<f64>, y_train: Array1<f64>) -> Result<(), KRRFitError> {
    let n: usize = y_train.len();
    let mut k_train: Array2<f64> = Array::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let kxy = self.kernel.compute(x_train.row(i), x_train.row(j));
            k_train[(i, j)] = kxy;
            k_train[(j, i)] = kxy;
        }
    }

    let identity_n = Array2::eye(n);
    let a: Array2<f64> = k_train + self.lambda * identity_n;
    let alpha = a
        .solve_into(y_train)
        .map_err(|e| KRRFitError::LinAlgError(e.to_string()))?;

    self.x_train = Some(x_train);
    self.alpha = Some(alpha);

    Ok(())
}
```

Additional notes:
* `x_train` and `y_train` are not passed by reference and are therefore moved into the _fit function. This is fine because we do not need them afterward, and `x_train` is stored into `self.x_train` at the end of the function.

* The method `x_train.row(i)` extracts the i-th row of the training matrix as an `ArrayView1<f64>`, which is exactly the input type expected by our `kernel.compute` method.

* The line that computes `alpha` ends with a question mark ?, which is Rust syntax for propagating errors. If `solve_into()` fails (for instance, due to an ill-conditioned matrix), the function returns early with a `KRRFitError::LinAlgError`. If it succeeds, the result is assigned to alpha, and we continue toward returning `Ok(())`, consistent with the declared return type `Result<(), KRRFitError>`.

* `x_train` and `alpha` are wrapped in `Some(...)` because the fields in the KRRModel struct are declared as `Option`.

## The `fit` function

The fit function serves as the public interface for training the model. It takes ownership of the training data and performs validation before delegating the actual computation to the private `_fit` method. Its signature is:

```rust
pub fn fit(&mut self, x_train: Array2<f64>, y_train: Array1<f64>) -> Result<(), KRRFitError> {
    let n: usize = x_train.nrows();
    let m: usize = y_train.len();

    if n != m {
        eprintln!("[KRR::fit] Shape mismatch: x_train has {n} rows, y_train has {m} elements");
        return Err(KRRFitError::ShapeMismatch { x_n: n, y_n: m });
    }

    match self._fit(x_train, y_train) {
        Ok(_) => {
            eprintln!("[KRR::fit] Model successfully fitted.");
            Ok(())
        }
        Err(e) => {
            eprintln!("[KRR::fit] Fitting failed: {e}");
            Err(e)
        }
    }
}
```

Here's how it works step-by-step:

* It extracts the number of training samples in `x_train` (n) and compares it to the number of targets in `y_train` (m).

* If these sizes do not match, it logs a message and returns a `KRRFitError::ShapeMismatch` variant. This early return prevents proceeding with inconsistent inputs.

* If the shapes are consistent, the function calls the `_fit` method to perform the actual kernel ridge regression fitting.

* It logs whether the fitting was successful or not, and returns a Result accordingly.

This design separates concerns:

* `fit` is responsible for input checking and logging,
* `_fit` performs the mathematical computations.

This modular approach makes it easier to write clean tests, and to report errors in a structured and maintainable way.

## Unit tests

The `test_ok_fit_and_predict` test verifies that a valid fit and prediction workflow runs without errors.

```rust
#[test]
fn test_ok_fit_and_predict() {
    let kernel = RBFKernel::new(1.0);
    let mut model: KRRModel<RBFKernel> = KRRModel::new(kernel, 1.0);
    let x_train: Array2<f64> = array![[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]];
    let y_train: Array1<f64> = array![0.9, 0.6];

    let res = model.fit(x_train, y_train);
    assert!(res.is_ok());

    let x_test: Array2<f64> = array![[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]];
    let y_pred = model.predict(&x_test);
    assert!(y_pred.is_ok());
}
```

The `test_dim_mismatch` test confirms that the model returns an appropriate error when input shapes are inconsistent:

```rust
#[test]
fn test_dim_mismatch() {
    let x_train = array![[1.0, 2.0], [3.0, 4.0]];
    let y_train = array![1.0, 2.0, 3.0];
    let res = model.fit(x_train, y_train);
    assert!(res.is_err());
}
```