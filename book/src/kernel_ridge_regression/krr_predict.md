# Predict function

This section describes the implementation of the `predict` method, which uses the trained model to make predictions on new inputs. The full code is given below, and we break it down in the sequel of this section.

```rust
{{#include ../../../crates/krr_ndarray/src/model.rs:predict_function}}
```

It takes a reference to a two-dimensional test array and returns either the predicted values as a one-dimensional array or an error if the model is not yet fitted.

## Logic of the predict method

The function first checks whether the model has been fitted. This involves verifying that the fields `self.alpha` and `self.x_train` are set (i.e., not `None`). If either is missing, the function returns a `KRRPredictError::NotFitted` variant.

```rust
let alpha = self.alpha.as_ref().ok_or(KRRPredictError::NotFitted)?;
let x_train = self.x_train.as_ref().ok_or(KRRPredictError::NotFitted)?;
```

The actual prediction then proceeds by computing the kernel value between each training point and each test point:

```rust
for i in 0..n_test {
    for j in 0..n_train {
        let k_val = self.kernel.compute(x_train.row(j), x_test.row(i));
        y_pred[i] += alpha[j] * k_val;
    }
}
```

This implements the inference equation:

$$
\hat{y}(x) = \sum_{i=1}^n \alpha_i K(x_i, x)
$$

where $x_i$ are the training samples, $\alpha_i$ are the learned dual coefficients, and $K$ is the kernel function.

## KRRPredictError

The `KRRPredictError` is an enum used to indicate that the model has not been fitted yet. It is defined in the `errors.rs` module using the `thiserror` crate:

```rust
#[derive(Debug, Error)]
pub enum KRRPredictError {
    #[error("Model not fitted")]
    NotFitted,
}
```

This enum allows the `predict` function to return a `Result` type, making error propagation idiomatic and clean.

## Use of `?` operator

The function uses the `?` operator to simplify error handling. For example:

```rust
let alpha = self.alpha.as_ref().ok_or(KRRPredictError::NotFitted)?;
```

This line either extracts the `alpha` reference if it exists or returns early with an error. This pattern keeps the code concise and expressive.

## Unit tests

The `test_unfitted_predict_error_type` unit test checks that the correct error is returned when attempting to call `predict` before fitting the model:

```rust
#[test]
fn test_unfitted_predict_error_type() {
    use crate::errors::KRRPredictError;

    let kernel = RBFKernel::new(1.0);
    let model: KRRModel<RBFKernel> = KRRModel::new(kernel, 1.0);
    let x_test: Array2<f64> = array![[1.0, 2.0, 3.0]];

    let result = model.predict(&x_test);
    assert!(matches!(result, Err(KRRPredictError::NotFitted)));
}
```
