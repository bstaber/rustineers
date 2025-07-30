# KRR Model

This section describes the definition of the `KRRModel` struct and its constructor.

<details>
<summary>Click here to view the full model: <b>model.rs</b>.</summary>

```rust
{{#include ../../../crates/krr_ndarray/src/model.rs}}
```
</details>

## The `KRRModel` struct

The `KRRModel` struct represents a kernel ridge regression model parameterized by a kernel type `K` that implements the `Kernel` trait. It includes the following fields:

- `kernel`: an instance of the kernel function to be used (e.g., RBF kernel).
- `lambda`: the regularization parameter.
- `x_train`: optional training inputs stored after fitting.
- `alpha`: optional dual coefficients computed during fitting.

These fields are marked `pub` depending on whether they are exposed to the user.

## The `new` method

The `new` method is a constructor for creating a new instance of the model. It takes a kernel instance and a regularization parameter as arguments, and initializes an unfitted model:

```rust
pub fn new(kernel: K, lambda: f64) -> Self {
    Self {
        kernel,
        lambda,
        x_train: None,
        alpha: None,
    }
}
```

## Unit test

The `test_krr_constructor` unit test validates that the constructor sets the `lambda` and kernel `lengthscale` fields correctly:

```rust
#[test]
fn test_krr_constructor() {
    let kernel = RBFKernel::new(1.0);
    let model: KRRModel<RBFKernel> = KRRModel::new(kernel, 1.0);

    assert_eq!(model.lambda, 1.0);
    assert_eq!(model.kernel.lengthscale, 1.0);
}
```