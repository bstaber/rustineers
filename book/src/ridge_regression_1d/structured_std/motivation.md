# Structured: introduction

This section focuses on implementinng the 1D Ridge problem using functions, structures, traits and Rust standard library only. It's divided into 3 subsections:

1) [Closed-form solution](closed_form_solution.md): Implements the closed-form solution of the Ridge optimization problem using a `struct` to define a `RidgeEstimator` type. It shows how to implement a constructor together with `fit` and `predict` functions.
2) [Gradient descent](gradient_descent.md): Solves the Ridge problem using gradient descent using a `struct` as well to define a `RidgeGradientDescent` type.
3) [Trait Ridge model](traits.md): Explains how to define a trait `RidgeModel`, which describes the shared behavior of any Ridge estimator like our `RidgeEstimator` and `RidgeGradientDescent`.

# What we're building here

The aim of this chapter is to build a small crate with the following layout:

```text
crates/ridge_1d_struct/
├── Cargo.toml
└── src
    ├── regressor.rs    # Closed-form solution of the Ridge estimator
    └── lib.rs          # Main entry point for the library
```

It is made of a single module `regressor.rs` which implements both the closed-form Ridge estimator and the gradient descent-based estimator using structs, respectively called `RidgeEstimator` and `RidgeGradientDescent`.

These types are exposed in the library entry point `lib.rs`. We end up with the following user interfaces.

**Closed-form estimator**:

```rust
use ridge_1d_struct::RidgeEstimator;

let mut model: RidgeEstimator = RidgeEstimator::new(0.0);

let x: Vec<f64> = vec![1.0, 2.0];
let y: Vec<f64> = vec![0.1, 0.2];
let lambda2 = 0.001;

model.fit(&x, &y, lambda2);
let preds = model.predict(&x);
```

**Gradient descent-based estimator**:

```rust
use ridge_1d_struct::RidgeGradientDescent;

let mut model: RidgeGradientDescent = RidgeGradientDescent::new(0.0, 1000, 1e-2);

let x: Vec<f64> = vec![1.0, 2.0];
let y: Vec<f64> = vec![0.1, 0.2];
let lambda2 = 0.001;

model.fit(&x, &y, lambda2);
let preds = model.predict(&x);
```


# What's next

Up to this stage, we implemented everything using the `f64` precision for all our variables. In the [next section](../generics_std/motivation.md), we will see how to make our code independent of the floating-point types by leveraging generics.