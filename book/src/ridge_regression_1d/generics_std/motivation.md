# Generics: introduction

The aim of this section is to generalize our estimators so they work with any numeric type, not just `f64`. Rust makes this possible through generics and trait bounds. It's divided into 2 subsections:

1) [Generics & trait bounds](what_are_generics.md): Introduces generics and trait bounds. The floating-point type `f64` is replaced by a generic type `F` that can either be `f32` or `f64`. In Rust, generic types have no behavior by default, and we must tell the compiler which traits `F` should implement.
2) [Closed-form solution](closed_form_solution.md): Explains how to implement the closed-form solution with generics and traits.

In the [next final section](../structured_ndarray/motivation.md), we finally explore how to use the external crate `ndarray` for linear algebra, and how to incorporate additional Rust features such as optional values, pattern matching, and error handling.

# What we're building here

The aim of this chapter is to build a small crate with the following layout:

```text
crates/ridge_1d_generic/
├── Cargo.toml
└── src
    ├── regressor.rs    # Closed-form solution of the Ridge estimator
    └── lib.rs          # Main entry point for the library
```

The module `regressor.rs` implements the closed-form Ridge estimator using the generic type `Float`. As usual, the resulting regressor, here called `GenRidgeEstimator`, is exposed through the library entry point `lib.rs`.

In contrast to the previous implementations, this can be used with `f32` or `f64` floating-point types.

**Example of usage**:

```rust
use ridge_1d_generic::GenRidgeEstimator;

let mut model: GenRidgeEstimator<f32> = GenRidgeEstimator::new(1.0);

let x: Vec<f32> = vec![1.0, 2.0];
let y: Vec<f32> = vec![0.1, 0.2];
let lambda2 = 0.001;

model.fit(&x, &y, lambda2);
let preds: Vec<f32> = model.predict(&x);
```