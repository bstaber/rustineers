
# Chapter: Generic types and trait bounds in Rust

## What are generics?

Generics let you write code that works with many types, not just one.

Instead of writing:

```rust
struct RidgeEstimator {
    beta: f64,
}
```

You can write:

```rust
struct RidgeEstimator<F> {
    beta: F,
}
```

Here, `F` is a type parameter — it could be `f32`, `f64`, or another type. In Rust, generic types have no behavior by default.

For example:

```rust
fn sum(xs: &[F]) -> F {
    xs.iter().sum() // This will not compile
}
```

The compiler gives an error: “`F` might not implement `Sum`, so I don’t know how to `.sum()` over it.”

## Trait bounds

To fix that, we must tell the compiler which traits `F` should implement.

For example:

```rust
impl<F: Float + Sum> RidgeModel<F> for GenRidgeEstimator<F> {
    ...
}
```

This means:
- `F` must implement `Float` (it must behave like a floating point number: support `powi`, `abs`, etc.)
- `F` must implement `Sum` (so we can sum an iterator of `F`)

This allows code like:

```rust
let mean = xs.iter().copied().sum::<F>() / F::from(xs.len()).unwrap();
```

Using generic bounds allows the estimator to work with `f32`, `f64`, or any numeric type implementing `Float`. The compiler can generate specialized code for each concrete type.

## Generic Ridge estimator

Our Ridge estimator that works with `f32` and `f64` takes this form:

```rust
{{#include ../../../../crates/ridge_regression_1d/src/generics_std/gen_regressor.rs}}
```

Notice that the trait bounds `<F: Float + Sum> RidgeModel<F>` are defined after the name of a `trait` or `struct`, or right next to an `impl`.

## Summary

- Generics support type-flexible code.
- Trait bounds like `<F: Float + Sum>` constrain what operations are valid.
- Without `Sum`, the compiler does not allow `.sum()` on iterators of `F`.

Try removing `Sum` from the bound:

```rust
impl<F: Float> RidgeModel<F> for GenRidgeEstimator<F>
```

And keep a call to `.sum()`. The compiler should complain:

```
error[E0599]: the method `sum` exists for iterator `std::slice::Iter<'_, F>`,
              but its trait bounds were not satisfied
```

To resolve this, add `+ Sum` to the bound.
