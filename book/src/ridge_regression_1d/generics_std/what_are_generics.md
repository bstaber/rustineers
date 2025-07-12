# What are generics?

Generics let you write code that works with many types, not just one.

Instead of writing:

```rust
struct RidgeEstimator {
    beta: f64,
}
```

You can write:

```rust
use num_traits::Float;

struct RidgeEstimator<F> {
    beta: F,
}
```

Here, `F` is a type parameter — it could be `f32`, `f64`, or another type. In Rust, generic types have no behavior by default.

~~~admonish bug
```rust
fn sum(xs: &[F]) -> F {
    xs.iter().sum() // This will not compile
}
```

The compiler gives an error: "`F` might not implement `Sum`, so I don’t know how to `.sum()` over it."
~~~

## Trait bounds

To fix that, we must tell the compiler which traits `F` should implement.

For example:

```rust
use num_traits::Float;
use std::iter::Sum;

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