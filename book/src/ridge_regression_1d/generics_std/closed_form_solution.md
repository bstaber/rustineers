# Generic Ridge estimator

In this section, we implement a Ridge estimator that works with `f32` and `f64` using generics and trait bounds.

To make this work, we need to import the following:

```rust
use num_traits::Float;
use std::iter::Sum;
```

## Ridge model trait

We start by defining a trait, `RidgeModel`, that describes the core behavior expected from any Ridge regression model. We tell the compiler that `F` must implement the traits `Float` and `Sum`.

```rust
{{#include ../../../../crates/ridge_1d_generic/src/regressor.rs:ridge_model_trait}}
```

## Ridge estimator structure

We next define a Ridge structure as usual but using our generic type `F`. The model only stores the Ridge coefficient `beta`.

```rust
{{#include ../../../../crates/ridge_1d_generic/src/regressor.rs:gen_ridge_estimator}}
```

We implement the constructor as usual as well.

```rust
{{#include ../../../../crates/ridge_1d_generic/src/regressor.rs:gen_ridge_estimator_impl}}
```

## Fit and predict methods

We can finally implement the trait `RidgeModel` for our `GenRidgeEstimator`.

```rust
{{#include ../../../../crates/ridge_1d_generic/src/regressor.rs:gen_ridge_estimator_trait_impl}}
```
Notice that the trait bounds `<F: Float + Sum> RidgeModel<F>` are defined after the name of a `trait` or `struct`, or right next to an `impl`.

````admonish title="Why do we need the Sum trait bound"

Without `Sum`, the compiler does not allow `.sum()` on iterators of `F`. Try removing `Sum` from the bound:

```rust
impl<F: Float> RidgeModel<F> for GenRidgeEstimator<F>
```

And keep a call to `.sum()`. The compiler should complain:

```
error[E0599]: the method `sum` exists for iterator `std::slice::Iter<'_, F>`,
              but its trait bounds were not satisfied
```
````

```admonish title="The copied() method"

The `copied()` method in `.iter().copied().sum::<F>()` is necessary because we're iterating over a slice of `F`, and `F` is a generic type that implements the `Copy` trait but not the `Clone` trait by default. 

Without this, `x.iter()` yields references `&F` while `sum::<F>()` expects owned values of type `F`. We could have used `cloned()` instead but since `Float` already requires `Copy`, this works without adding the `Clone` trait bound.

Note that in the `predict` function, we don't need to use `Copy` because we manually dereference each item, `*xi`, inside the `.map()`. It could have been possible to use `copied()` there as well and modify the mapping closure accordingly.

```

````admonish title="The unwrap() method"

The `unwrap` in `F::from(n).unwrap()`. The length `n` of the slice `x` is of type `usize`, as usual. We need `n_f` to be of type `F` so that we can perform operations like division with other `F`-typed values. 

The conversion is done using `F::from(n)` which returns an `Option<F>`, not a plain `F`. We assumed that the conversion always succeeds or crashes by using `unwrap()`. Since `n` is from `x.len()`, it might easily be representable as `f32` or `f64`, so unwrapping seems safe.

Note that we could have handled the error explicitly:

```rust
let n_f: F = F::from(n).expect("Length too large to convert to float");
````