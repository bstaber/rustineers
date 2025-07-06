# Optimizers using ndarray

This section introduces a modular and idiomatic way to implement optimization algorithms in Rust using `ndarray` and traits. It is intended for readers who are already comfortable with basic Rust syntax and want to learn how to build reusable, extensible components in numerical computing.

## 1. Trait-based design

We define a trait called `Optimizer` to represent any optimizer that can update model weights based on gradients. In contrast to the previous sections where we mostly implemented `step`functions, here the trait requires implementors to define a `run` method with the following signature:

```rust
{{#include ../../../crates/simple_optimizers/src/traits_and_ndarray/optimizers.rs:trait}}
```

This method takes:
- A mutable reference to a vector of weights (`Array1<f64>`).
- A function that computes the gradient of the loss with respect to the weights. This `grad_fn` function takes itself a borrowed reference to the weights `&Array1<f64>` and outputs a new array `Array1<f64>`.
- The number of iterations to perform.

This trait `run` defines the whole optimization algorithm.

## 2. Gradient descent (GD)

The `GD` struct implements basic gradient descent with a fixed step size:

```rust
{{#include ../../../crates/simple_optimizers/src/traits_and_ndarray/optimizers.rs:struct_gd}}
```

It has a constructor:

```rust
{{#include ../../../crates/simple_optimizers/src/traits_and_ndarray/optimizers.rs:impl_gd_new}}
```

And implements `Optimizer` by subtracting the gradient scaled by the step size from the weights at each iteration.

```rust
{{#include ../../../crates/simple_optimizers/src/traits_and_ndarray/optimizers.rs:impl_gd_run}}
```

Some notes:

- At each iteration, we compute the gradient with `let grads = grad_fn(weights)`, which is fine but it reallocates a new vector at each call. If we wanted to optimize the gradient computation, we could pre-allocate a buffer outside the loop and pass a mutable reference into the gradient function to avoid repeated allocations. This would require to change the signature of the `grad_fn`.

- `weights.zip_mut_with(&grads, |w, &g| {{ ... }})`: This is a mutable zip operation from the `ndarray` crate. It walks over `weights` and `grads`, applying the closure to each pair.

- [zip_mut_with](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.zip_mut_with) is a method defined by the `Zip` trait, which is implemented for [ArrayBase](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html), and in particular for [Array1<f64>](https://docs.rs/ndarray/latest/ndarray/type.Array1.html). That’s why we can call it directly on `weights`.

- In the closure statement we wrote: `|w, &g| *w -= self.step_size * g;`. Here, `w` is a mutable reference to each weight element, so we dereference it using `*w` to update its value. The `&g` in the closure means we’re pattern-matching by reference to avoid cloning or copying each `f64`.


## 3. Accelerated gradient descent (AGD)

The `AGD` struct extends gradient descent with a classical momentum term:

```rust
{{#include ../../../crates/simple_optimizers/src/traits_and_ndarray/optimizers.rs:struct_agd}}
```


## 4. Nesterov-based adaptive AGD

The `AdaptiveAGD` struct implements a more advanced optimizer based on Nesterov's method and FISTA:

```rust
{{#include ../../../crates/simple_optimizers/src/traits_and_ndarray/optimizers.rs:AdaptiveAGD_struct}}
```

The implementation includes an extrapolation step based on previous iterates. It uses a sequence `t_k` to adaptively control the extrapolation coefficient. This is common in smooth convex optimization literature.

## Summary

This design demonstrates a few Rust programming techniques:
- Traits for abstraction and polymorphism
- Structs to encapsulate algorithm-specific state
- Use of the `ndarray` crate for numerical data
- Generic functions using closures for computing gradients
