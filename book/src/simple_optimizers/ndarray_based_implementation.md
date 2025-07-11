# Optimizers using ndarray

This section introduces a modular and idiomatic way to implement optimization algorithms in Rust using `ndarray` and traits. It is intended for readers who are already comfortable with basic Rust syntax and want to learn how to build reusable, extensible components in numerical computing.

In this example, we need to import the following types and traits:
```rust
use ndarray::Array;
use ndarray::Array1;
use ndarray::Zip;
```

You can inspect the full module that we're about the break down over here:
<details>
<summary>Click to view <b>optimizers.rs</b></summary>

```rust
{{#include ../../../crates/simple_optimizers/src/traits_and_ndarray/optimizers.rs}}
```
</details>

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

It has a constructor:

```rust
{{#include ../../../crates/simple_optimizers/src/traits_and_ndarray/optimizers.rs:impl_agd_new}}
```

This algorithm adds momentum to classical gradient descent. Instead of updating weights using just the current gradient, it maintains a velocity vector that accumulates the influence of past gradients. This helps smooth the trajectory and accelerates convergence on convex problems.

```rust
{{#include ../../../crates/simple_optimizers/src/traits_and_ndarray/optimizers.rs:impl_agd_run}}
```

Some notes:

- We initialize a vector of zeros to track the momentum (velocity) across steps. It has the same length as the weights. This is achieved with: `let mut velocity: Array1<f64> = Array::zeros(n)`. Note that we could have defined the velocity as an internal state variable within the struct defintion.

- We use a triple nested zip to unpack the values of the weights, gradients, and velocity: `for ((w, g), v) in weights.iter_mut().zip(grads.iter()).zip(velocity.iter_mut())`. Here,
  - `weights.iter_mut()` gives a mutable reference to each weight,
  - `grads.iter()` provides read-only access to each gradient,
  - `velocity.iter_mut()` allows in-place updates of the velocity vector.

  This pattern allows us to update everything in one pass, element-wise.

- Within the nested zip closure, we update the velocity using the momentum term and current gradient: `*v = self.momentum * *v - self.step_size * g;`
  
- The weight is updated using the new velocity: `*w += *v;`. Again, we dereference `w` because it's a mutable reference.

## 4. Nesterov-based adaptive AGD

The `AdaptiveAGD` struct implements a more advanced optimizer based on Nesterov's method and FISTA:

```rust
{{#include ../../../crates/simple_optimizers/src/traits_and_ndarray/optimizers.rs:AdaptiveAGD_struct}}
```

It has a constructor:

```rust
{{#include ../../../crates/simple_optimizers/src/traits_and_ndarray/optimizers.rs:AdaptiveAGD_impl_new}}
```

This algorithm implements an accelerated method inspired by Nesterov’s momentum and the FISTA algorithm. The key idea is to introduce an extrapolation step between iterates, controlled by a sequence `t_k`. This helps the optimizer "look ahead" and converge faster in smooth convex problems.

Update steps:
- Compute a temporary point `y_{k+1}` by taking a gradient step from `x_k`.
- Update the extrapolation coefficient `t_{k+1}`.
- Combine `y_{k+1}` and `y_k` using a weighted average to get the new iterate `x_{k+1}`.

```rust
{{#include ../../../crates/simple_optimizers/src/traits_and_ndarray/optimizers.rs:AdaptiveAGD_impl_run}}
```

Some notes:

- We deliberately re-allocate multiple variables within the for loop (`grad`, `y_next`, `t_next`) but we could have pre-allocated buffers before the for loop.

- The algorithm keeps track of two sequences: the main iterate (`weights`) and the extrapolated one (`y_k`). Before starting the for loop, we initialize `y_k` by cloning the weights: `let mut y_k = weights.clone();`.

- The gradient is evaluated at the current weights, as in standard gradient descent: `let grad = grad_fn(weights);`. Since `weights` is a mutable reference, we can pass it straightaway to our `grad_fn`.
  
- A temporary variable to store the new extrapolated point. This is again a full allocation and clone for clarity. `let mut y_next = weights.clone();`.

- We next compute: `y_{k+1} = x_k - α ∇f(x_k)` using an element-wise operation: `Zip::from(&mut y_next).and(&grad).for_each(|y, &g| { *y -= self.step_size * g; });`. This time, we rely on the `Zip::from` trait implement by `ndarray`.
  
- The new weights are obtained by combining `y_{k+1}` and `y_k`. The triple zip walks over the current weights and both extrapolation points: `Zip::from(&mut *weights)...`.

This optimizer is more involved than basic gradient descent but still relies on the same functional building blocks: closures, element-wise iteration, and vector arithmetic with `ndarray`.

## Summary

This design demonstrates a few Rust programming techniques:
- Traits for abstraction and polymorphism
- Structs to encapsulate algorithm-specific state
- Use of the `ndarray` crate for numerical data
- Generic functions using closures for computing gradients
