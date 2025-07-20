# Nesterov accelerated GD

The `AdaptiveAGD` struct implements a more advanced optimizer based on Nesterov's method and FISTA:

```rust
{{#include ../../../../crates/simple_optimizers_ndarray/src/optimizers.rs:AdaptiveAGD_struct}}
```

It has a constructor:

```rust
{{#include ../../../../crates/simple_optimizers_ndarray/src/optimizers.rs:AdaptiveAGD_impl_new}}
```

This algorithm implements an accelerated method inspired by Nesterov’s momentum and the FISTA algorithm. The key idea is to introduce an extrapolation step between iterates, controlled by a sequence `t_k`. This helps the optimizer "look ahead" and converge faster in smooth convex problems.

Update steps:
- Compute a temporary point `y_{k+1}` by taking a gradient step from `x_k`.
- Update the extrapolation coefficient `t_{k+1}`.
- Combine `y_{k+1}` and `y_k` using a weighted average to get the new iterate `x_{k+1}`.

```rust
{{#include ../../../../crates/simple_optimizers_ndarray/src/optimizers.rs:AdaptiveAGD_impl_run}}
```

Some notes:

- We deliberately re-allocate multiple variables within the for loop (`grad`, `y_next`, `t_next`) but we could have pre-allocated buffers before the for loop.

- The algorithm keeps track of two sequences: the main iterate (`weights`) and the extrapolated one (`y_k`). Before starting the for loop, we initialize `y_k` by cloning the weights: `let mut y_k = weights.clone();`.

- The gradient is evaluated at the current weights, as in standard gradient descent: `let grad = grad_fn(weights);`. Since `weights` is a mutable reference, we can pass it straightaway to our `grad_fn`.
  
- A temporary variable to store the new extrapolated point. This is again a full allocation and clone for clarity. `let mut y_next = weights.clone();`.

- We next compute: `y_{k+1} = x_k - α ∇f(x_k)` using an element-wise operation: `Zip::from(&mut y_next).and(&grad).for_each(|y, &g| { *y -= self.step_size * g; });`. This time, we rely on the `Zip::from` trait implement by `ndarray`.
  
- The new weights are obtained by combining `y_{k+1}` and `y_k`. The triple zip walks over the current weights and both extrapolation points: `Zip::from(&mut *weights)...`.

This optimizer is more involved than basic gradient descent but still relies on the same functional building blocks: closures, element-wise iteration, and vector arithmetic with `ndarray`.