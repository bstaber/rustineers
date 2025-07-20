# Accelerated gradient descent

The `AGD` struct extends gradient descent with a classical momentum term:

```rust
{{#include ../../../../crates/simple_optimizers_ndarray/src/optimizers.rs:struct_agd}}
```

It has a constructor:

```rust
{{#include ../../../../crates/simple_optimizers_ndarray/src/optimizers.rs:impl_agd_new}}
```

This algorithm adds momentum to classical gradient descent. Instead of updating weights using just the current gradient, it maintains a velocity vector that accumulates the influence of past gradients. This helps smooth the trajectory and accelerates convergence on convex problems.

```rust
{{#include ../../../../crates/simple_optimizers_ndarray/src/optimizers.rs:impl_agd_run}}
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