
# Gradient descent

Recall that the gradient descent algorithm is given by:

$$
x \leftarrow x - \epsilon \nabla L(x)
$$

where $\epsilon$ denotes the step size, and $L$ is the objective function to minimize. We first define the structure for the gradient descent algorithm. It only stores the learning rate as a `f64`.

```rust
{{#include ../../../../crates/simple_optimizers_traits/src/optimizers.rs:gd_struct}}
```

We then implement a constructor. In this case, it simply consists of choosing the learning rate.

```rust
{{#include ../../../../crates/simple_optimizers_traits/src/optimizers.rs:impl_optimizer_gd}}
```

Next, we implement the `step` method required by the `Optimizer` trait:

```rust
{{#include ../../../../crates/simple_optimizers_traits/src/optimizers.rs:impl_optimizer_gd_step}}
```

This function updates each entry of `weights` by looping over the elements and applying the gradient descent update. The weight `w` inside the loop must be dereferenced as it is passed as a mutable reference.

We use elementwise operations because `Vec` doesn't provide built-in arithmetic methods. External crates such as `ndarray` or `nalgebra` could help write this more expressively.
