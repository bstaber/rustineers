# Gradient descent with momentum

Recall that the gradient descent algorithm with momentum is given by:

$$
\begin{align*}
& v \leftarrow \mu v + \epsilon \nabla L(x) \\
& x \leftarrow x - v
\end{align*}
$$

where $v$, $\mu$, and $\epsilon$ denote the velocity, momentum and step size, respectively. The structure we define stores the learning rate, the momentum factor, and an internal velocity buffer:

```rust
{{#include ../../../../crates/simple_optimizers_traits/src/optimizers.rs:momentum_struct}}
```

We define the constructor by taking the required parameters, and we initialize the velocity to a zero vector:

```rust
{{#include ../../../../crates/simple_optimizers_traits/src/optimizers.rs:impl_optimizer_momentum}}
```

The `step` function is slightly more complex, as it performs elementwise operations over the weights, velocity, and gradients:

```rust
{{#include ../../../../crates/simple_optimizers_traits/src/optimizers.rs:impl_optimizer_momentum_step}}
```

The internal state of the velocity is updated as well, which is possible because we pass a mutable reference `&self`. At this point, we've defined two optimizers using structs and a shared trait. To complete the module, we define a training loop that uses any optimizer implementing the trait.