# Gradient descent with momentum

Now let’s implement gradient descent with momentum. The structure stores the learning rate, the momentum factor, and an internal velocity buffer:

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

At this point, we've defined two optimizers using structs and a shared trait. To complete the module, we define a training loop that uses any optimizer implementing the trait.