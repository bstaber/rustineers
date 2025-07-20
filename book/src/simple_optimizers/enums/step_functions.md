# Implementing the step method

The `step` method applies one optimization update depending on the variant. The method uses pattern matching to extract variant-specific behavior.

```rust
{{#include ../../../../crates/simple_optimizers_enums/src/optimizers.rs:step}}
```

Here, `GradientDescent` simply applies the learning rate times the gradient. The `Momentum` variant updates and stores the velocity vector before updating the weights.