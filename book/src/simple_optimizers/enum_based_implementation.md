# Optimizers as enums with internal state and methods

This chapter builds on the previous enum-based optimizer design. We now give each variant its own internal state and encapsulate behavior using methods. This pattern is useful when you want enum-based control flow with encapsulated logic.

## Defining the optimizer enum

Each optimizer variant includes its own parameters and, when needed, its internal state.

```rust
{{#include ../../../crates/simple_optimizers_enums/src/optimizers.rs:enum_definition}}
```

Here, `GradientDescent` stores only the learning rate, while `Momentum` additionally stores its velocity vector.