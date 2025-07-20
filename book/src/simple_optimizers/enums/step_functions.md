# Implementing the step method

The `step` method is responsible for updating the model parameters (`weights`) according to the specific optimization strategy in use. It uses pattern matching to dispatch the correct behavior depending on whether the optimizer is a `GradientDescent` or a `Momentum` variant. 

```rust
{{#include ../../../../crates/simple_optimizers_enums/src/optimizers.rs:step}}
```

The `match` expression identifies which optimizer variant is being used. This pattern can be a clean alternative to trait-based designs when you want:
- A small number of well-known variants
- Built-in state encapsulation
- Exhaustive handling via pattern matching

It keeps related logic grouped under one type and can be extended easily with new optimizers.