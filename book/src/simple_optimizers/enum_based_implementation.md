# Optimizers as enums with internal state and methods

This chapter builds on the previous enum-based optimizer design. We now give each variant its own internal state and encapsulate behavior using methods. This pattern is useful when you want enum-based control flow with encapsulated logic.

## Defining the optimizer enum

Each optimizer variant includes its own parameters and, when needed, its internal state.

```rust
{{#include ../../../crates/simple_optimizers/src/enum_based/optimizers.rs:enum_definition}}
```

Here, `GradientDescent` stores only the learning rate, while `Momentum` additionally stores its velocity vector.


## Constructors

We define convenience constructors for each optimizer. These make usage simpler and avoid manually writing match arms.

```rust
{{#include ../../../crates/simple_optimizers/src/enum_based/optimizers.rs:constructors}}
```

This helps create optimizers in a more idiomatic and clean way.

## Implementing the step method

The `step` method applies one optimization update depending on the variant. The method uses pattern matching to extract variant-specific behavior.

```rust
{{#include ../../../crates/simple_optimizers/src/enum_based/optimizers.rs:step}}
```

Here, `GradientDescent` simply applies the learning rate times the gradient. The `Momentum` variant updates and stores the velocity vector before updating the weights.

## Summary

This pattern can be a clean alternative to trait-based designs when you want:
- A small number of well-known variants
- Built-in state encapsulation
- Exhaustive handling via pattern matching

It keeps related logic grouped under one type and can be extended easily with new optimizers.