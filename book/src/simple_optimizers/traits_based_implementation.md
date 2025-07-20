
# Optimizers using traits

This chapter illustrates how to use traits for implementing a module of optimizers. This approach is useful when you want polymorphism or when each optimizer requires its own state and logic.

It's similar to what you might do in other languages such as Python or C++, and it's a good fit for applications that involve multiple algorithm variants.

## Trait definition

We define a common trait `Optimizer`, which describes the shared behavior of any optimizer. Let's assume that our optimizers only need a `step` function.

```rust
{{#include ../../../crates/simple_optimizers_traits/src/optimizers.rs:optimizer_trait}}
```

Any type that implements this trait must provide a `step` method. Let's illustrate how to use this by implementing two optimizers: gradient descent with and without momentum.
