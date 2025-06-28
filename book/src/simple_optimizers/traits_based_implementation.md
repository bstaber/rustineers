# Optimizers using traits

This chapter shows how to implement optimizers in Rust using traits. This approach is useful when you want polymorphism, possibly at runtime, or if each optimizer requires its own state and logic.

## Trait definition

```rust
{{#include ../../../crates/simple_optimizers/src/traits_based/optimizers.rs:optimizer_trait}}
```

## Implementing gradient descent

```rust
{{#include ../../../crates/simple_optimizers/src/traits_based/optimizers.rs:gd_struct}}
```

```rust
{{#include ../../../crates/simple_optimizers/src/traits_based/optimizers.rs:impl_optimizer_gd}}
```

```rust
{{#include ../../../crates/simple_optimizers/src/traits_based/optimizers.rs:impl_optimizer_gd_update}}
```

## Implementing gradient descent with momentum

```rust
{{#include ../../../crates/simple_optimizers/src/traits_based/optimizers.rs:momentum_struct}}
```

```rust
{{#include ../../../crates/simple_optimizers/src/traits_based/optimizers.rs:impl_optimizer_momentum}}
```

## Exposed public helper

```rust
{{#include ../../../crates/simple_optimizers/src/lib.rs}}
```