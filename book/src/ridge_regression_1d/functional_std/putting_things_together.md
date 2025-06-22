# Putting things together

To wrap up our 1D Ridge Regression example, let's see how all the parts fit together into a real Rust crate.

## Project layout

Here’s the directory structure for our `ridge_regression_1d` crate:

```text
crates/ridge_regression_1d/
├── Cargo.toml
└── src
    ├── grad_functions.rs      # Gradient of the loss
    ├── lib.rs                 # Main entry point for the library
    ├── loss_functions.rs      # Loss function implementations
    ├── optimizer.rs           # Gradient descent
    └── utils.rs               # Utility functions (e.g., dot product)
```

---

## What's `lib.rs`?

The `lib.rs` file is the entry point for the crate as a **library**. This is where we declare which modules (i.e., other `.rs` files) are exposed to the outside world.

```rust
pub mod grad_functions;
pub mod loss_functions;
pub mod optimizer;
pub mod utils;
```

Each line tells Rust:

> “There is a file called `X.rs` that defines a module `X`. Please include it in the crate.”

By default, items inside a module are private. That’s where `pub` comes in.


## Why `pub`?

If you want to use a function from another module or crate, you must declare it `pub` (public). For example:

```rust
// In utils.rs
pub fn dot(a: &[f64], b: &[f64]) -> f64 { ... }
```

If `dot` is not marked as `pub`, you can’t use it outside `utils.rs`, even from `optimizer.rs`.

## Importing between modules

Rust requires explicit imports between modules. For example, in `optimizer.rs`, we want to use the `dot` function from `utils.rs`:

```rust
use crate::utils::dot;
```

Here, `crate` refers to the root of this library crate—`lib.rs`.


## Example usage in `main.rs`

Now let's see how you could use the library from a binary crate:

```rust
{{#include ../../../../crates/ridge_regression_1d/src/main.rs}}
```

## Summary

This chapter demonstrated how to:

- Implement the 1D Ridge regression in sample ways by relying on Rust standard library only.
- Organize a crate into multiple source files (modules)
- Use `pub` to expose functions
- Import functions from other modules
- Call everything together from a `main.rs`

This is idiomatic Rust structure and prepares you to scale beyond toy examples while staying modular and readable.
