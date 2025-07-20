# Simple optimizers

This chapter explores how to implement a small module of optimization algorithms in Rust. It is divided into three sections:

- In the [first section](traits_based_implementation.md), we begin by defining a common interface for optimizers and show how different strategies like gradient descent and momentum-based methods can be implemented using Rust's `trait` system.
- In the [second section](enum_based_implementation.md), we explore an alternative design using enums, which can be helpful when working with simpler control flow or dynamic dispatch.
- In the [last section](ndarray_based_implementation.md), we demonstrate how to replace `Vec<f64>` with `ndarray` structures, which allows for more expressive and efficient numerical code, especially for larger-scale or matrix-based computations.

The goal is to gradually expose the design space for writing numerical algorithms idiomatically in Rust.

In each section, we implement a small crate with the following layout:

```text
├── Cargo.toml
└── src
    ├── optimizers.rs
    └── lib.rs
```

The module `optimizers.rs` implements classical gradient descent with and without momentum, and eventually a Nesterov accelerated gradient descent.