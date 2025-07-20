# Functional: introduction

This section focuses on implementinng the 1D Ridge problem using functions and Rust standard library only. It's divided into 5 subsections:

1) [Loss function](loss_function.md): Shows how to implement the Ridge loss function in two simple ways.
2) [Closed-form solution](closed_form_solution.md): Implements the closed-form solution of the Ridge optimization problem.
3) [Gradient-descent](gradient_descent.md): Solves the Ridge problem using gradient descent to illustrate how to perform for loops.
4) [Putting things together](putting_things_together.md): Explains how to assemble everything into a simple library.
5) [Exposing API](exposing_api.md): Explains how to use `lib.rs` to define what is made available to the user.

# What we're building here

The aim of this chapter is to build a small crate with the following layout:

```text
crates/ridge_1d_fn/
├── Cargo.toml
└── src
    ├── estimator.rs           # Closed-form solution of the Ridge estimator
    ├── gradient_descent.rs    # Gradient descent solution
    ├── lib.rs                 # Main entry point for the library
    └── loss_functions.rs      # Loss function implementations
```

It is made of three modules: `estimator.rs`, `gradient_descent.rs`, and `loss_fnctions.rs`. At the end of the chapter, we end up with a crate that can be used as follows:

```rust
use ridge_1d_fn::{fit, predict};

fn main() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];

    let beta = fit(&x, &y, 0.1, 0.01, 1000, 0.0);
    let preds = predict(&x, beta);

    println!("Learned beta: {}", beta);
    println!("Predictions: {:?}", preds);
}
```

The fit and predict functions are implemented in the library entry point `lib.rs`.

Note that the `gradient_descent.rs`, and `loss_fnctions.rs` modules mostly serve as additional illustrations.

# What's next

After this first chapter, we explore how to implement the same things using [structs and traits](../structured_std/motivation.md) to make our code more modular.