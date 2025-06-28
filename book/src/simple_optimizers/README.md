# Simple optimizers

This section explores how to implement a small module of optimization algorithms in Rust.

We begin by defining a common interface for optimizers and show how different strategies like gradient descent and momentum-based methods can be implemented using Rust's `trait` system.

Then, we explore an alternative design using enums, which can be helpful when working with simpler control flow or dynamic dispatch.

Finally, we demonstrate how to replace `Vec<f64>` with `ndarray` structures, which allows for more expressive and efficient numerical code, especially for larger-scale or matrix-based computations.

The goal is to gradually expose the design space for writing numerical algorithms idiomatically in Rust.