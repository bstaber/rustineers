# Exposing a clean API

Until now, we've manually chained together the loss, gradient, and optimization steps. This is great for learning, but in real projects, we often want a simplified and reusable API.

Rust gives us a clean way to do this by leveraging the `lib.rs` file as the public interface to our crate.

## `lib.rs` as a public API

In your crate, `lib.rs` is responsible for organizing and exposing the components we want users to interact with.

We can re-export key functions and define top-level utilities like `fit` and `predict`. The complete `lib.rs` file now looks like this:

```rust
{{#include ../../../../crates/ridge_regression_1d/src/lib.rs:exports_and_fit}}
```

Everything declared `pub` is available to the user.

## Example of usage

You can update your binary entry point to try out the public API.

```rust
use ridge_regression_1d::{fit, predict};

fn main() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];

    let beta = fit(&x, &y, 0.1, 0.01, 1000, 0.0);
    let preds = predict(&x, beta);

    println!("Learned beta: {}", beta);
    println!("Predictions: {:?}", preds);
}
```