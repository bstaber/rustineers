# Putting things together

To wrap up our 1D Ridge Regression example, let's see how all the parts fit together into a real Rust crate.

## Project layout

Here’s the directory structure for our `ridge_1d_fn` crate:

```text
crates/ridge_1d_fn/
├── Cargo.toml
└── src
    ├── estimator.rs           # Closed-form solution of the Ridge estimator
    ├── gradient_descent.rs    # Gradient descent solution
    ├── lib.rs                 # Main entry point for the library
    └── loss_functions.rs      # Loss function implementations
```

All the functions discussed in the previous sections are implemented in `estimator.rs`, `loss_functions.rs`, `gradient_descent.rs`. You can inspect each of these files below.

<details>
<summary>Click to view <b>estimator.rs</b></summary>

```rust
{{#include ../../../../crates/ridge_1d_fn/src/estimator.rs}}
```
</details>

<details>
<summary>Click to view <b>gradient_descent.rs</b></summary>

```rust
{{#include ../../../../crates/ridge_1d_fn/src/gradient_descent.rs}}
```
</details>

<details>
<summary>Click to view <b>loss_functions.rs</b></summary>

```rust
{{#include ../../../../crates/ridge_1d_fn/src/loss_functions.rs}}
```
</details>

<details>
<summary>Click to view <b>lib.rs</b></summary>

```rust
{{#include ../../../../crates/ridge_1d_fn/src/lib.rs}}
```
</details>

Note that the layout can be more complicated by introducing modules and submodules. This will be covered in the next chapter when we implement a structured-oriented version of the 1D Ridge regression.

## What's `lib.rs`?

The `lib.rs` file is the entry point for the crate as a **library**. This is where we declare which modules (i.e., other `.rs` files) are exposed to the outside world.

```rust
pub mod estimator;
pub mod gradient_descent;
pub mod loss_functions;

pub use estimator::ridge_estimator;
```

Each line tells Rust:

> “There is a file called `X.rs` that defines a module `X`. Please include it in the crate.”

By default, items inside a module are private. That’s where `pub` comes in.

We will dive deeper into `lib.rs` in the [2.1.5 Exposing API](exposing_api.md) chapter.


## Why `pub`?

If you want to use a function from another module or crate, you must declare it `pub` (public). For example:

```rust
// In utils.rs
pub fn dot(a: &[f64], b: &[f64]) -> f64 { ... }
```

If `dot` is not marked as `pub`, you can’t use it outside `utils.rs`, even from `optimizer.rs`.

## Importing between modules

Rust requires explicit imports between modules. For example, let's say we want to use the `dot` function from `gradient_descent.rs`. We can import it as follows:

```rust
use crate::utils::dot;
```

Here, `crate` refers to the root of this library crate `lib.rs`.


## Example of usage

Now let's see how you could use the library from a binary crate:

```rust
use ridge_1d_fn::ridge_estimator;

let x: Vec<f64> = vec![1.0, 2.0];
let y: Vec<f64> = vec![0.1, 0.2];
let lambda2 = 0.001;

let beta = ridge_estimator(&x, &y, lambda2);
```
