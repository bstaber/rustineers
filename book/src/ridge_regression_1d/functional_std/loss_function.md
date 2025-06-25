# Ridge loss

In this example, we implement one-dimensional Ridge Regression loss using only the **Rust standard library**, without any external crates. This lets us focus on core Rust features such as slices, iterators, and type safety.

Although the loss function by itself isn't really useful to solve the Ridge problem, implementing it provides a simple and focused introduction to Rust.

## Naive implementation

We now present a straightforward implementation of the **Ridge regression loss function**:

```rust
{{#include ../../../../crates/ridge_regression_1d/src/functional_std/loss_functions.rs:loss_function_naive}}
```

In this example, we use two helper functions that we implement ourselves. A helper function for multiplying a vector by a scalar:

```rust
{{#include ../../../../crates/ridge_regression_1d/src/utils.rs:mul_scalar_vec}}
```

We also defined a helper that subtracts two slices element-wise:

```rust
{{#include ../../../../crates/ridge_regression_1d/src/utils.rs:subtract_vectors}}
```

Rather than using explicit loops, this implementation uses Rust’s iterator combinators, which the compiler optimizes into efficient code. This zero-cost abstraction keeps the code both readable and fast.

### Why use `&[f64]` instead of `Vec<f64>`? 

Rust distinguishes between:

- `&Vec<f64>` – an owned, growable vector on the heap
- `&[f64]` – a **slice**, i.e., a borrowed view into an array or vector

The type `&[f64]` seems to be commonly used in function signatures for several reasons:

- It's **more general**: it works with both arrays and vectors.
- It **avoids ownership transfer**: keeps our functions composable and efficient.
- It's **idiomatic Rust** and satisfies linters like Clippy.

The only time we return a `Vec<f64>` is when we allocate a **new output vector**, like in `mul_scalar_vec`.

## Inlined iterator-based implementation

In this version, we present a **more compact implementation** of the Ridge regression loss function.

Unlike the previous example where we broke the computation into multiple intermediate steps—like computing `y_hat`, `residuals`, and then squaring each residual—here we **inline all computations** into a single expression using iterators and closures.

This is ideal for demonstrating the expressive power of Rust's iterator API, especially once you're comfortable with basic slice handling and `.map()` chaining.

```rust
{{#include ../../../../crates/ridge_regression_1d/src/functional_std/loss_functions.rs:loss_function_line}}
```

This implementation computes the mean squared error in a single iteration, minimizing allocations and abstraction overhead. In particular:
* We use `.iter().zip()` to iterate over two slices.
* We define a full code block inside the `.map()` closure, which makes it easier to write intermediate expressions like `let residual = yi - beta * xi;` before returning the squared value.
