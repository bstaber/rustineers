# Ridge loss

In this example, we implement one-dimensional Ridge Regression loss using only the **Rust standard library**, without any external crates. This lets us focus on core Rust features such as slices, iterators, and type safety.

Although the loss function by itself isn't really useful to solve the Ridge problem, implementing it provides a simple and focused introduction to Rust.

## Naive implementation

We now present a straightforward implementation of the **Ridge regression loss function**:

```rust
{{#include ../../../../crates/ridge_1d_fn/src/loss_functions.rs:loss_function_naive}}
```

In this example, we use two helper functions that we implement ourselves. A helper function for multiplying a vector by a scalar:

```rust
{{#include ../../../../crates/ridge_1d_fn/src/loss_functions.rs:mul_scalar_vec}}
```

We also defined a helper that subtracts two slices element-wise:

```rust
{{#include ../../../../crates/ridge_1d_fn/src/loss_functions.rs:subtract_vectors}}
```

Rather than using explicit loops, this implementation uses Rust’s iterator combinators, which the compiler optimizes into efficient code. This zero-cost abstraction keeps the code both readable and fast.

### Ownership and borrowing

In Rust, every value has a single owner. When you assign a value to a new variable or pass it to a function by value, ownership is transferred (moved).

Borrowing allows you to use a value without taking ownership of it. Borrowing is done using references:

- `&T` is a shared (read-only) reference.
- `&mut T` is a mutable reference.

These references allow access to data without moving it.

A function like this:

```rust
fn mul_scalar_vec(scalar: f64, vector: &[f64]) -> Vec<f64> {
    vector.iter().map(|x| x * scalar).collect()
}
```

does not take ownership of the input `vector`. Instead, it borrows it for the duration of the function call. This makes it easier to reuse the input vector later.

If we instead defined:

```rust
fn mul_scalar_vec(scalar: f64, vector: Vec<f64>) -> Vec<f64> { ... }
```

then passing a vector would move ownership:

```rust
let v = vec![1.0, 2.0, 3.0];
let result = mul_scalar_vec(2.0, v); // v is moved here
let v2 = v; // error: value borrowed after move
```

### Why use `&[f64]` instead of `Vec<f64>`? 

The type `&[f64]` seems to be commonly used in function signatures because it works with both arrays and vectors.

Finally, note that:

- `Vec<f64>` is an owned, growable vector on the heap. The only time we return a `Vec<f64>` is when we allocate a new output vector, like in `mul_scalar_vec`.
- `&Vec<f64>` is a shared reference to a `Vec<f64>`.
- `&[f64]` is a slice, i.e., a borrowed view into an array or vector.

In this chapter, we will mostly use these types but things can easily get more tricky.

## Inlined iterator-based implementation

Let's implement the loss function in a more compact way. Instead of breaking the computation into multiple intermediate steps like computing `y_hat`, `residuals`, and then squaring each residual, here we inline all computations into a single expression using iterators and closures.

This is ideal for demonstrating the expressive power of Rust's iterator API, especially once you're comfortable with basic slice handling and `.map()` chaining.

```rust
{{#include ../../../../crates/ridge_1d_fn/src/loss_functions.rs:loss_function_line}}
```

This implementation computes the mean squared error in a single iteration, minimizing allocations and abstraction overhead. In particular:
* We use `.iter().zip()` to iterate over two slices.
* We define a full code block inside the `.map()` closure, which makes it easier to write intermediate expressions like `let residual = yi - beta * xi;` before returning the squared value.
