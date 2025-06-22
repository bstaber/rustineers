# Gradient of the Ridge loss

To perform optimization—such as gradient descent—we need the derivative of the loss function with respect to the model parameters. In the case of Ridge regression, the gradient has a closed-form expression and can be efficiently computed.

In this section, we implement the gradient in two different ways to highlight trade-offs between clarity and performance.

## Naive implementation

This version breaks the computation into two separate steps:  
* Compute the residuals $r_i := y_i - \beta x_i$
* Compute the dot product between the residuals and the inputs: $\sum_{i=1}^{n} r_i x_i$
* Then assemble the get the gradient value

This is easier to follow and useful for educational purposes.

```rust
{{#include ../../../crates/ridge_regression_1d/src/grad_functions.rs:grad_loss_function_naive}}
```

## Inlined iterator-based implementation
In this version, we fuse the residual and gradient computation into a single iterator chain. This avoids intermediate memory allocations and takes full advantage of Rust’s zero-cost abstraction model.

```rust
{{#include ../../../crates/ridge_regression_1d/src/grad_functions.rs:grad_loss_function_inline}}
```

Key differences:
* The naive version allocates a temporary vector for the residuals and is closer to how you'd write this in a high-level math language like NumPy or MATLAB.
* The inlined version is more idiomatic Rust: it avoids allocation and achieves better performance through iterator fusion.