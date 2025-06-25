# Gradient descent

As an exercise, we now implement the gradient descent algorithm to optimize the Ridge regression loss. 

In the case of Ridge regression, the gradient has a closed-form expression and can be efficiently computed. We implement the gradient in two different ways as we did for the loss function.

## Gradient descent implementation

Gradient descent iteratively updates the parameter `β` using the gradient of the loss function:

$$
\beta \leftarrow \beta - \eta \cdot \nabla_\beta \mathcal{L}(\beta)
$$

Where `η` is the learning rate, and `∇βL(β)` is the gradient of the loss.

We allow flexible experimentation by passing the gradient function as parameters:

```rust
{{#include ../../../../crates/ridge_regression_1d/src/optimizer.rs:gradient_descent}}
```

This version is generic, letting us plug in any valid `grad_fn`.

## Gradient function: naive implementation

This version breaks the computation into two separate steps:  
* Compute the residuals $r_i := y_i - \beta x_i$
* Compute the dot product between the residuals and the inputs: $\sum_{i=1}^{n} r_i x_i$
* Then assemble the get the gradient value

We first start by implementing our own `dot` function by relying on iterators, map chaining, and summing the results.

```rust
{{#include ../../../../crates/ridge_regression_1d/src/utils.rs:dot}}
```

Our first implementation takes the following form:


```rust
{{#include ../../../../crates/ridge_regression_1d/src/functional_std/grad_functions.rs:grad_loss_function_naive}}
```

## Gradient function: inlined iterator-based implementation
In this version, we fuse the residual and gradient computation into a single iterator chain. This avoids intermediate memory allocations and takes full advantage of Rust’s zero-cost abstraction model.

```rust
{{#include ../../../../crates/ridge_regression_1d/src/functional_std/grad_functions.rs:grad_loss_function_inline}}
```

Key differences:
* The naive version allocates a temporary vectosr for the residuals and the dot product.
* The inlined version is more idiomatic Rust: it avoids allocation and achieves better performance through iterator fusion.