# Gradient Descent

We now implement the **gradient descent algorithm** to optimize the Ridge regression loss.

Gradient descent iteratively updates the parameter `β` using the gradient of the loss function:

$$
\beta \leftarrow \beta - \eta \cdot \nabla_\beta \mathcal{L}(\beta)
$$

Where `η` is the learning rate, and `∇βL(β)` is the gradient of the loss.

## Full Implementation

We allow flexible experimentation by passing the gradient function as parameters:

```rust
{{#include ../../../crates/ridge_regression_1d/src/optimizer.rs:gradient_descent}}
```

This version is generic, letting us plug in any valid `grad_fn`.