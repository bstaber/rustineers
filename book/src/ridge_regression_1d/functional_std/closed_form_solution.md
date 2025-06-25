# Closed-form solution

The one-dimensional Ridge regression problem admits a simple closed-form solution.  
Given a dataset $(x_i, y_i)$ for $i = 1, \ldots, n$, and a regularization parameter $\lambda > 0$, the Ridge estimator is:

$$\hat{w}_\lambda = \frac{\sum_{i=1}^n x_i y_i}{\sum_{i=1}^n x_i^2 + \lambda}$$

We now implement this solution in Rust, using only the standard library:

```rust
{{#include ../../../../crates/ridge_regression_1d/src/functional_std/analytical.rs}}
```

This implementation avoids any external dependencies and highlights core Rust features like slices, iterators, and closures.

You can probably implement it as a single iterator chain like we did for the loss function.