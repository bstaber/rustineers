# Closed-form solution

The one-dimensional Ridge regression problem admits a simple closed-form solution.  
Given a dataset $(x_i, y_i)$ for $i = 1, \ldots, n$, and a regularization parameter $\lambda > 0$, the Ridge estimator is:

$$\hat{\beta}_\lambda = \frac{\sum_{i=1}^n x_i y_i}{\sum_{i=1}^n x_i^2 + n\lambda}$$

This form assumes that the data has no intercept term, i.e., the model is $y = \beta x$, or equivalently, that the data is centered around zero. In practice, it is common to subtract the means of both $x$ and $y$ before computing the estimator. This removes the intercept and gives:

$$\hat{\beta}_\lambda = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2 + n\lambda}$$

We now implement this solution in Rust, using only the standard library.

```rust
{{#include ../../../../crates/ridge_regression_1d/src/functional_std/analytical.rs}}
```

You could also express it as a single iterator chain, similar to how we implemented the loss function earlier.