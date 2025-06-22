# Ridge regression 1D

Here, we implement one-dimensional Ridge Regression*in several styles, using increasing levels of abstraction. It's designed as a learning path for beginners, and focuses on writing idiomatic, clear, and type-safe Rust code. We focus on minimizing this loss function:

$$
\mathcal{L}(\beta) = \frac{1}{2n} \sum_{i=1}^n (y_i - \beta x_i)^2 + \lambda \beta^2\,,
$$

where: $x_i \in \mathbb{R}$ is an input covariate, $y_i \in \mathbb{R}$ is the associated output, $\beta$ is the Ridge coefficient, $\lambda$ is the $L^2$ regularization strength.

## What you'll learn

This module is perfect if you're just starting with Rust and want to:

- Write beginner-friendly numerical code
- Understand how to manipulate vectors (`Vec<f64>`) and slices (`&[f64]`)
- Write clean and safe code using the standard library only
- Structure your functions with clear responsibilities
- Learn good Rust patterns: immutability, iterators, ownership
- How to make a simple crate out of it 
- How to expose a public API
