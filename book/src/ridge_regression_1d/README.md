# Ridge regression 1D

Here, we implement one-dimensional Ridge Regression in several styles, using increasing levels of abstraction. It's designed as a learning path for beginners, and focuses on writing idiomatic, clear, and type-safe Rust code. We focus on minimizing this loss function:

$$
\mathcal{L}(\beta) = \frac{1}{n} \sum_{i=1}^n (y_i - \beta x_i)^2 + \lambda \beta^2\,,
$$

where: $x_i \in \mathbb{R}$ is an input covariate, $y_i \in \mathbb{R}$ is the associated output, $\beta$ is the Ridge coefficient, $\lambda$ is the $L^2$ regularization strength.

## How this chapter is organized

This chapter introduces several useful concepts for Rust beginners. It is divided into four sections, each solving the same problem (1D Ridge regression) using different tools and with slightly increasing complexity.

* The [first section](functional_std/motivation.md) shows how to use basic functions and the Rust standard library to build a simple library. In particular, it shows how to manipulate vectors (`Vec<f64>`) and slices (`&[f64]`).

* The [next section](structured_std/motivation.md) explains how to solve the same problem using structs and traits to make the code more modular and extensible.

* The [third section](generics_std/motivation.md) introduces generics, allowing the code to work with different floating-point types (`f32` and `f64`). 

* Finally, the [last section](structured_ndarray/motivation.md) goes further by using `ndarray` for linear algebra and incorporating additional Rust features such as optional values, pattern matching, and error handling.

If you want to implement and run this example while you read but are not familiar with Cargo yet, have a look at [Cargo 101](../CARGO_TUTORIAL.md) for how to set up your project.