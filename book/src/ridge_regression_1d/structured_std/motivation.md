# Structured: introduction

This section focuses on implementinng the 1D Ridge problem using functions, structures, traits and Rust standard library only. It's divided into 3 subsections:

1) [Closed-form solution](closed_form_solution.md): Implements the closed-form solution of the Ridge optimization problem using a `struct` to define a `RidgeEstimator` type. It shows how to implement a constructor together with `fit` and `predict` functions.
2) [Gradient descent](gradient_descent.md): Solves the Ridge problem using gradient descent using a `struct` as well to define a `RidgeGradientDescent` type.
3) [Trait Ridge model](traits.md): Explains how to define a trait `RidgeModel`, which describes the shared behavior of any Ridge estimator like our `RidgeEstimator` and `RidgeGradientDescent`.

Up to this stage, we implemented everything using the `f64` precision for all our variables. In the [next section](../generics_std/motivation.md), we will see how to make our code independent of the floating-point types by leveraging generics.