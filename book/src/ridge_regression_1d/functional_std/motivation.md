# Functional: introduction

This section focuses on implementinng the 1D Ridge problem using functions and Rust standard library only. It's divided into 5 subsections:

1) [Loss function](loss_function.md): Shows how to implement the Ridge loss function in two simple ways.
2) [Closed-form solution](closed_form_solution.md): Implements the closed-form solution of the Ridge optimization problem.
3) [Gradient-descent](gradient_descent.md): Solves the Ridge problem using gradient descent to illustrate how to perform for loops.
4) [Putting things together](putting_things_together.md): Explains how to assemble everything into a simple library.
5) [Exposing API](exposing_api.md): Explains how to use `lib.rs` to define what is made available to the user.

After this first section, we explore how to implement the same things using [structs and traits](../structured_std/motivation.md) to make our code more modular.