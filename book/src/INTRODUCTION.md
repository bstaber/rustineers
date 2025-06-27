# Welcome!

Welcome to Rustineers, a dive into the Rust programming language through the lens of applied mathematics and science. There are already several high-quality resources available for learning Rust:

- [The Book](https://doc.rust-lang.org/book/) – a comprehensive introduction to Rust.
- [Rustlings](https://github.com/rust-lang/rustlings/) – hands-on exercises to reinforce learning.
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) – learn by studying runnable examples.

You can find even more learning material at [rust-lang.org](https://www.rust-lang.org/).

This book is meant to be complementary to those great resources. Our goal is to learn Rust by implementing practical examples drawn from applied mathematics, including:

- Machine Learning  
- Statistics and Probability  
- Optimization  
- Ordinary Differential Equations (ODEs)  
- Partial Differential Equations (PDEs)  
- And other topics from engineering and physics  

Each chapter centers around a specific scientific algorithm or computational problem. We explore how to implement it idiomatically in Rust—sometimes in multiple styles.

## Difficulty Levels

To help you navigate the material, each chapter is marked with a difficulty level using 🦀 emojis:

- 🦀 — Beginner-friendly  
- 🦀🦀 — Intermediate  
- 🦀🦀🦀 — Advanced

As this is a work in progress, the difficulty levels might not always be well chosen.

## Roadmap

Here's an unordered list of examples of applications that could be added to the book:

- **Statistics & Probability**
  - Mean, variance, and standard deviation
  - Random sampling with `rand` and `rand_distr`
  - Basic probability distributions (Gaussian, Bernoulli)
  - Monte Carlo integration

- **Regression & Optimization**
  - Ridge Regression with `ndarray`
  - Multivariate Ridge Regression
  - Lasso regression (proximal gradient)
  - Gradient descent in 1D
  - Multivariate optimization
  - Constrained optimization (e.g., Lagrange multipliers)
  - Projected gradient methods

- **Machine Learning**
  - Logistic regression
  - K-means clustering
  - Simple neural networks (from scratch)

- **Bayesian Inference & MCMC**
  - Random number engines and reproducibility
  - Langevin Monte Carlo (LMC)
  - Metropolis-adjusted Langevin (MALA)
  - Hamiltonian Monte Carlo

- **Differential Equations**
  - Euler’s method
  - Runge-Kutta 4
  - Systems of ODEs
  - 1D Heat Equation (finite difference)
  - 2D Poisson Equation
  - Sparse matrices and solvers

Let us know if you have other ideas.