# Welcome!

Welcome to Rustineers, a dive into the Rust programming language through the lens of applied mathematics and science. There are already several high-quality resources available for learning Rust:

- [The Book](https://doc.rust-lang.org/book/) – a comprehensive introduction to Rust.
- [Rustlings](https://github.com/rust-lang/rustlings/) – hands-on exercises to reinforce learning.
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) – learn by studying runnable examples.
- [Rust Language Cheat Sheet](https://cheats.rs/) - really useful when you're trying to remember something without asking your favorite LLM. 

You can find even more learning material at [rust-lang.org](https://www.rust-lang.org/).

This book is meant to be complementary to those great resources. Our goal is to learn Rust by implementing practical examples drawn from applied mathematics, including:

- Machine Learning  
- Statistics and Probability  
- Optimization  
- Ordinary Differential Equations (ODEs)  
- Partial Differential Equations (PDEs)  
- And other topics from engineering and physics  

Each chapter centers around a specific scientific algorithm or computational problem. We explore how to implement it idiomatically in Rust and sometimes in multiple styles.

Hopefully, we manage to go through the [core concepts of Rust](https://youtu.be/06CVZKbNvgE?list=PLLWK4pUHYDYSnFTR7PPPAN1YtDgbP9h_z), namely:
- Ownership / borrowing
- Data types
- Traits
- Modules
- Error handling
- Macros

Most examples being with [Rust's standard library](https://doc.rust-lang.org/std/) which seems to be a solid foundation for learning both the language and its ecosystem.

## Difficulty Levels

To help you navigate the material, each chapter is marked with a difficulty level using 🦀 emojis:

- 🦀 — Beginner-friendly  
- 🦀🦀 — Intermediate  
- 🦀🦀🦀 — Advanced

As this is a work in progress, the difficulty levels might not always be well chosen.

## Roadmap

Here's an unordered list of examples of topics that could be added to the book:

- [x] 1D Ridge regression.
- [x] Simple first-order gradient descent algorithms.
- [x] Kernel methods: multivariate kernel Ridge regression.
- [x] Scientific computing: Solving the 2D Poisson problem with the finite element method.
- [ ] Classification algorithms: logistic regression.
- [ ] Some clustering algorithms: K-means, Gaussian mixtures.
- [ ] Some MCMC algorithms: MH, LMC, MALA.
- [ ] Numerical methods for solving ODEs: Euler, Runge-Kutta.
- [ ] Optimization algorithms: gradient-based, derivative-free.
- [ ] Divergences and distances for probability distributions: KL divergence, total variation, Wasserstein.

Let us know if you have other ideas or if you want to improve any existing chapter.