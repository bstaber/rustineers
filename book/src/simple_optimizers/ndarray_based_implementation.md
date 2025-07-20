# Optimizers using ndarray

This section introduces a modular and idiomatic way to implement optimization algorithms in Rust using `ndarray` and traits. It is intended for readers who are already comfortable with basic Rust syntax and want to learn how to build reusable, extensible components in numerical computing.

In this example, we need to import the following types and traits:
```rust
use ndarray::Array;
use ndarray::Array1;
use ndarray::Zip;
```

You can inspect the full module that we're about the break down over here:
<details>
<summary>Click to view <b>optimizers.rs</b></summary>

```rust
{{#include ../../../crates/simple_optimizers_ndarray/src/optimizers.rs}}
```
</details>

## Trait-based design

We define a trait called `Optimizer` to represent any optimizer that can update model weights based on gradients. In contrast to the previous sections where we mostly implemented `step`functions, here the trait requires implementors to define a `run` method with the following signature:

```rust
{{#include ../../../crates/simple_optimizers_ndarray/src/optimizers.rs:trait}}
```

This method takes:
- A mutable reference to a vector of weights (`Array1<f64>`).
- A function that computes the gradient of the loss with respect to the weights. This `grad_fn` function takes itself a borrowed reference to the weights `&Array1<f64>` and outputs a new array `Array1<f64>`.
- The number of iterations to perform.

This trait `run` defines the whole optimization algorithm.