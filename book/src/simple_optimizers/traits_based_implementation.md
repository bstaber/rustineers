
# Optimizers using traits

This chapter illustrates how to use traits for implementing a module of optimizers. This approach is useful when you want polymorphism or when each optimizer requires its own state and logic.

It's similar to what you might do in other languages such as Python or C++, and it's a good fit for applications that involve multiple algorithm variants.

## Trait definition

We define a common trait `Optimizer`, which describes the shared behavior of any optimizer. Let's assume that our optimizers only need a `step` function.

```rust
{{#include ../../../crates/simple_optimizers/src/traits_based/optimizers.rs:optimizer_trait}}
```

Any type that implements this trait must provide a `step` method. Let's illustrate how to use this by implementing two optimizers: gradient descent with and without momentum.

## Gradient descent

We first define the structure for the gradient descent algorithm. It only stores the learning rate as a `f64`.

```rust
{{#include ../../../crates/simple_optimizers/src/traits_based/optimizers.rs:gd_struct}}
```

We then implement a constructor. In this case, it simply consists of choosing the learning rate.

```rust
{{#include ../../../crates/simple_optimizers/src/traits_based/optimizers.rs:impl_optimizer_gd}}
```

Next, we implement the `step` method required by the `Optimizer` trait:

```rust
{{#include ../../../crates/simple_optimizers/src/traits_based/optimizers.rs:impl_optimizer_gd_step}}
```

This function updates each entry of `weights` by looping over the elements and applying the gradient descent update. We use elementwise operations because `Vec` doesn't provide built-in arithmetic methods. External crates such as `ndarray` or `nalgebra` could help write this more expressively.

## Gradient descent with momentum

Now let’s implement gradient descent with momentum. The structure stores the learning rate, the momentum factor, and an internal velocity buffer:

```rust
{{#include ../../../crates/simple_optimizers/src/traits_based/optimizers.rs:momentum_struct}}
```

We define the constructor by taking the required parameters, and we initialize the velocity to a zero vector:

```rust
{{#include ../../../crates/simple_optimizers/src/traits_based/optimizers.rs:impl_optimizer_momentum}}
```

The `step` function is slightly more complex, as it performs elementwise operations over the weights, velocity, and gradients:

```rust
{{#include ../../../crates/simple_optimizers/src/traits_based/optimizers.rs:impl_optimizer_momentum_step}}
```

At this point, we've defined two optimizers using structs and a shared trait. To complete the module, we define a training loop that uses any optimizer implementing the trait.

## Public API

We expose the training loop in `lib.rs` as the public API. The function `run_optimization` takes a generic `Optimizer`, a gradient function, an initial weight vector, and a maximum number of iterations.

```rust
{{#include ../../../crates/simple_optimizers/src/lib.rs}}
```

### Example of usage

Here’s a simple example where we minimize the function $\mathbf{x} \mapsto \|\mathbf{x}\|_2^2$ in $\mathbb{R}^n$:

```rust
use traits_based::optimizers::Momentum;

fn grad_fn(w: &[f64]) -> Vec<f64> {
    w.iter().map(|wi| 2.0 * wi.powi(2)).collect()
}

let n: usize = 10;
let mut weights = vec![1.0; n];
let mut optimizer = Momentum::new(0.01, 0.9, n);

run_optimization(&mut optimizer, &mut weights, grad_fn, 100);

// Final weights after optimization
println!("{:?}", weights); 
```

Some notes:
- Both `weights` and `optimizer` must be mutable, because we perform in-place updates.
- We pass mutable references into `run_optimization`, matching its function signature.
- The example uses a closure-based gradient function, which you can easily replace.
