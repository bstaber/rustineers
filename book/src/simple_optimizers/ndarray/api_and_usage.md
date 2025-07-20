# API and usage

Here’s how you can use each of the three optimizers `GD`, `Momentum`, and `NAG` to minimize a simple quadratic function.

We'll try to minimize the function:

$$
f(w) = \frac{1}{2} \|w - 3\|^2
$$

Its gradient is:

$$
\nabla f(w) = w - 3
$$

We expect convergence toward the vector `[3.0, 3.0, 3.0]`.

## Using gradient descent

```rust
use optimizers::GD;
use ndarray::{array, Array1};

fn main() {
    let mut weights = array![0.0, 0.0, 0.0];
    let grad_fn = |w: &Array1<f64>| w - 3.0;

    let gd = GD::new(0.1);
    gd.run(&mut weights, grad_fn, 100);

    println!("GD result: {:?}", weights);
}
```

## Using momentum

```rust
use optimizers::Momentum;
use ndarray::{array, Array1};

fn main() {
    let mut weights = array![0.0, 0.0, 0.0];
    let grad_fn = |w: &Array1<f64>| w - 3.0;

    let momentum = Momentum::new(0.1, 0.9);
    momentum.run(&mut weights, grad_fn, 100);

    println!("Momentum result: {:?}", weights);
}
```

## Using Nesterov’s Accelerated Gradient (NAG)

```rust
use optimizers::NAG;
use ndarray::{array, Array1};

fn main() {
    let mut weights = array![0.0, 0.0, 0.0];
    let grad_fn = |w: &Array1<f64>| w - 3.0;

    let nag = NAG::new(0.1);
    nag.run(&mut weights, grad_fn, 100);

    println!("NAG result: {:?}", weights);
}
```

## Summary

This design demonstrates a few Rust programming techniques:
- Traits for abstraction and polymorphism
- Structs to encapsulate algorithm-specific state
- Use of the `ndarray` crate for numerical data
- Generic functions using closures for computing gradients
