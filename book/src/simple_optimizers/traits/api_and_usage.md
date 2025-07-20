# API and usage

We expose the training loop in `lib.rs` as the public API. The function `run_optimization` takes a generic `Optimizer`, a gradient function, an initial weight vector, and a maximum number of iterations.

```rust
{{#include ../../../../crates/simple_optimizers_traits/src/lib.rs:entry_point}}
```

Here, the trait bound `<Opt: Optimizer>` tells the compiler that type of the given `optimizer` must implement the trait `Optimizer`. This ensures that `optimizer` has the required step function.

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

Some final reminders:
- Both `weights` and `optimizer` must be mutable, because we perform in-place updates.
- We pass mutable references into `run_optimization`, matching its function signature.
- The example uses a closure-based gradient function, which you can easily replace.
