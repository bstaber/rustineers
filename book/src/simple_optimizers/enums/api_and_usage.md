# API and usage

Once you have implemented the optimizers and the `step` logic, it's time to expose a public API to run optimization from your crate's `lib.rs`. This typically involves defining a helper function like `run_optimization`.

## Define `run_optimization` in `lib.rs`

Similarly to the trait-based implementation, we can define a function `run_optimization` that performs the optimization. However, here, `Optimizer` is a enum instead of a trait, hence we can't define a generic type and write `<Opt: Optimizer>` (see [trait-based run_optimization function](../traits/api_and_usage.md) if you don't remember.). Instead, we simply pass a concrete mutable reference.

```rust
{{#include ../../../../crates/simple_optimizers_enums/src/lib.rs}}
```


## Example of usage

Here's a basic example of using `run_optimization` to minimize a simple quadratic loss.

```rust
fn main() {
    let grad_fn = |w: &[f64]| vec![2.0 * (w[0] - 3.0)];
    
    let mut weights = vec![0.0];
    let mut optimizer = Optimizer::gradient_descent(0.1);

    run_optimization(&mut optimizer, &mut weights, grad_fn, 100);

    println!("Optimized weight: {:?}", weights[0]);
}
```