# Gradient descent

The `GD` struct implements basic gradient descent with a fixed step size:

```rust
{{#include ../../../../crates/simple_optimizers_ndarray/src/optimizers.rs:struct_gd}}
```

It has a constructor:

```rust
{{#include ../../../../crates/simple_optimizers_ndarray/src/optimizers.rs:impl_gd_new}}
```

And implements `Optimizer` by subtracting the gradient scaled by the step size from the weights at each iteration.

```rust
{{#include ../../../../crates/simple_optimizers_ndarray/src/optimizers.rs:impl_gd_run}}
```

Some notes:

- At each iteration, we compute the gradient with `let grads = grad_fn(weights)`, which is fine but it reallocates a new vector at each call. If we wanted to optimize the gradient computation, we could pre-allocate a buffer outside the loop and pass a mutable reference into the gradient function to avoid repeated allocations. This would require to change the signature of the `grad_fn`.

- `weights.zip_mut_with(&grads, |w, &g| {{ ... }})`: This is a mutable zip operation from the `ndarray` crate. It walks over `weights` and `grads`, applying the closure to each pair.

- [zip_mut_with](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.zip_mut_with) is a method defined by the `Zip` trait, which is implemented for [ArrayBase](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html), and in particular for [Array1<f64>](https://docs.rs/ndarray/latest/ndarray/type.Array1.html). That’s why we can call it directly on `weights`.

- In the closure statement we wrote: `|w, &g| *w -= self.step_size * g;`. Here, `w` is a mutable reference to each weight element, so we dereference it using `*w` to update its value. The `&g` in the closure means we’re pattern-matching by reference to avoid cloning or copying each `f64`.
