# Instantiation

To instantiate a `GradientDescent`, the user has to write:

```rust
let optimizer = Optimizer::GradientDescent {
    learning_rate: 0.1,
};
```

For the momentum-based gradient descent, the instantiation becomes more cumbersome:

```rust
let optimizer = Optimizer::Momentum {
    learning_rate: 0.1,
    momentum: 0.9,
    velocity: vec![0.0; 3],
};
```

To make this more user-friendly, we can define more convenient constrcutors such as:

```rust
let optimizer = Optimizer::gradient_descent(0.1);
```

and

```rust
let optimizer = Optimizer::momentum(0.1, 0.9, 3);
```

This can be achieved by adding two implementations:

```rust
{{#include ../../../../crates/simple_optimizers_enums/src/optimizers.rs:constructors}}
```

This is optional but it helps create optimizers easily.

