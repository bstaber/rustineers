# Pattern matching

In Rust, the match keyword is used to compare a value against a set of patterns and execute code based on which pattern matches. This is especially useful with enums like `Option`.

```rust
let maybe_number: Option<i32> = Some(42);

match maybe_number {
    Some(n) => println!("The number is: {}", n),
    None => println!("No number available."),
}
```

This pattern ensures we safely handle both the presence and absence of a value.

## Predict function

We use the exact same technique in our model to check whether it has been fitted. Since beta is of type `Option<f64>`, we can match on its value to determine whether a prediction can be made:

```rust
match self.beta {
    Some(beta) => Ok(x * beta),
    None => Err("Model not fitted".to_string()),
}
```

The full function takes this form:

```rust
{{#include ../../../../crates/ridge_regression_1d/src/structured_ndarray/regressor.rs:ridge_estimator_impl_predict}}
```

Here, we also decide to explicitly raise an error if the model has not been fitted. To do this in a type-safe way, we use Rust’s `Result` enum, which is commonly used for functions that may fail. The `Result` enum can be either `Ok(value)` (indicating success) or `Err(error)` (indicating failure). More details about error handling are given in the next section.