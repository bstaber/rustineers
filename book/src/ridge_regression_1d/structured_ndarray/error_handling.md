# Error handling with `Result`

We decided to raise an error if the model hasn't been fitted yet using the pattern matching:

```rust
match self.beta {
    Some(beta) => Ok(x * beta),
    None => Err("Model not fitted".to_string()),
}
```

This makes sense because our `predict` function returns a `Result<Array1<f64>, String>`. We use the `to_string()` method to convert the string literal to a regular `String` as requested. In practice, the user can use this function as follows:

```rust
let y_pred = model.predict(&x).unwrap();
```

which will panic if the model is not fitted, or


```
let y_pred = model.predict(&x).expect("Model not fitted yet");
```

to add a custom error message. These methods will make the code crash. Another strategy is to handle the `Result` with a `match` too, i.e., 

```rust
match model.predict(&x) {
    Ok(y_pred) => println!("Predicted values: {:?}", y_pred),
    Err(e) => eprintln!("Prediction failed: {}", e),
}
```

In summary, we have two matches that serve different roles:
- Internal match: is `beta` available ?
- External match: did `predict` work ?

We could also handle other kinds of errors such as dimensionality mismatch. To do, we can implement our own types of errors.

## More advanced error handling

We could have gone even further by defining a custom `ModelError` type as follows.

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Model is not fitted yet")]
    NotFitted,
    #[error("Dimension mismatch")]
    DimensionMismatch,
}
```

This approach uses the `thiserror` crate to simplify the implementation of the standard `Error` trait.

By deriving `#[derive(Debug, Error)]` and annotating each variant with `#[error("...")]`, we define error messages rightaway.

The predict function would be rewritten as:

```rust
pub fn predict(&self, x: &Array1<f64>) -> Result<f64, ModelError> {
    match &self.beta {
        Some(beta) => {
            if beta.len() != x.len() {
                return Err(ModelError::DimensionMismatch);
            }
            Ok(beta.dot(x))
        }
        None => Err(ModelError::NotFitted),
    }
}
```

and could be used as follows:

```rust
match model.predict(&x) {
    Ok(y_pred) => println!("Prediction: {}", y_pred),
    Err(ModelError::NotFitted) => eprintln!("Model is not fitted yet."),
    Err(ModelError::DimensionMismatch) => eprintln!("Input dimension doesn't match."),
}
```