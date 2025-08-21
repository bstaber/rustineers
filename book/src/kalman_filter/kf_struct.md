# Kalman filter struct

## Struct defintion

The `KalmanFilter` struct is the central object of our implementation. It bundles together the current state of the filter, its uncertainty, and the matrices that govern its dynamics and observations.

```rust
{{#include ../../../crates/kalman_filter/src/algorithm.rs:struct}}
```

We rely on the [`nalgebra`](https://nalgebra.org/) crate for linear algebra operations. Its `DVector` and `DMatrix` types are dynamically sized, meaning their dimensions are not fixed at compile time but can be determined at runtime.   Together, these fields provide all the ingredients needed to perform the two alternating steps of the Kalman filter algorithm: prediction (using $A, Q$) and correction (using $H, R$).

## Custom error Enum

```rust
{{#include ../../../crates/kalman_filter/src/algorithm.rs:error_enum}}
```

The `KalmanError` enum defines the possible errors that may arise when running the filter.

- **`InnovationNotSPD`**  
  This error is triggered when the innovation covariance matrix is not symmetric positive definite (SPD).

- **`Dim(String)`**  
  This error indicates a mismatch in matrix or vector dimensions. For example, if the state vector size does not match the number of rows in the transition matrix, the filter cannot proceed. The `String` message provides more context about the mismatch.

By defining a custom error type, we make the library easier to debug and integrate into larger applications. Instead of panicking on invalid inputs, we return descriptive error messages that the user can handle.