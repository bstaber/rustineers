# Constructor

## Implementation of the **new** method

```rust
{{#include ../../../crates/kalman_filter/src/algorithm.rs:new}}
```

### `Result<Self, KalmanError>`?

The constructor returns a `Result` because building a valid Kalman filter requires matrix and vector invariants that can fail at runtime (e.g., incompatible dimensions). 
Instead of panicking, we make these failures explicit and type-checked:

- On success we return `Ok(Self { ... })` with a fully-formed filter.
- On failure we return `Err(KalmanError::...)` describing exactly what went wrong (e.g., `Dim("H must be m x n")`).

This makes the API predictable and easy to integrate in larger applications:
```rust
let kf = KalmanFilter::new(/* ... */)?; // propagate errors with `?`
```

### Optional inputs with `Option<...>`

Two inputs are optional in the constructor: the initial state and the initial covariance.

- `init_state: Option<DVector<f64>>`
- `init_covariance: Option<DMatrix<f64>>`

We handle them using `Option`’s combinators:

```rust
let state = init_state.unwrap_or_else(|| {
    // Lazily create a default x0 ~ N(0, I) if none is provided
    let mut rng = thread_rng();
    DVector::from_iterator(n, (0..n).map(|_| StandardNormal.sample(&mut rng)))
});

let covariance = init_covariance.unwrap_or_else(|| DMatrix::identity(n, n));
```

Why `unwrap_or_else` and not `unwrap_or`? Because the defaults are not free (they allocate and may call RNG). `unwrap_or_else` takes a closure, so the work is done only when needed.

### Inferring sizes and checking invariants

The constructor derives the state dimension from the transition matrix:

```rust
let n: usize = state_transition_matrix.ncols();
```

Then it validates every object against this `n` and against the measurement size `m` inferred from `H`:

- `A` must be square: `A.shape() == (n, n)`
- `Q` must be `n x n`
- `H` must be `m x n` with `m = H.nrows()`
- `R` must be `m x m`
- If provided, `x0.len() == n`
- If provided, `P0.shape() == (n, n)`

On violation, we return dimension errors such as:
```rust
return Err(KalmanError::Dim("A must be square".to_string()));
```

This early validation guarantees the object is internally consistent once constructed.

### The role of `Ok(...)`

At the end of the constructor we wrap the freshly created struct in `Ok(...)`:
```rust
Ok(Self {
    _state: state,
    _covariance: covariance,
    _state_transition_matrix: state_transition_matrix,
    _observation_matrix: observation_matrix,
    _state_noise_covariance: state_noise_covariance,
    _observation_noise_covariance: observation_noise_covariance,
})
```

`Ok` is the success variant of `Result`. Returning `Ok(Self { ... })` signals that all checks passed and the filter is ready to use.

## Accessors
 
In Rust, struct fields can be made `pub` to expose them directly, but here we decided to keep them private and provide accessor methods instead. This gives us more flexibility to change internal representation later without breaking user code.

```rust
{{#include ../../../crates/kalman_filter/src/algorithm.rs:accessors}}
```

These accessors can be used as follows:

```rust
let kf = KalmanFilter::new(None, None, A, H, Q, R)?;

// Access the current state estimate
let x = kf.state();
println!("Current state: {x:?}");

// Access the current covariance
let p = kf.covariance();
println!("Current covariance: {p:?}");
```

If you do want to expose modification, you can add a mutable accessor:

```rust
pub fn state_mut(&mut self) -> &mut DVector<f64> {
    &mut self._state
}
```