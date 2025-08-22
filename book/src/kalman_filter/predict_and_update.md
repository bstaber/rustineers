# Predict and update

The Kalman filter works in two main phases:  
1. Prediction: we project the system state forward in time, based on the model.  
2. Update (or correction): we adjust this prediction using the latest observation.  

Finally, the step function combines both into one routine, making it convenient to advance the filter by a single time step.

<details>
<summary>Click here to view the full implementation: <b>algorithm.rs</b>. We break into down in the sequel of this section. </summary>

```rust
{{#include ../../../crates/kalman_filter/src/algorithm.rs}}
```
</details>

## Predict step

The predict step function simply updates the state and its covariance. 

The method takes `&mut self`. This means we are borrowing the filter mutably, because the internal state (`_state` and `_covariance`) will be modified in place. A shared reference `&self` would not allow us to update fields.  

The code uses `&self._state_transition_matrix * &self._state`. In `nalgebra`, the multiplication operator (`*`) is overloaded for matrices and vectors. We pass references to avoid unnecessary cloning of large matrices/vectors.

```rust
{{#include ../../../crates/kalman_filter/src/algorithm.rs:predict}}
```

## Update/correction step

This method ingests a new observation and corrects the predicted state and covariance. For the sake of clarity, the code is hidden but you can view it below.

Our implementation has two notable features:

- `&mut self`: the method mutates the internal fields (`_state`, `_covariance`) of the filter in place.
- `Result<(), KalmanError>`: numerically, the innovation covariance `S` must be symmetric positive definite (SPD) to invert/factorize. If `S` is not SPD, the update is invalid and we signal that with a pecific error rather than panic.

Regarding `nalgebra` details:

- Borrowing to avoid clones: expressions like `h_matrix * &self._covariance` or `&kalman_gain * r_matrix` pass references so large temporaries aren’t cloned.
- Operator overloading: `*` performs matrix–matrix or matrix–vector multiplication and `transpose()` creates a transposed view when possible.
- Dimension expectations: if state dimension is `n` and measurement dimension is `m`, then  
  `H: m×n`, `R: m×m`, `P: n×n`, `K: n×m`, `S: m×m`. Mismatches will cause compile‑time or runtime panics depending on how they occur.
- SPD check via Cholesky: `Cholesky::new(s)` returns `None` when `s` is not SPD. That’s why this method can fail and returns a `KalmanError`.

<details>
<summary><b>Click here to view the update function</b> </summary>

```rust
{{#include ../../../crates/kalman_filter/src/algorithm.rs:update}}
```
</details>


## Step function

```rust
{{#include ../../../crates/kalman_filter/src/algorithm.rs:step}}
```

The `step` method combines prediction and update into a single call, representing one full iteration of the Kalman filter. It works as follows:

- First, it always performs a predict step, advancing the state forward in time.
- Then, if an observation is provided, it runs the update step with that measurement.
- Finally, it returns `Ok(())` if successful, or propagates any error from `update_step`.

**Implementation details**

- `Option<DVector<f64>>`: Observations may not always be available (e.g., missing sensor data). By using `Option`, we can handle both cases:  
  - `Some(obs)` → we have a measurement to incorporate.  
  - `None` → we skip the update and rely on prediction only.  

- Pattern matching with `if let Some(obs)`: This is a concise way to branch only when the observation is present.  

- Error propagation (`?`): If `update_step` returns an error (e.g., covariance not SPD), the `?` operator bubbles it up immediately, stopping execution.  

**Summary**

The `step` function is the public API entry point for advancing the filter by one timestep.  
It hides the internal details of calling `predict_step` and `update_step` separately, making the filter easier to use in simulations or real-time systems:

- Call `step(None)` when no measurement is available.  
- Call `step(Some(obs))` when you do have a measurement.  
