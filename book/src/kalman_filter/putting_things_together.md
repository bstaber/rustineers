# Putting things together

Here is how to set up and run this model using our `KalmanFilter` struct.

```rust
use nalgebra::{DMatrix, DVector};
use kalman_filter::KalmanFilter;

fn main() {
    // Define matrices
    let a_state = DMatrix::from_row_slice(2, 2, &[1.0, 1.0,
                                                  0.0, 1.0]);
    let h_obs = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
    let q = DMatrix::identity(2, 2) * 0.01;
    let r = DMatrix::identity(1, 1) * 0.1;

    // Initial state (position=0, velocity=1)
    let init_state = DVector::from_row_slice(&[0.0, 1.0]);
    let init_cov = DMatrix::identity(2, 2);

    // Build the filter
    let mut kf = KalmanFilter::new(
        Some(init_state),
        Some(init_cov),
        a_state,
        h_obs,
        q,
        r,
    ).unwrap();

    // Simulate one step with an observation
    let observation = DVector::from_row_slice(&[0.9]);
    kf.step(Some(observation)).unwrap();

    println!("Updated state: {:?}", kf.state());
}
```