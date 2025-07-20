// ANCHOR: lib_rs
pub mod estimator;
pub mod gradient_descent;
pub mod loss_functions;

pub use estimator::ridge_estimator;

/// Fits a Ridge regression model.
///
/// # Arguments
///
/// * `x` - Input features (`&[f64]`)
/// * `y` - Target values (`&[f64]`)
/// * `lambda2` - Regularization strength
///
/// # Returns
///
/// The optimized coefficient `beta` as `f64`.
pub fn fit(x: &[f64], y: &[f64], lambda2: f64) -> f64 {
    ridge_estimator(x, y, lambda2)
}

/// Predicts output values using a trained Ridge regression coefficient.
///
/// # Arguments
///
/// * `x` - Input features (`&[f64]`)
/// * `beta` - Trained coefficient
///
/// # Returns
///
/// A `Vec<f64>` with predicted values.
pub fn predict(x: &[f64], beta: f64) -> Vec<f64> {
    x.iter().map(|xi| xi * beta).collect()
}
// ANCHOR_END: lib_rs

// ANCHOR: run_demo
pub fn run_demo() {
    println!("-----------------------------------------------------");
    println!("Running ridge_1d_fn::run_demo");
    let x: Vec<f64> = vec![1.0, 2.0];
    let y: Vec<f64> = vec![0.1, 0.2];
    let lambda2 = 0.001;

    let beta = fit(&x, &y, lambda2);
    let preds = predict(&x, beta);

    println!("Learned beta: {beta}, true solution: 0.1!");
    println!("Predictions: {preds:?}");
    println!("-----------------------------------------------------");
}
// ANCHOR_END: run_demo
