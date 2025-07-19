pub mod regressor;
pub use regressor::{RidgeEstimator, RidgeGradientDescent, RidgeModel};

pub fn run_demo() {
    println!("-----------------------------------------------------");
    println!("Running ridge_1d_struct::run_demo");
    let mut model: RidgeEstimator = RidgeEstimator::new(0.0);

    let x: Vec<f64> = vec![1.0, 2.0];
    let y: Vec<f64> = vec![0.1, 0.2];
    let lambda2 = 0.001;

    model.fit(&x, &y, lambda2);
    let preds = model.predict(&x);

    println!("Learned beta: {}, true solution: 0.1!", model.beta);
    println!("Predictions: {preds:?}");
    println!("-----------------------------------------------------");
}
