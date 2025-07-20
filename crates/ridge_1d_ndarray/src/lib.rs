pub mod regressor;
use ndarray::array;
pub use regressor::RidgeEstimator;

pub fn run_demo() {
    println!("-----------------------------------------------------");
    println!("Running ridge_1d_ndarray::run_demo");

    let mut model = RidgeEstimator::new();

    let x = array![1.0, 2.0];
    let y = array![0.1, 0.2];
    let lambda2 = 0.001;

    model.fit(&x, &y, lambda2);
    let preds = model.predict(&x);

    match model.beta {
        Some(beta) => println!("Learned beta: {beta}, true solution: 0.1!"),
        None => println!("Model not fitted!"),
    }
    println!("Predictions: {preds:?}");
    println!("-----------------------------------------------------");
}
