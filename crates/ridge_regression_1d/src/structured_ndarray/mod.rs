pub mod regressor;
pub use self::regressor::RidgeEstimator;
use ndarray::array;

pub fn run_demo() {
    println!("-----------------------------------------------------");
    println!("Running ridge_regression_1d::structured_ndarray::run_demo");

    let mut model = RidgeEstimator::new();

    let x = array![1.0, 2.0];
    let y = array![0.1, 0.2];
    let lambda2 = 0.001;

    model.fit(&x, &y, lambda2);
    let preds = model.predict(&x);
    
    match model.beta {
        Some(beta) => println!("Learned beta: {}, true solution: 0.1!", beta),
        None => println!("Model not fitted!"),
    }
    println!("Predictions: {preds:?}");
    println!("-----------------------------------------------------");
}
