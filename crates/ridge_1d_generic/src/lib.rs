pub mod regressor;
pub use regressor::{GenRidgeEstimator, RidgeModel};

pub fn run_demo() {
    println!("-----------------------------------------------------");
    println!("Running ridge_regression_1d::generics_std::run_demo");

    let mut model: GenRidgeEstimator<f32> = GenRidgeEstimator::new(1.0);

    let x: Vec<f32> = vec![1.0, 2.0];
    let y: Vec<f32> = vec![0.1, 0.2];
    let lambda2 = 0.001;

    model.fit(&x, &y, lambda2);
    let preds: Vec<f32> = model.predict(&x);

    println!("Learned beta: {}, true solution: 0.1!", model.beta);
    println!("Predictions: {preds:?}");
    println!("-----------------------------------------------------");
}
