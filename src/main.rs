use ridge_regression_1d::loss_functions;

fn main() {
    let x: Vec<f64> = vec![1.0, 2.0];
    let y: Vec<f64> = vec![0.1, 0.2];
    let beta: f64 = 0.5;
    let lambda2: f64 = 0.001;

    let value1 = loss_functions::loss_function_naive(&x, &y, beta, lambda2);
    let value2 = loss_functions::loss_function_inline(&x, &y, beta, lambda2);

    println!("Value with first implementation: {:#?}", value1);
    println!("Value with second implementation: {:#?}", value2);
}
