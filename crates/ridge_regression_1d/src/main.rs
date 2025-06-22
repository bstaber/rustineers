use ridge_regression_1d::grad_functions::grad_loss_function_inline as grad_fn;
use ridge_regression_1d::optimizer::gradient_descent;

fn main() {
    let x: Vec<f64> = vec![1.0, 2.0];
    let y: Vec<f64> = vec![0.1, 0.2];

    let lambda2 = 0.001;
    let step_size = 0.1;
    let n_iters = 100;
    let init_beta = 0.5;

    let beta = gradient_descent(grad_fn, &x, &y, lambda2, step_size, n_iters, init_beta);

    println!("Learned beta: {beta}, true solution: 0.1!");
}
