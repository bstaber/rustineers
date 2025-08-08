use crate::errors::KRRPredictError;
use crate::kernel::Kernel;
use crate::model::KRRModel;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;

//ANCHOR: loo_cv_error
pub fn loo_cv_error<K: Kernel>(model: &KRRModel<K>) -> Result<f64, KRRPredictError> {
    let alpha = model.alpha.as_ref().ok_or(KRRPredictError::NotFitted)?;
    let x_train = model.x_train.as_ref().ok_or(KRRPredictError::NotFitted)?;

    let n = x_train.nrows();
    let mut k_train = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let kxy = model.kernel.compute(x_train.row(i), x_train.row(j));
            k_train[(i, j)] = kxy;
            k_train[(j, i)] = kxy;
        }
    }

    let identity_n = Array2::eye(n);
    let a = k_train + model.lambda * identity_n;
    let a_inv = a.inv().expect("Inversion failed");

    let mut loo_error = 0.0;
    for i in 0..n {
        let ai = alpha[i];
        let di = a_inv[(i, i)];
        let res = ai / di;
        loo_error += res.powi(2);
    }

    Ok(loo_error / (n as f64))
}
//ANCHOR_END: loo_cv_error

//ANCHOR: tune_lengthscale
pub fn tune_lengthscale<K: Kernel + Clone>(
    x_train: Array2<f64>,
    y_train: Array1<f64>,
    lambda: f64,
    lengthscales: &[f64],
    kernel_builder: impl Fn(f64) -> K,
) -> Result<(K, f64), String> {
    let mut best_error = f64::INFINITY;
    let mut best_kernel = None;

    for &l in lengthscales {
        let kernel = kernel_builder(l);
        let mut model = KRRModel::new(kernel.clone(), lambda);

        if model.fit(x_train.clone(), y_train.clone()).is_err() {
            continue;
        }

        if let Ok(err) = loo_cv_error(&model)
            && err < best_error
        {
            best_error = err;
            best_kernel = Some(kernel);
        }
    }

    best_kernel
        .map(|k| (k, best_error))
        .ok_or("Tuning failed".to_string())
}
//ANCHOR_END: tune_lengthscale

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::RBFKernel;
    use crate::model::KRRModel;
    use ndarray::array;

    #[test]
    fn test_loo_cv_and_tuning() {
        let x_train = array![[1.0], [2.0], [3.0], [4.0]];
        let y_train = array![1.0, 2.0, 3.0, 4.0];
        let lambda = 1e-2;
        let lengthscales = [0.1, 0.5, 1.0, 2.0, 5.0];

        let (best_kernel, err) = tune_lengthscale(
            x_train.clone(),
            y_train.clone(),
            lambda,
            &lengthscales,
            RBFKernel::new,
        )
        .expect("Tuning failed");

        let mut model = KRRModel::new(best_kernel, lambda);
        model.fit(x_train.clone(), y_train).unwrap();
        let loo = loo_cv_error(&model).unwrap();

        assert!(
            (err - loo).abs() < 1e-6,
            "Mismatch between stored and recomputed error"
        );
    }
}
