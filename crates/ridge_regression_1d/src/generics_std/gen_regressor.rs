use num_traits::Float;
use std::iter::Sum;

pub trait RidgeModel<F: Float + Sum> {
    fn fit(&mut self, x: &[F], y: &[F], lambda2: F);
    fn predict(&self, x: &[F]) -> Vec<F>;
}

pub struct GenRidgeEstimator<F: Float + Sum> {
    beta: F,
}

impl<F: Float + Sum> GenRidgeEstimator<F> {
    pub fn new(init_beta: F) -> Self {
        Self { beta: init_beta }
    }
}

impl<F: Float + Sum> RidgeModel<F> for GenRidgeEstimator<F> {
    fn fit(&mut self, x: &[F], y: &[F], lambda2: F) {
        let n: usize = x.len();
        let n_f: F = F::from(n).unwrap();
        assert_eq!(x.len(), y.len(), "x and y must have the same length");

        let x_mean: F = x.iter().copied().sum::<F>() / n_f;
        let y_mean: F = y.iter().copied().sum::<F>() / n_f;

        let num: F = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (*xi - x_mean) * (*yi - y_mean))
            .sum::<F>();

        let denom: F = x.iter().map(|xi| (*xi - x_mean).powi(2)).sum::<F>() + lambda2 * n_f;

        self.beta = num / denom;
    }

    fn predict(&self, x: &[F]) -> Vec<F> {
        x.iter().map(|xi| *xi * self.beta).collect()
    }
}
