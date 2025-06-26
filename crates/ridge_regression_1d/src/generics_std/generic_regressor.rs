use num_traits::{Float, FromPrimitive};
use std::iter::Sum;

pub struct GenRidgeRegressor<F: Float> {
    beta: F,
    lambda2: F,
}

impl<F: Float + FromPrimitive + Sum> GenRidgeRegressor<F> {
    pub fn loss_function(&self, x: &[F], y: &[F]) -> F {
        let n: usize = x.len();
        let n_f: F = F::from(n).expect("usize to F conversion failed");

        let mse: F = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| {
                let residual = *yi - self.beta * *xi;
                residual * residual
            })
            .sum::<F>()
            / (F::from(2.0).unwrap() * n_f);
        mse + self.lambda2 * self.beta * self.beta
    }
}
