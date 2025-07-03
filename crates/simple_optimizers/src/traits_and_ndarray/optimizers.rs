use ndarray::{Array, Array1};

pub trait Optimizer {
    fn run(
        &mut self,
        weights: &mut Array1<f64>,
        grad_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
        n_steps: usize,
    );
}

pub struct GD {
    step_size: f64,
}

pub struct AGD {
    step_size: f64,
    momentum: f64,
}

// pub struct AdaptiveAGD {
//     step_size: f64,
//     momentum: f64,
//     velocity: Array1<f64>,
// }

impl GD {
    pub fn new(step_size: f64) -> Self {
        Self { step_size }
    }
}

impl AGD {
    pub fn new(step_size: f64, momentum: f64) -> Self {
        Self {
            step_size,
            momentum,
        }
    }
}

// impl AdaptiveAGD {
//     pub fn new(step_size: f64, dim: usize) -> Self {
//         let zeros = Array1::<f64>::zeros(dim);
//         Self {
//             step_size,
//             momentum: 1.0,
//             velocity: zeros.clone(),
//         }
//     }
// }

impl Optimizer for GD {
    fn run(
        &mut self,
        weights: &mut Array1<f64>,
        grad_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
        n_steps: usize,
    ) {
        for _ in 0..n_steps {
            let grads = grad_fn(weights);
            weights.zip_mut_with(&grads, |w, g| {
                *w -= self.step_size * g;
            });
        }
    }
}

impl Optimizer for AGD {
    fn run(
        &mut self,
        weights: &mut Array1<f64>,
        grad_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
        n_steps: usize,
    ) {
        let n: usize = weights.len();
        let mut velocity: Array1<f64> = Array::zeros(n);
        for _ in 0..n_steps {
            let grads = grad_fn(weights);
            for ((w, g), v) in weights
                .iter_mut()
                .zip(grads.iter())
                .zip(velocity.iter_mut())
            {
                *v = self.momentum * *v - self.step_size * g;
                *w += *v;
            }
        }
    }
}

// impl Optimizer for AdaptiveAGD {
//     fn run(
//         &mut self,
//         weights: &mut Array1<f64>,
//         grad_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
//         n_steps: usize,
//     ) {
//         self.velocity.assign(weights);

//         for _ in 0..n_steps {
//             let grad = grad_fn(&self.velocity);

//             let mut x_next = self.velocity.clone();
//         }
//     }
// }
