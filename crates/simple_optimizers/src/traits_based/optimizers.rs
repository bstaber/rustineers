// ANCHOR: optimizer_trait
pub trait Optimizer {
    fn update(&mut self, weights: &mut [f64], grads: &[f64]);
}
// ANCHOR_END: optimizer_trait

// ANCHOR: gd_struct
pub struct GradientDescent {
    pub learning_rate: f64,
}
// ANCHOR_END: gd_struct

// ANCHOR: impl_optimizer_gd
impl GradientDescent {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

// ANCHOR_END: impl_optimizer_gd

// ANCHOR: impl_optimizer_gd_update
impl Optimizer for GradientDescent {
    fn update(&mut self, weights: &mut [f64], grads: &[f64]) {
        for (w, g) in weights.iter_mut().zip(grads.iter()) {
            *w -= self.learning_rate * g;
        }
    }
}
// ANCHOR_END: impl_optimizer_gd_update

// ANCHOR: momentum_struct
pub struct Momentum {
    pub learning_rate: f64,
    pub momentum: f64,
    pub velocity: Vec<f64>,
}
// ANCHOR_END: momentum_struct

// ANCHOR: impl_optimizer_momentum
impl Momentum {
    pub fn new(learning_rate: f64, momentum: f64, dim: usize) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: vec![0.0; dim],
        }
    }
}
// ANCHOR_END: impl_optimizer_momentum

// ANCHOR: impl_optimizer_momentum_update
impl Optimizer for Momentum {
    fn update(&mut self, weights: &mut [f64], grads: &[f64]) {
        for ((w, g), v) in weights
            .iter_mut()
            .zip(grads.iter())
            .zip(self.velocity.iter_mut())
        {
            *v = self.momentum * *v + self.learning_rate * *g;
            *w -= *v;
        }
    }
}
// ANCHOR_END: impl_optimizer_momentum_update
