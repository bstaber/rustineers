use ndarray::ArrayView1;

pub trait Kernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64;
}

#[derive(Clone)]
pub struct RBFKernel {
    pub lengthscale: f64,
}

impl RBFKernel {
    pub fn new(lengthscale: f64) -> Self {
        assert!(lengthscale > 0.0, "Lengthscale must be positive");
        Self { lengthscale }
    }
}

impl Kernel for RBFKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let diff = &x - &y;
        (-diff.dot(&diff) / (2.0 * self.lengthscale.powi(2))).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_rbf_kernel_xx() {
        let kernel = RBFKernel::new(1.0);
        let x = array![1.0, 2.0, 3.0];
        let kxx = kernel.compute(x.view(), x.view());
        assert_eq!(kxx, 1.0, "Expected k(x, x) to be equal to 1.0, got {}", kxx);
    }

    #[test]
    fn test_fb_kernel_xy() {
        let kernel = RBFKernel::new(1.0);
        let x = array![1.0, 2.0, 3.0];
        let y = array![4.0, 5.0, 6.0];
        let kxy = kernel.compute(x.view(), y.view());
        assert!(kxy < 1.0, "Expected k(x, y) < 1.0, got {}", kxy);
    }
}
