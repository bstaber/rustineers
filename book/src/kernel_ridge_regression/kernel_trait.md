### The kernel module

The `kernel.rs` module defines the core abstraction used to compute similarity between data points in Kernel Ridge Regression (KRR). This abstraction is formalized as a trait, and a specific instance of this trait is implemented using the radial basis function (RBF) kernel, a popular choice in kernel methods. The module also includes unit tests to validate correctness.

<details>
<summary>Click here to view to full module: <b>kernel.rs</b>. We break into down in the sequel of this section. </summary>

```rust
{{#include ../../../crates/krr_ndarray/src/kernel.rs}}
```
</details>

#### The `Kernel` trait

We first define a `Kernel` trait:

```rust
pub trait Kernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64;
}
```

This trait represents a generic kernel function. It requires a single method, `compute`, which takes two inputs `x` and `y` as one-dimensional views (`ArrayView1<f64>`) and returns a scalar similarity score of type `f64`. By using views instead of owned arrays, this interface avoids unnecessary data copying and supports efficient evaluation.

This trait enables polymorphism: any kernel function that implements `Kernel` can be used within the rest of the KRR pipeline.

#### The `RBFKernel` struct

To provide a concrete implementation of the `Kernel` trait, the module defines the `RBFKernel` struct:

```rust
pub struct RBFKernel {
    pub lengthscale: f64,
}
```

The `lengthscale` parameter controls how quickly the similarity between two points decays with distance. A smaller lengthscale produces more localized kernels, while a larger one results in smoother, more global effects.

The constructor `new` is implemented as:

```rust
impl RBFKernel {
    pub fn new(lengthscale: f64) -> Self {
        assert!(lengthscale > 0.0, "Lengthscale must be positive");
        Self { lengthscale }
    }
}
```

This method ensures that the lengthscale is strictly positive, preventing ill-posed kernel evaluations.

#### Kernel evaluation

The `Kernel` trait is implemented for `RBFKernel` as follows:

```rust
impl Kernel for RBFKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let diff = &x - &y;
        (-diff.dot(&diff) / (2.0 * self.lengthscale.powi(2))).exp()
    }
}
```

This implementation computes the squared Euclidean distance between `x` and `y`, scales it by the squared lengthscale, and applies the exponential function. The result is the value of the Gaussian kernel:

$$
k(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\ell^2}\right)
$$

This function satisfies the requirements of a positive definite kernel and is commonly used in many kernel-based algorithms.

#### Unit tests

The module includes two unit tests that validate the behavior of the RBF kernel:

```rust
#[test]
fn test_rbf_kernel_xx() {
    let kernel = RBFKernel::new(1.0);
    let x = array![1.0, 2.0, 3.0];
    let kxx = kernel.compute(x.view(), x.view());
    assert_eq!(kxx, 1.0, "Expected k(x, x) to be equal to 1.0, got {}", kxx);
}
```

This test checks that the kernel evaluated at the same point yields 1.0, as expected from the RBF formula.

```rust
#[test]
fn test_fb_kernel_xy() {
    let kernel = RBFKernel::new(1.0);
    let x = array![1.0, 2.0, 3.0];
    let y = array![4.0, 5.0, 6.0];
    let kxy = kernel.compute(x.view(), y.view());
    assert!(kxy < 1.0, "Expected k(x, y) < 1.0, got {}", kxy);
}
```

This test confirms that the similarity between two distinct vectors is strictly less than 1.0, reflecting the decay property of the RBF kernel.

#### Summary

The `kernel.rs` module introduces a reusable kernel interface and demonstrates a concrete implementation using the RBF kernel. It serves as a foundation for computing Gram matrices and enables modularity in the design of the KRR model. The use of traits and parametric polymorphism makes it easy to experiment with other kernel functions in future extensions.
