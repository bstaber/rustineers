# Gram matrix

Once the `Kernel` trait is defined and implemented, it can be used to construct the Gram matrix required for kernel ridge regression. The Gram matrix contains all pairwise kernel evaluations between training inputs. It is symmetric by definition, since $k(x_i, x_j) = k(x_j, x_i)$ for common kernels such as the RBF.

In the KRR implementation, the Gram matrix is computed as follows:

```rust
use crate::kernel::RBFKernel;
use ndarray::{Array, Array2};

let n: usize = y_train.len();
let mut k_train: Array2<f64> = Array::zeros((n, n));
let kernel: RBFKernel = RBFKernel::new(1.0);

for i in 0..n {
    for j in 0..=i {
        let kxy: f64 = kernel.compute(x_train.row(i), x_train.row(j));
        k_train[(i, j)] = kxy;
        k_train[(j, i)] = kxy;
    }
}
```

Here, `x_train.row(i)` returns a value of type `ArrayView1<f64>`, which is exactly the type expected by the `Kernel::compute` method. The loop only computes the lower triangle of the matrix and mirrors it to the upper triangle to avoid redundant computation, exploiting the symmetry of the kernel. This approach is efficient and idiomatic in Rust using `ndarray`.

We will use this piece of code in our fit function.