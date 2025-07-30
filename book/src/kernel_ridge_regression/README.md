# Kernel Ridge regression

In this chapter, we implement [Kernel Ridge Regression (KRR)](https://teazrq.github.io/SMLR/kernel-ridge-regression.html) in Rust using the `ndarray` and `ndarray-linalg` crates. The implementation is broken into the following sections:

* In the [Kernel module section](kernel_trait.md), we define a `Kernel` trait and implement the radial basis function (RBF) kernel.

* In the [Gram matrix section](gram_matrix.md), we construct the symmetric Gram matrix needed to solve the KRR problem using `Array2` and `ArrayView1`.

* In the [KRR model section](krr_model.md), we define the `KRRModel` struct and its constructor, making the model generic over any type that implements the Kernel trait.

* In the [fit function section](krr_fit.md), we implement the logic for training the model, including matrix assembly, regularization, and linear system solving. We introduce a custom error enum `KRRFitError` to manage common issues.

* In the [predict function section](krr_predict.md), we implement inference for new samples and introduce the `KRRPredictError` enum to handle the unfitted model case.

* In the [hyperparameter tuning section](hparams_tuning.md), we implement leave-one-out cross-validation (LOOCV) to select a good value for the kernel’s lengthscale.

At the end of the chapter, we obtain a small standalone crate with the following layout:
```text
├── Cargo.toml
└── src
    ├── errors.rs
    ├── kernel.rs
    ├── lib.rs
    └── model.rs
```

where the `Cargo.toml` configuration file is given by:

```toml
[package]
name = "krr_ndarray"
version = "0.1.0"
edition = "2024"

[dependencies]
rustineers = { path = "../../" }
ndarray = "0.15.2"
ndarray-linalg = { version = "0.16", features = ["openblas-static"] }
thiserror = "1.0"
```

We enable the `openblas-static` feature to ensure OpenBLAS is built within the crate, avoiding reliance on system-wide BLAS libraries. The `thiserror` crate is used to define ergonomic and readable custom error types.