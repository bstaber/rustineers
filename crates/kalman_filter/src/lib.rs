//! Kalman filtering primitives.
//!
//! # Overview
//! This crate provides a minimal Kalman filter for linear-Gaussian state-space
//! models. The core type is [`KalmanFilter`], which exposes a `step` method
//! performing predict (+ optional update).
//!
//! # Example
//!
//! use nalgebra::{DMatrix, DVector};
//! use kalman_filter::prelude::*
//!
//! // 1D constant-velocity toy (just to exercise the API)
//! let a = DMatrix::<f64>::identity(1, 1);
//! let h = DMatrix::<f64>::identity(1, 1);
//! let q = DMatrix::<f64>::identity(1, 1) * 1e-3;
//! let r = DMatrix::<f64>::identity(1, 1) * 1e-2;
//!
//! let mut kf = KalmanFilter::new(
//!     Some(DVector::from_element(1, 0.0)), // x0
//!     None,                                // P0 defaults to I
//!     a, h, q, r
//! );
//!
//! // no observation → pure predict
//! kf.step(None);
//!
//! // with observation → predict + update
//! let z = DVector::from_element(1, 0.5);
//! kf.step(Some(z));
//!
//!

mod algorithm;
pub use algorithm::KalmanFilter;

pub mod prelude {
    pub use super::KalmanFilter;
}
