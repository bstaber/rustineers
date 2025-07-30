use thiserror::Error;

#[derive(Debug, Error)]
pub enum KRRFitError {
    #[error("Shape mismatch: x has {x_n} rows but y has {y_n} elements")]
    ShapeMismatch { x_n: usize, y_n: usize },

    #[error("Solving the linear system failed")]
    LinAlgError(String),
}

#[derive(Debug, Error)]
pub enum KRRPredictError {
    #[error("Model not fitted")]
    NotFitted,
}
