use nalgebra::{Cholesky, DMatrix, DVector};
use rand::thread_rng;
use rand_distr::{Distribution, StandardNormal};

#[derive(Debug, thiserror::Error)]
pub enum KalmanError {
    /// Innovation covariance was not symmetric and positive definite
    #[error("innovation covariance is not SPD")]
    InnovationNotSpd,
    /// Dimension mismatch error
    #[error("dimension mismatch: {0}")]
    Dim(String),
}
pub struct KalmanFilter {
    _state: DVector<f64>,
    _covariance: DMatrix<f64>,
    _state_transition_matrix: DMatrix<f64>,
    _observation_matrix: DMatrix<f64>,
    _state_noise_covariance: DMatrix<f64>,
    _observation_noise_covariance: DMatrix<f64>,
}

impl KalmanFilter {
    pub fn new(
        init_state: Option<DVector<f64>>,
        init_covariance: Option<DMatrix<f64>>,
        state_transition_matrix: DMatrix<f64>,
        observation_matrix: DMatrix<f64>,
        state_noise_covariance: DMatrix<f64>,
        observation_noise_covariance: DMatrix<f64>,
    ) -> Result<Self, KalmanError> {
        let n: usize = state_transition_matrix.ncols();

        let state = init_state.unwrap_or_else(|| {
            let mut rng = thread_rng();
            DVector::from_iterator(n, (0..n).map(|_| StandardNormal.sample(&mut rng)))
        });

        let covariance = init_covariance.unwrap_or_else(|| DMatrix::identity(n, n));

        // Check square matrices
        if state_transition_matrix.nrows() != n {
            return Err(KalmanError::Dim("A must be square".to_string()));
        }
        if state_noise_covariance.shape() != (n, n) {
            return Err(KalmanError::Dim("Q must be n × n".to_string()));
        }

        // Check observation dimensions
        let m: usize = observation_matrix.nrows();
        if observation_matrix.ncols() != n {
            return Err(KalmanError::Dim("H must be m x n".to_string()));
        };
        if observation_noise_covariance.shape() != (m, m) {
            return Err(KalmanError::Dim("R must be m × m".to_string()));
        }

        // Check state init
        if state.len() != n {
            return Err(KalmanError::Dim("x0 must have length n".to_string()));
        }
        if covariance.shape() != (n, n) {
            return Err(KalmanError::Dim("P0 must be n×n".to_string()));
        }

        Ok(Self {
            _state: state,
            _covariance: covariance,
            _state_transition_matrix: state_transition_matrix,
            _observation_matrix: observation_matrix,
            _state_noise_covariance: state_noise_covariance,
            _observation_noise_covariance: observation_noise_covariance,
        })
    }
}

impl KalmanFilter {
    pub fn state(&self) -> &DVector<f64> {
        &self._state
    }

    pub fn covariance(&self) -> &DMatrix<f64> {
        &self._covariance
    }
}

impl KalmanFilter {
    fn predict_step(&mut self) {
        self._state = &self._state_transition_matrix * &self._state;
        self._covariance = &self._state_transition_matrix
            * &self._covariance
            * &self._state_transition_matrix.transpose()
            + &self._state_noise_covariance;
    }

    fn update_step(&mut self, observation: DVector<f64>) -> Result<(), KalmanError> {
        let h_matrix = &self._observation_matrix;
        let r_matrix = &self._observation_noise_covariance;

        let innovation = observation - h_matrix * &self._state;

        // Innovation covariance: S = H P^- H^T + R   (SPD)
        let hp = h_matrix * &self._covariance;
        let s = &hp * h_matrix.transpose() + r_matrix;

        // K^T = S^{-1} (H P^-)
        let Some(chol) = Cholesky::new(s) else {
            return Err(KalmanError::InnovationNotSpd);
        };
        let k_t = chol.solve(&hp);
        let kalman_gain = k_t.transpose();

        // State update: x = x^- + K y
        self._state = &self._state + &kalman_gain * innovation;

        // Joseph covariance update
        // P = (I - K H) P^- (I - K H)^T + K R K^T
        let n = self._covariance.nrows();
        let i = DMatrix::<f64>::identity(n, n);
        let ikh = &i - &kalman_gain * h_matrix;
        self._covariance = &ikh * &self._covariance * ikh.transpose()
            + &kalman_gain * r_matrix * kalman_gain.transpose();

        Ok(())
    }

    pub fn step(&mut self, observation: Option<DVector<f64>>) -> Result<(), KalmanError> {
        self.predict_step();
        if let Some(obs) = observation {
            self.update_step(obs)?;
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_kf_constructor() {
        let a_state = DMatrix::<f64>::identity(2, 2);
        let h_obs = DMatrix::<f64>::identity(2, 2);
        let state_noise = DMatrix::<f64>::identity(2, 2) * 1e-3;
        let obs_noise = DMatrix::<f64>::identity(2, 2) * 1e-2;

        // no initial state or covariance provided
        let kf_model = KalmanFilter::new(
            None,       // init_state
            None,       // init_covariance
            a_state,
            h_obs,
            state_noise,
            obs_noise,
        );
        // check kf_model is not an error
        assert!(kf_model.is_ok());
        let kf_model = kf_model.unwrap();

        // check dimensions
        assert_eq!(kf_model.state().len(), 2);
        assert_eq!(kf_model.covariance().shape(), (2, 2));
    }

    #[test]
    fn test_kf_constructor_misspecified() {
        let a_state = DMatrix::<f64>::identity(2, 2);
        let h_obs = DMatrix::<f64>::identity(2, 3);
        let state_noise = DMatrix::<f64>::identity(2, 2) * 1e-3;
        let obs_noise = DMatrix::<f64>::identity(2, 2) * 1e-2;

        // no initial state or covariance provided
        let kf_model = KalmanFilter::new(
            None,       // init_state
            None,       // init_covariance
            a_state,
            h_obs,
            state_noise,
            obs_noise,
        );
        // check kf_model is not an error
        assert!(kf_model.is_err());
    }
}