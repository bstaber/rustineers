use nalgebra::{Cholesky, DMatrix, DVector};
use rand::thread_rng;
use rand_distr::{Distribution, StandardNormal};

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
    ) -> Self {
        let n: usize = state_transition_matrix.ncols();

        let state = init_state.unwrap_or_else(|| {
            let mut rng = thread_rng();
            DVector::from_iterator(n, (0..n).map(|_| StandardNormal.sample(&mut rng)))
        });

        let covariance = init_covariance.unwrap_or_else(|| DMatrix::identity(n, n));

        Self {
            _state: state,
            _covariance: covariance,
            _state_transition_matrix: state_transition_matrix,
            _observation_matrix: observation_matrix,
            _state_noise_covariance: state_noise_covariance,
            _observation_noise_covariance: observation_noise_covariance,
        }
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

    fn update_step(&mut self, observation: DVector<f64>) {
        let h_matrix = &self._observation_matrix;
        let r_matrix = &self._observation_noise_covariance;

        let innovation = observation - h_matrix * &self._state;

        // Innovation covariance: S = H P^- H^T + R   (SPD)
        let hp = h_matrix * &self._covariance;
        let s = &hp * h_matrix.transpose() + r_matrix;

        // K^T = S^{-1} (H P^-)
        let chol = Cholesky::new(s).expect("Innovation covariance not SPD");
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
    }

    pub fn step(&mut self, observation: Option<DVector<f64>>) {
        self.predict_step();
        if let Some(obs) = observation {
            self.update_step(obs);
        }
    }
}
