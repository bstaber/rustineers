pub mod functional_std;
pub mod generics_std;
pub mod optimizer;
pub mod structured_ndarray;
pub mod structured_std;
pub mod utils;

pub use functional_std::fit as fit_functional_std;
pub use functional_std::predict as predict_functional_std;
pub use functional_std::run_demo as run_demo_functional_std;

pub use structured_std::RidgeEstimator as StructRidgeEstimator;
pub use structured_std::RidgeGradientDescent;
pub use structured_std::run_demo as run_demo_structured_std;

pub use generics_std::GenRidgeEstimator;
pub use generics_std::run_demo as run_demo_generics_std;

pub use structured_ndarray::RidgeEstimator as NDArrayRidgeEstimator;
pub use structured_ndarray::run_demo as run_demo_structured_ndarray;
