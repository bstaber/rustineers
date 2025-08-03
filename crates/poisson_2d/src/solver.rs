use crate::mesh::Mesh2d;
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::conjugate_gradient;

pub fn _assemble_system_dense<F>(mesh: &Mesh2d, _f: &F) -> (DMatrix<f64>, DVector<f64>)
where
    F: Fn(f64, f64) -> f64,
{
    let num_vertices = mesh.vertices().len();
    let a = DMatrix::zeros(num_vertices, num_vertices);
    let b = DVector::zeros(num_vertices);
    (a, b)
}

pub fn _assemble_system_sparse<F>(mesh: &Mesh2d, _f: &F) -> (CsrMatrix<f64>, DVector<f64>)
where
    F: Fn(f64, f64) -> f64,
{
    let num_vertices = mesh.vertices().len();

    let mut coo = CooMatrix::new(num_vertices, num_vertices);

    for i in 0..num_vertices {
        coo.push(i, i, 1.0);
    }

    let a = CsrMatrix::from(&coo);
    let b = DVector::zeros(num_vertices);
    (a, b)
}

pub fn _apply_dirichlet_dense() {}

pub fn _apply_dirichlet_sparse() {}

pub fn _solve_dense(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    let chol = a.clone().cholesky()?;
    Some(chol.solve(b))
}

pub fn _solve_sparse(a: &CsrMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    conjugate_gradient::solve(a, b, 1000, 1e-10)
}
