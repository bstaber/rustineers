//! This file is part of the Poisson 2D crate.
//!! It provides functionality for solving the 2D Poisson equation using finite element methods (FEM).
//!
//! The crate includes modules for elements, mesh, quadrature rules, and solvers.

pub mod element;
pub mod mesh;
pub mod quadrature;
pub mod solver;

pub use solver::{assemble_and_solve_dense, assemble_and_solve_sparse};

pub use mesh::Mesh2d;
pub use nalgebra::DVector;

/// Enum representing the type of solver to use
pub enum SolverType {
    Dense,
    Sparse,
}

/// Helper function for solving the 2D Poisson problem
///
/// This function takes a mesh, boundary nodes, boundary function, source function, and solver type.
/// Arguments:
/// - `mesh`: The mesh representing the domain.
/// - `boundary_nodes`: Indices of the nodes on the boundary.
/// - `boundary_fn`: Function defining the boundary condition.
/// - `source_fn`: Function defining the source term.
/// - `solver_type`: Type of solver to use (Dense or Sparse).
///
/// Returns:
/// - A vector containing the solution at the mesh nodes.
pub fn solve_poisson_2d<F>(
    mesh: &Mesh2d,
    boundary_nodes: &[usize],
    boundary_fn: &F,
    source_fn: &F,
    solver_type: SolverType,
) -> DVector<f64>
where
    F: Fn(f64, f64) -> f64,
{
    match solver_type {
        SolverType::Dense => assemble_and_solve_dense(mesh, boundary_nodes, boundary_fn, source_fn),
        SolverType::Sparse => {
            assemble_and_solve_sparse(mesh, boundary_nodes, boundary_fn, source_fn)
        }
    }
}
