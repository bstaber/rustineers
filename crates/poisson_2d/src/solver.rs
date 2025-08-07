use crate::element::ReferenceElement;
use crate::mesh::{ElementType, Mesh2d};
use crate::quadrature::QuadRule;
use nalgebra::{DMatrix, DVector, Point2, Vector2};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::conjugate_gradient;

/// Function that assembles the FEM system using a dense matrix.
pub fn assemble_system_dense<F>(mesh: &Mesh2d, source_fn: &F) -> (DMatrix<f64>, DVector<f64>)
where
    F: Fn(f64, f64) -> f64,
{
    let num_vertices = mesh.vertices().len();
    let mut a = DMatrix::zeros(num_vertices, num_vertices);
    let mut b = DVector::zeros(num_vertices);

    // Pick the right reference element based on the element type in the mesh.
    let ref_element = match mesh.element_type() {
        ElementType::P1 => ReferenceElement::Tri3,
        ElementType::Q1 => ReferenceElement::Quad4,
    };

    // Pick the right quadrature rule based on the element type in the mesh.
    // We use second-order quadrature rules by default.
    let quad_rule = match mesh.element_type() {
        ElementType::P1 => QuadRule::triangle(2),
        ElementType::Q1 => QuadRule::quadrilateral(2),
    };

    let n: usize = ref_element.num_nodes();
    for element in mesh.elements() {
        // Get the coordinates of the element nodes
        let mut nodes: Vec<Point2<f64>> = Vec::with_capacity(n);
        for vid in &element.indices {
            let vertex = mesh.vertices()[*vid];
            nodes.push(vertex);
        }

        // Compute the local stiff and load vectors
        let mut ke = vec![vec![0.0; n]; n];
        let mut fe = vec![0.0; n];
        for (quad_points, quad_weights) in quad_rule.points.iter().zip(quad_rule.weights.iter()) {
            // Compute local quantities in the reference element
            let grads_ref = ref_element.shape_gradients(quad_points);
            let jac_ref = ref_element.jacobian(&nodes, quad_points);
            let det_jac_ref = jac_ref.determinant();
            let jac_inv_t = jac_ref.try_inverse().unwrap().transpose();

            // Compute gradient in the physical space
            let mut grads_global: Vec<Vector2<f64>> = Vec::with_capacity(n);
            for grad_ref in grads_ref {
                let grad = jac_inv_t * grad_ref;
                grads_global.push(grad);
            }

            // Evaluate physical coordinates of quadrature point
            let shape_vals = ref_element.shape_functions(quad_points);
            let mut x = 0.0;
            let mut y = 0.0;
            for (val, vtx) in shape_vals.iter().zip(&nodes) {
                x += val * vtx.x;
                y += val * vtx.y;
            }

            // Fill ke and fe
            let f_val = source_fn(x, y);
            let weight = quad_weights * det_jac_ref.abs();
            for i in 0..n {
                for j in 0..n {
                    ke[i][j] += grads_global[i].dot(&grads_global[j]) * weight;
                }
                fe[i] += shape_vals[i] * f_val * weight;
            }
        }

        // Assemble into global matrix/vector
        for (local_i, &global_i) in element.indices.iter().enumerate() {
            b[global_i] += fe[local_i];
            for (local_j, &global_j) in element.indices.iter().enumerate() {
                a[(global_i, global_j)] += ke[local_i][local_j];
            }
        }
    }

    (a, b)
}

/// Function that assembles the FEM using a sparse matrix.
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

/// Function that applies Dirichlet boundary conditions to the dense FEM system.
pub fn _apply_dirichlet_dense() {}

/// Function that applies Dirichlet boundary conditions to the sparse FEM system.
pub fn _apply_dirichlet_sparse() {}

/// Function that solves the dense FEM system.
pub fn _solve_dense(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    let chol = a.clone().cholesky()?;
    Some(chol.solve(b))
}

/// Function that solves the sparse FEM system.
pub fn _solve_sparse(a: &CsrMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    conjugate_gradient::solve(a, b, 1000, 1e-10)
}
