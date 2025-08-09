use crate::element::{ElementType, ReferenceElement};
use crate::mesh::Mesh2d;
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
pub fn assemble_system_sparse<F>(mesh: &Mesh2d, source_fn: &F) -> (CsrMatrix<f64>, DVector<f64>)
where
    F: Fn(f64, f64) -> f64,
{
    let num_vertices = mesh.vertices().len();
    let mut coo = CooMatrix::new(num_vertices, num_vertices);
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

        for (local_i, &global_i) in element.indices.iter().enumerate() {
            b[global_i] += fe[local_i];
            for (local_j, &global_j) in element.indices.iter().enumerate() {
                let stiffness: f64 = ke[local_i][local_j];
                coo.push(global_i, global_j, stiffness);
            }
        }
    }

    let a = CsrMatrix::from(&coo);
    (a, b)
}

/// Function that applies Dirichlet boundary conditions to the dense FEM system.
pub fn apply_dirichlet_dense<G>(
    a: &mut DMatrix<f64>,
    b: &mut DVector<f64>,
    boundary_nodes: &[usize],
    mesh: &Mesh2d,
    g: G,
) where
    G: Fn(f64, f64) -> f64,
{
    // Compute the boundary conditions values at each boundary node
    let mut values = Vec::with_capacity(boundary_nodes.len());
    for &i in boundary_nodes {
        let v = &mesh.vertices()[i];
        values.push((i, g(v.x, v.y)));
    }
    let n = a.nrows();

    // For each boundary node j, update rhs: b_i -= a_ij * g_j for all i
    for &(j, g_j) in &values {
        for i in 0..n {
            b[i] -= a[(i, j)] * g_j;
        }
    }

    // Zero out rows and columns and set diagonal
    for &(j, g_j) in &values {
        for k in 0..n {
            a[(j, k)] = 0.0;
            a[(k, j)] = 0.0;
        }
        a[(j, j)] = 1.0;
        b[j] = g_j;
    }
}

/// Function that applies Dirichlet boundary conditions to the sparse FEM system.
pub fn apply_dirichlet_sparse<G>(
    a: &mut CsrMatrix<f64>,
    b: &mut DVector<f64>,
    boundary_nodes: &[usize],
    mesh: &Mesh2d,
    g: G,
) where
    G: Fn(f64, f64) -> f64,
{
    // Compute the boundary conditions values at each boundary node
    let mut bc_vals = Vec::with_capacity(boundary_nodes.len());
    for &j in boundary_nodes {
        let v = &mesh.vertices()[j];
        bc_vals.push((j, g(v.x, v.y)));
    }

    let n = a.nrows();

    for &(j, g_j) in &bc_vals {
        // For each boundary node j, update rhs: b_i -= a_ij * g_j for all i

        for i in 0..n {
            let row = a.row(i);
            let cols = row.col_indices();
            let vals = row.values();
            if let Some(pos) = cols.iter().position(|&c| c == j) {
                b[i] -= vals[pos] * g_j;
            }
        }

        // Zero out row j
        for v in a.row_mut(j).values_mut() {
            *v = 0.0;
        }

        // Zero out column j to preserve symmetry
        // We first collect the positions
        let mut to_zero: Vec<(usize, usize)> = Vec::new();
        for i in 0..n {
            let row_i = a.row(i);
            let cols = row_i.col_indices();
            if let Some(pos) = cols.iter().position(|&c| c == j) {
                to_zero.push((i, pos));
            }
        }
        // Zero out the collected positions
        for (i, pos) in to_zero {
            a.row_mut(i).values_mut()[pos] = 0.0;
        }

        // Set diagonal to 1.0
        if let Some(pos) = a.row(j).col_indices().iter().position(|&c| c == j) {
            a.row_mut(j).values_mut()[pos] = 1.0;
        }

        b[j] = g_j;
    }
}

/// Function that solves the dense FEM system.
pub fn dense_solver(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    let chol = a.clone().cholesky()?;
    Some(chol.solve(b))
}

/// Function that solves the sparse FEM system.
pub fn sparse_solver(a: &CsrMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    conjugate_gradient::solve(a, b, 1000, 1e-10)
}

/// Dense Poisson solver
pub fn assemble_and_solve_dense<F>(
    mesh: &Mesh2d,
    boundary_nodes: &[usize],
    boundary_fn: F,
    source_fn: F,
) -> DVector<f64>
where
    F: Fn(f64, f64) -> f64,
{
    // Assemble dense system
    let (mut a, mut b) = assemble_system_dense(mesh, &source_fn);

    // Apply BCs
    apply_dirichlet_dense(&mut a, &mut b, boundary_nodes, mesh, boundary_fn);

    // Solve linear system
    dense_solver(&a, &b).expect("failed to solve")
}

pub fn assemble_and_solve_sparse<F>(
    mesh: &Mesh2d,
    boundary_nodes: &[usize],
    boundary_fn: F,
    source_fn: F,
) -> DVector<f64>
where
    F: Fn(f64, f64) -> f64,
{
    // Assemble sparse system
    let (mut a, mut b) = assemble_system_sparse(mesh, &source_fn);

    // Apply BCs
    apply_dirichlet_sparse(&mut a, &mut b, boundary_nodes, mesh, boundary_fn);

    // Solve linear system
    sparse_solver(&a, &b).expect("failed to solve")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::Element;

    #[test]
    fn test_assemble_system_dense() {
        let vertices = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let elements = vec![Element {
            indices: vec![0, 1, 2, 3],
        }];
        let mesh = Mesh2d::new(vertices, elements, ElementType::Q1);

        let source_fn = |x: f64, y: f64| x + y;
        let (a, b) = assemble_system_dense(&mesh, &source_fn);

        assert_eq!(a.nrows(), 4);
        assert_eq!(b.len(), 4);
    }

    #[test]
    fn test_assemble_system_sparse() {
        let vertices = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let elements = vec![Element {
            indices: vec![0, 1, 2, 3],
        }];
        let mesh = Mesh2d::new(vertices, elements, ElementType::Q1);

        let source_fn = |x: f64, y: f64| x + y;
        let (a, b) = assemble_system_sparse(&mesh, &source_fn);

        assert_eq!(a.nrows(), 4);
        assert_eq!(b.len(), 4);
    }
}
