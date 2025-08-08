//! Module that implements two classical finite element types: tri3 and quad4.
use nalgebra::{Matrix2, Point2, Vector2};

#[derive(Debug, Clone)]
pub enum ReferenceElement {
    /// 3-node reference triangle
    Tri3,
    /// 4-node reference quadrangle
    Quad4,
}

impl ReferenceElement {
    pub fn num_nodes(&self) -> usize {
        match self {
            ReferenceElement::Tri3 => 3,
            ReferenceElement::Quad4 => 4,
        }
    }
}

impl ReferenceElement {
    pub fn shape_functions(&self, local_coordinates: &Point2<f64>) -> Vec<f64> {
        match self {
            ReferenceElement::Tri3 => {
                let xi = local_coordinates.x;
                let eta = local_coordinates.y;
                vec![1.0 - xi - eta, xi, eta]
            }
            ReferenceElement::Quad4 => {
                let xi = local_coordinates.x;
                let eta = local_coordinates.y;
                let n1 = 0.25 * (1.0 - xi) * (1.0 - eta);
                let n2 = 0.25 * (1.0 + xi) * (1.0 - eta);
                let n3 = 0.25 * (1.0 + xi) * (1.0 + eta);
                let n4 = 0.25 * (1.0 - xi) * (1.0 + eta);
                vec![n1, n2, n3, n4]
            }
        }
    }

    pub fn shape_gradients(&self, local_coordinates: &Point2<f64>) -> Vec<Vector2<f64>> {
        match self {
            ReferenceElement::Tri3 => {
                vec![
                    Vector2::new(-1.0, 1.0),
                    Vector2::new(1.0, 0.0),
                    Vector2::new(0.0, 1.0),
                ]
            }
            ReferenceElement::Quad4 => {
                let xi = local_coordinates.x;
                let eta = local_coordinates.y;
                let dn1_dxi = -0.25 * (1.0 - eta);
                let dn1_deta = -0.25 * (1.0 - xi);
                let dn2_dxi = 0.25 * (1.0 - eta);
                let dn2_deta = -0.25 * (1.0 + xi);
                let dn3_dxi = 0.25 * (1.0 + eta);
                let dn3_deta = 0.25 * (1.0 + xi);
                let dn4_dxi = -0.25 * (1.0 + eta);
                let dn4_deta = 0.25 * (1.0 - xi);
                vec![
                    Vector2::new(dn1_dxi, dn1_deta),
                    Vector2::new(dn2_dxi, dn2_deta),
                    Vector2::new(dn3_dxi, dn3_deta),
                    Vector2::new(dn4_dxi, dn4_deta),
                ]
            }
        }
    }

    pub fn jacobian(
        &self,
        vertices_coordinates: &[Point2<f64>],
        local_coordinates: &Point2<f64>,
    ) -> Matrix2<f64> {
        match self {
            ReferenceElement::Tri3 => {
                let v0 = vertices_coordinates[0];
                let v1 = vertices_coordinates[1];
                let v2 = vertices_coordinates[2];
                let dx_dxi = v1.x - v0.x;
                let dy_dxi = v1.y - v0.y;
                let dx_deta = v2.x - v0.x;
                let dy_deta = v2.y - v0.y;
                Matrix2::new(dx_dxi, dx_deta, dy_dxi, dy_deta)
            }
            ReferenceElement::Quad4 => {
                let grads = self.shape_gradients(local_coordinates);
                let mut jac = Matrix2::zeros();
                for (grad, vertex) in grads.iter().zip(vertices_coordinates.iter()) {
                    jac[(0, 0)] += grad.x * vertex.x;
                    jac[(1, 0)] += grad.x * vertex.y;
                    jac[(0, 1)] += grad.y * vertex.x;
                    jac[(1, 1)] += grad.y * vertex.y;
                }
                jac
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_element() {
        let tri3 = ReferenceElement::Tri3;
        let quad4 = ReferenceElement::Quad4;

        assert_eq!(tri3.num_nodes(), 3);
        assert_eq!(quad4.num_nodes(), 4);

        let local_coords = Point2::new(0.5, 0.5);
        let tri_shape_funcs = tri3.shape_functions(&local_coords);
        let quad_shape_funcs = quad4.shape_functions(&local_coords);

        assert_eq!(tri_shape_funcs.len(), 3);
        assert_eq!(quad_shape_funcs.len(), 4);
    }
}
