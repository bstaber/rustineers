use nalgebra::Point2;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ElementType {
    /// 3-node triangle
    P1,
    /// 4-node quadrangle
    Q1,
}

/// An element stores a vector containing its global indices.
#[derive(Clone, Debug)]
pub struct Element {
    pub indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Mesh2d {
    vertices: Vec<Point2<f64>>,
    elements: Vec<Element>,
    element_type: ElementType,
}

impl Mesh2d {
    pub fn new(
        vertices: Vec<Point2<f64>>,
        elements: Vec<Element>,
        element_type: ElementType,
    ) -> Self {
        Self {
            vertices,
            elements,
            element_type,
        }
    }
    pub fn vertices(&self) -> &[Point2<f64>] {
        &self.vertices
    }

    pub fn elements(&self) -> &[Element] {
        &self.elements
    }

    pub fn element_type(&self) -> &ElementType {
        &self.element_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh2d() {
        let vertices = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let elements = vec![Element {
            indices: vec![0, 1, 2, 3],
        }];
        let mesh = Mesh2d {
            vertices,
            elements,
            element_type: ElementType::Q1,
        };

        assert_eq!(mesh.vertices().len(), 4);
        assert_eq!(mesh.elements().len(), 1);
        assert_eq!(*mesh.element_type(), ElementType::Q1);
    }
}
