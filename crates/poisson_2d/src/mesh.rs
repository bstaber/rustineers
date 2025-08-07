use nalgebra::Point2;

#[derive(Clone, Debug)]
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
