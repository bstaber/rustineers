use nalgebra::Point2;

#[derive(Clone, Debug)]
pub enum ElementType {
    Triangle,
    Quadrilateral,
}

#[derive(Clone, Debug)]
pub struct Element {
    pub indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Mesh2d {
    vertices: Vec<Point2<f64>>,
    elements: Vec<Element>,
}

impl Mesh2d {
    pub fn vertices(&self) -> &[Point2<f64>] {
        &self.vertices
    }

    pub fn elements(&self) -> &[Element] {
        &self.elements
    }
}
