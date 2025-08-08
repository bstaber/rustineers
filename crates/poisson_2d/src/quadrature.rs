use nalgebra::Point2;

#[derive(Clone, Debug)]
pub struct QuadRule {
    pub points: Vec<Point2<f64>>,
    pub weights: Vec<f64>,
}

impl QuadRule {
    pub fn triangle(order: usize) -> Self {
        match order {
            1 => QuadRule {
                points: vec![Point2::new(1.0 / 3.0, 1.0 / 3.0)],
                weights: vec![0.5],
            },
            2 => QuadRule {
                points: vec![
                    Point2::new(1.0 / 6.0, 1.0 / 6.0),
                    Point2::new(2.0 / 3.0, 1.0 / 6.0),
                    Point2::new(1.0 / 6.0, 2.0 / 3.0),
                ],
                weights: vec![1.0 / 6.0; 3],
            },
            _ => panic!("triangle quadratule of order > 2 not implemented"),
        }
    }

    pub fn quadrilateral(n: usize) -> Self {
        match n {
            1 => QuadRule {
                points: vec![Point2::new(0.0, 0.0)],
                weights: vec![4.0],
            },
            2 => {
                let a = 1.0 / 3.0f64.sqrt();
                let pts = [-a, a];
                let mut points = Vec::with_capacity(4);
                let mut weights = Vec::with_capacity(4);
                for xi in pts {
                    for eta in pts {
                        points.push(Point2::new(xi, eta));
                        weights.push(1.0);
                    }
                }
                QuadRule { points, weights }
            }
            _ => panic!("quadrilateral quadrature with n > 2 points not implemented"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_quadrature() {
        let rule = QuadRule::triangle(2);
        assert_eq!(rule.points.len(), 3);
        assert_eq!(rule.weights.len(), 3);
    }

    #[test]
    fn test_quadrilateral_quadrature() {
        let rule = QuadRule::quadrilateral(2);
        assert_eq!(rule.points.len(), 4);
        assert_eq!(rule.weights.len(), 4);
    }
}
