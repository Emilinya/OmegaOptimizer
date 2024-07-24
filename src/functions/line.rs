use nalgebra::{Matrix2, Vector2};

use super::Differentiated;

pub struct Line;

impl Differentiated<2> for Line {
    const PARAMETER_NAMES: [&'static str; 2] = ["a", "b"];
    const NAME: &'static str = "line";

    fn f(x: f64, params: &Vector2<f64>) -> f64 {
        let (a, b) = (params.x, params.y);
        a * x + b
    }

    fn grad(x: f64, _params: &Vector2<f64>) -> Vector2<f64> {
        Vector2::new(x, 1.0)
    }

    fn hess(_x: f64, _params: &Vector2<f64>) -> Matrix2<f64> {
        Matrix2::new(0.0, 0.0, 0.0, 0.0)
    }
}
