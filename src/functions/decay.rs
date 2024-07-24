use nalgebra::{Matrix2, Vector2};

use super::Differentiated;

pub struct Decay;

impl Differentiated<2> for Decay {
    const PARAMETER_NAMES: [&'static str; 2] = ["a", "λ"];
    const NAME: &'static str = "decay";

    fn f(x: f64, params: &Vector2<f64>) -> f64 {
        let (a, l) = (params.x, params.y);
        a * (-l * x).exp()
    }

    fn grad(x: f64, params: &Vector2<f64>) -> Vector2<f64> {
        let (a, l) = (params.x, params.y);
        let exp = (-l * x).exp();
        Vector2::new(1.0, -a * x) * exp
    }

    fn hess(x: f64, params: &Vector2<f64>) -> Matrix2<f64> {
        let (a, l) = (params.x, params.y);
        let exp = (-l * x).exp();
        Matrix2::new(0.0, -x, -x, a * x * x) * exp
    }
}
