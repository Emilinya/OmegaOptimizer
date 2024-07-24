use nalgebra::{Matrix4, RowVector4, Vector4};

use super::Differentiated;

pub struct Sqrt;

impl Differentiated<4> for Sqrt {
    const PARAMETER_NAMES: [&'static str; 4] = ["a", "b", "c", "d"];
    const NAME: &'static str = "sqrt";

    fn f(x: f64, params: &Vector4<f64>) -> f64 {
        let (a, b, c, d) = (params.x, params.y, params.z, params.w);
        a * (b * x + c).abs().sqrt() + d
    }

    fn grad(x: f64, params: &Vector4<f64>) -> Vector4<f64> {
        let (a, b, c, _d) = (params.x, params.y, params.z, params.w);
        let sign = (b * x + c).signum();
        let sqrt = (b * x + c).abs().sqrt();
        let hisqrt = 0.5 * sign / sqrt;
        Vector4::new(sqrt, a * x * hisqrt, a * hisqrt, 1.0)
    }

    fn hess(x: f64, params: &Vector4<f64>) -> Matrix4<f64> {
        let (a, b, c, _d) = (params.x, params.y, params.z, params.w);
        let sign = (b * x + c).signum();
        let isqrt = sign / (b * x + c).abs().sqrt();
        let isqrt3 = isqrt.powi(3);

        let hisqrt = 0.5 * isqrt;
        let qisqrt3 = 0.25 * isqrt3;
        Matrix4::from_rows(&[
            RowVector4::new(0.0, x * hisqrt, hisqrt, 0.0),
            RowVector4::new(x * hisqrt, -a * x * x * qisqrt3, -a * x * qisqrt3, 0.0),
            RowVector4::new(hisqrt, -a * x * qisqrt3, -a * qisqrt3, 0.0),
            RowVector4::new(0.0, 0.0, 0.0, 0.0),
        ])
    }
}
