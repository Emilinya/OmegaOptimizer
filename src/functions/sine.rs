use nalgebra::{Matrix4, RowVector4, Vector4};

use super::Differentiated;

pub struct Sine;

impl Differentiated<4> for Sine {
    const PARAMETER_NAMES: [&'static str; 4] = ["ω", "φ", "a", "b"];
    const NAME: &'static str = "sine";

    fn f(t: f64, params: &Vector4<f64>) -> f64 {
        let (omega, phi, a, b) = (params.x, params.y, params.z, params.w);
        a * (omega * t + phi).sin() + b
    }

    fn grad(t: f64, params: &Vector4<f64>) -> Vector4<f64> {
        let (omega, phi, a, _b) = (params.x, params.y, params.z, params.w);
        let (sin, cos) = (omega * t + phi).sin_cos();
        Vector4::new(a * t * cos, a * cos, sin, 1.0)
    }

    fn hess(t: f64, params: &Vector4<f64>) -> Matrix4<f64> {
        let (omega, phi, a, _b) = (params.x, params.y, params.z, params.w);
        let (sin, cos) = (omega * t + phi).sin_cos();
        Matrix4::from_rows(&[
            RowVector4::new(-a * t * t * sin, -a * t * sin, t * cos, 0.0),
            RowVector4::new(-a * t * sin, -a * sin, cos, 0.0),
            RowVector4::new(t * cos, cos, 0.0, 0.0),
            RowVector4::new(0.0, 0.0, 0.0, 0.0),
        ])
    }
}
