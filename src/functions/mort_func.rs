use nalgebra::{Matrix4, RowVector4, Vector4};

use super::Differentiated;

pub struct MortFunc;

impl Differentiated<4> for MortFunc {
    const PARAMETER_NAMES: [&'static str; 4] = ["a", "b", "c", "n"];
    const NAME: &'static str = "mort_func";

    fn f(x: f64, params: &Vector4<f64>) -> f64 {
        let (a, b, c, n) = (params.x, params.y, params.z, params.w);
        a * x.powf(n) / (b * x.powf(n) + 1.0) + c
    }

    fn grad(x: f64, params: &Vector4<f64>) -> Vector4<f64> {
        let (a, b, _c, n) = (params.x, params.y, params.z, params.w);
        let xn = x.powf(n);

        Vector4::new(
            xn / (b * xn + 1.0),
            -a * (xn / (b * xn + 1.0)).powi(2),
            1.0,
            a * xn * x.ln() / (b * xn + 1.0).powi(2),
        )
    }

    fn hess(x: f64, params: &Vector4<f64>) -> Matrix4<f64> {
        let (a, b, _c, n) = (params.x, params.y, params.z, params.w);
        let xn = x.powf(n);
        let xln = x.ln();

        let h11 = 0.0;
        let h12 = -(xn / (b * xn + 1.0)).powi(2);
        let h13 = 0.0;
        let h14 = xln * xn / (b * xn + 1.0).powi(2);

        let h22 = 2.0 * a * (xn / (b * xn + 1.0)).powi(3);
        let h23 = 0.0;
        let h24 = -2.0 * a * xln * xn.powi(2) / (b * xn + 1.0).powi(3);

        let h33 = 0.0;
        let h34 = 0.0;

        let h44 = -xln.powi(2) * xn * (b * xn - 1.0) * a / (b * xn + 1.0).powi(3);

        Matrix4::from_rows(&[
            RowVector4::new(h11, h12, h13, h14),
            RowVector4::new(h12, h22, h23, h24),
            RowVector4::new(h13, h23, h33, h34),
            RowVector4::new(h14, h24, h34, h44),
        ])
    }
}
