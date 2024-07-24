use nalgebra::{Matrix3, RowVector3, Vector3};

use super::Differentiated;

pub struct Normal;

impl Differentiated<3> for Normal {
    const PARAMETER_NAMES: [&'static str; 3] = ["a", "μ", "σ"];
    const NAME: &'static str = "normal";

    fn f(x: f64, params: &Vector3<f64>) -> f64 {
        let (a, x0, sigma) = (params.x, params.y, params.z);
        a * (-0.5 * (((x - x0) / sigma).powi(2))).exp()
    }

    fn grad(x: f64, params: &Vector3<f64>) -> Vector3<f64> {
        let (a, x0, sigma) = (params.x, params.y, params.z);
        let core = (x - x0) / sigma;
        let exp = (-0.5 * core.powi(2)).exp();
        Vector3::new(exp, a * exp * core / sigma, a * exp * core.powi(2) / sigma)
    }

    fn hess(x: f64, params: &Vector3<f64>) -> Matrix3<f64> {
        let (a, x0, sigma) = (params.x, params.y, params.z);
        let core = (x - x0) / sigma;
        let core2 = core.powi(2);
        let exp = (-0.5 * core2).exp();

        let h11 = 0.0;
        let h12 = exp * core / sigma;
        let h13 = exp * core2 / sigma;

        let h22 = a * (core2 - 1.0) * exp / sigma.powi(2);
        let h23 = a * core * (core2 - 2.0) * exp / sigma.powi(2);

        let h33 = a * core2 * (core2 - 3.0) * exp / sigma.powi(2);

        Matrix3::from_rows(&[
            RowVector3::new(h11, h12, h13),
            RowVector3::new(h12, h22, h23),
            RowVector3::new(h13, h23, h33),
        ])
    }
}
