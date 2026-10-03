use itertools::izip;
use ndarray::prelude::*;

type InFType = fn(f64, &Array1<f64>) -> f64;
type InGradFType = fn(f64, &Array1<f64>) -> Array1<f64>;
type InHessFType = fn(f64, &Array1<f64>) -> Array2<f64>;

fn outer_self(array: &Array1<f64>) -> Array2<f64> {
    let mut outer: Array2<f64> = Array::zeros((array.len(), array.len()));
    for (j, vy) in array.iter().enumerate() {
        for (i, vx) in array.iter().enumerate() {
            outer[(j, i)] = vx * vy;
        }
    }
    outer
}

type FunctionBox<T> = Box<dyn Fn(&Array1<f64>) -> T>;

pub fn calc_err(x_ray: Array1<f64>, y_ray: Array1<f64>, f: InFType) -> FunctionBox<f64> {
    let ray_len = x_ray.len();

    Box::new(move |a_ray: &Array1<f64>| -> f64 {
        let mut sum = 0f64;
        for (x, y) in izip!(&x_ray, &y_ray) {
            sum += (y - f(*x, a_ray)).powi(2) / ray_len as f64;
        }
        sum
    })
}

pub fn calc_grad_err(
    x_ray: Array1<f64>,
    y_ray: Array1<f64>,
    f: InFType,
    grad_f: InGradFType,
) -> FunctionBox<Array1<f64>> {
    let ray_len = x_ray.len();

    Box::new(move |a_ray: &Array1<f64>| -> Array1<f64> {
        let mut gradient: Array1<f64> = Array::zeros(a_ray.raw_dim());
        for (x, y) in izip!(&x_ray, &y_ray) {
            gradient = gradient + (y - f(*x, a_ray)) * grad_f(*x, a_ray);
        }
        -2.0 * (gradient / ray_len as f64)
    })
}

pub fn calc_hess_err(
    x_ray: Array1<f64>,
    y_ray: Array1<f64>,
    f: InFType,
    grad_f: InGradFType,
    hess_f: InHessFType,
) -> FunctionBox<Array2<f64>> {
    let ray_len = x_ray.len();

    Box::new(move |a_ray: &Array1<f64>| -> Array2<f64> {
        let mut hess: Array2<f64> = Array::zeros(hess_f(x_ray[0], a_ray).raw_dim());
        for (x, y) in izip!(&x_ray, &y_ray) {
            hess = hess + ((y - f(*x, a_ray)) * hess_f(*x, a_ray) - outer_self(&grad_f(*x, a_ray)));
        }
        (-2_f64 / ray_len as f64) * hess
    })
}
