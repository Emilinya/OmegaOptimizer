use itertools::izip;
use nalgebra::{SMatrix, SVector};
use std::rc::Rc;

use crate::functions::{Differentiated, Functions};

type EBox<const D: usize> = Box<dyn Fn(&SVector<f64, D>) -> f64>;
type EGradBox<const D: usize> = Box<dyn Fn(&SVector<f64, D>) -> SVector<f64, D>>;
type EHessBox<const D: usize> = Box<dyn Fn(&SVector<f64, D>) -> SMatrix<f64, D, D>>;

pub struct ErrorFunction<const D: usize> {
    pub f: EBox<D>,
    pub grad: EGradBox<D>,
    pub hess: EHessBox<D>,
}

/// Computes the outer product of a column vector
#[inline]
pub fn outer<const D: usize>(vector: &SVector<f64, D>) -> SMatrix<f64, D, D> {
    vector * vector.transpose()
}

pub fn get_error_functions<const D: usize, F: Differentiated<D>>(
    x_ray: Rc<Vec<f64>>,
    y_ray: Rc<Vec<f64>>,
) -> ErrorFunction<D> {
    let ray_len = x_ray.len() as f64;

    let (error_x_ray, error_y_ray) = (x_ray.clone(), y_ray.clone());
    let error = Box::new(move |a_ray: &SVector<f64, D>| -> f64 {
        let mut sum = 0.0;
        for (x, y) in izip!(error_x_ray.clone().iter(), error_y_ray.clone().iter()) {
            sum += (y - F::f(*x, a_ray)).powi(2);
        }
        sum / ray_len
    });

    let (gradient_x_ray, gradient_y_ray) = (x_ray.clone(), y_ray.clone());
    let gradient = Box::new(move |a_ray: &SVector<f64, D>| -> SVector<f64, D> {
        let mut gradient = SVector::<f64, D>::zeros();
        for (x, y) in izip!(gradient_x_ray.iter(), gradient_y_ray.iter()) {
            gradient += (y - F::f(*x, a_ray)) * F::grad(*x, a_ray);
        }
        (-2.0 / ray_len) * gradient
    });

    let hessian = Box::new(move |a_ray: &SVector<f64, D>| -> SMatrix<f64, D, D> {
        let mut hess = SMatrix::<f64, D, D>::zeros();
        for (x, y) in izip!(x_ray.iter(), y_ray.iter()) {
            hess += (y - F::f(*x, a_ray)) * F::hess(*x, a_ray) - outer(&F::grad(*x, a_ray));
        }
        (-2.0 / ray_len) * hess
    });

    ErrorFunction {
        f: error,
        grad: gradient,
        hess: hessian,
    }
}

pub fn error(x_ray: &[f64], y_ray: &[f64], function: &Functions, parameters: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (x, y) in izip!(x_ray.iter(), y_ray.iter()) {
        sum += (y - function.f(*x, parameters)).powi(2);
    }
    sum / (x_ray.len() as f64)
}
