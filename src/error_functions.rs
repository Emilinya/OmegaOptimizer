use itertools::izip;
use nalgebra::{SMatrix, SVector};
use std::marker::PhantomData;

use crate::functions::{Differentiated, Functions};

/// Computes the outer product of a column vector
#[inline]
pub fn outer<const D: usize>(vector: &SVector<f64, D>) -> SMatrix<f64, D, D> {
    vector * vector.transpose()
}

pub struct ErrorFunction<const D: usize, F: Differentiated<D>> {
    x_ray: Vec<f64>,
    y_ray: Vec<f64>,
    ray_len: f64,
    function: PhantomData<F>,
}

impl<const D: usize, F: Differentiated<D>> ErrorFunction<D, F> {
    pub fn new(x_ray: &[f64], y_ray: &[f64]) -> Self {
        let ray_len = x_ray.len() as f64;
        Self {
            x_ray: x_ray.to_vec(),
            y_ray: y_ray.to_vec(),
            ray_len,
            function: PhantomData,
        }
    }

    pub fn f(&self, params: &SVector<f64, D>) -> f64 {
        let mut sum = 0.0;
        for (x, y) in izip!(self.x_ray.iter(), self.y_ray.iter()) {
            sum += (y - F::f(*x, params)).powi(2);
        }
        sum / self.ray_len
    }

    pub fn grad(&self, params: &SVector<f64, D>) -> SVector<f64, D> {
        let mut gradient = SVector::<f64, D>::zeros();
        for (x, y) in izip!(self.x_ray.iter(), self.y_ray.iter()) {
            gradient += (y - F::f(*x, params)) * F::grad(*x, params);
        }
        (-2.0 / self.ray_len) * gradient
    }

    pub fn hess(&self, params: &SVector<f64, D>) -> SMatrix<f64, D, D> {
        let mut hess = SMatrix::<f64, D, D>::zeros();
        for (x, y) in izip!(self.x_ray.iter(), self.y_ray.iter()) {
            hess += (y - F::f(*x, params)) * F::hess(*x, params) - outer(&F::grad(*x, params));
        }
        (-2.0 / self.ray_len) * hess
    }
}

pub fn error(x_ray: &[f64], y_ray: &[f64], function: &Functions, parameters: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (x, y) in izip!(x_ray.iter(), y_ray.iter()) {
        sum += (y - function.f(*x, parameters)).powi(2);
    }
    sum / (x_ray.len() as f64)
}
