use itertools::izip;
use nalgebra::{DMatrix, SMatrix, SVector};

use crate::error_functions::outer;
use crate::functions::Differentiated;

/// Estimate variance of the experimental error
fn calculate_variance<const D: usize, F: Differentiated<D>>(
    x_ray: &[f64],
    y_ray: &[f64],
    parameters: &SVector<f64, D>,
) -> f64 {
    let mut variance = 0.0;
    for (x, y) in izip!(x_ray, y_ray) {
        variance += (y - F::f(*x, parameters)).powi(2);
    }
    variance / ((x_ray.len() - D) as f64)
}

fn calculate_covariance<const D: usize, F: Differentiated<D>>(
    x_ray: &[f64],
    y_ray: &[f64],
    parameters: &SVector<f64, D>,
) -> SMatrix<f64, D, D> {
    let mut outer_sum = SMatrix::<f64, D, D>::zeros();
    for x in x_ray {
        let g = F::grad(*x, parameters);
        outer_sum += outer(&g);
    }
    if outer_sum.iter().any(|v| v.is_nan()) {
        return SMatrix::<f64, D, D>::from_element(f64::NAN);
    }

    // Why must I use a DMatrix to calculate the pseudo inverse? IDK
    let outer_sum_dynamic = DMatrix::from_row_slice(D, D, outer_sum.data.as_slice());

    // I use the pseudo inverse instead of the true inverse as I found that in some cases,
    // M * M.inverse() != Identity. I don't know why this is.
    let Ok(outer_inverse_dynamic) = outer_sum_dynamic.clone().pseudo_inverse(1e-18) else {
        panic!("Sum of outer products is not invertible!");
    };

    // Back to SMatrix, yay!
    let outer_inverse = SMatrix::from_row_slice(outer_inverse_dynamic.data.as_slice());

    outer_inverse * calculate_variance::<D, F>(x_ray, y_ray, parameters)
}

pub fn get_uncertainties<const D: usize, F: Differentiated<D>>(
    x_ray: &[f64],
    y_ray: &[f64],
    parameters: &SVector<f64, D>,
) -> SVector<f64, D> {
    calculate_covariance::<D, F>(x_ray, y_ray, parameters)
        .diagonal()
        .map(|v| v.sqrt())
}
