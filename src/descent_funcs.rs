use ndarray::prelude::*;
use ndarray_linalg::cholesky::*;
use ndarray_stats::QuantileExt;

type FType = dyn Fn(&Array1<f64>) -> f64;
type GradFType = dyn Fn(&Array1<f64>) -> Array1<f64>;
type HessFType = dyn Fn(&Array1<f64>) -> Array2<f64>;

pub fn varying_gradient_descent(
	x0: &Array1<f64>, f: &FType, grad_f: &GradFType, max_steps: i32
) -> Array1<f64> {
    let epsilon = 1e-15;

	let mut dt = 1.0;
    let dt_diffs = Array::linspace(-2., 2., 5).map(|v| 10f64.powf(*v));
    
	let mut x = x0.to_owned();
	for _ in 0..max_steps {
		let v = -grad_f(&x);
		if v.dot(&v).sqrt() < epsilon {
			break;
        }

		loop {
			let prev_f = f(&(&x + dt * &v));

			let f_diffs = dt_diffs.mapv(|dt_diff| f(&(&x + dt * dt_diff * &v)) - prev_f);
			let min_idx = f_diffs.argmin_skipnan().unwrap();
			let dt_diff = dt_diffs[min_idx];

			if dt_diff == 1.0 {
				break;
            }
			
			dt *= dt_diff;
			if dt < 1e-14 {
				return x;
            }
        }

		x = &x + dt * &v;
    }

	return x;
}

pub fn newton_descent(
	x0: &Array1<f64>, grad_f: &GradFType, inv_hess_f: &HessFType, max_steps: i32
) -> Array1<f64> {
    fn matrix_sign(mat: &Array2<f64>) -> f64 {
		match mat.cholesky(UPLO::Lower) {
			Ok(_) => { return 1_f64 },
			Err(_) => { return -1_f64 },
		}
	}

	let epsilon = 1e-15;

	let mut x = x0.to_owned();
	for _ in 0..max_steps {
		let v = -grad_f(&x);
		let inv_hess = inv_hess_f(&x);

		let dx = matrix_sign(&inv_hess)*(inv_hess.dot(&v));

		if v.dot(&v).sqrt() < epsilon {
			break;
        }

		x = x + dx;
	}
	return x;
}

pub fn combined_descent(
	x0: &Array1<f64>, f: &FType, grad_f: &GradFType, inv_hess_f: &HessFType, print: bool
) -> Array1<f64> {
	if print {
		println!("I: {}, err={}", x0, f(&x0));
	}
	let mut improved_x = varying_gradient_descent(x0, f, grad_f, 25);
	if print {
		println!("G: {}, err={}", improved_x, f(&improved_x));
	}
	if f(&improved_x) > f(x0) {
		improved_x = x0.to_owned();
	}

	let mut optimal_x = newton_descent(&improved_x, grad_f, inv_hess_f, 50);
	if print {
		println!("N: {}, err={}", optimal_x, f(&optimal_x));
	}

	if f(&optimal_x) > f(&improved_x) {
		// Newton increased error, try once more with many more steps
		improved_x = varying_gradient_descent(&improved_x, f, grad_f, 1000);
		if print {
			println!("G: {}, err={}", improved_x, f(&improved_x));
		}

		optimal_x = newton_descent(&improved_x, grad_f, inv_hess_f, 50);
		if print {
			println!("N: {}, err={}", optimal_x, f(&optimal_x));
		}

		if f(&optimal_x) > f(&improved_x) {
			// Newton increased error again, give up on newton
			optimal_x = varying_gradient_descent(&improved_x, f, grad_f, 10000);
			if print {
				println!("G: {}, err={}", optimal_x, f(&optimal_x));
			}
		}
	}

	return optimal_x;
}