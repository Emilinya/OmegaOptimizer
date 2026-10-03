#[cfg(test)]
use nalgebra::{DMatrix, DVector};
use nalgebra::{SMatrix, SVector};
use strum::VariantNames;
use strum_macros::{EnumIter, EnumString, VariantNames};

use std::{path::PathBuf, str::FromStr};

use crate::utils::prettify_list;
use crate::{OptimizinateResult, optimizinate};

pub trait Differentiated<const D: usize> {
    const PARAMETER_NAMES: [&'static str; D];
    const NAME: &'static str;

    fn f(x: f64, params: &SVector<f64, D>) -> f64;

    fn grad(x: f64, params: &SVector<f64, D>) -> SVector<f64, D>;

    fn hess(x: f64, params: &SVector<f64, D>) -> SMatrix<f64, D, D>;
}

macro_rules! create_function_enum {
    ($($file:ident::$typename:ident<$D:literal>),*,) => {
        $(pub mod $file);*;

        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumString, EnumIter, VariantNames)]
        #[strum(serialize_all = "snake_case")]
        pub enum Functions {
            $($typename),*
        }

        impl Functions {
            /// Tries to create a function from a function name, returns a string with
            /// a descriptive error message if the function name is invalid.
            pub fn descriptive_from_str(s: &str) -> Result<Functions, String> {
                Self::from_str(&s.to_lowercase()).or_else(|_| Err(format!(
                    "Got malformed function name '{}'. Legal \
                    function names are {}.", s, prettify_list(Self::VARIANTS)
                )))
            }

            pub fn parameter_count(&self) -> usize {
                match self {
                    $(Self::$typename => $D),*
                }
            }

            pub fn parameter_names(&self) -> Vec<&'static str> {
                match self {
                    $(Self::$typename => $file::$typename::PARAMETER_NAMES.to_vec()),*
                }
            }

            pub fn name(&self) -> &'static str {
                match self {
                    $(Self::$typename => $file::$typename::NAME),*
                }
            }

            pub fn f(&self, x: f64, params: &[f64]) -> f64 {
                match self {
                    $(Self::$typename => {
                        let params = SVector::<f64, $D>::from_column_slice(params);
                        $file::$typename::f(x, &params)
                    }),*
                }
            }

            #[cfg(test)]
            fn grad(&self, x: f64, params: &[f64]) -> DVector<f64> {
                match self {
                    $(Self::$typename => {
                        let params = SVector::<f64, $D>::from_column_slice(params);
                        DVector::from_column_slice($file::$typename::grad(x, &params).data.as_slice())
                    }),*
                }
            }

            #[cfg(test)]
            fn hess(&self, x: f64, params: &[f64]) -> DMatrix<f64> {
                match self {
                    $(Self::$typename => {
                        let params = SVector::<f64, $D>::from_column_slice(params);
                        DMatrix::from_row_slice($D, $D, $file::$typename::hess(x, &params).data.as_slice())
                    }),*
                }
            }

            pub fn optimizinate(
                &self,
                datafile: &PathBuf,
                initial_parameter_opt: Option<&[f64]>,
                plot_result: bool,
            ) -> OptimizinateResult {
                match self {
                    $(Self::$typename => {
                        let initial_parameters = if let Some(parameters) = initial_parameter_opt {
                            SVector::<f64, $D>::from_vec(parameters.to_vec())
                        } else {
                            SVector::<f64, $D>::from_element(1.0)
                        };
                        optimizinate::<$D, $file::$typename>(
                            datafile, initial_parameters, plot_result
                        )
                    }),*
                }
            }
        }
    };
}

create_function_enum!(
    line::Line<2>,
    sine::Sine<4>,
    sqrt::Sqrt<4>,
    normal::Normal<3>,
    decay::Decay<2>,
    mort_func::MortFunc<4>,
);

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SVD;
    use rand::prelude::{SeedableRng, StdRng};
    use rand_distr::{Distribution, Uniform};
    use std::ops::{Index, Sub};

    use crate::utils::format_vector;

    #[test]
    fn test_descriptive_from_str() {
        for name in Functions::VARIANTS {
            assert!(Functions::descriptive_from_str(name).is_ok());
        }

        let err = Functions::descriptive_from_str("does_not_exist");
        assert!(err.is_err());
        assert!(err.err().unwrap().contains("Legal function names are"));
    }

    #[test]
    fn test_gradients() {
        test_derivative(Mode::Gradient);
    }

    #[test]
    fn test_hessians() {
        test_derivative(Mode::Hessian);
    }

    fn test_derivative(mode: Mode) {
        let mut rng = StdRng::seed_from_u64(80085);
        let uniform = Uniform::new(0.0, 1.0).unwrap();

        let mut should_panic = false;
        for name in Functions::VARIANTS {
            let x = uniform.sample(&mut rng);
            let parameters: Vec<f64> = uniform.sample_iter(&mut rng).take(10).collect();
            let function = Functions::from_str(name).unwrap();

            let (derivative, size) = match mode {
                Mode::Gradient => {
                    let grad = function.grad(x, &parameters);
                    let size = grad.len();
                    (ModeContainer::Gradient(grad), size)
                }
                Mode::Hessian => {
                    let hess = function.hess(x, &parameters);
                    let size = hess.nrows();
                    (ModeContainer::Hessian(hess), size)
                }
            };

            let true_params: Vec<f64> = parameters.iter().copied().take(size).collect();

            let hf = match mode {
                Mode::Gradient => 1.0,
                Mode::Hessian => 0.5,
            };
            let h_values: Vec<f64> = (2..=6).map(|i| 10_f64.powf(-i as f64 * hf)).collect();

            let mut error_values = Vec::<ModeContainer>::with_capacity(h_values.len());
            for h in h_values.iter() {
                let numeric_derivative = match mode {
                    Mode::Gradient => ModeContainer::Gradient(get_numeric_gradient(
                        &function,
                        x,
                        &true_params,
                        *h,
                    )),
                    Mode::Hessian => {
                        ModeContainer::Hessian(get_numeric_hessian(&function, x, &true_params, *h))
                    }
                };

                error_values.push((numeric_derivative - &derivative).abs());
            }

            let total_size = match mode {
                Mode::Gradient => size,
                Mode::Hessian => size * size,
            };

            let mut convergence_rates = Vec::<f64>::with_capacity(total_size);
            for i in 0..total_size {
                if error_values[0][i] < 1e-12 {
                    convergence_rates.push(999.0);
                    continue;
                }

                let error_vector: Vec<f64> = error_values.iter().map(|v| v[i]).collect();
                convergence_rates.push(get_convergence(&h_values, &error_vector));
            }

            const THRESHOLD: f64 = 0.75;
            if convergence_rates.iter().any(|v| v < &THRESHOLD) {
                println!("{} of {} is implemented incorrectly!", mode.name(), name);
                for (i, r) in convergence_rates
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v < &&THRESHOLD)
                {
                    let errors: Vec<f64> = error_values.iter().map(|v| v[i]).collect();
                    println!(
                        "  Value at index {} is implemented incorrectly! \
                        Got convergence rate of {:.5}, expected {}.",
                        mode.to_index_str(i, size),
                        r,
                        THRESHOLD
                    );
                    println!("  The values are {:?}.\n", format_vector(&errors, 3));
                }
                should_panic = true;
            }
        }
        if should_panic {
            panic!("Some {}s are implemented incorrectly!", mode.name());
        }
    }

    fn get_numeric_gradient(function: &Functions, x: f64, params: &[f64], h: f64) -> DVector<f64> {
        let f = function.f(x, params);
        let size = params.len();

        let mut p_plus_dx = params.to_vec();
        let mut numeric_gradient = DVector::zeros(size);
        for j in 0..size {
            p_plus_dx[j] += h;
            let f_dx = function.f(x, &p_plus_dx);

            numeric_gradient[j] = (f_dx - f) / h;
            p_plus_dx[j] -= h;
        }

        numeric_gradient
    }

    fn get_numeric_hessian(function: &Functions, x: f64, params: &[f64], h: f64) -> DMatrix<f64> {
        let f = function.f(x, params);
        let size = params.len();

        let mut p_plus_step = params.to_vec();
        let mut numeric_hessian = DMatrix::zeros(size, size);
        for yi in 0..size {
            p_plus_step[yi] += h;
            let f_dy = function.f(x, &p_plus_step);

            for xi in 0..size {
                p_plus_step[xi] += h;
                let f_dxdy = function.f(x, &p_plus_step);

                p_plus_step[yi] -= h;
                let f_dx = function.f(x, &p_plus_step);

                numeric_hessian[(yi, xi)] = (f_dxdy - f_dx - f_dy + f) / h.powi(2);
                p_plus_step[xi] -= h;
                p_plus_step[yi] += h;
            }
            p_plus_step[yi] -= h;
        }

        numeric_hessian
    }

    /// fits linear polynomial to the points (ln(x), ln(y)) and returns the slope.
    fn get_convergence(x_values: &[f64], y_values: &[f64]) -> f64 {
        const DEGREE: usize = 1;

        let log_x_values: Vec<f64> = x_values.iter().map(|v| v.ln()).collect();
        let log_y_values: Vec<f64> = y_values.iter().map(|v| v.ln()).collect();

        let ncols = DEGREE + 1;
        let nrows = x_values.len();
        let mut a = DMatrix::zeros(nrows, ncols);

        for (row, &log_x) in log_x_values.iter().enumerate() {
            a[(row, 0)] = 1.0;
            for col in 1..ncols {
                a[(row, col)] = log_x.powi(col as i32);
            }
        }

        let b = DVector::from_vec(log_y_values);
        let a_svd = SVD::new(a, true, true);

        match a_svd.solve(&b, 1e-18) {
            Ok(mat) => mat[1],
            Err(error) => panic!("{}", error),
        }
    }

    enum Mode {
        Gradient,
        Hessian,
    }

    impl Mode {
        fn to_index_str(&self, i: usize, size: usize) -> String {
            match self {
                Mode::Gradient => format!("{}", i),
                Mode::Hessian => {
                    let (yi, xi) = (i / size, i % size);
                    format!("({}, {})", xi, yi)
                }
            }
        }

        fn name(&self) -> String {
            match self {
                Mode::Gradient => "gradient",
                Mode::Hessian => "hessian",
            }
            .into()
        }
    }

    enum ModeContainer {
        Gradient(DVector<f64>),
        Hessian(DMatrix<f64>),
    }

    impl Sub<&Self> for ModeContainer {
        type Output = ModeContainer;

        fn sub(self, rhs: &Self) -> Self::Output {
            match self {
                Self::Gradient(lv) => match rhs {
                    Self::Gradient(rv) => Self::Gradient(lv - rv),
                    Self::Hessian(_) => panic!("Cant subtract hessian from gradient!"),
                },
                Self::Hessian(lv) => match rhs {
                    Self::Hessian(rv) => Self::Hessian(lv - rv),
                    Self::Gradient(_) => panic!("Cant subtract gradient from hessian!"),
                },
            }
        }
    }

    impl Index<usize> for ModeContainer {
        type Output = f64;

        fn index(&self, index: usize) -> &Self::Output {
            match self {
                Self::Gradient(v) => &v[index],
                Self::Hessian(v) => &v[index],
            }
        }
    }

    impl ModeContainer {
        fn abs(self) -> Self {
            match self {
                ModeContainer::Gradient(v) => ModeContainer::Gradient(v.abs()),
                ModeContainer::Hessian(v) => ModeContainer::Hessian(v.abs()),
            }
        }
    }
}
