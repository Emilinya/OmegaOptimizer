use crate::error_functions::ErrorFunction;
use log::info;
use nalgebra::SVector;
use std::convert::Into;

type MinimizerOut<const D: usize, E> = (SVector<f64, D>, Option<E>);

#[derive(Debug)]
enum NewtonError {
    DidNotConverge,
    NotPositiveDefinite,
    Singular,
    StepIncreased,
}

impl From<NewtonError> for String {
    fn from(value: NewtonError) -> Self {
        match value {
            NewtonError::DidNotConverge => "Iteration did not converge",
            NewtonError::NotPositiveDefinite => "Hessian is not positive-definite",
            NewtonError::Singular => "Hessian is singular",
            NewtonError::StepIncreased => "Step increased error",
        }
        .into()
    }
}

fn convertify<const D: usize>(out: MinimizerOut<D, NewtonError>) -> MinimizerOut<D, String> {
    let (v, e) = out;
    (v, e.map(Into::into))
}

fn newton_descent<const D: usize>(
    x0: &SVector<f64, D>,
    function: &ErrorFunction<D>,
    max_steps: usize,
) -> MinimizerOut<D, NewtonError> {
    let threshold = 1e-12;

    let f = &function.f;
    let grad_f = &function.grad;
    let hess_f = &function.hess;

    // TODO: See if the descent will find minima or maxima
    if hess_f(x0).cholesky().is_none() {
        return (*x0, Some(NewtonError::NotPositiveDefinite));
    }

    let mut prev_f = f64::INFINITY;
    let mut x = *x0;
    for _ in 0..max_steps {
        let g = grad_f(&x);
        if g.dot(&g).sqrt() < threshold {
            info!("Newton did it!");
            return (x, None);
        }

        let Some(inv_hess) = hess_f(&x).try_inverse() else {
            return (x, Some(NewtonError::Singular));
        };

        // ensure step decreases function value by damping step if it does not
        let mut damping = 1.0;
        loop {
            let next_x = x - damping * inv_hess * g;

            if f(&next_x) < prev_f {
                x = next_x;
                prev_f = f(&x);
                break;
            } else {
                damping *= 0.5;
            }

            if damping < f64::EPSILON {
                return (x, Some(NewtonError::StepIncreased));
            }
        }
    }

    (x, Some(NewtonError::DidNotConverge))
}

struct BacktrackArgs {
    c: f64,
    tau: f64,
    alpha_0: f64,
}

impl BacktrackArgs {
    fn new(c: f64, tau: f64, alpha_0: f64) -> Self {
        Self { c, tau, alpha_0 }
    }

    fn default() -> Self {
        Self::new(0.5, 0.5, 1.0)
    }
}

#[derive(Debug)]
enum BacktrackError {
    DidNotConverge,
    TooSmallAlpha,
}

fn backtrack_descent<const D: usize>(
    x0: &SVector<f64, D>,
    function: &ErrorFunction<D>,
    max_steps: usize,
    backtrack_args: &BacktrackArgs,
) -> MinimizerOut<D, BacktrackError> {
    let threshold = 1e-12;

    let f = &function.f;
    let grad_f = &function.grad;

    let c = backtrack_args.c;
    let tau = backtrack_args.tau;
    let alpha_0 = backtrack_args.alpha_0;

    let mut x = *x0;
    let mut prev_alpha = alpha_0;

    for _ in 0..max_steps {
        let g = grad_f(&x);
        let g_norm = g.dot(&g).sqrt();
        if g_norm < threshold {
            info!("Backtrack did it!");
            return (x, None);
        }

        let f_val = f(&x);
        let t = c * g_norm.powi(2);
        let mut alpha = prev_alpha;

        let accept = |alpha: f64| f_val - f(&(x - alpha * g)) >= alpha * t;

        // try to increase alpha in case previous value is too small
        let mut increased_alpha = false;
        while accept(alpha) {
            increased_alpha = true;
            alpha /= tau;
        }

        // we now know that accept(alpha) == False
        alpha *= tau;

        if !increased_alpha {
            // decrease alpha until it is acceptable
            while !accept(alpha) && alpha > f64::EPSILON {
                alpha *= tau;
            }
        }

        prev_alpha = alpha;

        if alpha <= f64::EPSILON {
            return (x, Some(BacktrackError::TooSmallAlpha));
        }

        // gradient descent using optimal step size
        x -= alpha * g;
    }

    (x, Some(BacktrackError::DidNotConverge))
}

pub fn combined_descent<const D: usize>(
    x0: &SVector<f64, D>,
    function: &ErrorFunction<D>,
) -> MinimizerOut<D, String> {
    const STEP_COUNTS: [usize; 4] = [10, 100, 1000, 10_000];

    const MIN_STEPS: usize = STEP_COUNTS[0];
    const MAX_STEPS: usize = STEP_COUNTS[STEP_COUNTS.len() - 1];

    let f = &function.f;

    let mut best_params = *x0;
    let mut best_f = f(&best_params);

    for step_count in STEP_COUNTS {
        info!("Trying {} iterations...", step_count);

        // try a quick backtrack minimization to find approximate minimum
        let backtrack_out = match backtrack_descent(
            &best_params,
            function,
            step_count,
            &BacktrackArgs::default(),
        ) {
            (p, None) => return (p, None),
            (p, Some(BacktrackError::DidNotConverge)) => p,
            (_, Some(BacktrackError::TooSmallAlpha)) => {
                return (best_params, Some("Backtrack can't improve".into()))
            }
        };
        let f_backtrack = f(&backtrack_out);
        if f_backtrack > best_f {
            return (best_params, Some("Backtrack increased error?!".into()));
        }

        // try to use newton to improve result
        let (newton_out, newton_error) = newton_descent(&backtrack_out, function, MIN_STEPS);
        if f(&newton_out) < f_backtrack {
            match newton_error {
                None => return (newton_out, None),
                Some(NewtonError::DidNotConverge) => {
                    return convertify(newton_descent(&newton_out, function, MAX_STEPS))
                }
                Some(_) => {
                    best_params = newton_out;
                    best_f = f(&newton_out);
                    continue;
                }
            }
        }

        best_params = backtrack_out;
        best_f = f_backtrack;
    }

    (best_params, Some("Combined descent never converged".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        error_functions::get_error_functions,
        functions::{line::Line, Differentiated},
    };
    use core::f64::consts::{E, PI};
    use nalgebra::Vector2;

    #[test]
    fn test_newton_descent() {
        test_minimizer(Mode::Newton);
    }

    #[test]
    fn test_backtrack_descent() {
        test_minimizer(Mode::Backtrack);
    }

    #[test]
    fn test_combined_descent() {
        test_minimizer(Mode::Combined);
    }

    fn test_minimizer(mode: Mode) {
        let parameters = Vector2::new(E, PI);

        // Create data with no noise, as we then should get 'parameters' exactly.
        const N: usize = 100;
        let mut x_ray = Vec::with_capacity(N);
        let mut y_ray = Vec::with_capacity(N);
        for i in 0..N {
            let x = i as f64 / ((N - 1) as f64);
            x_ray.push(x);
            y_ray.push(Line::f(x, &parameters));
        }

        let p0 = Vector2::from_element(1.0);
        let error_function = get_error_functions::<2, Line>(x_ray.into(), y_ray.into());
        let (optimal_parameters, error) = match mode {
            Mode::Backtrack => {
                let (p, e) =
                    backtrack_descent(&p0, &error_function, 1000, &BacktrackArgs::default());
                (p, e.map(|e| format!("{:?}", e)))
            }
            Mode::Newton => {
                let (p, e) = newton_descent(&p0, &error_function, 10);
                (p, e.map(|e| format!("{:?}", e)))
            }
            Mode::Combined => combined_descent(&p0, &error_function),
        };

        if let Some(error) = error {
            panic!("{:?} got error: {:?}", mode, error);
        }

        let threshold = match mode {
            Mode::Backtrack => 1e-10,
            Mode::Newton => 1e-14,
            Mode::Combined => 0.0,
        };

        if (optimal_parameters - parameters).abs().max() > threshold {
            panic!(
                "{:?} got wrong parameters! {:?} > {}",
                mode,
                (optimal_parameters - parameters).abs(),
                threshold,
            );
        }
    }

    #[derive(Debug)]
    enum Mode {
        Backtrack,
        Newton,
        Combined,
    }
}
