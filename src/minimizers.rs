use log::{debug, info};
use nalgebra::SVector;

use crate::Differentiated;
use crate::error_functions::ErrorFunction;

pub enum MinimizerMessage {
    Success,
    TimedOut,
    Error(&'static str),
}
type MinimizerOut<const D: usize> = (SVector<f64, D>, MinimizerMessage);

fn newton_descent<const D: usize, F: Differentiated<D>>(
    x0: &SVector<f64, D>,
    function: &ErrorFunction<D, F>,
    max_steps: usize,
) -> MinimizerOut<D> {
    let threshold = 1e-12;

    // TODO: See if the descent will find minima or maxima
    if function.hess(x0).cholesky().is_none() {
        return (
            *x0,
            MinimizerMessage::Error("Hessian is not positive-definite!"),
        );
    }

    let mut prev_f = f64::INFINITY;
    let mut x = *x0;
    for _ in 0..max_steps {
        let g = function.grad(&x);
        if g.dot(&g).sqrt() < threshold {
            info!("Newton converged!");
            return (x, MinimizerMessage::Success);
        }

        let Some(inv_hess) = function.hess(&x).try_inverse() else {
            return (x, MinimizerMessage::Error("Hessian is singular!"));
        };

        // ensure step decreases function value by damping step if it does not
        let mut damping = 1.0;
        loop {
            let next_x = x - damping * inv_hess * g;

            if function.f(&next_x) < prev_f {
                x = next_x;
                prev_f = function.f(&x);
                break;
            } else {
                damping *= 0.5;
            }

            if damping < f64::EPSILON {
                // Should this be a success?
                info!("Newton got a damping factor of zero");
                return (x, MinimizerMessage::Success);
            }
        }
    }

    (x, MinimizerMessage::TimedOut)
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

fn backtrack_descent<const D: usize, F: Differentiated<D>>(
    x0: &SVector<f64, D>,
    function: &ErrorFunction<D, F>,
    max_steps: usize,
    backtrack_args: &BacktrackArgs,
) -> MinimizerOut<D> {
    let threshold = 1e-12;

    let c = backtrack_args.c;
    let tau = backtrack_args.tau;
    let alpha_0 = backtrack_args.alpha_0;

    let mut x = *x0;
    let mut prev_alpha = alpha_0;

    for _ in 0..max_steps {
        let g = function.grad(&x);
        let g_norm = g.dot(&g).sqrt();
        if g_norm < threshold {
            info!("Backtrack converged!");
            return (x, MinimizerMessage::Success);
        }

        let f_val = function.f(&x);
        let t = c * g_norm.powi(2);
        let mut alpha = prev_alpha;

        let accept = |alpha: f64| f_val - function.f(&(x - alpha * g)) >= alpha * t;

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
            // Should this be a success?
            info!("Backtrack got a step size of zero");
            return (x, MinimizerMessage::Success);
        }

        // gradient descent using optimal step size
        x -= alpha * g;
    }

    (x, MinimizerMessage::TimedOut)
}

pub fn combined_descent<const D: usize, F: Differentiated<D>>(
    x0: &SVector<f64, D>,
    function: &ErrorFunction<D, F>,
) -> MinimizerOut<D> {
    const STEP_COUNTS: [usize; 4] = [10, 100, 1000, 10_000];

    const MIN_STEPS: usize = STEP_COUNTS[0];
    const MAX_STEPS: usize = STEP_COUNTS[STEP_COUNTS.len() - 1];

    let mut best_params = *x0;
    let mut best_f = function.f(&best_params);

    for step_count in STEP_COUNTS {
        debug!("Trying {} iterations...", step_count);

        // try a quick backtrack minimization to find approximate minimum
        let backtrack_out = match backtrack_descent(
            &best_params,
            function,
            step_count,
            &BacktrackArgs::default(),
        ) {
            (p, MinimizerMessage::Success) => return (p, MinimizerMessage::Success),
            (p, MinimizerMessage::TimedOut) => p,
            _ => {
                return (
                    best_params,
                    MinimizerMessage::Error("Backtrack can't improve"),
                );
            }
        };
        let f_backtrack = function.f(&backtrack_out);
        if f_backtrack > best_f {
            return (
                best_params,
                MinimizerMessage::Error("Backtrack increased error?!"),
            );
        }

        // try to use newton to improve result
        let (newton_out, newton_message) = newton_descent(&backtrack_out, function, MIN_STEPS);
        if function.f(&newton_out) < f_backtrack {
            match newton_message {
                MinimizerMessage::Success => return (newton_out, MinimizerMessage::Success),
                MinimizerMessage::TimedOut => {
                    return newton_descent(&newton_out, function, MAX_STEPS);
                }
                _ => {
                    best_params = newton_out;
                    best_f = function.f(&newton_out);
                    continue;
                }
            }
        }

        best_params = backtrack_out;
        best_f = f_backtrack;
    }

    (
        best_params,
        MinimizerMessage::Error("Combined descent never converged"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::functions::line::Line;
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
        let error_function = ErrorFunction::<2, Line>::new(&x_ray, &y_ray);
        let (optimal_parameters, message) = match mode {
            Mode::Backtrack => {
                backtrack_descent(&p0, &error_function, 1000, &BacktrackArgs::default())
            }
            Mode::Newton => newton_descent(&p0, &error_function, 10),
            Mode::Combined => combined_descent(&p0, &error_function),
        };

        let MinimizerMessage::Success = message else {
            let error = match message {
                MinimizerMessage::TimedOut => "Timed out",
                MinimizerMessage::Error(s) => s,
                _ => "???",
            };
            panic!("{:?} got error: {:?}", mode, error);
        };

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
