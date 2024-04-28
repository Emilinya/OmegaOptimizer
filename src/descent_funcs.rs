use std::fs::File;
use std::io::Write;

use ndarray::prelude::*;
use ndarray_linalg::cholesky::*;
use ndarray_linalg::Inverse;
use ndarray_stats::QuantileExt;

type FType = dyn Fn(&Array1<f64>) -> f64;
type GradFType = dyn Fn(&Array1<f64>) -> Array1<f64>;
type HessFType = dyn Fn(&Array1<f64>) -> Array2<f64>;

enum GradMsg {
    Ok,
    Timeout,
    DTErr, // f_diffs is all NaN or dt == 0
}
pub struct GradRes {
    val: Array1<f64>,
    msg: GradMsg,
}

enum NewtMsg {
    Ok,
    Timeout,
    NotPos, // Hessian is not positive definite
    NotInv, // Hessian is not invertible
    ErrInc, // Newton step increased error
}
pub struct NewtRes {
    val: Array1<f64>,
    msg: NewtMsg,
}

pub fn varying_gradient_descent(
    x0: &Array1<f64>,
    f: &FType,
    grad_f: &GradFType,
    max_steps: i32,
    param_path_file: &mut File,
) -> GradRes {
    let epsilon = 1e-12;

    let mut dt = 1.0;
    // let dt_diffs = Array::linspace(-2., 2., 31).mapv(|v| 10_f64.powf(v));
    let dt_diffs = array![0.01, 0.1, 0.5, 0.9, 1., 1.1, 2., 10., 100.];
    // let dt_diffs = array![0.5, 1., 2.];

    writeln!(param_path_file, "Gradient descent\n{}", x0).unwrap();

    let mut x = x0.to_owned();
    for _ in 0..max_steps {
        let v = -grad_f(&x);
        if v.dot(&v).sqrt() < epsilon {
            return GradRes {
                val: x,
                msg: GradMsg::Ok,
            };
        }

        loop {
            let prev_f = f(&(&x + dt * &v));

            let f_diffs = dt_diffs.mapv(|dt_diff| f(&(&x + dt * dt_diff * &v)) - prev_f);
            let min_idx = match f_diffs.argmin_skipnan() {
                Ok(v) => v,
                Err(_) => {
                    return GradRes {
                        val: x,
                        msg: GradMsg::DTErr,
                    }
                }
            };
            let dt_diff = dt_diffs[min_idx];

            if dt_diff == 1.0 {
                break;
            }
            dt *= dt_diff;

            if dt == 0.0 {
                return GradRes {
                    val: x,
                    msg: GradMsg::DTErr,
                };
            }
        }

        x = &x + dt * &v;
        writeln!(param_path_file, "{}", x).unwrap();
    }

    GradRes {
        val: x,
        msg: GradMsg::Timeout,
    }
}

pub fn newton_descent(
    x0: &Array1<f64>,
    f: &FType,
    grad_f: &GradFType,
    hess_f: &HessFType,
    max_steps: i32,
    param_path_file: &mut File,
) -> NewtRes {
    let epsilon = 1e-12;

    // See if the descent will find minima or maxima
    match hess_f(x0).cholesky(UPLO::Lower) {
        Ok(_) => {}
        Err(_) => {
            return NewtRes {
                val: x0.to_owned(),
                msg: NewtMsg::NotPos,
            }
        }
    };

    writeln!(param_path_file, "Netwon's method\n{}", x0).unwrap();

    let mut x = x0.to_owned();
    for _ in 0..max_steps {
        let v = -grad_f(&x);
        if v.dot(&v).sqrt() < epsilon {
            return NewtRes {
                val: x,
                msg: NewtMsg::Ok,
            };
        }

        let inv_hess = match hess_f(&x).inv() {
            Ok(inv) => inv,
            Err(_) => {
                return NewtRes {
                    val: x,
                    msg: NewtMsg::NotInv,
                }
            }
        };

        let x_new = &x + inv_hess.dot(&v);
        if f(&x_new) > f(&x) {
            return NewtRes {
                val: x,
                msg: NewtMsg::ErrInc,
            };
        }

        x = x_new;
        writeln!(param_path_file, "{}", x).unwrap();
    }

    NewtRes {
        val: x,
        msg: NewtMsg::Timeout,
    }
}

pub fn combined_descent(
    x0: &Array1<f64>,
    f: &FType,
    grad_f: &GradFType,
    hess_f: &HessFType,
    print: bool,
    param_path_file_name: &str,
) -> (Array1<f64>, bool) {
    fn is_worse(new_err: f64, old_err: f64) -> bool {
        new_err.is_nan() || new_err > old_err
    }
    fn print_grad_result(msg: &GradMsg, x: &Array1<f64>, err: f64, print: bool) {
        if print {
            print!("G: {}, err={}\n  msg: ", x, err);
            match msg {
                GradMsg::Ok => println!("Gradient minimized error!"),
                GradMsg::Timeout => println!("Gradient timed out"),
                GradMsg::DTErr => println!("Gradient could not find good dt"),
            };
        }
    }
    fn print_newt_result(msg: &NewtMsg, x: &Array1<f64>, err: f64, print: bool) {
        if print {
            print!("N: {}, err={}\n  msg: ", x, err);
            match msg {
                NewtMsg::Ok => println!("Newton minimized error!"),
                NewtMsg::Timeout => println!("Newton timed out"),
                NewtMsg::NotPos => println!("Hessian is not positive definite"),
                NewtMsg::NotInv => println!("Hessian is not invertible"),
                NewtMsg::ErrInc => println!("Newton step increased error"),
            };
        }
    }

    let mut param_path_file = match File::create(param_path_file_name) {
        Ok(file) => file,
        Err(err) => {
            panic!(
                "Got error when opening file {}: {}",
                param_path_file_name, err
            );
        }
    };
    if print {
        println!("I: {}, err={}", x0, f(x0));
    }

    // Try with only Newton first
    let newton_result = newton_descent(x0, f, grad_f, hess_f, 50, &mut param_path_file);
    let (optimal_x, msg) = (newton_result.val, newton_result.msg);

    print_newt_result(&msg, &optimal_x, f(&optimal_x), print);

    let mut start_x = x0.to_owned();
    if !is_worse(f(&optimal_x), f(x0)) {
        match msg {
            NewtMsg::Ok => return (optimal_x, true),
            _ => {
                start_x = optimal_x;
            }
        }
    }

    // Approximate xmin with some gradient descent steps
    let gradient_result = varying_gradient_descent(&start_x, f, grad_f, 25, &mut param_path_file);
    let (improved_x, msg) = (gradient_result.val, gradient_result.msg);
    print_grad_result(&msg, &improved_x, f(&improved_x), print);

    if is_worse(f(&improved_x), f(&start_x)) {
        // Gradient descent increased error, maybe newton will work?
        let newton_result = newton_descent(x0, f, grad_f, hess_f, 50, &mut param_path_file);
        let (optimal_x, msg) = (newton_result.val, newton_result.msg);
        print_newt_result(&msg, &optimal_x, f(&optimal_x), print);

        if is_worse(f(&optimal_x), f(x0)) {
            // Newton also increased error, abort
            return (x0.to_owned(), false);
        }

        match msg {
            NewtMsg::Ok => return (optimal_x, true),
            _ => return (improved_x, false),
        }
    }

    // If the message is Ok, the initial gradient descent reached the bottom!
    if let GradMsg::Ok = msg {
        return (improved_x, true);
    }

    // Improve initial x with Newton's method
    let newton_result = newton_descent(&improved_x, f, grad_f, hess_f, 50, &mut param_path_file);
    let (optimal_x, msg) = (newton_result.val, newton_result.msg);
    print_newt_result(&msg, &optimal_x, f(&optimal_x), print);

    if !is_worse(f(&optimal_x), f(&improved_x)) {
        if let NewtMsg::Ok = msg {
            return (optimal_x, true);
        }

        // Newton decreased error, but is still not happy. Maybe using gradient descent again wil work?
        let gradient_result =
            varying_gradient_descent(&optimal_x, f, grad_f, 1000, &mut param_path_file);
        let (improved_x, msg) = (gradient_result.val, gradient_result.msg);
        print_grad_result(&msg, &improved_x, f(&improved_x), print);

        if is_worse(f(&improved_x), f(&optimal_x)) {
            // Nope, error increased. Abort
            return (optimal_x, false);
        }

        if let GradMsg::Ok = msg {
            return (improved_x, true);
        }

        // Now try Newton again!
        let newton_result = newton_descent(&improved_x, f, grad_f, hess_f, 50, &mut param_path_file);
        let (optimal_x, msg) = (newton_result.val, newton_result.msg);
        print_newt_result(&msg, &optimal_x, f(&optimal_x), print);

        if is_worse(f(&optimal_x), f(&improved_x)) {
            // Now Newton increased error? Oh well, abort
            return (improved_x, false);
        }

        // Newton is still not happy, but what can you do?
        match msg {
            NewtMsg::Ok => (optimal_x, true),
            _ => (optimal_x, false),
        }
    } else {
        // Newton increased error, try again with more descent steps
        let gradient_result =
            varying_gradient_descent(&improved_x, f, grad_f, 1000, &mut param_path_file);
        let (new_improved_x, msg) = (gradient_result.val, gradient_result.msg);
        print_grad_result(&msg, &new_improved_x, f(&new_improved_x), print);

        if is_worse(f(&new_improved_x), f(&improved_x)) {
            // Gradient descent increased error now? Abort
            return (improved_x, false);
        }

        if let GradMsg::Ok = msg {
            return (new_improved_x, true);
        }

        // Improve new x with Newton's method
        let newton_result = newton_descent(&new_improved_x, f, grad_f, hess_f, 50, &mut param_path_file);
        let (optimal_x, msg) = (newton_result.val, newton_result.msg);
        print_newt_result(&msg, &optimal_x, f(&optimal_x), print);

        if !is_worse(f(&optimal_x), f(&new_improved_x)) {
            match msg {
                NewtMsg::Ok => (optimal_x, true),
                _ => (new_improved_x, false),
            }
        } else {
            (improved_x, false)
        }
    }
}
