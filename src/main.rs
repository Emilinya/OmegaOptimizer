use std::io::{Write, BufReader, BufRead};
use std::process::Command;
use std::time::Instant;
use std::fs::File;

use ndarray::prelude::*;
use itertools::izip;

mod function;
mod error_funcs;
mod descent_funcs;

use function::*;
use error_funcs::*;
use descent_funcs::combined_descent;

fn plot(
    x_ray: &Array1<f64>, y_ray: &Array1<f64>,
    f: fn(f64, &Array1<f64>) -> f64,
    amin: &Array1<f64>, filename: &str
) {
    let mut file = File::create("plotting/data.dat").unwrap();
    writeln!(&mut file, "{}", filename).unwrap();
    for (x, y) in izip!(x_ray, y_ray) {
        writeln!(&mut file, "{} {} {}", x, y, f(*x, amin)).unwrap();
    }
    let mut run_python = {
        if cfg!(target_os = "windows") {
            Command::new("python")
        } else {
            Command::new("python3")
        }
    };
    run_python.arg("plotting/plotter.py").spawn().expect(":(").wait().unwrap();
}

fn main() {
    let file = File::open("data/datafile.dat").unwrap();
    let mut reader = BufReader::new(file);
    let mut buf: String = "".to_string();
    reader.read_line(&mut buf).unwrap();
    let num_params = buf.split(" = ").next().unwrap().split("; ").last().unwrap().split(", ").count();

    let (mut x_vec, mut y_vec) = (Vec::new(), Vec::new());
    for line in reader.lines().map(|l| l.unwrap()) {
        let vals: Vec<&str> = line.split(" ").collect();
        let (x, y) = (vals[0].parse::<f64>(), vals[1].parse::<f64>());
        match (x, y) {
            (Ok(x), Ok(y)) => { x_vec.push(x); y_vec.push(y); },
            _ => { },
        }
    }

    let x_ray = Array::from(x_vec);
    let y_ray = Array::from(y_vec);

    let err_f = calc_err(x_ray.to_owned(), y_ray.to_owned(), f);
    let grad_err_f = calc_grad_err(x_ray.to_owned(), y_ray.to_owned(), f, grad_f);
    let inv_hess_err_f = calc_inv_hess_err(x_ray.to_owned(), y_ray.to_owned(), f, hess_f, outer_f);


    let now = Instant::now();
    let initial_params = Array::ones(num_params);
    let optimal_params = combined_descent(&initial_params, &*err_f, &*grad_err_f, &*inv_hess_err_f, true);
    println!("Descent took {} ms", (now.elapsed().as_nanos() as f64) / 1_000_000_f64);
    println!("Got the parameters {}", optimal_params);

    plot(&x_ray, &y_ray, f, &optimal_params, "result.png")
}
