use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::Command;
use std::time::Instant;

use itertools::izip;
use ndarray::prelude::*;
use regex::Regex;

mod descent_funcs;
mod error_funcs;
mod function;
mod utils;

use descent_funcs::combined_descent;
use error_funcs::*;
use function::*;
use utils::*;

fn plot_path(datafile: &str, descent_path_file: &str, img_folder: &str, img_prefix: &str) {
    let mut run_python = {
        if cfg!(target_os = "windows") {
            Command::new("python")
        } else {
            Command::new("python3")
        }
    };
    run_python
        .args(vec![
            "plotting/path_plotter.py",
            &format!("data/{}", datafile),
            descent_path_file,
            img_folder,
            img_prefix,
        ])
        .spawn()
        .unwrap()
        .wait()
        .unwrap();
}

fn plot(
    x_ray: &Array1<f64>,
    y_ray: &Array1<f64>,
    f: fn(f64, &Array1<f64>) -> f64,
    amin: &Array1<f64>,
    filename: &str,
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
    run_python
        .arg("plotting/plotter.py")
        .spawn()
        .unwrap()
        .wait()
        .unwrap();
}

fn get_initial_params(datafile: &str, num_params: usize) -> Array1<f64> {
    // See if user has found initial params
    let file = File::open("preprocessor/initial_params.dat");
    if let Ok(file) = file {
        let mut reader = BufReader::new(file);
        let mut buf = "".to_string();
        if let Err(err) = reader.read_line(&mut buf) {
            eprintln!("Got error when reading initial parameters: {}", err);
            buf = "".to_string();
        }

        if buf.is_empty() {
            let param_results: Result<Vec<f64>, _> =
                buf.split(' ').map(|s| s.parse::<f64>()).collect();
            if let Ok(params) = param_results {
                if params.len() == num_params {
                    return Array::from(params);
                } else {
                    println!("initial_params.dat does not have the correct amount of values");
                }
            } else {
                println!("initial_params.dat is not formated correctly");
            }
        }
    }

    // If not, see if datafile has any parameter values
    let mut initial_params = Array::ones(num_params);

    let file = match File::open(format!("data/{}", datafile)) {
        Ok(file) => file,
        Err(_) => {
            return initial_params;
        }
    };

    let mut buf = "".to_string();
    let mut header = "".to_string();
    let mut reader = BufReader::new(file);

    // we first create a map from variable to index
    if let Err(err) = reader.read_line(&mut header) {
        eprintln!("Got error when reading datafile: {}", err);
    }

    let re = Regex::new(r".*?\((.*?);\s*(.*?)\)\s*=\s*(.*?)").unwrap();
    let caps = re.captures(&header).unwrap();
    let varable_iter = caps[2].split(',').enumerate().map(|(i, v)| (v.trim(), i));
    let variable_map: HashMap<&str, usize> = HashMap::from_iter(varable_iter);

    if let Err(err) = reader.read_line(&mut buf) {
        eprintln!("Got error when reading datafile: {}", err);
    }

    let params: Vec<Vec<&str>> = buf
        .split(',')
        .map(|s| s.trim().split('=').collect())
        .collect();

    if params.iter().all(|vec| vec.len() == 2) {
        for pair in params {
            let (var, val) = (pair[0], pair[1]);
            let val = &val.replace(&['\n', '\r'][..], "");
            if variable_map.contains_key(var) {
                if let Ok(v) = val.parse::<f64>() {
                    initial_params[variable_map[var]] = v;
                } else {
                    println!("Malformed parameter value for parameter {}: {:?}", var, val);
                }
            } else {
                println!("Unknown parameter {:?}", var);
            }
        }
    };

    initial_params
}

fn main() {
    let mut arg_map: HashMap<String, String> = HashMap::new();
    env::args().enumerate().for_each(|(i, v)| {
        if i != 0 {
            let arg_vec: Vec<&str> = v.split('=').collect();
            if arg_vec.len() == 2 {
                arg_map.insert(arg_vec[0].to_owned(), arg_vec[1].to_owned());
            } else {
                println!("Malformed argument: {}", v);
            }
        }
    });

    let datafile = match arg_map.get("datafile") {
        Some(v) => v,
        None => "datafile.dat",
    };
    println!("{}", datafile);

    let file = match File::open(format!("data/{}", datafile)) {
        Ok(file) => file,
        Err(err) => {
            panic!("Got error when opening data/{}: {}", datafile, err)
        }
    };
    let mut reader = BufReader::new(file);
    let mut buf: String = "".to_string();
    match reader.read_line(&mut buf) {
        Ok(_) => {}
        Err(err) => {
            panic!("Got error when reading first line of datafile: {}", err)
        }
    };
    let re = Regex::new(r".*?\((.*?);\s*(.*?)\)\s*=\s*(.*?)").unwrap();
    let caps = re.captures(&buf).unwrap();
    let num_params = caps[2].split(", ").count();

    let (mut x_vec, mut y_vec) = (Vec::new(), Vec::new());
    for line in reader.lines().map_while(Result::ok) {
        let vals: Vec<&str> = line.split(' ').collect();
        if vals.len() == 2 {
            if let (Ok(x), Ok(y)) = (vals[0].parse::<f64>(), vals[1].parse::<f64>()) {
                x_vec.push(x);
                y_vec.push(y);
            }
        }
    }

    let x_ray = Array::from(x_vec);
    let y_ray = denoise(&Array::from(y_vec));

    let err_f = calc_err(x_ray.to_owned(), y_ray.to_owned(), f);
    let grad_err_f = calc_grad_err(x_ray.to_owned(), y_ray.to_owned(), f, grad_f);
    let hess_err_f = calc_hess_err(x_ray.to_owned(), y_ray.to_owned(), f, grad_f, hess_f);

    let now = Instant::now();
    let descent_path_file = "plotting/descent_path.dat";
    let initial_params = get_initial_params(datafile, num_params);
    let (optimal_params, params_are_good) = combined_descent(
        &initial_params,
        &*err_f,
        &*grad_err_f,
        &*hess_err_f,
        true,
        descent_path_file,
    );
    if !params_are_good {
        eprintln!(
            "Parameter optimizer might not have found optimal parameters. If the result looks wrong, try again with different initial parameters"
        );
    }
    let err = err_f(&optimal_params);

    println!("Descent took {} ms", get_ms(now));
    println!("Got the parameters: {}", optimal_params);
    println!("The error is: {}", err);

    let name = match arg_map.get("name") {
        Some(v) => v,
        None => "",
    };

    println!("Plotting ...");
    plot_path(datafile, descent_path_file, "figures", name);
    plot(&x_ray, &y_ray, f, &optimal_params, "figures/result.png");
}
