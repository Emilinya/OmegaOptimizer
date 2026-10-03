use std::{
    fs::{remove_file, File},
    io::Write,
    process::Command,
};

use itertools::{izip, Itertools, MinMaxResult};
use nalgebra::SVector;

use crate::utils::{format_vector, format_with_uncertainty};

pub fn plot_static<const D: usize>(
    x_ray: &[f64],
    y_ray: &[f64],
    f: fn(f64, &SVector<f64, D>) -> f64,
    optimal_parameters: &SVector<f64, D>,
    uncertainties: &SVector<f64, D>,
    filename: &str,
) {
    let datafile = "src/plotting/data.dat";
    let mut file = File::create(datafile).unwrap();

    // save parameters
    writeln!(
        &mut file,
        "{}",
        format_with_uncertainty(
            optimal_parameters.data.as_slice(),
            uncertainties.data.as_slice()
        )
    )
    .unwrap();

    // save input data
    writeln!(&mut file, "{}\n{}", filename, x_ray.len()).unwrap();
    for (x, y) in izip!(x_ray, y_ray) {
        writeln!(&mut file, "{} {}", x, y).unwrap();
    }

    // save high quality best fit model
    const N: usize = 1000;
    writeln!(&mut file, "{}", N).unwrap();
    let (min, max) = match x_ray.iter().minmax() {
        MinMaxResult::MinMax(min, max) => (*min, *max),
        _ => panic!("x_ray must have more than one item!"),
    };
    for i in 0..N {
        let x = (i as f64 / (N - 1) as f64) * (max - min) + min;
        writeln!(&mut file, "{} {}", x, f(x, optimal_parameters)).unwrap();
    }

    call_and_remove(datafile);
}

pub fn plot_slice(
    x_ray: &[f64],
    y_ray: &[f64],
    f: impl Fn(f64, &[f64]) -> f64,
    optimal_parameters: &[f64],
    uncertainties: &Option<Vec<f64>>,
    filename: &str,
) {
    let datafile = "src/plotting/data.dat";
    let mut file = File::create(datafile).unwrap();

    // save parameters
    if let Some(uncertainties) = uncertainties {
        writeln!(
            &mut file,
            "{}",
            format_with_uncertainty(optimal_parameters, uncertainties)
        )
        .unwrap();
    } else {
        writeln!(&mut file, "{}", format_vector(optimal_parameters, 3)).unwrap();
    }

    // save input data
    writeln!(&mut file, "{}\n{}", filename, x_ray.len()).unwrap();
    for (x, y) in izip!(x_ray, y_ray) {
        writeln!(&mut file, "{} {}", x, y).unwrap();
    }

    // save high quality best fit model
    const N: usize = 1000;
    writeln!(&mut file, "{}", N).unwrap();
    let (min, max) = match x_ray.iter().minmax() {
        MinMaxResult::MinMax(min, max) => (*min, *max),
        _ => panic!("x_ray must have more than one item!"),
    };
    for i in 0..N {
        let x = (i as f64 / (N - 1) as f64) * (max - min) + min;
        writeln!(&mut file, "{} {}", x, f(x, optimal_parameters)).unwrap();
    }

    call_and_remove(datafile);
}

fn call_and_remove(datafile: &str) {
    let mut run_python = {
        if cfg!(target_os = "windows") {
            Command::new("python")
        } else {
            Command::new("python3")
        }
    };
    run_python
        .arg("src/plotting/plotter.py")
        .spawn()
        .unwrap()
        .wait()
        .unwrap();

    remove_file(datafile).unwrap();
}
