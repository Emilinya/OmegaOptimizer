use std::{io::Write, process::Command};

use anyhow::Context;
use fs_err::File;
use itertools::{Itertools, MinMaxResult, izip};
use nalgebra::SVector;

use crate::utils::{format_vector, format_with_uncertainty};

pub fn plot_static<const D: usize>(
    x_ray: &[f64],
    y_ray: &[f64],
    f: fn(f64, &SVector<f64, D>) -> f64,
    optimal_parameters: &SVector<f64, D>,
    uncertainties: &SVector<f64, D>,
    filename: &str,
) -> anyhow::Result<()> {
    let datafile = "src/plotting/data.dat";
    let mut file = File::create(datafile)?;

    // save parameters
    writeln!(
        &mut file,
        "{}",
        format_with_uncertainty(
            optimal_parameters.data.as_slice(),
            uncertainties.data.as_slice()
        )
    )?;

    // save input data
    writeln!(&mut file, "{}\n{}", filename, x_ray.len())?;
    for (x, y) in izip!(x_ray, y_ray) {
        writeln!(&mut file, "{} {}", x, y)?;
    }

    // save high quality best fit model
    const N: usize = 1000;
    writeln!(&mut file, "{}", N)?;
    let (min, max) = match x_ray.iter().minmax() {
        MinMaxResult::MinMax(min, max) => (*min, *max),
        _ => anyhow::bail!("x_ray must have more than one item!"),
    };
    for i in 0..N {
        let x = (i as f64 / (N - 1) as f64) * (max - min) + min;
        writeln!(&mut file, "{} {}", x, f(x, optimal_parameters))?;
    }

    call_and_remove(datafile)
}

pub fn plot_slice(
    x_ray: &[f64],
    y_ray: &[f64],
    f: impl Fn(f64, &[f64]) -> f64,
    optimal_parameters: &[f64],
    uncertainties: Option<&[f64]>,
    filename: &str,
) -> anyhow::Result<()> {
    let datafile = "src/plotting/data.dat";
    let mut file = File::create(datafile)?;

    // save parameters
    if let Some(uncertainties) = uncertainties {
        writeln!(
            &mut file,
            "{}",
            format_with_uncertainty(optimal_parameters, uncertainties)
        )?;
    } else {
        writeln!(&mut file, "{}", format_vector(optimal_parameters, 3))?;
    }

    // save input data
    writeln!(&mut file, "{}\n{}", filename, x_ray.len())?;
    for (x, y) in izip!(x_ray, y_ray) {
        writeln!(&mut file, "{} {}", x, y)?;
    }

    // save high quality best fit model
    const N: usize = 1000;
    writeln!(&mut file, "{}", N)?;
    let (min, max) = match x_ray.iter().minmax() {
        MinMaxResult::MinMax(min, max) => (*min, *max),
        _ => anyhow::bail!("x_ray must have more than one item!"),
    };
    for i in 0..N {
        let x = (i as f64 / (N - 1) as f64) * (max - min) + min;
        writeln!(&mut file, "{} {}", x, f(x, optimal_parameters))?;
    }

    call_and_remove(datafile)
}

fn call_and_remove(datafile: &str) -> anyhow::Result<()> {
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
        .context("Failed to spawn plotter")?
        .wait()
        .context("Failed to wait for plotter")?;

    Ok(fs_err::remove_file(datafile)?)
}
