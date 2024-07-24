mod error_functions;
mod functions;
mod minimizers;
mod parameter_gui;
mod plotting;
mod statistics;
mod utils;

use clap::Parser;
use log::{info, warn};
use nalgebra::SVector;
use std::{path::PathBuf, rc::Rc, time::Instant};
use strum::VariantNames;

use error_functions::get_error_functions;
use functions::{Differentiated, Functions};
use minimizers::combined_descent;
use parameter_gui::create_gui;
use plotting::plotter::plot_static;
use statistics::get_uncertainties;

fn parse_initial_parameters(parameter_string: &str) -> Result<f64, String> {
    if parameter_string == "None" {
        Ok(1.0)
    } else {
        match parameter_string.parse::<f64>() {
            Ok(v) => Ok(v),
            Err(e) => Err(e.to_string()),
        }
    }
}

fn optimizinate<const D: usize, F: Differentiated<D>>(
    datafile: &PathBuf,
    initial_parameters: SVector<f64, D>,
    plot_result: bool,
) -> (SVector<f64, D>, SVector<f64, D>, f64) {
    let (x_ray, y_ray) = utils::load_txt(datafile).unwrap();
    let x_ray = Rc::new(x_ray);
    let y_ray = Rc::new(y_ray);
    let error_function = get_error_functions::<D, F>(x_ray.clone(), y_ray.clone());

    let (optimal_parameters, error) = combined_descent(&initial_parameters, &error_function);
    if let Some(error) = error {
        warn!("{}", error);
    }

    let parameter_uncertainties = get_uncertainties::<D, F>(&x_ray, &y_ray, &optimal_parameters);
    let error = (error_function.f)(&optimal_parameters);

    if plot_result {
        let data_name = datafile.file_stem().unwrap().to_string_lossy();
        let figure_name = format!("figures/{}-{}.png", data_name, F::NAME);

        plot_static(
            &x_ray,
            &y_ray,
            F::f,
            &optimal_parameters,
            &parameter_uncertainties,
            &figure_name,
        );
    }

    (optimal_parameters, parameter_uncertainties, error)
}

#[derive(Parser)]
struct Args {
    /// Path to the file containing data you want to fit a function to
    datafile: PathBuf,
    /// Name of the function you want to fit to your data.
    /// Use the -p flag to get a list of valid function names
    #[arg(value_parser=Functions::descriptive_from_str)]
    function: Option<Functions>,
    /// An optional space separated list of initial parameters. Number
    /// of initial parameters must match the number of parameters in the function
    /// you choose. Parameters in the list can also be 'None', in which case they
    /// are set to a default value.
    #[arg(value_parser=parse_initial_parameters)]
    initial_parameters: Option<Vec<f64>>,
    /// Run program without a gui.
    #[arg(short, long)]
    fast: bool,
    /// Print a list of all valid function names
    #[arg(short, long)]
    print_function_names: bool,
}

fn main() {
    let args = Args::parse();
    if args.print_function_names {
        println!(
            "Valid function names are {}.",
            utils::prettify_list(Functions::VARIANTS)
        );
    }

    pretty_env_logger::init();

    if args.initial_parameters.is_some() && args.function.is_some() {
        let (parameters, function) = (
            args.initial_parameters.as_ref().unwrap(),
            args.function.as_ref().unwrap(),
        );
        if parameters.len() != function.parameter_count() {
            panic!(
                "Got invalid number of initial parameters. \
                {:?} takes {} parameters, but got {}.",
                function,
                function.parameter_count(),
                parameters.len()
            );
        }
    }

    if !args.fast {
        create_gui(&args.datafile, args.function, args.initial_parameters);
    } else {
        let Some(function) = args.function else {
            panic!("You must specify a function when running program headless!");
        };
        let start = Instant::now();
        let (optimal_parameters, parameter_uncertainties, error) =
            function.optimizinate(&args.datafile, &args.initial_parameters, true);

        info!("Descent took {}", utils::format_duration(start.elapsed()));

        println!(
            "Got optimal parameters: {}, which gives an error of {}",
            utils::format_with_uncertainty(&optimal_parameters, &parameter_uncertainties),
            utils::g_format(error, 5)
        );
    }
}
