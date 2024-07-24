use core::array::from_fn;
use eframe::egui;
use egui::{Color32, Widget};
use egui_plot::{Line, Plot, PlotPoints, Points};
use itertools::{izip, Itertools, MinMaxResult};
use nalgebra::SVector;

use std::{
    cell::RefCell,
    marker::PhantomData,
    path::{Path, PathBuf},
    rc::Rc,
};

use crate::functions::Differentiated;
use crate::utils::load_txt;

pub fn create_gui<const D: usize, F: Differentiated<D> + 'static>(
    datafile: &Path,
    initial_parameters: SVector<f64, D>,
) -> SVector<f64, D> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([640.0, 420.0]),
        ..Default::default()
    };

    // why does the eframe closure think it is static? You end! Why are you so difficult!
    let good_parameters_rc_rc: Rc<RefCell<SVector<f64, D>>> =
        Rc::new(RefCell::new(initial_parameters));
    let datafile_clone = datafile.to_path_buf();
    let good_parameters_rc_rc_clone = good_parameters_rc_rc.clone();

    eframe::run_native(
        "Parameter Finder",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(MyApp::<D, F>::new(
                datafile_clone,
                good_parameters_rc_rc_clone,
            )))
        }),
    )
    .unwrap();

    // why can't I just return this? Why are you so mean to me borrow checker?
    let good_parameters =
        { SVector::<f64, D>::from_iterator(good_parameters_rc_rc.borrow().iter().copied()) };

    good_parameters
}

struct MyApp<const D: usize, F: Differentiated<D>> {
    x_ray: Vec<f64>,
    y_ray: Vec<f64>,
    parameter_ptr: Rc<RefCell<SVector<f64, D>>>,
    parameter_names: [&'static str; D],
    parameter_strings: [String; D],
    parameter_values: [Option<f64>; D],
    _phantom: PhantomData<F>,
}

impl<const D: usize, F: Differentiated<D>> MyApp<D, F> {
    fn new(datafile: PathBuf, parameter_ptr: Rc<RefCell<SVector<f64, D>>>) -> Self {
        let (x_ray, y_ray) = load_txt(&datafile).unwrap();
        Self {
            x_ray,
            y_ray,
            parameter_ptr: parameter_ptr.clone(),
            parameter_names: F::PARAMETER_NAMES,
            parameter_strings: from_fn(|i| format!("{}", parameter_ptr.borrow()[i])),
            parameter_values: from_fn(|i| Some(parameter_ptr.borrow()[i])),
            _phantom: PhantomData,
        }
    }
}

impl<const D: usize, F: Differentiated<D>> eframe::App for MyApp<D, F> {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Find good initial parameters");
            ui.vertical(|ui| {
                for (i, parameter) in self.parameter_names.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("{}: ", parameter));
                        let mut text_edit =
                            egui::TextEdit::singleline(&mut self.parameter_strings[i]);
                        if self.parameter_values[i].is_none() {
                            text_edit = text_edit.text_color(Color32::RED);
                        }

                        if text_edit.ui(ui).changed() {
                            self.parameter_values[i] =
                                self.parameter_strings[i].parse::<f64>().ok();
                            if let Some(v) = self.parameter_values[i] {
                                self.parameter_ptr.borrow_mut()[i] = v;
                            }
                        }
                    });
                }
            });

            let data: PlotPoints = izip!(&self.x_ray, &self.y_ray)
                .map(|(x, y)| [*x, *y])
                .collect();
            let data_points = Points::new(data)
                .radius(4.0)
                .color(Color32::from_hex("#1f77b4").unwrap());

            let line = if self.parameter_values.iter().all(Option::is_some) {
                let params = SVector::<f64, D>::from_iterator(
                    self.parameter_values.iter().map(|v| v.unwrap()),
                );

                const N: usize = 1000;
                let (min, max) = match self.x_ray.iter().minmax() {
                    MinMaxResult::MinMax(min, max) => (*min, *max),
                    _ => panic!("x_ray must have more than one item!"),
                };
                let function: PlotPoints = (0..N)
                    .map(|i| {
                        let x = (i as f64 / (N - 1) as f64) * (max - min) + min;
                        [x, F::f(x, &params)]
                    })
                    .collect();
                Some(Line::new(function).color(Color32::from_hex("#ff7f0e").unwrap()))
            } else {
                None
            };

            Plot::new("my_plot").show(ui, |plot_ui| {
                plot_ui.points(data_points);
                if let Some(line) = line {
                    plot_ui.line(line);
                }
            });
        });
    }
}
