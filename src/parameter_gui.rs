use eframe::egui;
use egui::{Color32, Widget};
use egui_plot::{Line, Plot, PlotPoints, Points};
use itertools::{izip, Itertools, MinMaxResult};
use strum::IntoEnumIterator;

use std::{
    collections::HashMap,
    iter::repeat,
    path::{Path, PathBuf},
};

use crate::error_functions::error;
use crate::functions::Functions;
use crate::plotting::plotter::plot_slice;
use crate::utils::{format_with_uncertainty, load_txt};

pub fn create_gui(
    datafile: &Path,
    function: Option<Functions>,
    initial_parameters: Option<Vec<f64>>,
) {
    const SCALE: f32 = 1.25;

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([640.0 * SCALE, 520.0 * SCALE]),
        follow_system_theme: false,
        ..Default::default()
    };

    let datafile_clone = datafile.to_path_buf();

    eframe::run_native(
        "Parameter Finder",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_zoom_factor(SCALE);
            Ok(Box::new(MyApp::new(
                datafile_clone,
                function,
                initial_parameters,
            )))
        }),
    )
    .unwrap();
}

enum Message {
    None,
    Ok(String),
    Error(String),
}

struct ParameterStore {
    names: Vec<&'static str>,
    strings: Vec<String>,
    values: Vec<Option<f64>>,
    uncertainties: Option<Vec<f64>>,
}

impl ParameterStore {
    fn slice_to_values(slice: &[f64]) -> (Vec<String>, Vec<Option<f64>>) {
        let strings = slice.iter().map(|v| format!("{}", v)).collect();
        let values = slice.iter().map(|v| Some(*v)).collect();

        (strings, values)
    }

    fn new(function: &Functions, values: &Option<Vec<f64>>) -> Self {
        let count = function.parameter_count();
        let values = values
            .clone()
            .unwrap_or_else(|| repeat(1.0).take(count).collect());

        let names = function.parameter_names();
        let (strings, values) = Self::slice_to_values(&values);

        Self {
            names,
            strings,
            values,
            uncertainties: None,
        }
    }

    fn get_parameters(&self) -> Option<Vec<f64>> {
        if self.values.iter().all(Option::is_some) {
            Some(self.values.iter().filter_map(|v| *v).collect())
        } else {
            None
        }
    }

    fn update_values(&mut self, new_values: &[f64], uncertainties: &[f64]) {
        (self.strings, self.values) = Self::slice_to_values(new_values);
        self.uncertainties = Some(uncertainties.to_vec());
    }

    fn reset(&mut self) {
        let ones: Vec<f64> = repeat(1.0).take(self.names.len()).collect();
        (self.strings, self.values) = Self::slice_to_values(&ones);
        self.uncertainties = None;
    }
}

struct MyApp {
    x_ray: Vec<f64>,
    y_ray: Vec<f64>,
    message: Message,
    datafile: PathBuf,
    function: Functions,
    parameter_store_map: HashMap<Functions, ParameterStore>,
}

impl MyApp {
    fn new(
        datafile: PathBuf,
        function: Option<Functions>,
        initial_parameters: Option<Vec<f64>>,
    ) -> Self {
        let function = function.unwrap_or(Functions::Line);

        let parameter_store_map = HashMap::from_iter(Functions::iter().map(|f| {
            let store = if f == function && initial_parameters.is_some() {
                ParameterStore::new(&f, &initial_parameters)
            } else {
                ParameterStore::new(&f, &None)
            };
            (f, store)
        }));

        let (x_ray, y_ray) = load_txt(&datafile).unwrap();
        Self {
            x_ray,
            y_ray,
            message: Message::None,
            datafile,
            function,
            parameter_store_map,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Find good initial parameters");

            ui.add_space(2.0);

            // Function combo box
            egui::ComboBox::from_label("Select a function")
                .selected_text(format!("{:?}", self.function))
                .show_ui(ui, |ui| {
                    for variant in Functions::iter() {
                        let text = format!("{:?}", variant);
                        if ui
                            .selectable_value(&mut self.function, variant, text)
                            .changed()
                        {
                            self.message = Message::None;
                        }
                    }
                });

            ui.add_space(5.0);

            // Parameter selection boxes
            let parameter_store = self.parameter_store_map.get_mut(&self.function).unwrap();
            ui.vertical(|ui| {
                for (i, parameter) in parameter_store.names.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("{}: ", parameter));
                        let mut text_edit =
                            egui::TextEdit::singleline(&mut parameter_store.strings[i]);
                        if parameter_store.values[i].is_none() {
                            text_edit = text_edit.text_color(Color32::RED);
                        }

                        if text_edit.ui(ui).changed() {
                            parameter_store.values[i] =
                                parameter_store.strings[i].parse::<f64>().ok();
                        }

                        ui.add_space(2.0);

                        if let Some(uncertainties) = &parameter_store.uncertainties {
                            ui.label(format!("Δ{}: {}", parameter, uncertainties[i]));
                        }
                    });
                }
            });

            ui.add_space(5.0);

            // Run/Reset/Save buttons
            ui.horizontal(|ui| {
                if ui.button("Run").clicked() {
                    if let Some(parameters) = parameter_store.get_parameters() {
                        let (optimal_parameters, uncertainties, _) =
                            self.function
                                .optimizinate(&self.datafile, &Some(parameters), false);
                        parameter_store.update_values(&optimal_parameters, &uncertainties);
                        self.message = Message::Ok(format!(
                            "Got parameters {}",
                            format_with_uncertainty(&optimal_parameters, &uncertainties,)
                        ));
                    } else {
                        self.message = Message::Error("Some parameters are malformed.".into());
                    }
                }
                ui.add_space(5.0);
                if ui.button("Reset").clicked() {
                    parameter_store.reset();
                }
                ui.add_space(5.0);
                if ui.button("Save Figure").clicked() {
                    if let Some(parameters) = parameter_store.get_parameters() {
                        let data_name = self.datafile.file_stem().unwrap().to_string_lossy();
                        let figure_name =
                            format!("figures/{}-{}.png", data_name, self.function.name());
                        plot_slice(
                            &self.x_ray,
                            &self.y_ray,
                            |x, p| self.function.f(x, p),
                            &parameters,
                            &parameter_store.uncertainties,
                            &figure_name,
                        );
                        self.message = Message::Ok(format!("Saved figure '{}'", figure_name));
                    } else {
                        self.message = Message::Error("Some parameters are malformed.".into());
                    }
                }
            });

            ui.add_space(5.0);

            // Message
            let message = match &self.message {
                Message::None => "",
                Message::Ok(s) => s,
                Message::Error(s) => &format!("Error: {}", s),
            };
            ui.label(message);

            ui.add_space(10.0);

            // Approximation error
            let error = if let Some(parameters) = parameter_store.get_parameters() {
                format!(
                    "{}",
                    error(&self.x_ray, &self.y_ray, &self.function, &parameters)
                )
            } else {
                "NaN".into()
            };
            ui.label(format!("Error: {}", error));

            ui.add_space(2.0);

            // Data figure
            let data: PlotPoints = izip!(&self.x_ray, &self.y_ray)
                .map(|(x, y)| [*x, *y])
                .collect();
            let data_points = Points::new(data)
                .radius(4.0)
                .color(Color32::from_hex("#1f77b4").unwrap());

            let line = if parameter_store.values.iter().all(Option::is_some) {
                let params: Vec<f64> = parameter_store.values.iter().filter_map(|v| *v).collect();

                const N: usize = 1000;
                let (min, max) = match self.x_ray.iter().minmax() {
                    MinMaxResult::MinMax(min, max) => (*min, *max),
                    _ => panic!("x_ray must have more than one item!"),
                };
                let function: PlotPoints = (0..N)
                    .map(|i| {
                        let x = (i as f64 / (N - 1) as f64) * (max - min) + min;
                        [x, self.function.f(x, &params)]
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
