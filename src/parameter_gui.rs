use anyhow::Context;
use eframe::egui::{self, Ui};
use egui::{Color32, Widget};
use egui_plot::{Line, Plot, PlotPoints, Points};
use itertools::{Itertools, MinMaxResult, izip, repeat_n};
use strum::IntoEnumIterator;

use std::{
    collections::HashMap,
    fmt::Display,
    path::{Path, PathBuf},
    sync::mpsc::{self, TryRecvError},
    thread,
    time::Duration,
};

use crate::functions::Functions;
use crate::plotting::plotter::plot_slice;
use crate::utils::{format_with_uncertainty, load_txt};
use crate::{OptimizinateResult, error_functions::error};

pub fn create_gui(
    datafile: Option<PathBuf>,
    function: Option<Functions>,
    initial_parameters: Option<&[f64]>,
) -> anyhow::Result<()> {
    const SCALE: f32 = 1.25;
    const ICON: &[u8; 64 * 64 * 4] = include_bytes!("../media/icon.raw");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_icon(egui::IconData {
                rgba: ICON.to_vec(),
                width: 64,
                height: 64,
            })
            .with_inner_size([640.0 * SCALE, 520.0 * SCALE]),
        ..Default::default()
    };

    eframe::run_native(
        "Omega Optimizer",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_zoom_factor(SCALE);
            Ok(Box::new(MyApp::new(
                datafile,
                function,
                initial_parameters,
            )?))
        }),
    )
    .map_err(|err| anyhow::anyhow!("{}", err))
}

#[derive(Debug)]
enum Message {
    None,
    Ok(String),
    Error(String),
}

impl Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::None => Ok(()),
            Self::Ok(s) => f.write_str(s),
            Self::Error(s) => write!(f, "Error: {s}"),
        }
    }
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

    fn new(function: Functions, values: Option<Vec<f64>>) -> Self {
        let count = function.parameter_count();
        let values = values.unwrap_or_else(|| repeat_n(1.0, count).collect());

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
        let ones: Vec<f64> = repeat_n(1.0, self.names.len()).collect();
        (self.strings, self.values) = Self::slice_to_values(&ones);
        self.uncertainties = None;
    }
}

struct ParameterStoreMap {
    map: HashMap<Functions, ParameterStore>,
}

impl ParameterStoreMap {
    fn new(function: Functions, initial_parameters: Option<&[f64]>) -> Self {
        let map = HashMap::from_iter(Functions::iter().map(|f| {
            let store = if f == function
                && let Some(parameters) = initial_parameters
            {
                ParameterStore::new(f, Some(parameters.to_vec()))
            } else {
                ParameterStore::new(f, None)
            };
            (f, store)
        }));
        Self { map }
    }

    fn get(&self, function: Functions) -> &ParameterStore {
        self.map
            .get(&function)
            .expect("map should contain parameters for all functions")
    }

    fn get_mut(&mut self, function: Functions) -> &mut ParameterStore {
        self.map
            .get_mut(&function)
            .expect("map should contain parameters for all functions")
    }

    fn reset(&mut self) {
        for patameter_store in self.map.values_mut() {
            patameter_store.reset();
        }
    }
}

struct RunThread {
    thread: Option<thread::JoinHandle<()>>,
    receiver: mpsc::Receiver<OptimizinateResult>,
}

impl RunThread {
    fn start(function: Functions, datafile: &Path, mut parameters: Vec<f64>) -> Self {
        let (result_tx, result_rx) = mpsc::channel();

        let datafile = datafile.to_owned();
        let thread = thread::spawn(move || {
            let mut i = 0;
            let mut previous_error = f64::INFINITY;
            loop {
                let result = match function.optimizinate(&datafile, Some(&parameters), false) {
                    Ok(result) => result,
                    Err(err) => {
                        log::error!("Failed to optimizinate: {:#}", err);
                        return;
                    }
                };
                let _ = result_tx.send(result.clone());

                i += 1;

                if previous_error - result.error < 1e-8 {
                    log::debug!("Error is not decreasing, stopping after {i} iterations");
                    return;
                }
                previous_error = result.error;
                parameters = result.parameters;
            }
        });

        Self {
            thread: Some(thread),
            receiver: result_rx,
        }
    }
}

impl Drop for RunThread {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            log::warn!("Failed to join RunThread");
        }
    }
}

struct Data {
    datafile: PathBuf,
    x_ray: Box<[f64]>,
    y_ray: Box<[f64]>,
}

impl Data {
    fn new(datafile: PathBuf) -> anyhow::Result<Self> {
        let (x_ray, y_ray) = load_txt(&datafile)
            .with_context(|| format!("Failed to read {}", datafile.display()))?;
        Ok(Self {
            datafile,
            x_ray: x_ray.into(),
            y_ray: y_ray.into(),
        })
    }
}

struct MyApp {
    data: Option<Data>,
    message: Message,
    function: Functions,
    run_thread: Option<RunThread>,
    parameter_store_map: ParameterStoreMap,
}

impl MyApp {
    fn new(
        datafile: Option<PathBuf>,
        function: Option<Functions>,
        initial_parameters: Option<&[f64]>,
    ) -> anyhow::Result<Self> {
        let function = function.unwrap_or(Functions::Line);
        let parameter_store_map = ParameterStoreMap::new(function, initial_parameters);

        let data = datafile.map(Data::new).transpose()?;
        Ok(Self {
            data,
            message: Message::None,
            function,
            run_thread: None,
            parameter_store_map,
        })
    }

    fn run(
        data: &Data,
        function: Functions,
        run_thread: &mut Option<RunThread>,
        parameter_store: &ParameterStore,
    ) -> Message {
        if let Some(parameters) = parameter_store.get_parameters() {
            *run_thread = Some(RunThread::start(function, &data.datafile, parameters));
            Message::None
        } else {
            Message::Error("Some parameters are malformed.".into())
        }
    }

    fn read_run_thread(
        function: Functions,
        run_thread: &mut Option<RunThread>,
        parameter_store_map: &mut ParameterStoreMap,
    ) -> Option<Message> {
        if let Some(thread) = run_thread {
            // read messages in a loop to ensure we use the latest message
            let mut result = None;
            loop {
                match thread.receiver.try_recv() {
                    Ok(message) => {
                        result = Some(message);
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        *run_thread = None;
                        break;
                    }
                }
            }
            let result = result?;

            parameter_store_map
                .get_mut(function)
                .update_values(&result.parameters, &result.uncertainties);
            Some(Message::Ok(format!(
                "Got parameters {}",
                format_with_uncertainty(&result.parameters, &result.uncertainties)
            )))
        } else {
            None
        }
    }

    fn save_figure(data: &Data, function: Functions, parameter_store: &ParameterStore) -> Message {
        if let Some(parameters) = parameter_store.get_parameters() {
            let Some(file_stem) = data.datafile.file_stem() else {
                return Message::Error("Datafile has no file stem".into());
            };

            let data_name = file_stem.to_string_lossy();
            let figure_name = format!("figures/{}-{}.png", data_name, function.name());
            match plot_slice(
                &data.x_ray,
                &data.y_ray,
                |x, p| function.f(x, p),
                &parameters,
                parameter_store.uncertainties.as_deref(),
                &figure_name,
            ) {
                Ok(()) => Message::Ok(format!("Saved figure '{}'", figure_name)),
                Err(err) => Message::Error(format!("Failed to save figure: {:#}", err)),
            }
        } else {
            Message::Error("Some parameters are malformed.".into())
        }
    }

    fn show_figure(&self, ui: &mut Ui, data: &Data) -> anyhow::Result<()> {
        let parameter_store = self.parameter_store_map.get(self.function);
        let data_series: PlotPoints = izip!(&data.x_ray, &data.y_ray)
            .map(|(x, y)| [*x, *y])
            .collect();
        let data_points = Points::new("Data", data_series).radius(4.0f32).color(
            Color32::from_hex("#1f77b4").map_err(|_| anyhow::anyhow!("#1f77b4 is invalid hex?"))?,
        );

        let line = if parameter_store.values.iter().all(Option::is_some) {
            let params: Vec<f64> = parameter_store.values.iter().filter_map(|v| *v).collect();

            const N: usize = 1000;
            let (min, max) = match data.x_ray.iter().minmax() {
                MinMaxResult::MinMax(min, max) => (*min, *max),
                _ => anyhow::bail!("x_ray must have more than one item!"),
            };
            let function: PlotPoints = (0..N)
                .map(|i| {
                    let x = (i as f64 / (N - 1) as f64) * (max - min) + min;
                    [x, self.function.f(x, &params)]
                })
                .collect();
            Some(
                Line::new("Function", function).color(
                    Color32::from_hex("#ff7f0e")
                        .map_err(|_| anyhow::anyhow!("#ff7f0e is invalid hex?"))?,
                ),
            )
        } else {
            None
        };

        Plot::new("my_plot").show(ui, |plot_ui| {
            plot_ui.points(data_points);
            if let Some(line) = line {
                plot_ui.line(line);
            }
        });
        Ok(())
    }

    fn set_file(&mut self, path: PathBuf) {
        match Data::new(path) {
            Ok(data) => {
                self.data = Some(data);
                self.parameter_store_map.reset();
                self.message = Message::None;
            }
            Err(err) => self.message = Message::Error(format!("{err:#}")),
        }
    }
}

impl eframe::App for MyApp {
    fn logic(&mut self, _ctx: &egui::Context, _frame: &mut eframe::Frame) {}

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("top_panel").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open").clicked()
                        && let Some(path) = rfd::FileDialog::new().pick_file()
                    {
                        self.set_file(path);
                    }
                    if ui.button("Quit").clicked() {
                        ui.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            let Some(data) = &self.data else {
                ui.heading("Open a file containing datapoints");

                ui.add_space(5.0);

                if ui.button("Open file…").clicked()
                    && let Some(path) = rfd::FileDialog::new().pick_file()
                {
                    self.set_file(path);
                }

                ui.label(self.message.to_string());
                return;
            };
            let parameter_store = self.parameter_store_map.get_mut(self.function);

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
            let message = &mut self.message;
            let function = self.function;
            let run_thread = &mut self.run_thread;
            ui.horizontal(|ui| {
                if ui.button("Run").clicked() {
                    *message = Self::run(data, function, run_thread, parameter_store);
                }
                ui.add_space(5.0);
                if ui.button("Reset").clicked() {
                    parameter_store.reset();
                }
                ui.add_space(5.0);
                if ui.button("Save Figure").clicked() {
                    *message = Self::save_figure(data, function, parameter_store);
                }
            });

            ui.add_space(5.0);

            // Message
            if let Some(message) = Self::read_run_thread(
                self.function,
                &mut self.run_thread,
                &mut self.parameter_store_map,
            ) {
                self.message = message;
            }
            ui.label(self.message.to_string());

            ui.add_space(10.0);

            // Approximation error
            let error = if let Some(parameters) =
                self.parameter_store_map.get(self.function).get_parameters()
            {
                format!(
                    "{}",
                    error(&data.x_ray, &data.y_ray, self.function, &parameters)
                )
            } else {
                "NaN".into()
            };
            ui.label(format!("Error: {}", error));

            ui.add_space(2.0);

            // Data figure
            if let Err(err) = self.show_figure(ui, data) {
                self.message = Message::Error(format!("Failed to show figure: {}", err));
            }
        });

        // normally, the GUI only updates when necessary, but when we have a run thread,
        // we want to read from it every once in a while.
        if self.run_thread.is_some() {
            ui.request_repaint_after(Duration::from_millis(100));
        }
    }
}
