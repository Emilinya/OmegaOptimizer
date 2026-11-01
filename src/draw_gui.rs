use std::{
    io::Write,
    path::{Path, PathBuf},
};

use eframe::{
    egui::{self, Button, Color32, Id, Modal, Pos2, Sense, Widget},
    epaint::CircleShape,
};

struct PointText {
    point: Option<Pos2>,
    text: String,
}

impl PointText {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            point: Some(Pos2::new(x, y)),
            text: format!("({}, {})", x, y),
        }
    }

    pub fn as_input(&mut self, ui: &mut egui::Ui, label: &str) {
        let label = ui.label(label);
        let mut input = egui::TextEdit::singleline(&mut self.text);
        if self.point.is_none() {
            input = input.text_color(Color32::RED);
        }
        if input.ui(ui).labelled_by(label.id).changed() {
            self.point = if let Some(stripped) = self.text.strip_prefix('(')
                && let Some(stripped) = stripped.strip_suffix(')')
                && let Some((x, y)) = stripped.split_once(',')
                && let (Ok(x), Ok(y)) = (x.trim().parse(), y.trim().parse())
            {
                Some(Pos2::new(x, y))
            } else {
                None
            };
        }
    }
}

pub struct Draw {
    modal_open: bool,
    filename: String,
    bottom_left: PointText,
    top_right: PointText,
    points: Vec<Pos2>,
    path: Option<PathBuf>,
}

impl Draw {
    pub fn new() -> Self {
        Self {
            modal_open: false,
            filename: String::new(),
            bottom_left: PointText::new(-1.0, -1.0),
            top_right: PointText::new(1.0, 1.0),
            points: Vec::new(),
            path: None,
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) -> Option<PathBuf> {
        ui.horizontal(|ui| {
            if ui.button("Save").clicked() {
                self.modal_open = true;
            }
            ui.separator();
            if ui.button("Clear").clicked() {
                self.points.clear();
            }
        });

        if self.modal_open {
            let modal = Modal::new(Id::unique("Save modal")).show(ui.ctx(), |ui| {
                ui.set_width(250.0);

                let label = ui.label("Name:");
                ui.text_edit_singleline(&mut self.filename)
                    .labelled_by(label.id);

                ui.separator();

                ui.label("Bounding box:");
                ui.add_space(2.0);

                self.bottom_left.as_input(ui, "Bottom Left");
                self.top_right.as_input(ui, "Top Right");

                ui.separator();

                egui::Sides::new().show(
                    ui,
                    |_ui| {},
                    |ui| {
                        if let (Some(top_right), Some(bottom_left)) =
                            (self.top_right.point, self.bottom_left.point)
                        {
                            if ui.button("Save").clicked() {
                                match self.save(top_right, bottom_left) {
                                    Ok(path) => self.path = Some(path),
                                    Err(err) => log::error!("{err:#}"),
                                }
                                ui.close();
                            }
                        } else {
                            ui.add_enabled(false, Button::new("Save"));
                        }
                        if ui.button("Cancel").clicked() {
                            ui.close();
                        }
                    },
                );
            });

            if modal.should_close() {
                self.modal_open = false;
            }
        }

        egui::Frame::canvas(ui.style()).show(ui, |ui| {
            let (mut response, painter) =
                ui.allocate_painter(ui.available_size_before_wrap(), Sense::drag());

            if let Some(pointer_pos) = response.interact_pointer_pos()
                && self
                    .points
                    .last()
                    .is_none_or(|last| last.distance(pointer_pos) > 5.0)
            {
                self.points.push(pointer_pos);
                response.mark_changed();
            }

            let shapes = self
                .points
                .iter()
                .map(|point| egui::Shape::Circle(CircleShape::filled(*point, 2.0, Color32::GRAY)));
            painter.extend(shapes);

            response
        });

        self.path.take()
    }

    fn save(&self, top_right: Pos2, bottom_left: Pos2) -> anyhow::Result<PathBuf> {
        let path = Path::new("data").join(&self.filename);
        let mut file = fs_err::File::create(&path)?;

        let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
        let (mut max_x, mut max_y) = (-f32::INFINITY, -f32::INFINITY);
        for point in &self.points {
            min_x = f32::min(min_x, point.x);
            min_y = f32::min(min_y, point.y);
            max_x = f32::max(max_x, point.x);
            max_y = f32::max(max_y, point.y);
        }

        for point in &self.points {
            let x = (point.x - min_x) / (max_x - min_x);
            let x = x * (top_right.x - bottom_left.x) + bottom_left.x;

            let y = 1.0 - (point.y - min_y) / (max_y - min_y);
            let y = y * (top_right.y - bottom_left.y) + bottom_left.y;

            writeln!(file, "{} {}", x, y)?;
        }

        Ok(path)
    }
}
