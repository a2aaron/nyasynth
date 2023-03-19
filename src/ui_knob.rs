use std::f32::consts::TAU;

use nih_plug::prelude::{Param, ParamSetter};
use nih_plug_egui::egui::{
    epaint::PathShape, pos2, vec2, Align2, Color32, FontId, Id, Pos2, Rect, Response, Rgba, Sense,
    Shape, Stroke, Ui, Widget,
};
use once_cell::sync::Lazy;

use crate::common::lerp;

static DRAG_AMOUNT_MEMORY_ID: Lazy<Id> = Lazy::new(|| Id::new("drag_amount_memory_id"));

struct SliderRegion<'a, P: Param> {
    param: &'a P,
    param_setter: &'a ParamSetter<'a>,
}

impl<'a, P: Param> SliderRegion<'a, P> {
    fn new(param: &'a P, param_setter: &'a ParamSetter) -> Self {
        SliderRegion {
            param,
            param_setter,
        }
    }

    // Handle the input for a given response. Returns an f32 containing the normalized value of
    // the parameter.
    fn handle_response(&self, ui: &Ui, response: &Response) -> f32 {
        let value = self.param.unmodulated_normalized_value();
        if response.drag_started() {
            self.param_setter.begin_set_parameter(self.param);
            ui.memory().data.insert_temp(*DRAG_AMOUNT_MEMORY_ID, value)
        }

        if response.dragged() {
            // Invert the y axis, since we want dragging up to increase the value and down to
            // decrease it, but drag_delta() has the y-axis increasing downwards.
            let delta = -response.drag_delta().y;
            let mut memory = ui.memory();
            let value = memory.data.get_temp_mut_or(*DRAG_AMOUNT_MEMORY_ID, value);
            *value = (*value + delta / 100.0).clamp(0.0, 1.0);
            self.param_setter
                .set_parameter_normalized(self.param, *value);
        }

        if response.drag_released() {
            self.param_setter.end_set_parameter(self.param);
        }
        value
    }

    fn get_string(&self) -> String {
        self.param.to_string()
    }
}

pub struct ArcKnob<'a, P: Param> {
    slider_region: SliderRegion<'a, P>,
    radius: f32,
    center: Pos2,
}

impl<'a, P: Param> ArcKnob<'a, P> {
    pub fn for_param(param: &'a P, param_setter: &'a ParamSetter, radius: f32, pos: Pos2) -> Self {
        ArcKnob {
            slider_region: SliderRegion::new(param, param_setter),
            radius,
            center: pos,
        }
    }
}

impl<'a, P: Param> Widget for ArcKnob<'a, P> {
    fn ui(self, ui: &mut Ui) -> Response {
        let size = vec2(self.radius * 2.0, self.radius * 2.0);
        let rect = Rect::from_center_size(self.center, size);
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        let value = self.slider_region.handle_response(&ui, &response);

        let painter = ui.painter_at(response.rect);
        let center = response.rect.center();

        // Draw the arc
        let stroke_width = 5.0;
        let radius = self.radius - stroke_width - 2.0;
        let stroke = Stroke::new(stroke_width, Rgba::from_rgb(1.0, 1.0, 0.0));
        let shape = Shape::Path(PathShape {
            points: get_arc_points(center, radius, value, 0.03),
            closed: false,
            fill: Color32::TRANSPARENT,
            stroke,
        });
        painter.add(shape);
        response
    }
}

fn get_arc_points(center: Pos2, radius: f32, value: f32, max_arc_distance: f32) -> Vec<Pos2> {
    let start_turns: f32 = 0.625;
    let arc_length = lerp(0.0, -0.75, value);
    let end_turns = start_turns + arc_length;

    let points = (arc_length.abs() / max_arc_distance).ceil() as usize;
    let points = points.max(1);
    (0..=points)
        .map(|i| {
            let t = i as f32 / (points - 1) as f32;
            let angle = lerp(start_turns * TAU, end_turns * TAU, t);
            let x = radius * angle.cos();
            let y = -radius * angle.sin();
            pos2(x, y) + center.to_vec2()
        })
        .collect()
}

pub struct TextSlider<'a, P: Param> {
    slider_region: SliderRegion<'a, P>,
    location: Rect,
}

impl<'a, P: Param> TextSlider<'a, P> {
    pub fn for_param(param: &'a P, param_setter: &'a ParamSetter, location: Rect) -> Self {
        TextSlider {
            slider_region: SliderRegion::new(param, param_setter),
            location,
        }
    }
}

impl<'a, P: Param> Widget for TextSlider<'a, P> {
    fn ui(self, ui: &mut Ui) -> Response {
        let response = ui.allocate_rect(self.location, Sense::click_and_drag());
        self.slider_region.handle_response(&ui, &response);

        let painter = ui.painter_at(self.location);
        let center = self.location.center();

        // Draw the text
        let text = self.slider_region.get_string();
        let anchor = Align2::CENTER_CENTER;
        let color = Color32::from(Rgba::WHITE);
        let font = FontId::monospace(16.0);
        painter.text(center, anchor, text, font, color);
        response
    }
}
