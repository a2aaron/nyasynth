use std::f32::consts::TAU;

use nih_plug::prelude::{Param, ParamSetter};
use nih_plug_egui::egui::{
    epaint::PathShape, pos2, vec2, Color32, Pos2, Response, Rgba, Sense, Shape, Stroke, Ui, Widget,
};

use crate::ease::lerp;

pub struct ArcKnob<'a, P: Param> {
    param: &'a P,
    param_setter: &'a ParamSetter<'a>,
    radius: f32,
}

impl<'a, P: Param> ArcKnob<'a, P> {
    pub fn for_param(param: &'a P, param_setter: &'a ParamSetter) -> Self {
        ArcKnob {
            param,
            param_setter,
            radius: 20.0,
        }
    }
}

impl<'a, P: Param> Widget for ArcKnob<'a, P> {
    fn ui(self, ui: &mut Ui) -> Response {
        let size = vec2(self.radius * 2.0 + 4.0, self.radius * 2.0 + 4.0);
        let response = ui.allocate_response(size, Sense::click_and_drag());
        if response.drag_started() {
            self.param_setter.begin_set_parameter(self.param);
        }

        let value = self.param.unmodulated_normalized_value();
        if response.dragged() {
            // Invert the y axis, since we want dragging up to increase the value and down to
            // decrease it, but drag_delta() has the y-axis increasing downwards.
            let delta = -response.drag_delta().y;
            let new_value = (value + delta / 100.0).clamp(0.0, 1.0);
            self.param_setter
                .set_parameter_normalized(self.param, new_value);
        }

        if response.drag_released() {
            self.param_setter.end_set_parameter(self.param);
        }

        let painter = ui.painter_at(response.rect);
        let center = response.rect.center();
        // Draw the background for the knob
        painter.circle_filled(center, self.radius, Rgba::BLACK);

        // Draw the arc
        let radius = self.radius - 4.0;
        let stroke = Stroke::new(4.0, Rgba::RED);
        let shape = Shape::Path(PathShape {
            points: get_arc_points(center, radius, value, 0.05),
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
