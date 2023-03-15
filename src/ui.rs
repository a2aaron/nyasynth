use std::sync::Arc;

use nih_plug::prelude::{Editor, Param, ParamSetter};
use nih_plug_egui::{
    create_egui_editor,
    egui::{self, vec2, Label, Layout, Ui, WidgetText},
    EguiState,
};

use crate::{params::Parameters, ui_knob::ArcKnob};

fn make_arc_knob_no_label(ui: &mut Ui, param_setter: &ParamSetter, param: &impl Param) {
    ui.add(ArcKnob::for_param(param, param_setter));
}

fn make_arc_knob(
    ui: &mut Ui,
    param_setter: &ParamSetter,
    label: impl Into<WidgetText>,
    param: &impl Param,
) {
    let layout = Layout::top_down(egui::Align::Center).with_cross_justify(false);
    ui.allocate_ui_with_layout(vec2(50.0, 50.0), layout, |ui| {
        ui.add(Label::new(label).wrap(false));
        ui.add(ArcKnob::for_param(param, param_setter));
    });
}

fn make_slider(
    ui: &mut Ui,
    param_setter: &ParamSetter,
    label: impl Into<WidgetText>,
    param: &impl Param,
) {
    ui.vertical(|ui| {
        ui.label(label);
        ui.add(nih_plug_egui::widgets::ParamSlider::for_param(
            param,
            param_setter,
        ));
    });
}

pub fn get_editor(params: Arc<Parameters>) -> Option<Box<dyn Editor>> {
    let egui_state = EguiState::from_size(800, 600);
    let user_state = ();

    create_egui_editor(
        egui_state,
        user_state,
        |_cx, _user_state| {},
        move |cx, param_setter, _user_state| {
            egui::CentralPanel::default().show(cx, |ui| {
                cx.set_debug_on_hover(true);
                ui.vertical(|ui| {
                    ui.label("MEOW ENVELOPE");
                    ui.horizontal_top(|ui| {
                        make_arc_knob_no_label(ui, param_setter, &params.meow_attack);
                        make_arc_knob_no_label(ui, param_setter, &params.meow_decay);
                        make_arc_knob_no_label(ui, param_setter, &params.meow_sustain);
                        make_arc_knob_no_label(ui, param_setter, &params.meow_release);
                    });
                    ui.label("VIBRATO");
                    ui.horizontal_top(|ui| {
                        make_arc_knob(ui, param_setter, "AMNT", &params.vibrato_amount);
                        make_arc_knob(ui, param_setter, "ATCK", &params.vibrato_attack);
                        make_slider(ui, param_setter, "Vibrato Rate", &params.vibrato_rate);
                    });
                    ui.horizontal_top(|ui| {
                        make_arc_knob(ui, param_setter, "PORTA", &params.portamento_time);
                        make_arc_knob(ui, param_setter, "NOISE", &params.noise_mix);
                        make_arc_knob(ui, param_setter, "CHORUS", &params.chorus_mix);
                        make_slider(ui, param_setter, "P. BEND", &params.pitch_bend);
                    });
                    make_slider(ui, param_setter, "POLYCAT", &params.polycat);
                })
            });
        },
    )
}
