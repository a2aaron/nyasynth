use std::sync::Arc;

use nih_plug::prelude::{Editor, Param, ParamSetter};
use nih_plug_egui::{
    create_egui_editor,
    egui::{self, Ui, WidgetText},
    EguiState,
};

use crate::params::Parameters;

fn make_slider(
    ui: &mut Ui,
    param_setter: &ParamSetter,
    label: impl Into<WidgetText>,
    param: &impl Param,
) {
    ui.label(label);
    ui.add(nih_plug_egui::widgets::ParamSlider::for_param(
        param,
        param_setter,
    ));
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
                make_slider(ui, param_setter, "Meow Attack", &params.meow_attack);
                make_slider(ui, param_setter, "Meow Decay", &params.meow_decay);
                make_slider(ui, param_setter, "Meow Sustain", &params.meow_sustain);
                make_slider(ui, param_setter, "Meow Release", &params.meow_release);
                make_slider(ui, param_setter, "Vibrato Amount", &params.vibrato_amount);
                make_slider(ui, param_setter, "Vibrato Attack", &params.vibrato_attack);
                make_slider(ui, param_setter, "Vibrato Rate", &params.vibrato_rate);
                make_slider(ui, param_setter, "Portamento Time", &params.portamento_time);
                make_slider(ui, param_setter, "Noise Mix", &params.noise_mix);
                make_slider(ui, param_setter, "Chorus Mix", &params.chorus_mix);
                make_slider(ui, param_setter, "Pitch Bend", &params.pitch_bend);
                make_slider(ui, param_setter, "Polycat", &params.polycat);
            });
        },
    )
}
