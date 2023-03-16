use std::sync::Arc;

use nih_plug::prelude::{Editor, Param, ParamSetter};
use nih_plug_egui::{
    create_egui_editor,
    egui::{self, vec2, ColorImage, Label, Rgba, RichText, TextureHandle, Ui, WidgetText},
    EguiState,
};

use crate::{params::Parameters, ui_knob::ArcKnob};

const YELLOW: Rgba = Rgba::from_rgb(1.0, 1.0, 0.0);

fn make_arc_knob_no_label(ui: &mut Ui, param_setter: &ParamSetter, param: &impl Param) {
    ui.add(ArcKnob::for_param(param, param_setter));
}

fn make_arc_knob(
    ui: &mut Ui,
    param_setter: &ParamSetter,
    label: impl Into<WidgetText>,
    param: &impl Param,
) {
    ui.vertical(|ui| {
        ui.label(label);
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

struct EditorState {
    cat_image: Option<TextureHandle>,
    polycat_state: bool,
}

impl EditorState {
    fn cat_image(&self) -> &TextureHandle {
        self.cat_image.as_ref().unwrap()
    }
}

pub fn get_editor(params: Arc<Parameters>) -> Option<Box<dyn Editor>> {
    let egui_state = EguiState::from_size(450, 300);
    let user_state = EditorState {
        cat_image: None,
        polycat_state: params.polycat.value(),
    };

    create_egui_editor(
        egui_state,
        user_state,
        |cx, editor_state| {
            let image = ColorImage::example();
            editor_state.cat_image =
                Some(cx.load_texture("cat-image", image, egui::TextureFilter::Linear));
        },
        move |cx, param_setter, editor_state| {
            cx.set_debug_on_hover(true);
            egui::CentralPanel::default().show(cx, |ui| {
                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    ui.vertical(|ui| {
                        ui.add_space(18.0);
                        ui.image(editor_state.cat_image(), vec2(200.0, 160.0));
                        ui.add_space(12.0);
                        let background_color = Rgba::from_white_alpha(0.1);
                        let nyasynth = RichText::new("NYASynth")
                            .color(YELLOW)
                            .background_color(background_color)
                            .size(50.0);
                        let label = Label::new(nyasynth);
                        ui.add_sized(vec2(210.0, 75.0), label);
                    });
                    ui.vertical(|ui| {
                        ui.add_space(18.0);
                        ui.label("MEOW ENVELOPE");
                        ui.horizontal(|ui| {
                            make_arc_knob_no_label(ui, param_setter, &params.meow_attack);
                            make_arc_knob_no_label(ui, param_setter, &params.meow_decay);
                            make_arc_knob_no_label(ui, param_setter, &params.meow_sustain);
                            make_arc_knob_no_label(ui, param_setter, &params.meow_release);
                        });
                        ui.label("VIBRATO");
                        ui.horizontal(|ui| {
                            make_arc_knob(ui, param_setter, "AMNT", &params.vibrato_amount);
                            make_arc_knob(ui, param_setter, "ATCK", &params.vibrato_attack);
                            make_slider(ui, param_setter, "SPEED", &params.vibrato_rate);
                        });
                        ui.horizontal(|ui| {
                            make_arc_knob(ui, param_setter, "PORTA", &params.portamento_time);
                            make_arc_knob(ui, param_setter, "NOISE", &params.noise_mix);
                            make_arc_knob(ui, param_setter, "CHORUS", &params.chorus_mix);
                            make_slider(ui, param_setter, "P. BEND", &params.pitch_bend);
                        });
                        let button = ui.toggle_value(&mut editor_state.polycat_state, "POLYCAT");
                        if button.clicked() {
                            param_setter.begin_set_parameter(&params.polycat);
                            param_setter.set_parameter(&params.polycat, editor_state.polycat_state);
                            param_setter.end_set_parameter(&params.polycat);
                        }
                    });
                });
            });
        },
    )
}
