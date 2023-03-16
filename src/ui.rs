use std::sync::Arc;

use nih_plug::prelude::{Editor, Param, ParamSetter};
use nih_plug_egui::{
    create_egui_editor,
    egui::{
        self, vec2, ColorImage, Frame, Label, Rgba, RichText, TextureHandle, Ui, Vec2, WidgetText,
    },
    EguiState,
};

use crate::{
    params::Parameters,
    ui_knob::{ArcKnob, TextSlider},
};

const YELLOW: Rgba = Rgba::from_rgb(1.0, 1.0, 0.0);

fn make_arc_knob_no_label(ui: &mut Ui, ctx: &EditorContext, param: &impl Param) {
    ui.add(ArcKnob::for_param(
        param,
        ctx.param_setter,
        ctx.editor.metal_knob(),
    ));
}

fn make_arc_knob(
    ui: &mut Ui,
    ctx: &EditorContext,
    label: impl Into<WidgetText>,
    param: &impl Param,
) {
    ui.vertical(|ui| {
        ui.label(label);
        ui.add(ArcKnob::for_param(
            param,
            ctx.param_setter,
            ctx.editor.metal_knob(),
        ));
    });
}

fn make_text_slider(
    ui: &mut Ui,
    param_setter: &ParamSetter,
    label: impl Into<WidgetText>,
    param: &impl Param,
    size: Vec2,
) {
    ui.vertical(|ui| {
        ui.label(label);
        ui.add(TextSlider::for_param(param, param_setter, size));
    });
}

struct EditorContext<'a> {
    editor: &'a EditorState,
    param_setter: &'a ParamSetter<'a>,
}

struct EditorState {
    cat_image: Option<TextureHandle>,
    metal_knob: Option<TextureHandle>,
    polycat_state: bool,
}

impl EditorState {
    fn cat_image(&self) -> TextureHandle {
        self.cat_image.clone().unwrap()
    }

    fn metal_knob(&self) -> TextureHandle {
        self.metal_knob.clone().unwrap()
    }
}

fn load_image_from_memory(image_data: &[u8]) -> Result<ColorImage, image::ImageError> {
    let image = image::load_from_memory(image_data)?;
    let size = [image.width() as _, image.height() as _];
    let image_buffer = image.to_rgba8();
    let pixels = image_buffer.as_flat_samples();
    Ok(ColorImage::from_rgba_unmultiplied(size, pixels.as_slice()))
}

pub fn get_editor(params: Arc<Parameters>) -> Option<Box<dyn Editor>> {
    let egui_state = EguiState::from_size(450, 300);
    let user_state = EditorState {
        cat_image: None,
        metal_knob: None,
        polycat_state: params.polycat.value(),
    };

    create_egui_editor(
        egui_state,
        user_state,
        |cx, editor_state| {
            let image = ColorImage::example();
            editor_state.cat_image =
                Some(cx.load_texture("cat-image", image, egui::TextureFilter::Linear));

            let knob_texture =
                load_image_from_memory(include_bytes!("../assets/metal_knob_color.png")).unwrap();
            editor_state.metal_knob =
                Some(cx.load_texture("metal-knob", knob_texture, egui::TextureFilter::Linear))
        },
        move |cx, param_setter, editor_state| {
            cx.set_debug_on_hover(true);
            egui::CentralPanel::default()
                .frame(
                    Frame::none()
                        .fill(Rgba::from_srgba_premultiplied(0x89, 0xA9, 0xBD, 0xFF).into()),
                )
                .show(cx, |ui| {
                    ui.horizontal(|ui| {
                        ui.add_space(16.0);
                        ui.vertical(|ui| {
                            ui.add_space(18.0);
                            ui.image(&editor_state.cat_image(), vec2(200.0, 160.0));
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
                            ui.vertical(|ui| {
                                let ctx = EditorContext {
                                    editor: &editor_state,
                                    param_setter,
                                };

                                ui.horizontal(|ui| {
                                    make_arc_knob_no_label(ui, &ctx, &params.meow_attack);
                                    make_arc_knob_no_label(ui, &ctx, &params.meow_decay);
                                    make_arc_knob_no_label(ui, &ctx, &params.meow_sustain);
                                    make_arc_knob_no_label(ui, &ctx, &params.meow_release);
                                });
                                ui.label("VIBRATO");
                                ui.horizontal(|ui| {
                                    make_arc_knob(ui, &ctx, "AMNT", &params.vibrato_amount);
                                    make_arc_knob(ui, &ctx, "ATCK", &params.vibrato_attack);
                                    make_text_slider(
                                        ui,
                                        param_setter,
                                        "SPEED",
                                        &params.vibrato_rate,
                                        vec2(64.0, 25.0),
                                    );
                                });
                                ui.horizontal(|ui| {
                                    make_arc_knob(ui, &ctx, "PORTA", &params.portamento_time);
                                    make_arc_knob(ui, &ctx, "NOISE", &params.noise_mix);
                                    make_arc_knob(ui, &ctx, "CHORUS", &params.chorus_mix);
                                    make_text_slider(
                                        ui,
                                        param_setter,
                                        "P. BEND",
                                        &params.pitch_bend,
                                        vec2(25.0, 25.0),
                                    );
                                });
                                ui.vertical_centered(|ui| {
                                    let polycat_text = RichText::new("POLYCAT").size(32.0);
                                    let button = ui.toggle_value(
                                        &mut editor_state.polycat_state,
                                        polycat_text,
                                    );
                                    if button.clicked() {
                                        param_setter.begin_set_parameter(&params.polycat);
                                        param_setter.set_parameter(
                                            &params.polycat,
                                            editor_state.polycat_state,
                                        );
                                        param_setter.end_set_parameter(&params.polycat);
                                    }
                                })
                            });
                        });
                    });
                });
        },
    )
}
