use std::sync::Arc;

use nih_plug::prelude::{Editor, Param, ParamSetter};
use nih_plug_egui::{
    create_egui_editor,
    egui::{
        self, pos2, vec2, Color32, ColorImage, Frame, Pos2, Rect, Rgba, Sense, Shape,
        TextureHandle, Ui, Vec2,
    },
    EguiState,
};

use crate::{
    params::Parameters,
    ui_knob::{ArcKnob, TextSlider},
};

const SCREEN_WIDTH: u32 = 450;
const SCREEN_HEIGHT: u32 = 300;

fn make_arc_knob(ui: &mut Ui, param_setter: &ParamSetter, param: &impl Param, center: Pos2) {
    // Knobs are 140.0x140.0 px, but need to scaled down by a factor of 4.
    let radius = 140.0 / 2.0 / 4.0;
    ui.add(ArcKnob::for_param(param, param_setter, radius, center));
}

fn make_text_slider(ui: &mut Ui, param_setter: &ParamSetter, param: &impl Param, location: Rect) {
    ui.add(TextSlider::for_param(param, param_setter, location));
}

struct WidgetLocations {
    meow_attack: Pos2,
    meow_decay: Pos2,
    meow_sustain: Pos2,
    meow_release: Pos2,
    vibrato_attack: Pos2,
    vibrato_amount: Pos2,
    vibrato_speed: Rect,
    portamento_time: Pos2,
    noise_mix: Pos2,
    chorus_mix: Pos2,
    pitch_bend: Rect,
    polycat_button: Rect,
    polycat_on: Rect,
    cat_image: Rect,
}
impl WidgetLocations {
    fn from_spine_json(value: serde_json::Value) -> WidgetLocations {
        use serde_json::Value;
        fn unwrap_object(value: &Value) -> &serde_json::Map<String, Value> {
            match value {
                Value::Object(map) => map,
                _ => panic!("Expected an object, got {}", value.to_string()),
            }
        }

        fn unwrap_f64(value: &Value) -> f64 {
            match value {
                Value::Number(number) => number.as_f64().unwrap(),
                _ => panic!("Expected a number, got {}", value.to_string()),
            }
        }

        /// The position data from Affinty Photo's Spine JSON export
        /// See http://en.esotericsoftware.com/spine-json-format for details
        /// It seems that the origin is located at the bottom left corner of the entire image
        /// and the y-axis increases going up.
        struct SpineData {
            /// The x-coordinate of the bottom left corner of the AABB
            x: f64,
            /// The y-coordinate of the bottom left corner of the AABB
            y: f64,
            /// The AABB width
            width: f64,
            /// The AABB height
            height: f64,
        }
        impl SpineData {
            fn center(&self, original_size: (f64, f64)) -> Pos2 {
                let x_scale_factor = SCREEN_WIDTH as f32 / original_size.0 as f32;
                let y_scale_factor = SCREEN_HEIGHT as f32 / original_size.1 as f32;
                pos2(
                    self.x as f32 * x_scale_factor,
                    (original_size.1 - self.y) as f32 * y_scale_factor,
                )
            }
            fn size(&self, original_size: (f64, f64)) -> Vec2 {
                let x_scale_factor = SCREEN_WIDTH as f32 / original_size.0 as f32;
                let y_scale_factor = SCREEN_HEIGHT as f32 / original_size.1 as f32;
                vec2(
                    self.width as f32 * x_scale_factor,
                    self.height as f32 * y_scale_factor,
                )
            }
            fn as_rect(&self, original_size: (f64, f64)) -> Rect {
                Rect::from_center_size(self.center(original_size), self.size(original_size))
            }
        }
        fn unwrap_data(data: &Value) -> SpineData {
            let obj = unwrap_object(data);
            let x = unwrap_f64(obj.get("x").unwrap());
            let y = unwrap_f64(obj.get("y").unwrap());
            let width = unwrap_f64(obj.get("width").unwrap());
            let height = unwrap_f64(obj.get("height").unwrap());
            SpineData {
                x,
                y,
                width,
                height,
            }
        }

        fn get_data(default_map: &serde_json::Map<String, Value>, key: &str) -> SpineData {
            let object = unwrap_object(default_map.get(key).unwrap());
            unwrap_data(object.get(key).unwrap())
        }

        let main_obj = unwrap_object(&value);
        let skins = unwrap_object(main_obj.get("skins").unwrap());
        let default = unwrap_object(skins.get("default").unwrap());
        let bg = get_data(default, "background");
        let original_size = (bg.width, bg.height);

        nih_plug::nih_log!(
            "rect: {:?}",
            get_data(default, "Image Emboss").as_rect(original_size)
        );

        WidgetLocations {
            meow_attack: get_data(default, "Meow Attack Knob").center(original_size),
            meow_decay: get_data(default, "Meow Decay Knob").center(original_size),
            meow_sustain: get_data(default, "Meow Sustain Knob").center(original_size),
            meow_release: get_data(default, "Meow Release Knob").center(original_size),
            vibrato_attack: get_data(default, "Vibrato Attack Knob").center(original_size),
            vibrato_amount: get_data(default, "Vibrato Amount Knob").center(original_size),
            vibrato_speed: get_data(default, "Vibrato Speed Emboss").as_rect(original_size),
            portamento_time: get_data(default, "Portamento Knob").center(original_size),
            noise_mix: get_data(default, "Noise Knob").center(original_size),
            chorus_mix: get_data(default, "Chorus Knob").center(original_size),
            pitch_bend: get_data(default, "Pitchbend Emboss").as_rect(original_size),
            polycat_button: get_data(default, "Polycat BG").as_rect(original_size),
            polycat_on: get_data(default, "POLYCAT ON").as_rect(original_size),
            cat_image: get_data(default, "Image Emboss").as_rect(original_size),
        }
    }
}

struct EditorState {
    cat_image: Option<TextureHandle>,
    brushed_metal: Option<TextureHandle>,
    polycat_on: Option<TextureHandle>,
    polycat_state: bool,
    widget_location: WidgetLocations,
}

impl EditorState {
    fn cat_image(&self) -> TextureHandle {
        self.cat_image.clone().unwrap()
    }

    fn brushed_metal(&self) -> TextureHandle {
        self.brushed_metal.clone().unwrap()
    }

    fn polycat_on(&self) -> TextureHandle {
        self.polycat_on.clone().unwrap()
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
    let egui_state = EguiState::from_size(SCREEN_WIDTH, SCREEN_HEIGHT);
    let user_state = EditorState {
        cat_image: None,
        brushed_metal: None,
        polycat_on: None,
        polycat_state: params.polycat.value(),
        widget_location: WidgetLocations::from_spine_json(
            serde_json::from_str(include_str!("../assets/spine_json/Spine.json")).unwrap(),
        ),
    };

    create_egui_editor(
        egui_state,
        user_state,
        |cx, editor_state| {
            let load_image = |name: &str, image: ColorImage| -> Option<TextureHandle> {
                Some(cx.load_texture(name, image, egui::TextureFilter::Linear))
            };
            let image = ColorImage::example();
            editor_state.cat_image = load_image("cat-image", image);

            let brushed_metal =
                load_image_from_memory(include_bytes!("../assets/ui_2x_v2.png")).unwrap();
            editor_state.brushed_metal = load_image("metal-knob", brushed_metal);

            let polycat_on =
                load_image_from_memory(include_bytes!("../assets/spine_json/POLYCAT ON.png"))
                    .unwrap();
            editor_state.polycat_on = load_image("polycat-on", polycat_on);
        },
        move |cx, param_setter, editor_state| {
            cx.set_debug_on_hover(true);
            egui::CentralPanel::default()
                .frame(
                    Frame::none()
                        .fill(Rgba::from_srgba_premultiplied(0x89, 0xA9, 0xBD, 0xFF).into()),
                )
                .show(cx, |ui| {
                    let background = image_shape(editor_state.brushed_metal(), ui.max_rect());
                    ui.painter().add(background);

                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            let image = image_shape(
                                editor_state.cat_image(),
                                editor_state.widget_location.cat_image,
                            );
                            ui.painter().add(image);
                        });
                        ui.vertical(|ui| {
                            ui.vertical(|ui| {
                                {
                                    let locations = &editor_state.widget_location;

                                    ui.horizontal(|ui| {
                                        make_arc_knob(
                                            ui,
                                            &param_setter,
                                            &params.meow_attack,
                                            locations.meow_attack,
                                        );
                                        make_arc_knob(
                                            ui,
                                            &param_setter,
                                            &params.meow_decay,
                                            locations.meow_decay,
                                        );
                                        make_arc_knob(
                                            ui,
                                            &param_setter,
                                            &params.meow_sustain,
                                            locations.meow_sustain,
                                        );
                                        make_arc_knob(
                                            ui,
                                            &param_setter,
                                            &params.meow_release,
                                            locations.meow_release,
                                        );
                                    });
                                    ui.horizontal(|ui| {
                                        make_arc_knob(
                                            ui,
                                            &param_setter,
                                            &params.vibrato_amount,
                                            locations.vibrato_amount,
                                        );
                                        make_arc_knob(
                                            ui,
                                            &param_setter,
                                            &params.vibrato_attack,
                                            locations.vibrato_attack,
                                        );
                                        make_text_slider(
                                            ui,
                                            param_setter,
                                            &params.vibrato_rate,
                                            locations.vibrato_speed,
                                        );
                                    });
                                    ui.horizontal(|ui| {
                                        make_arc_knob(
                                            ui,
                                            &param_setter,
                                            &params.portamento_time,
                                            locations.portamento_time,
                                        );
                                        make_arc_knob(
                                            ui,
                                            &param_setter,
                                            &params.noise_mix,
                                            locations.noise_mix,
                                        );
                                        make_arc_knob(
                                            ui,
                                            &param_setter,
                                            &params.chorus_mix,
                                            locations.chorus_mix,
                                        );
                                        make_text_slider(
                                            ui,
                                            param_setter,
                                            &params.pitch_bend,
                                            locations.pitch_bend,
                                        );
                                    });
                                }
                                ui.vertical_centered(|ui| {
                                    let button = ui.allocate_rect(
                                        editor_state.widget_location.polycat_button,
                                        Sense::click(),
                                    );
                                    if button.clicked() {
                                        editor_state.polycat_state = !editor_state.polycat_state;
                                        param_setter.begin_set_parameter(&params.polycat);
                                        param_setter.set_parameter(
                                            &params.polycat,
                                            editor_state.polycat_state,
                                        );
                                        param_setter.end_set_parameter(&params.polycat);
                                    }
                                    if editor_state.polycat_state {
                                        let shape = image_shape(
                                            editor_state.polycat_on(),
                                            editor_state.widget_location.polycat_on,
                                        );
                                        ui.painter().add(shape);
                                    };
                                    button
                                });
                            });
                        });
                    });
                });
        },
    )
}

fn image_shape(texture_handle: TextureHandle, rect: Rect) -> Shape {
    Shape::image(
        texture_handle.id(),
        rect,
        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
        Color32::WHITE,
    )
}
