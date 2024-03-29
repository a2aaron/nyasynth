use std::{
    error::Error,
    sync::{atomic::Ordering, Arc},
};

use atomic_float::AtomicF32;
use nih_plug::{
    nih_error,
    prelude::{Editor, Param, ParamSetter},
};
use nih_plug_egui::{
    create_egui_editor,
    egui::{
        self, pos2, vec2, Color32, ColorImage, FontDefinitions, FontId, Frame, Pos2, Rect, Rgba,
        Rounding, Sense, Shape, TextureHandle, Ui, Vec2,
    },
    EguiState,
};

use crate::{
    params::Parameters,
    ui_knob::{ArcKnob, TextSlider},
};

const SCREEN_WIDTH: u32 = 450;
const SCREEN_HEIGHT: u32 = 300;

fn make_arc_knob(ui: &mut Ui, setter: &ParamSetter, param: &impl Param, center: Pos2) {
    // Knobs are 140.0x140.0 px, but need to scaled down by a factor of 4.
    let radius = 140.0 / 2.0 / 4.0;
    ui.add(ArcKnob::for_param(param, setter, radius, center));
}

fn make_text_slider(ui: &mut Ui, setter: &ParamSetter, param: &impl Param, location: Rect) {
    ui.add(TextSlider::for_param(param, setter, location));
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
    nyasynth_logo: Rect,
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

        let cat_image = get_data(default, "Image Emboss")
            .as_rect(original_size)
            .shrink(10.0);

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
            nyasynth_logo: get_data(default, "Logo Bg").as_rect(original_size),
            cat_image,
        }
    }
}

struct EditorState {
    cat_images: Vec<Vec<TextureHandle>>,
    which_cat: usize,
    brushed_metal: Option<TextureHandle>,
    polycat_on: Option<TextureHandle>,
    polycat_state: bool,
    logo_open: bool,
    widget_location: WidgetLocations,
    envelope_amount: Arc<AtomicF32>,
}

impl EditorState {
    fn new(polycat_state: bool, envelope_amount: Arc<AtomicF32>) -> EditorState {
        EditorState {
            widget_location: WidgetLocations::from_spine_json(
                serde_json::from_str(include_str!("../assets/Spine.json")).unwrap(),
            ),
            cat_images: vec![],
            which_cat: 0,
            brushed_metal: None,
            polycat_on: None,
            polycat_state,
            logo_open: false,
            envelope_amount,
        }
    }

    fn cat_image(&self) -> TextureHandle {
        let cat_images = &self.cat_images[self.which_cat];
        let amount = self.envelope_amount.load(Ordering::Relaxed);
        let i = (amount * (cat_images.len() - 1) as f32).floor() as usize;
        let i = i.clamp(0, cat_images.len() - 1);
        cat_images[i].clone()
    }

    fn brushed_metal(&self) -> TextureHandle {
        self.brushed_metal.clone().unwrap()
    }

    fn polycat_on(&self) -> TextureHandle {
        self.polycat_on.clone().unwrap()
    }
}

fn load_image_from_memory(image_data: &[u8]) -> Result<ColorImage, Box<dyn Error>> {
    // Always use PNG here--this is the only format that we use anyways.
    let format = image::ImageFormat::Png;
    let image = image::load_from_memory_with_format(image_data, format)?;
    let size = [image.width() as _, image.height() as _];
    let image_buffer = image.to_rgba8();
    let pixels = image_buffer.as_flat_samples();
    Ok(ColorImage::from_rgba_unmultiplied(size, pixels.as_slice()))
}

pub fn get_editor(
    params: Arc<Parameters>,
    envelope_amount: Arc<AtomicF32>,
) -> Option<Box<dyn Editor>> {
    let egui_state = EguiState::from_size(SCREEN_WIDTH, SCREEN_HEIGHT);
    let editor_state = EditorState::new(params.polycat.value(), envelope_amount);

    create_egui_editor(
        egui_state,
        editor_state,
        |cx, editor_state| {
            let load_image = |name: &str, image: &[u8]| -> TextureHandle {
                let image = load_image_from_memory(image);
                let image = match image {
                    Ok(image) => image,
                    Err(err) => {
                        nih_error!(
                            "Couldn't load image {}. Reason: {:?}. Falling back to example image",
                            name,
                            err
                        );
                        ColorImage::example()
                    }
                };
                cx.load_texture(name, image, egui::TextureFilter::Linear)
            };

            let load_cat_image = |name: &str, image: &[u8]| -> TextureHandle {
                let image = load_image_from_memory(image);
                let image = match image {
                    Ok(image) => image,
                    Err(err) => {
                        nih_error!(
                            "Couldn't load image {}, Reason: {:?}. Falling back to example image",
                            name,
                            err
                        );
                        ColorImage::example()
                    }
                };
                // use nearest neighbor scaling for added Comedy
                cx.load_texture(name, image, egui::TextureFilter::Nearest)
            };

            let cat_images = &mut editor_state.cat_images;
            let baksik = [
                include_bytes!("../assets/cat-imgs/baksik/0.png").as_slice(),
                include_bytes!("../assets/cat-imgs/baksik/1.png").as_slice(),
                include_bytes!("../assets/cat-imgs/baksik/2.png").as_slice(),
                include_bytes!("../assets/cat-imgs/baksik/3.png").as_slice(),
                include_bytes!("../assets/cat-imgs/baksik/4.png").as_slice(),
                include_bytes!("../assets/cat-imgs/baksik/5.png").as_slice(),
                include_bytes!("../assets/cat-imgs/baksik/6.png").as_slice(),
                include_bytes!("../assets/cat-imgs/baksik/7.png").as_slice(),
                include_bytes!("../assets/cat-imgs/baksik/8.png").as_slice(),
                include_bytes!("../assets/cat-imgs/baksik/9.png").as_slice(),
            ]
            .iter()
            .enumerate()
            .map(|(i, img)| load_cat_image(&format!("baksik-{}", i), img))
            .collect();
            cat_images.push(baksik);

            let severian = [
                include_bytes!("../assets/cat-imgs/severian/out8.png").as_slice(),
                include_bytes!("../assets/cat-imgs/severian/out10.png").as_slice(),
                include_bytes!("../assets/cat-imgs/severian/out13.png").as_slice(),
                include_bytes!("../assets/cat-imgs/severian/out16.png").as_slice(),
                include_bytes!("../assets/cat-imgs/severian/out19.png").as_slice(),
                include_bytes!("../assets/cat-imgs/severian/out22.png").as_slice(),
                include_bytes!("../assets/cat-imgs/severian/out24.png").as_slice(),
                include_bytes!("../assets/cat-imgs/severian/out26.png").as_slice(),
                include_bytes!("../assets/cat-imgs/severian/out28.png").as_slice(),
                include_bytes!("../assets/cat-imgs/severian/out30.png").as_slice(),
            ]
            .iter()
            .enumerate()
            .map(|(i, img)| load_cat_image(&format!("severian-{}", i), img))
            .collect();
            cat_images.push(severian);

            let chrysi = [
                include_bytes!("../assets/cat-imgs/chrysi/0.png").as_slice(),
                include_bytes!("../assets/cat-imgs/chrysi/1.png").as_slice(),
                include_bytes!("../assets/cat-imgs/chrysi/2.png").as_slice(),
                include_bytes!("../assets/cat-imgs/chrysi/3.png").as_slice(),
                include_bytes!("../assets/cat-imgs/chrysi/4.png").as_slice(),
                include_bytes!("../assets/cat-imgs/chrysi/5.png").as_slice(),
                include_bytes!("../assets/cat-imgs/chrysi/6.png").as_slice(),
                include_bytes!("../assets/cat-imgs/chrysi/7.png").as_slice(),
                include_bytes!("../assets/cat-imgs/chrysi/8.png").as_slice(),
                include_bytes!("../assets/cat-imgs/chrysi/9.png").as_slice(),
            ]
            .iter()
            .enumerate()
            .map(|(i, img)| load_cat_image(&format!("chrysi-{}", i), img))
            .collect();
            cat_images.push(chrysi);

            editor_state.brushed_metal = Some(load_image(
                "metal-knob",
                include_bytes!("../assets/ui_2x_v2.png"),
            ));

            editor_state.polycat_on = Some(load_image(
                "polycat-on",
                include_bytes!("../assets/POLYCAT ON.png"),
            ));

            let determination =
                egui::FontData::from_static(include_bytes!("../assets/DTM-Mono.otf"));

            let mut fonts = FontDefinitions::default();
            fonts
                .font_data
                .insert("Determination".to_string(), determination);

            // Put my font as last fallback for monospace:
            fonts
                .families
                .entry(egui::FontFamily::Monospace)
                .or_default()
                .insert(0, "Determination".to_owned());
            cx.set_fonts(fonts);
        },
        move |cx, setter, editor_state| {
            egui::CentralPanel::default()
                .frame(
                    Frame::none()
                        .fill(Rgba::from_srgba_premultiplied(0x89, 0xA9, 0xBD, 0xFF).into()),
                )
                .show(cx, |ui| {
                    let locs = &editor_state.widget_location;

                    // UI Background
                    let background = image_shape(editor_state.brushed_metal(), ui.max_rect());
                    ui.painter().add(background);

                    // Cat Image
                    let image = image_shape(editor_state.cat_image(), locs.cat_image);
                    ui.painter().add(image);
                    let cat_image = ui.allocate_rect(locs.cat_image, Sense::click());
                    if cat_image.clicked() {
                        editor_state.which_cat =
                            (editor_state.which_cat + 1) % editor_state.cat_images.len()
                    }

                    // Knobs
                    make_arc_knob(ui, &setter, &params.meow_attack, locs.meow_attack);
                    make_arc_knob(ui, &setter, &params.meow_decay, locs.meow_decay);
                    make_arc_knob(ui, &setter, &params.meow_sustain, locs.meow_sustain);
                    make_arc_knob(ui, &setter, &params.meow_release, locs.meow_release);
                    make_arc_knob(ui, &setter, &params.vibrato_amount, locs.vibrato_amount);
                    make_arc_knob(ui, &setter, &params.vibrato_attack, locs.vibrato_attack);
                    make_text_slider(ui, setter, &params.vibrato_rate, locs.vibrato_speed);
                    make_arc_knob(ui, &setter, &params.portamento_time, locs.portamento_time);
                    make_arc_knob(ui, &setter, &params.noise_mix, locs.noise_mix);
                    make_arc_knob(ui, &setter, &params.chorus_mix, locs.chorus_mix);
                    make_text_slider(ui, setter, &params.pitch_bend, locs.pitch_bend);

                    // Polycat Button
                    let button = ui.allocate_rect(locs.polycat_button, Sense::click());
                    if button.clicked() {
                        editor_state.polycat_state = !editor_state.polycat_state;
                        setter.begin_set_parameter(&params.polycat);
                        setter.set_parameter(&params.polycat, editor_state.polycat_state);
                        setter.end_set_parameter(&params.polycat);
                    }
                    if editor_state.polycat_state {
                        let shape = image_shape(editor_state.polycat_on(), locs.polycat_on);
                        ui.painter().add(shape);
                    };

                    // Nyasynth Logo

                    // Nyasynth Text
                    let nyasynth_logo = ui.allocate_rect(locs.nyasynth_logo, Sense::click());
                    if nyasynth_logo.clicked() {
                        editor_state.logo_open = !editor_state.logo_open;
                    }
                    if editor_state.logo_open {
                        let painter = ui.painter();
                        painter.rect_filled(
                            locs.cat_image,
                            Rounding::same(1.0),
                            Color32::from_black_alpha(200),
                        );
                        let font_id = FontId::monospace(11.0);
                        let color = Color32::WHITE;
                        let galley = painter.layout(
                            CREDITS_TEXT.to_string(),
                            font_id,
                            color,
                            locs.cat_image.width(),
                        );
                        painter.galley(locs.cat_image.left_top() + vec2(2.0, 2.0), galley);
                    }
                });
        },
    )
}

const CREDITS_TEXT: &str = r#"Determination font by Haley Wakamatsu
behance.net/JapanYoshi

Cat gif of Baksik originally from Meowsynth
web.archive.org/web/20120930004514/myspace.com/baksik

Nyasynth programmed by a2aaron
github.com/a2aaron/nyasynth
"#;

fn image_shape(texture_handle: TextureHandle, rect: Rect) -> Shape {
    Shape::image(
        texture_handle.id(),
        rect,
        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
        Color32::WHITE,
    )
}
