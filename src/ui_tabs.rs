use std::{hash::Hash, path::PathBuf};

use iced::{
    button, widget, Align, Background, Column, Element, HorizontalAlignment, Point, Rectangle, Row,
    Vector, VerticalAlignment,
};
use iced_audio::{
    knob::{self, LineCap},
    v_slider, Knob, NormalParam, VSlider,
};
use iced_graphics::{
    canvas::{Frame, LineJoin, Stroke},
    Backend, Primitive, Renderer,
};
use iced_native::{layout, mouse, Clipboard, Widget};

use crate::{
    ease::snap_float,
    params::{
        biquad_to_string, to_filter_type, EnvelopeParam, ModBankParameter, ModBankType,
        ModulationSend, ModulationType, OSCParameterType, OSCType, ParameterType, RawEnvelope,
        RawParameters,
    },
    presets::try_parse_file,
    sound_gen::NoteShape,
    ui::Message,
};

const LABEL_WIDTH: u16 = 45;
const KNOB_SPACING: u16 = 12;
const KNOB_SIZE: iced::Length = iced::Length::Units(29);
const PANE_SPACING: u16 = 15;
const PANE_PADDING: u16 = 15;
const INTERPANE_SPACING: u16 = 15;
const MOD_TYPE_SELECTOR_WIDTH: u16 = 200;
const MASTER_VOL_HEIGHT: u16 = 300;

const RED: iced::Color = iced::Color {
    r: 1.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};

const GREEN: iced::Color = iced::Color {
    r: 0.0,
    g: 1.0,
    b: 0.0,
    a: 1.0,
};

const BLACK_GREEN: iced::Color = iced::Color {
    r: 0.0,
    g: 0.2,
    b: 0.0,
    a: 1.0,
};

const GREY: iced::Color = iced::Color {
    r: 0.33,
    g: 0.33,
    b: 0.33,
    a: 1.0,
};

const GREEN_TRANS: iced::Color = iced::Color {
    r: 0.0,
    g: 1.0,
    b: 0.0,
    a: 0.5,
};

/// Use: widget!(identifier, state, param, title);
/// Create a widget (actually a column) that:
/// 1. Uses `$state` as the widget's `$widget::State` struct
/// 2. Sends the message ParameterChanged(normal, $parameter)
/// 3. Has the title `title` (or `widget_name($parameter)` if not provided)
macro_rules! widget {
    (VSlider, $state:expr, $parameter:expr, $title:expr) => {
        with_label(
            VSlider::new($state, move |normal| {
                Message::ParameterChanged($parameter, normal.as_f32())
            }),
            $title,
        )
    };

    ($widget:ident, $state:expr, $parameter:expr, $title:expr) => {
        with_label(
            $widget::new($state, move |normal| {
                Message::ParameterChanged($parameter, normal.as_f32())
            })
            .size(KNOB_SIZE),
            $title,
        )
    };

    ($widget:ident, $state:expr, $parameter:expr) => {
        widget!($widget, $state, $parameter, &widget_name($parameter))
    };
}

/// The main tab of the GUI, holding all the main knob stuff.
pub struct MainTab {
    pub master_vol: v_slider::State,
    pub osc_1: OSCKnobs,
    pub osc_2: OSCKnobs,
}

impl MainTab {
    pub fn new(params: &RawParameters) -> MainTab {
        MainTab {
            master_vol: v_slider::State::new(make_normal_param(
                params,
                ParameterType::MasterVolume,
            )),
            osc_1: OSCKnobs::new(params, OSCType::OSC1),
            osc_2: OSCKnobs::new(params, OSCType::OSC2),
        }
    }

    pub fn update_snapping_knobs(&mut self, osc_type: OSCType, osc_param: OSCParameterType) {
        let osc = match osc_type {
            OSCType::OSC1 => &mut self.osc_1,
            OSCType::OSC2 => &mut self.osc_2,
        };

        match osc_param {
            OSCParameterType::CoarseTune => osc.coarse_tune_range.snap_knob(&mut osc.coarse_tune.0),
            OSCParameterType::Shape => osc.note_shape_range.snap_knob(&mut osc.note_shape.0),
            OSCParameterType::FilterType => osc.filter_type_range.snap_knob(&mut osc.filter_type.0),
            _ => (),
        }
    }

    pub fn view(
        &mut self,
        screen_width: u32,
        screen_height: u32,
        params: &RawParameters,
    ) -> iced::Element<'_, Message> {
        let master_vol_widget = with_label(
            VSlider::new(&mut self.master_vol, move |normal| {
                Message::ParameterChanged(ParameterType::MasterVolume, normal.as_f32())
            })
            .height(iced::Length::Units(MASTER_VOL_HEIGHT)),
            &widget_name(ParameterType::MasterVolume),
        );

        let selected = match ModulationType::from(params.get(ParameterType::OSC2Mod)) {
            ModulationType::Mix => 0,
            ModulationType::AmpMod => 1,
            ModulationType::FreqMod => 2,
            ModulationType::PhaseMod => 3,
            ModulationType::WarpMod => 4,
        };
        // TODO: Consider a smarter way for messages that doesn't involve always casting to f32
        let osc_2_mod = iced::Element::new(ModTypeSelector::new(
            selected,
            vec!["Mix", "AM", "FM", "PM", "Warp"],
            iced::Length::Units(MOD_TYPE_SELECTOR_WIDTH),
            vec![
                Message::OSC2ModChanged(ModulationType::Mix),
                Message::OSC2ModChanged(ModulationType::AmpMod),
                Message::OSC2ModChanged(ModulationType::FreqMod),
                Message::OSC2ModChanged(ModulationType::PhaseMod),
                Message::OSC2ModChanged(ModulationType::WarpMod),
            ],
        ));

        let master_pane = master_vol_widget;
        let osc_1 = OSCKnobs::make_widget(&mut self.osc_1, OSCType::OSC1, None);
        let osc_2 = OSCKnobs::make_widget(&mut self.osc_2, OSCType::OSC2, Some(osc_2_mod));

        Row::new()
            .push(osc_1)
            .push(osc_2)
            .push(master_pane)
            .max_width(screen_width)
            .max_height(screen_height)
            .height(iced::Length::Shrink)
            .spacing(INTERPANE_SPACING)
            .into()
    }
}

/// The tab which has all the various modulation envelopes/LFOs
pub struct ModulationTab {
    pub env_1: EnvKnobGroup,
    pub env_2: EnvKnobGroup,
}

impl ModulationTab {
    pub fn new(params: &RawParameters) -> ModulationTab {
        ModulationTab {
            env_1: EnvKnobGroup::new(&params.get_mod_bank().env_1),
            env_2: EnvKnobGroup::new(&params.get_mod_bank().env_2),
        }
    }

    pub fn view(
        &mut self,
        _screen_width: u32,
        _screen_height: u32,
        params: &RawParameters,
    ) -> iced::Element<'_, Message> {
        let send_1 = params
            .get(ParameterType::ModBankSend(ModBankType::Env1))
            .into();
        let send_2 = params
            .get(ParameterType::ModBankSend(ModBankType::Env2))
            .into();
        column(vec![
            self.env_1.widgets(send_1, ModBankType::Env1, "ENV 1"),
            self.env_2.widgets(send_2, ModBankType::Env2, "ENV 2"),
        ])
        .padding(PANE_PADDING)
        .spacing(PANE_SPACING)
        .into()
    }
}

/// The tab which handles preset loading and saving.
pub struct PresetTab {
    // The list of presets available. The Option<String> is the name. If it is
    // none, then the preset failed to parse.
    presets: Vec<(PathBuf, button::State, Option<String>)>,
    save_preset: button::State,
}

impl PresetTab {
    pub fn new(_params: &RawParameters) -> PresetTab {
        PresetTab {
            presets: vec![],
            save_preset: button::State::new(),
        }
    }

    // pub fn update(&mut self, _message: Message, _params: &RawParameters) {
    // TODO!
    // }

    pub fn get_presets_from_folder(&mut self, path: impl AsRef<std::path::Path>) {
        self.presets.clear();
        match crate::presets::get_presets_from_folder(&path) {
            Ok(presets) => {
                for path in presets {
                    match try_parse_file(&path) {
                        Ok(preset) => {
                            self.presets
                                .push((path, button::State::new(), Some(preset.name)));
                        }
                        Err(err) => {
                            log::info!("Couldn't parse {}: {:?}", path.display(), err);
                            self.presets.push((path, button::State::new(), None));
                        }
                    }
                }
            }
            Err(err) => log::info!(
                "Couldn't load presets from path {}: {:?}",
                path.as_ref().display(),
                err
            ),
        }
        log::info!("Got {} presets!", self.presets.len());
    }

    pub fn view(
        &mut self,
        _screen_width: u32,
        _screen_height: u32,
        _params: &RawParameters,
    ) -> iced::Element<'_, Message> {
        let buttons: Vec<Element<_>> = self
            .presets
            .iter_mut()
            .map(|(preset, state, name)| {
                let button: iced::Button<_> = crate::ui::make_button(
                    state,
                    name.as_ref()
                        .unwrap_or(&"Invalid Preset".to_string())
                        .as_str(),
                    Message::LoadPreset(preset.clone()),
                );
                let button = if name.is_none() {
                    button.style(ButtonColor { color: RED })
                } else {
                    button
                };
                button.into()
            })
            .collect();

        let save_preset = crate::ui::make_button(
            &mut self.save_preset,
            "Save Preset",
            Message::SaveParamsAsPreset(crate::DATA_DIR.to_path_buf()),
        )
        .into();

        let preset_col = Column::with_children(buttons);
        Column::with_children(vec![
            preset_col.into(),
            widget::Space::new(iced::Length::Shrink, iced::Length::Fill).into(),
            save_preset,
        ])
        .into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tabs {
    Main,
    Preset,
    Modulation,
}

/// A struct keeping track of the various knob states an oscillator has.
/// The ranges are used for snapping the knobs.
pub struct OSCKnobs {
    volume: (knob::State, ParameterType),
    pan: (knob::State, ParameterType),
    phase: (knob::State, ParameterType),
    coarse_tune: (knob::State, ParameterType),
    fine_tune: (knob::State, ParameterType),
    attack: (knob::State, ParameterType),
    hold: (knob::State, ParameterType),
    decay: (knob::State, ParameterType),
    sustain: (knob::State, ParameterType),
    release: (knob::State, ParameterType),
    vol_lfo_amplitude: (knob::State, ParameterType),
    vol_lfo_period: (knob::State, ParameterType),
    note_shape: (knob::State, ParameterType),
    note_warp: (knob::State, ParameterType),
    pitch_attack: (knob::State, ParameterType),
    pitch_hold: (knob::State, ParameterType),
    pitch_decay: (knob::State, ParameterType),
    pitch_multiply: (knob::State, ParameterType),
    pitch_release: (knob::State, ParameterType),
    pitch_lfo_amplitude: (knob::State, ParameterType),
    pitch_lfo_period: (knob::State, ParameterType),
    filter_type: (knob::State, ParameterType),
    filter_freq: (knob::State, ParameterType),
    filter_q: (knob::State, ParameterType),
    filter_gain: (knob::State, ParameterType),
    coarse_tune_range: TruncatingIntRange,
    note_shape_range: TruncatingIntRange,
    filter_type_range: TruncatingIntRange,
}

impl OSCKnobs {
    fn new(params: &RawParameters, osc: OSCType) -> OSCKnobs {
        use EnvelopeParam::*;
        use OSCParameterType::*;
        OSCKnobs {
            volume: make_knob(params, (Volume, osc).into()),
            pan: make_knob(params, (Pan, osc).into()),
            phase: make_knob(params, (Phase, osc).into()),
            coarse_tune: make_knob(params, (CoarseTune, osc).into()),
            fine_tune: make_knob(params, (FineTune, osc).into()),
            attack: make_knob(params, (VolumeEnv(Attack), osc).into()),
            hold: make_knob(params, (VolumeEnv(Hold), osc).into()),
            decay: make_knob(params, (VolumeEnv(Decay), osc).into()),
            sustain: make_knob(params, (VolumeEnv(Sustain), osc).into()),
            release: make_knob(params, (VolumeEnv(Release), osc).into()),
            vol_lfo_amplitude: make_knob(params, (VolLFOAmplitude, osc).into()),
            vol_lfo_period: make_knob(params, (VolLFOPeriod, osc).into()),
            note_shape: make_knob(params, (Shape, osc).into()),
            note_warp: make_knob(params, (Warp, osc).into()),
            pitch_attack: make_knob(params, (PitchEnv(Attack), osc).into()),
            pitch_hold: make_knob(params, (PitchEnv(Hold), osc).into()),
            pitch_decay: make_knob(params, (PitchEnv(Decay), osc).into()),
            pitch_multiply: make_knob(params, (PitchEnv(Multiply), osc).into()),
            pitch_release: make_knob(params, (PitchEnv(Release), osc).into()),
            pitch_lfo_amplitude: make_knob(params, (PitchLFOAmplitude, osc).into()),
            pitch_lfo_period: make_knob(params, (PitchLFOPeriod, osc).into()),
            filter_type: make_knob(params, (FilterType, osc).into()),
            filter_freq: make_knob(params, (FilterFreq, osc).into()),
            filter_q: make_knob(params, (FilterQ, osc).into()),
            filter_gain: make_knob(params, (FilterGain, osc).into()),
            coarse_tune_range: TruncatingIntRange {
                num_regions: 24 * 2 + 1,
            },
            note_shape_range: TruncatingIntRange {
                num_regions: crate::sound_gen::NoteShape::VARIANT_COUNT,
            },
            filter_type_range: TruncatingIntRange {
                num_regions: crate::params::FILTER_TYPE_VARIANT_COUNT,
            },
        }
    }

    /// Create the set of widget for a particular oscillator.
    /// `osc` is used to set what type of ParamterType is fired for the
    /// `ParameterChanged(f32, ParameterType)` messages. Ex: is `osc` is
    /// OSCType::OSC1, then the volume knob will fire
    /// `ParameterChanged(value, ParameterType::OSC1(OSCParameterType::Volume))`
    /// When `on_off` is `Some(widget)`, the widget will be inserted to the right
    /// of the title of this widget block. This is usually the OSC 2 mod control.
    fn make_widget<'a>(
        &'a mut self,
        osc: OSCType,
        on_off: Option<Element<'a, Message>>,
    ) -> Element<'a, Message> {
        fn make_knob(knob: &mut (knob::State, ParameterType)) -> Element<'_, Message> {
            make_knob_with_label(knob, &widget_name(knob.1))
        }

        fn make_knob_with_label<'a>(
            knob: &'a mut (knob::State, ParameterType),
            title: &str,
        ) -> Element<'a, Message> {
            let param = knob.1;
            let knob = Knob::new(&mut knob.0, move |normal| {
                Message::ParameterChanged(param, normal.as_f32())
            })
            .size(KNOB_SIZE);
            with_label(knob, title)
        }

        let volume = make_knob(&mut self.volume);
        let pan = make_knob(&mut self.pan);
        let phase = make_knob(&mut self.phase);
        let fine_tune = make_knob(&mut self.fine_tune);
        let coarse_tune = make_knob(&mut self.coarse_tune);

        let shape_title = NoteShape::from_f32s(
            self.note_shape.0.normal().into(),
            self.note_warp.0.normal().into(),
        )
        .to_string();
        let shape = make_knob_with_label(&mut self.note_shape, &shape_title);

        let warp = make_knob(&mut self.note_warp);

        let attack = make_knob(&mut self.attack);
        let hold = make_knob(&mut self.hold);
        let decay = make_knob(&mut self.decay);
        let sustain = make_knob(&mut self.sustain);
        let release = make_knob(&mut self.release);

        let vol_lfo_amplitude = make_knob(&mut self.vol_lfo_amplitude);
        let vol_lfo_period = make_knob(&mut self.vol_lfo_period);

        let pitch_attack = make_knob(&mut self.pitch_attack);
        let pitch_hold = make_knob(&mut self.pitch_hold);
        let pitch_decay = make_knob(&mut self.pitch_decay);
        let pitch_multiply = make_knob(&mut self.pitch_multiply);
        let pitch_release = make_knob(&mut self.pitch_release);

        let pitch_lfo_amplitude = make_knob(&mut self.pitch_lfo_amplitude);
        let pitch_lfo_period = make_knob(&mut self.pitch_lfo_period);

        let filter_type_title = biquad_to_string(to_filter_type(
            self.filter_type.0.normal().into(),
            self.filter_gain.0.normal().into(),
        ));

        let filter_type = make_knob_with_label(&mut self.filter_type, &filter_type_title);
        let filter_freq = make_knob(&mut self.filter_freq);
        let filter_q = make_knob(&mut self.filter_q);
        let filter_gain = make_knob(&mut self.filter_gain);

        let title = match osc {
            OSCType::OSC1 => "OSC 1",
            OSCType::OSC2 => "OSC 2",
        };

        let osc_pane = make_pane_with_checkbox(
            title,
            on_off,
            vec![
                (vec![volume, pan, phase, coarse_tune, fine_tune], "Sound"),
                (vec![shape, warp], "Shape"),
                (vec![attack, hold, decay, sustain, release], "Envelope"),
                (vec![vol_lfo_amplitude, vol_lfo_period], "LFO"),
            ],
        );

        let pitch_pane = make_pane(
            "PITCH",
            vec![
                (
                    vec![
                        pitch_attack,
                        pitch_hold,
                        pitch_decay,
                        pitch_multiply,
                        pitch_release,
                    ],
                    "Envelope",
                ),
                (vec![pitch_lfo_amplitude, pitch_lfo_period], "LFO"),
            ],
        );

        let filter_pane = make_pane(
            "FILTER",
            vec![(vec![filter_type, filter_freq, filter_q, filter_gain], "")],
        );

        Column::with_children(vec![osc_pane.into(), pitch_pane.into(), filter_pane.into()])
            .padding(PANE_PADDING)
            .spacing(PANE_SPACING)
            .into()
    }

    pub fn set_knob(&mut self, param: OSCParameterType, value: f32) {
        fn set_knob(knob: &mut (knob::State, ParameterType), value: f32) {
            knob.0.set_normal(value.into())
        }

        /// Sets a knob while also the TruncatingIntRange to snap the value to
        /// the right spot.
        fn set_knob_with_range(
            knob: &mut (knob::State, ParameterType),
            range: &TruncatingIntRange,
            value: f32,
        ) {
            knob.0.set_normal(range.snap(value).into());
        }
        use OSCParameterType::*;
        match param {
            Volume => set_knob(&mut self.volume, value),
            Phase => set_knob(&mut self.phase, value),
            Pan => set_knob(&mut self.pan, value),
            Shape => set_knob_with_range(&mut self.note_shape, &self.note_shape_range, value),
            Warp => set_knob(&mut self.note_warp, value),
            FineTune => set_knob(&mut self.fine_tune, value),
            CoarseTune => {
                set_knob_with_range(&mut self.coarse_tune, &self.coarse_tune_range, value)
            }
            VolumeEnv(param) => match param {
                EnvelopeParam::Attack => set_knob(&mut self.attack, value),
                EnvelopeParam::Decay => set_knob(&mut self.decay, value),
                EnvelopeParam::Hold => set_knob(&mut self.hold, value),
                EnvelopeParam::Sustain => set_knob(&mut self.sustain, value),
                EnvelopeParam::Release => set_knob(&mut self.release, value),
                EnvelopeParam::Multiply => unreachable!(),
            },
            VolLFOAmplitude => set_knob(&mut self.vol_lfo_amplitude, value),
            VolLFOPeriod => set_knob(&mut self.vol_lfo_period, value),
            PitchEnv(param) => match param {
                EnvelopeParam::Attack => set_knob(&mut self.pitch_attack, value),
                EnvelopeParam::Decay => set_knob(&mut self.pitch_decay, value),
                EnvelopeParam::Hold => set_knob(&mut self.pitch_hold, value),
                EnvelopeParam::Sustain => unreachable!(),
                EnvelopeParam::Release => set_knob(&mut self.pitch_release, value),
                EnvelopeParam::Multiply => set_knob(&mut self.pitch_multiply, value),
            },
            PitchLFOAmplitude => set_knob(&mut self.pitch_lfo_amplitude, value),
            PitchLFOPeriod => set_knob(&mut self.pitch_lfo_period, value),
            FilterType => {
                set_knob_with_range(&mut self.filter_type, &self.filter_type_range, value)
            }
            FilterFreq => set_knob(&mut self.filter_freq, value),
            FilterQ => set_knob(&mut self.filter_q, value),
            FilterGain => set_knob(&mut self.filter_gain, value),
        }
    }
}

struct ModTypeSelector {
    // What message to send based on the currently selected element
    messages: Vec<Message>,
    text: Vec<&'static str>,
    text_size: f32,
    // Which element is currently selected
    pub selected: usize,
    width: iced::Length,
    height: iced::Length,
}

impl ModTypeSelector {
    fn new(
        selected: usize,
        text: Vec<&'static str>,
        width: iced::Length,
        messages: Vec<Message>,
    ) -> ModTypeSelector {
        ModTypeSelector {
            text,
            text_size: 15.0,
            selected,
            messages,
            width,
            height: iced::Length::Units(20),
        }
    }
}

impl<B: Backend> Widget<Message, Renderer<B>> for ModTypeSelector {
    fn width(&self) -> iced::Length {
        self.width
    }

    fn height(&self) -> iced::Length {
        self.height
    }

    fn layout(
        &self,
        _renderer: &Renderer<B>,
        limits: &iced_native::layout::Limits,
    ) -> layout::Node {
        let limits = limits.width(self.width).height(self.height);
        let size = limits.resolve(iced_native::Size::ZERO);
        layout::Node::new(size)
    }

    fn draw(
        &self,
        _renderer: &mut Renderer<B>,
        _defaults: &<iced_graphics::Renderer<B> as iced_native::Renderer>::Defaults,
        layout: iced_native::Layout<'_>,
        _cursor_position: iced::Point,
        _viewport: &iced::Rectangle,
    ) -> (Primitive, mouse::Interaction) {
        let rect = Rectangle::new(Point::ORIGIN, layout.bounds().size());
        let rects = split_rect_horiz(rect, self.text.len());

        let mut frame = Frame::new(rect.size());
        let mut stroke = Stroke {
            color: iced_native::Color::BLACK,
            width: 1.0,
            line_cap: LineCap::Round,
            line_join: LineJoin::Miter,
        };

        for (i, rect) in rects.iter().enumerate() {
            if self.selected == i {
                stroke = stroke.with_color(BLACK_GREEN);
            } else {
                stroke = stroke.with_color(GREY);
            }

            let text = iced_graphics::canvas::Text {
                content: self.text[i].to_string(),
                position: Point::new(((i as f32) + 0.5) * rect.width, rect.height / 2.0),
                color: if self.selected == i {
                    BLACK_GREEN
                } else {
                    GREY
                },
                size: self.text_size,
                font: iced_graphics::Font::Default,
                vertical_alignment: VerticalAlignment::Center,
                horizontal_alignment: HorizontalAlignment::Center,
            };
            frame.fill_text(text);
        }

        let rounded_rect = Primitive::Quad {
            bounds: rects[self.selected],
            background: Background::Color(GREEN_TRANS),
            border_radius: 3.0,
            border_width: 1.0,
            border_color: GREEN,
        };

        let primitive = Primitive::Group {
            primitives: vec![rounded_rect, frame.into_geometry().into_primitive()],
        };

        let primitive = Primitive::Translate {
            translation: Vector::new(layout.position().x, layout.position().y),
            content: Box::new(primitive),
        };

        (primitive, mouse::Interaction::Idle)
    }

    fn hash_layout(&self, state: &mut iced_native::Hasher) {
        struct Marker;
        std::any::TypeId::of::<Marker>().hash(state);
        // We only need to hash things which could cause a relayout, which is
        // just the width and height
        self.width.hash(state);
        self.height.hash(state);
    }

    fn on_event(
        &mut self,
        event: iced_native::Event,
        layout: iced_native::Layout<'_>,
        cursor_position: iced::Point,
        _renderer: &Renderer<B>,
        _clipboard: &mut dyn Clipboard,
        messages: &mut Vec<Message>,
    ) -> iced_native::event::Status {
        if let iced_native::Event::Mouse(mouse_event) = event {
            if mouse_event == mouse::Event::ButtonPressed(mouse::Button::Left) {
                let bounds = layout.bounds();
                let squares = split_rect_horiz(bounds, self.text.len());
                for (i, square) in squares.iter().enumerate() {
                    if square.contains(cursor_position) {
                        self.selected = i;
                        messages.push(self.messages[i].clone());
                        break;
                    }
                }
                return iced_native::event::Status::Captured;
            }
        }
        iced_native::event::Status::Ignored
    }
}

/// A group of knobs, arranged horizontally, reprsenting envelope knobs.
pub struct EnvKnobGroup {
    attack: knob::State,
    hold: knob::State,
    decay: knob::State,
    sustain: knob::State,
    release: knob::State,
    multiply: knob::State,
}

impl EnvKnobGroup {
    /// Create a new EnvKnobGroup out of the RawEnvelope
    fn new(envelope: &RawEnvelope) -> EnvKnobGroup {
        fn make_state(envelope: &RawEnvelope, param: EnvelopeParam) -> knob::State {
            knob::State::new(NormalParam {
                value: envelope.get(param).into(),
                default: EnvelopeParam::get_default(param).into(),
            })
        }
        EnvKnobGroup {
            attack: make_state(envelope, EnvelopeParam::Attack),
            hold: make_state(envelope, EnvelopeParam::Hold),
            decay: make_state(envelope, EnvelopeParam::Decay),
            sustain: make_state(envelope, EnvelopeParam::Sustain),
            release: make_state(envelope, EnvelopeParam::Release),
            multiply: make_state(envelope, EnvelopeParam::Multiply),
        }
    }

    /// Make a widget group out of the EnvKnobGroup
    /// TODO: Don't use ModBankType here
    pub fn widgets(
        &mut self,
        send: ModulationSend,
        param: ModBankType,
        title: &str,
    ) -> iced::Element<'_, Message> {
        /// Create a row of `knobs` with the title `title` and a selector next
        /// to the title.
        fn knob_row_with_selector<'a>(
            selector: Element<'a, Message>,
            knobs: Vec<Element<'a, Message>>,
            title: &str,
        ) -> Element<'a, Message> {
            Column::new()
                .push(row(vec![iced::Text::new(title).size(18).into(), selector]))
                .push(
                    Row::with_children(knobs)
                        .align_items(Align::Center)
                        .spacing(KNOB_SPACING),
                )
                .align_items(Align::Start)
                .spacing(2)
                .into()
        }

        let selected = match send {
            ModulationSend::Amplitude => 0,
            ModulationSend::Phase => 1,
            ModulationSend::Pitch => 2,
            ModulationSend::Warp => 3,
            ModulationSend::FilterFreq => 4,
        };

        // TODO: Consider a smarter way for messages that doesn't involve always casting to f32
        let selector = iced::Element::new(ModTypeSelector::new(
            selected,
            vec!["Amp", "Phase", "Pitch", "Warp", "Filter Freq"],
            iced::Length::Units(300),
            vec![
                Message::ModBankSendChanged(param, ModulationSend::Amplitude),
                Message::ModBankSendChanged(param, ModulationSend::Phase),
                Message::ModBankSendChanged(param, ModulationSend::Pitch),
                Message::ModBankSendChanged(param, ModulationSend::Warp),
                Message::ModBankSendChanged(param, ModulationSend::FilterFreq),
            ],
        ));

        let param = match param {
            ModBankType::Env1 => ModBankParameter::Env1,
            ModBankType::Env2 => ModBankParameter::Env2,
        };

        let attack = widget!(
            Knob,
            &mut self.attack,
            ParameterType::ModBank(param(EnvelopeParam::Attack))
        );
        let hold = widget!(
            Knob,
            &mut self.hold,
            ParameterType::ModBank(param(EnvelopeParam::Hold))
        );
        let decay = widget!(
            Knob,
            &mut self.decay,
            ParameterType::ModBank(param(EnvelopeParam::Decay))
        );
        let sustain = widget!(
            Knob,
            &mut self.sustain,
            ParameterType::ModBank(param(EnvelopeParam::Sustain))
        );
        let release = widget!(
            Knob,
            &mut self.release,
            ParameterType::ModBank(param(EnvelopeParam::Release))
        );
        let multiply = widget!(
            Knob,
            &mut self.multiply,
            ParameterType::ModBank(param(EnvelopeParam::Multiply))
        );
        knob_row_with_selector(
            selector,
            vec![attack, hold, decay, sustain, release, multiply],
            title,
        )
    }

    pub fn set_knob(&mut self, param: EnvelopeParam, value: f32) {
        let value = value.into();
        match param {
            EnvelopeParam::Attack => self.attack.set_normal(value),
            EnvelopeParam::Decay => self.decay.set_normal(value),
            EnvelopeParam::Hold => self.hold.set_normal(value),
            EnvelopeParam::Sustain => self.sustain.set_normal(value),
            EnvelopeParam::Release => self.release.set_normal(value),
            EnvelopeParam::Multiply => self.multiply.set_normal(value),
        }
    }
}

struct ButtonColor {
    color: iced::Color,
}

impl button::StyleSheet for ButtonColor {
    fn active(&self) -> button::Style {
        button::Style {
            background: Some(iced::Background::Color(self.color)),
            ..Default::default()
        }
    }
}

/// Makes a pane of widgets with the specified title.
/// `widgets` consists of a vector of (`inner_vec`, `row_title`). Each `inner_vec`
///  is a list of widgets that will be arranged in a row. The row will have the
/// heading of `row_title`.
/// If `on_off` is `Some(element)`. then the title for this pane will have the
/// `element` to the right of it.
fn make_pane_with_checkbox<'a>(
    title: &str,
    on_off: Option<Element<'a, Message>>,
    widgets: Vec<(Vec<Element<'a, Message>>, &str)>,
) -> Column<'a, Message> {
    // If there is no `on_off` element, just use a empty widget.
    // Picking `Length::Shrink` for a `Space` widget makes it have zero size.
    let on_off = on_off.unwrap_or_else(|| {
        iced::widget::Space::new(iced::Length::Shrink, iced::Length::Shrink).into()
    });

    let mut pane = column(vec![]).push(
        row(vec![
            iced::Text::new(title)
                .vertical_alignment(VerticalAlignment::Center)
                .into(),
            on_off,
        ])
        .align_items(Align::Center),
    );

    for (knobs, title) in widgets.into_iter() {
        pane = pane.push(knob_row(knobs, title));
    }
    pane
}

/// Make a pane without the additional `on_off` widget.
fn make_pane<'a>(
    title: &str,
    widgets: Vec<(Vec<Element<'a, Message>>, &str)>,
) -> Column<'a, Message> {
    make_pane_with_checkbox(title, None, widgets)
}

/// Create a row of `knobs` with the title `title`.
fn knob_row<'a>(knobs: Vec<Element<'a, Message>>, title: &str) -> Element<'a, Message> {
    Column::new()
        .push(iced::Text::new(title).size(18))
        .push(
            Row::with_children(knobs)
                .align_items(Align::Center)
                .spacing(KNOB_SPACING),
        )
        .align_items(Align::Start)
        .spacing(2)
        .into()
}

/// Create a combined element with the label `title` under a `widget`
fn with_label<'a>(widget: impl Into<Element<'a, Message>>, title: &str) -> Element<'a, Message> {
    let text = iced::Text::new(title)
        .size(15)
        .width(iced::Length::Units(LABEL_WIDTH))
        .horizontal_alignment(HorizontalAlignment::Center);
    Column::with_children(vec![widget.into(), text.into()])
        .align_items(Align::Center)
        .into()
}

/// Convience function to make `knob::State` out of a `RawParameters`
/// and `ParameterType`
fn make_knob(params: &RawParameters, param_type: ParameterType) -> (knob::State, ParameterType) {
    let knob = knob::State::new(make_normal_param(params, param_type));
    (knob, param_type)
}

/// Make a row of widgets with better default spacing.
fn row(widgets: Vec<Element<'_, Message>>) -> Row<'_, Message> {
    Row::with_children(widgets).spacing(5)
}

/// Make an empty column with better default spacing.
fn column(widgets: Vec<Element<'_, Message>>) -> Column<'_, Message> {
    Column::with_children(widgets).spacing(5)
}

/// Convience function to make a `NormalParam` using a `RawParameters`. The
/// `NormalParam` will have a default value of whatever the default value is for
/// a `RawParameters`.
fn make_normal_param(params: &RawParameters, param_type: ParameterType) -> NormalParam {
    NormalParam {
        value: params.get(param_type).into(),
        default: RawParameters::get_default(param_type).into(),
    }
}

/// A replacement struct for `iced_audio::IntRange`. This is used for snapping
/// knobs which only take on discrete integer values (for example, the Shape knob).
/// However, unlike `iced_audio::IntRange`, this knob sets the integer regions
/// via truncation, rather than rounding. This makes each int region have the
/// same size, which is needed because this matches with how `RawParameters`'s
/// `AtomicFloat` values are convereted to their discrete values.
/// For example, if we have a parameter that takes on 4 values, then we convert
/// from the `AtomicFloat` to the integer as follows:
/// 0.00 to 0.25 = 0
/// 0.25 to 0.50 = 1
/// 0.50 to 0.75 = 2
/// 0.75 to 1.00 = 3
/// However, iced_audio's `IntRange` wouldn't do this for the `Normal` that is
/// held in a `knob::State`, because it _rounds_ the float instead of truncating
/// it, which produces the wrong thing! It would end up having:
/// 0.00 to 0.16 = 0
/// 0.16 to 0.50 = 1
/// 0.50 to 0.83 = 2
/// 0.83 to 1.00 = 3
/// Which isn't evenly spaced. Thus, `TruncatingIntRange` instead snaps floats
/// using truncation instead.
/// See also: https://www.desmos.com/calculator/esnnnbfzml for a visualization
struct TruncatingIntRange {
    num_regions: usize,
}

impl TruncatingIntRange {
    /// Snap a normalized float to the nearest normalized float value corresponding
    /// to an integer.
    fn snap(&self, value: f32) -> f32 {
        snap_float(value, self.num_regions)
    }

    /// Snap a knob to the nearest integer value.
    fn snap_knob(&self, knob: &mut knob::State) {
        knob.normal_param.value = self.snap(knob.normal().into()).into();
    }
}

/// Split a rectangle horizontally
fn split_rect_horiz(rect: Rectangle, num_rects: usize) -> Vec<Rectangle> {
    let mut vec = Vec::with_capacity(num_rects);
    let new_width = rect.width / num_rects as f32;

    for i in 0..num_rects {
        vec.push(Rectangle {
            x: rect.x + i as f32 * new_width,
            y: rect.y,
            width: new_width,
            height: rect.height,
        })
    }
    vec
}

/// The widget name for a given parameter
fn widget_name(param: ParameterType) -> String {
    fn widget_name_env(param: EnvelopeParam) -> String {
        match param {
            EnvelopeParam::Attack => "A".to_string(),
            EnvelopeParam::Hold => "H".to_string(),
            EnvelopeParam::Decay => "D".to_string(),
            EnvelopeParam::Sustain => "S".to_string(),
            EnvelopeParam::Release => "R".to_string(),
            EnvelopeParam::Multiply => "M".to_string(),
        }
    }

    use OSCParameterType::*;
    match param {
        ParameterType::MasterVolume => "Master Volume".to_string(),
        ParameterType::OSC1(param) | ParameterType::OSC2(param) => match param {
            Volume => "Volume".to_string(),
            Phase => "Phase".to_string(),
            Pan => "Pan".to_string(),
            FineTune => "Fine".to_string(),
            CoarseTune => "Coarse".to_string(),
            Shape => "Shape".to_string(),
            Warp => "Warp".to_string(),
            VolumeEnv(param) | PitchEnv(param) => widget_name_env(param),
            VolLFOAmplitude => "Amount".to_string(),
            VolLFOPeriod => "Period".to_string(),
            PitchLFOAmplitude => "Amount".to_string(),
            PitchLFOPeriod => "Period".to_string(),
            FilterType => "Filter Type".to_string(),
            FilterFreq => "Freq.".to_string(),
            FilterQ => "Q".to_string(),
            FilterGain => "Gain".to_string(),
        },
        ParameterType::OSC2Mod => "OSC 2 Mod".to_string(),
        ParameterType::ModBank(ModBankParameter::Env1(param)) => widget_name_env(param),
        ParameterType::ModBank(ModBankParameter::Env2(param)) => widget_name_env(param),
        ParameterType::ModBankSend(ModBankType::Env1) => "Send 1".to_string(),
        ParameterType::ModBankSend(ModBankType::Env2) => "Send 2".to_string(),
    }
}
