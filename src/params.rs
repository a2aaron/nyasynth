use biquad::{Hertz, ToHertz};
use vst::{plugin::PluginParameters, util::AtomicFloat};

use crate::ease::Easing;
use crate::{
    common::{Decibel, I32Divable, Seconds},
    ease::{InvLerpable, Lerpable},
};

const IDENTITY: Easing<f32> = Easing::Linear {
    start: 0.0,
    end: 1.0,
};

const FILTER_TYPES: [biquad::Type<f32>; 4] = [
    biquad::Type::LowPass,
    biquad::Type::HighPass,
    biquad::Type::BandPass,
    biquad::Type::Notch,
];

// Default values for master volume
pub const DEFAULT_MASTER_VOL: f32 = 0.6875; // -3 dB

// Default values for volume envelope
pub const DEFAULT_MEOW_ATTACK: f32 = 0.1;
pub const DEFAULT_MEOW_DECAY: f32 = 0.5; // ~200 ms
pub const DEFAULT_MEOW_SUSTAIN: f32 = 0.75;
pub const DEFAULT_MEOW_RELEASE: f32 = 0.3;

pub const DEFAULT_VIBRATO_AMOUNT: f32 = 0.0;
pub const DEFAULT_VIBRATO_ATTACK: f32 = 0.0;
pub const DEFAULT_VIBRATO_RATE: f32 = 0.0;

pub const DEFAULT_FILTER_ATTACK: f32 = 0.0;
pub const DEFAULT_FILTER_DECAY: f32 = 0.0;
pub const DEFAULT_FILTER_ENVLOPE_MOD: f32 = 0.0;
pub const DEFAULT_FILTER_DRY_WET: f32 = 1.0; // 100% filter
pub const DEFAULT_FILTER_Q: f32 = 0.1;
pub const DEFAULT_FILTER_TYPE: f32 = 0.0; // Low Pass
pub const DEFAULT_FILTER_CUTOFF_FREQ: f32 = 0.0;

pub const DEFAULT_CHORUS_MIX: f32 = 0.0;
pub const DEFAULT_CHORUS_DEPTH: f32 = 0.0;
pub const DEFAULT_CHORUS_DISTANCE: f32 = 0.0;
pub const DEFAULT_CHORUS_RATE: f32 = 0.0;

pub const DEFAULT_PHASE: f32 = 0.0;

pub const DEFAULT_NOISE_MIX: f32 = 0.0;

pub const DEFAULT_PITCHBEND: f32 = 1.0; // +12 semis
pub const DEFAULT_PORTAMENTO: f32 = 0.3;
pub const DEFAULT_POLYCAT: f32 = 0.0; // Off

pub struct MeowParameters {
    // Public parameters (exposed in UI)
    meow_attack: Parameter<Seconds>,
    meow_decay: Parameter<Seconds>,
    meow_sustain: Parameter<Decibel>,
    meow_release: Parameter<Seconds>,
    vibrato_amount: Parameter<f32>,
    vibrato_attack: Parameter<Seconds>,
    vibrato_rate: Parameter<Seconds>,
    portamento_time: Parameter<Seconds>,
    noise_mix: Parameter<f32>,
    chorus_mix: Parameter<f32>,
    pitch_bend: Parameter<I32Divable>,
    polycat: Parameter<f32>,
    // Internal parametert not be exposed)
    gain: Parameter<Decibel>,
    filter_attack: Parameter<Seconds>,
    filter_decay: Parameter<Seconds>,
    filter_envlope_mod: Parameter<f32>,
    filter_dry_wet: Parameter<f32>,
    filter_q: Parameter<f32>,
    filter_type: Parameter<I32Divable>,
    filter_cutoff_freq: Parameter<f32>,
    chorus_depth: Parameter<f32>,
    chorus_distance: Parameter<f32>,
    chorus_rate: Parameter<f32>,
    phase: Parameter<f32>,
}

impl MeowParameters {
    pub const NUM_PARAMS: usize = 23;

    pub fn new() -> MeowParameters {
        let meow_attack = Seconds::ease_exp(0.001, 2.0);
        let meow_decay = Seconds::ease_exp(0.001, 5.0);
        let meow_sustain = Decibel::ease_db(-24.0, 0.0);
        let meow_release = Seconds::ease_exp(0.001, 5.0);
        let vibrato_attack = Seconds::ease_exp(0.001, 2.0);
        let vibrato_rate = Seconds::ease_exp(0.001, 2.0);
        let portamento_time = Seconds::ease_exp(0.001, 2.0);
        let pitch_bend = Easing::SteppedLinear {
            start: I32Divable(1),
            end: I32Divable(12),
            steps: 13,
        };
        let polycat = Easing::Linear {
            start: 0.0,
            end: 1.0,
        };
        let gain = Decibel::ease_db(-36.0, 12.0);
        let filter_attack = Seconds::ease_exp(0.001, 2.0);
        let filter_decay = Seconds::ease_exp(0.001, 5.0);
        let filter_type = Easing::SteppedLinear {
            start: I32Divable(0),
            end: I32Divable(FILTER_TYPES.len() as i32),
            steps: 3,
        };
        let filter_cutoff_freq = Easing::Exponential {
            start: 20.0,
            end: 22100.0,
        };
        MeowParameters {
            meow_attack: Parameter::time("Meow Attack", DEFAULT_MEOW_ATTACK, meow_attack),
            meow_decay: Parameter::time("Meow Decay", DEFAULT_MEOW_DECAY, meow_decay),
            meow_sustain: Parameter::decibel("Meow Sustain", DEFAULT_MEOW_SUSTAIN, meow_sustain),
            meow_release: Parameter::time("Meow Release", DEFAULT_MEOW_RELEASE, meow_release),
            vibrato_amount: Parameter::percent("Vibrato Amount", DEFAULT_VIBRATO_AMOUNT),
            vibrato_attack: Parameter::time(
                "Vibrato Attack",
                DEFAULT_VIBRATO_ATTACK,
                vibrato_attack,
            ),
            vibrato_rate: Parameter::time("Vibrato Rate", DEFAULT_VIBRATO_RATE, vibrato_rate),
            portamento_time: Parameter::time("Portamento", DEFAULT_PORTAMENTO, portamento_time),
            noise_mix: Parameter::percent("Noise", DEFAULT_NOISE_MIX),
            chorus_mix: Parameter::percent("Chorus", DEFAULT_CHORUS_MIX),
            pitch_bend: Parameter::new(
                "Pitchbend",
                NameFormatter::Semitones,
                DEFAULT_PITCHBEND,
                pitch_bend,
            ),
            polycat: Parameter::new("Polycat", NameFormatter::Boolean, DEFAULT_POLYCAT, polycat),
            // Internal parameters (might not be exposed)
            gain: Parameter::decibel("Master Volume", DEFAULT_MASTER_VOL, gain),
            filter_attack: Parameter::time("Filter Attack", DEFAULT_FILTER_ATTACK, filter_attack),
            filter_decay: Parameter::time("Filter Decay", DEFAULT_FILTER_DECAY, filter_decay),
            filter_envlope_mod: Parameter::percent("Filter EnvMod", DEFAULT_FILTER_ENVLOPE_MOD),
            filter_dry_wet: Parameter::percent("Filter DryWet", DEFAULT_FILTER_DRY_WET),
            filter_q: Parameter::unitless("Filter Q", DEFAULT_FILTER_Q),
            filter_type: Parameter::new(
                "Filter Type",
                NameFormatter::FilterType,
                DEFAULT_FILTER_TYPE,
                filter_type,
            ),
            filter_cutoff_freq: Parameter::new(
                "Filter Cutoff",
                NameFormatter::Frequency,
                DEFAULT_FILTER_CUTOFF_FREQ,
                filter_cutoff_freq,
            ),
            chorus_depth: Parameter::unitless("Chorus Depth", DEFAULT_CHORUS_DEPTH),
            chorus_distance: Parameter::unitless("Chorus Distance", DEFAULT_CHORUS_DISTANCE),
            chorus_rate: Parameter::unitless("Chorus Rate", DEFAULT_CHORUS_RATE),
            phase: Parameter::new("Phase", NameFormatter::Angle, DEFAULT_PHASE, IDENTITY),
        }
    }

    pub fn master_vol(&self) -> Decibel {
        self.gain.get()
    }

    pub fn phase(&self) -> f32 {
        self.phase.get()
    }

    pub fn noise_mix(&self) -> f32 {
        self.noise_mix.get()
    }

    pub fn portamento_time(&self) -> Seconds {
        self.portamento_time.get()
    }

    pub fn pitchbend_max(&self) -> usize {
        self.pitch_bend.get().0 as usize
    }

    pub fn polycat(&self) -> bool {
        self.polycat.get() > 0.5
    }

    pub fn vol_envelope(&self) -> VolumeEnvelopeParams {
        let attack = self.meow_attack.get();
        let decay = self.meow_decay.get();
        let sustain = self.meow_sustain.get();
        let release = self.meow_release.get();
        VolumeEnvelopeParams {
            attack,
            decay,
            sustain,
            release,
        }
    }

    pub fn filter(&self) -> FilterParams {
        let cutoff_freq = self.filter_cutoff_freq.get().hz();
        let q_value = self.filter_q.get();
        let dry_wet = self.filter_dry_wet.get();

        let filter_type = self.filter_type.get().0;
        assert!(0 <= filter_type && filter_type < FILTER_TYPES.len() as i32);
        let filter_type = FILTER_TYPES[filter_type as usize];
        FilterParams {
            cutoff_freq,
            q_value,
            filter_type,
            dry_wet,
        }
    }

    pub fn filter_envelope(&self) -> FilterEnvelopeParams {
        let attack = self.filter_attack.get();
        let decay = self.filter_decay.get();
        let env_mod = self.filter_envlope_mod.get();
        FilterEnvelopeParams {
            attack,
            decay,
            env_mod,
        }
    }

    pub fn chorus(&self) -> ChorusParams {
        todo!();
    }

    pub fn vibrato_lfo(&self) -> VibratoParams {
        todo!()
    }

    fn get(&self, index: i32) -> Option<ParameterView> {
        let view = match index {
            0 => self.meow_attack.view(),
            1 => self.meow_decay.view(),
            2 => self.meow_sustain.view(),
            3 => self.meow_release.view(),
            4 => self.vibrato_amount.view(),
            5 => self.vibrato_attack.view(),
            6 => self.vibrato_rate.view(),
            7 => self.portamento_time.view(),
            8 => self.noise_mix.view(),
            9 => self.chorus_mix.view(),
            10 => self.pitch_bend.view(),
            11 => self.polycat.view(),
            12 => self.filter_attack.view(),
            13 => self.filter_decay.view(),
            14 => self.filter_envlope_mod.view(),
            15 => self.filter_dry_wet.view(),
            16 => self.filter_q.view(),
            17 => self.filter_type.view(),
            18 => self.filter_cutoff_freq.view(),
            19 => self.chorus_depth.view(),
            20 => self.chorus_distance.view(),
            21 => self.chorus_rate.view(),
            22 => self.phase.view(),
            _ => return None,
        };
        Some(view)
    }
}

impl PluginParameters for MeowParameters {
    fn get_parameter_label(&self, index: i32) -> String {
        if let Some(parameter) = self.get(index) {
            parameter.get_label()
        } else {
            "".to_string()
        }
    }

    fn get_parameter_text(&self, index: i32) -> String {
        if let Some(parameter) = self.get(index) {
            parameter.get_text()
        } else {
            "".to_string()
        }
    }

    fn get_parameter_name(&self, index: i32) -> String {
        if let Some(parameter) = self.get(index) {
            parameter.name.to_string()
        } else {
            "".to_string()
        }
    }

    fn get_parameter(&self, index: i32) -> f32 {
        if let Some(parameter) = self.get(index) {
            parameter.get()
        } else {
            0.0
        }
    }

    fn set_parameter(&self, index: i32, value: f32) {
        if let Some(parameter) = self.get(index) {
            // This is needed because some VST hosts, such as Ableton, echo a
            // parameter change back to the plugin. This causes issues such as
            // weird knob behavior where the knob "flickers" because the user tries
            // to change the knob value, but ableton keeps sending back old, echoed
            // values.
            #[allow(clippy::float_cmp)]
            if parameter.get() == value {
                return;
            }
            parameter.set(value)
        } else {
            log::error!(
                "Cannot set value for parameter index {} (expected value in range 0 to {})",
                index,
                MeowParameters::NUM_PARAMS
            )
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ParameterView<'a> {
    formatter: &'a NameFormatter,
    name: &'a str,
    value: &'a AtomicFloat,
}

impl<'a> ParameterView<'a> {
    fn get_label(&self) -> String {
        self.formatter.get_label(self.get())
    }

    fn get_text(&self) -> String {
        self.formatter.get_text(self.get())
    }

    fn get(&self) -> f32 {
        self.value.get()
    }

    fn set(&self, value: f32) {
        self.value.set(value)
    }
}

#[derive(Debug)]
enum NameFormatter {
    Time,
    Percent,
    Decibel,
    Angle,
    Boolean,
    Unitless,
    Frequency,
    Semitones,
    FilterType,
}

impl NameFormatter {
    fn get_text(&self, value: f32) -> String {
        match self {
            NameFormatter::Time => {
                if value < 1.0 {
                    format!("{:.1}", value * 1000.0)
                } else {
                    format!("{:.2}", value)
                }
            }
            NameFormatter::Percent => format!("{:.2}", value * 100.0),
            NameFormatter::Decibel => {
                if value <= Decibel::NEG_INF_DB_THRESHOLD {
                    "-inf".to_string()
                } else if value < 0.0 {
                    format!("{:.2}", value)
                } else {
                    format!("+{:.2}", value)
                }
            }
            NameFormatter::Angle => format!("{:.2}", value * 360.0),
            NameFormatter::Boolean => {
                if value < 0.5 {
                    "Off".to_string()
                } else {
                    "On".to_string()
                }
            }
            NameFormatter::Unitless => format!("{:.3}", value),
            NameFormatter::Frequency => {
                if value < 1000.0 {
                    format!("{:.2}", value)
                } else {
                    format!("{:.2}", value / 1000.0)
                }
            }
            NameFormatter::Semitones => todo!(),
            NameFormatter::FilterType => todo!(),
        }
    }

    fn get_label(&self, value: f32) -> String {
        match self {
            NameFormatter::Time => {
                if value < 1.0 {
                    "ms".to_string()
                } else {
                    "sec".to_string()
                }
            }
            NameFormatter::Percent => "%".to_string(),
            NameFormatter::Decibel => "dB".to_string(),
            NameFormatter::Angle => "deg".to_string(),
            NameFormatter::Boolean => "".to_string(),
            NameFormatter::Unitless => "".to_string(),
            NameFormatter::Frequency => {
                if value < 1000.0 {
                    "Hz".to_string()
                } else {
                    "kHz".to_string()
                }
            }
            NameFormatter::Semitones => "semis".to_string(),
            NameFormatter::FilterType => "".to_string(),
        }
    }
}

#[derive(Debug)]
struct Parameter<T> {
    name: &'static str,
    /// The parameter text and label. The text is usually a number, such as "0.5" or "+7", and the
    /// label is usually a unit like "ms" or "semis".
    formatter: NameFormatter,
    value: AtomicFloat,
    easer: Easing<T>,
}

impl<T> Parameter<T> {
    fn get_raw(&self) -> f32 {
        self.value.get()
    }

    fn new(
        name: &'static str,
        formatter: NameFormatter,
        default: f32,
        easer: Easing<T>,
    ) -> Parameter<T> {
        Parameter {
            name,
            formatter,
            value: default.into(),
            easer,
        }
    }

    fn view(&self) -> ParameterView {
        ParameterView {
            formatter: &self.formatter,
            name: self.name,
            value: &self.value,
        }
    }
}

impl<T: Lerpable + InvLerpable> Parameter<T> {
    fn get(&self) -> T {
        let value = self.get_raw();
        self.easer.ease(value)
    }
}

impl Parameter<Seconds> {
    fn time(name: &'static str, default: f32, easer: Easing<Seconds>) -> Parameter<Seconds> {
        Parameter::new(name, NameFormatter::Time, default, easer)
    }
}

impl Parameter<Decibel> {
    fn decibel(name: &'static str, default: f32, easer: Easing<Decibel>) -> Parameter<Decibel> {
        Parameter::new(name, NameFormatter::Decibel, default, easer)
    }
}

impl Parameter<f32> {
    fn percent(name: &'static str, default: f32) -> Parameter<f32> {
        Parameter::new(name, NameFormatter::Percent, default, IDENTITY)
    }

    fn unitless(name: &'static str, default: f32) -> Parameter<f32> {
        Parameter::new(name, NameFormatter::Unitless, default, IDENTITY)
    }
}

pub struct ChorusParams {
    rate: Seconds,
    depth: f32,
    distance: f32,
}

// A set of immutable envelope parameters. The envelope is defined as follows:
// - In the attack phase, the envelope value goes from the `zero` value to the
//   `max` value.
// - In the decay phase, the envelope value goes from the `max` value to the
//   `sustain` value.
// - In the sustain phase, the envelope value is constant at the `sustain` value.
// - In the release phase, the envelope value goes from the `sustain` value to
//   `zero` value.
// The envelope value is then scaled by the `multiply` value
pub trait EnvelopeParams<T> {
    // In seconds, how long attack phase is
    fn attack(&self) -> Seconds;
    // In seconds, how long hold phase is
    fn hold(&self) -> Seconds;
    // In seconds, how long decay phase is
    fn decay(&self) -> Seconds;
    // The value to go to during sustain phase
    fn sustain(&self) -> T;
    // In seconds, how long release phase is
    fn release(&self) -> Seconds;
    // In -1.0 to 1.0 range usually. Multiplied by the value given by the ADSR
    fn multiply(&self) -> f32 {
        1.0
    }
}

pub struct VolumeEnvelopeParams {
    attack: Seconds,
    decay: Seconds,
    sustain: Decibel,
    release: Seconds,
}

impl EnvelopeParams<Decibel> for VolumeEnvelopeParams {
    fn attack(&self) -> Seconds {
        self.attack
    }

    fn hold(&self) -> Seconds {
        Seconds::ZERO
    }

    fn decay(&self) -> Seconds {
        self.decay
    }

    fn sustain(&self) -> Decibel {
        self.sustain
    }

    fn release(&self) -> Seconds {
        self.release
    }
}

pub struct FilterEnvelopeParams {
    attack: Seconds,
    decay: Seconds,
    pub env_mod: f32,
}

impl EnvelopeParams<f32> for FilterEnvelopeParams {
    fn attack(&self) -> Seconds {
        self.attack
    }

    fn hold(&self) -> Seconds {
        Seconds::ZERO
    }

    fn decay(&self) -> Seconds {
        self.decay
    }

    fn sustain(&self) -> f32 {
        0.0
    }

    fn release(&self) -> Seconds {
        Seconds::ZERO
    }
}

pub struct FilterParams {
    pub cutoff_freq: Hertz<f32>,
    pub q_value: f32,
    pub filter_type: biquad::Type<f32>,
    pub dry_wet: f32,
}

pub struct VibratoParams {
    speed: Hertz<f32>,
    amount: f32,
    attack: Seconds,
}

impl VibratoParams {
    pub fn freq(&self) -> Hertz<f32> {
        self.speed
    }
}

impl EnvelopeParams<f32> for VibratoParams {
    fn attack(&self) -> Seconds {
        self.attack
    }

    fn hold(&self) -> Seconds {
        Seconds::ZERO
    }

    fn decay(&self) -> Seconds {
        Seconds::ZERO
    }

    fn sustain(&self) -> f32 {
        self.amount
    }

    fn release(&self) -> Seconds {
        Seconds::ZERO
    }
}
