use biquad::Hertz;
use vst::{plugin::PluginParameters, util::AtomicFloat};

use crate::{
    common::{Decibel, Seconds},
    ease::Easing,
};

pub const IDENTITY: Easing<f32> = Easing::Linear {
    start: 0.0,
    end: 1.0,
};

pub const BIPOLAR: Easing<f32> = Easing::Linear {
    start: -1.0,
    end: 1.0,
};

// Default values for master volume
pub const DEFAULT_MASTER_VOL: f32 = 0.6875; // -3 dB

// Min and maximum ranges for volume knobs.
pub const MASTER_VOL_MIN_DB: f32 = -36.0;
pub const MASTER_VOL_MAX_DB: f32 = 12.0;
pub const SUSTAIN_MIN_DB: f32 = -24.0;

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
    meow_attack: Parameter,
    meow_decay: Parameter,
    meow_sustain: Parameter,
    meow_release: Parameter,
    vibrato_amount: Parameter,
    vibrato_attack: Parameter,
    vibrato_rate: Parameter,
    portamento_time: Parameter,
    noise_mix: Parameter,
    chorus_mix: Parameter,
    pitch_bend: Parameter,
    polycat: Parameter,
    // Internal parametert not be exposed)
    gain: Parameter,
    filter_attack: Parameter,
    filter_decay: Parameter,
    filter_envlope_mod: Parameter,
    filter_dry_wet: Parameter,
    filter_q: Parameter,
    filter_type: Parameter,
    filter_cutoff_freq: Parameter,
    chorus_depth: Parameter,
    chorus_distance: Parameter,
    chorus_rate: Parameter,
    phase: Parameter,
}

impl MeowParameters {
    pub const NUM_PARAMS: usize = 23;

    pub fn new() -> MeowParameters {
        MeowParameters {
            meow_attack: Parameter::time("Meow Attack", DEFAULT_MEOW_ATTACK),
            meow_decay: Parameter::time("Meow Decay", DEFAULT_MEOW_DECAY),
            meow_sustain: Parameter::decibel("Meow Sustain", DEFAULT_MEOW_SUSTAIN),
            meow_release: Parameter::time("Meow Release", DEFAULT_MEOW_RELEASE),
            vibrato_amount: Parameter::percent("Vibrato Amount", DEFAULT_VIBRATO_AMOUNT),
            vibrato_attack: Parameter::time("Vibrato Attack", DEFAULT_VIBRATO_ATTACK),
            vibrato_rate: Parameter::time("Vibrato Rate", DEFAULT_VIBRATO_RATE),
            portamento_time: Parameter::time("Portamento", DEFAULT_PORTAMENTO),
            noise_mix: Parameter::percent("Noise", DEFAULT_NOISE_MIX),
            chorus_mix: Parameter::percent("Chorus", DEFAULT_CHORUS_MIX),
            pitch_bend: Parameter::with_units("Pitchbend", " semis", DEFAULT_PITCHBEND),
            polycat: Parameter::new("Polycat", NameFormatter::Boolean, DEFAULT_POLYCAT),
            // Internal parameters (might not be exposed)
            gain: Parameter::decibel("Master Volume", DEFAULT_MASTER_VOL),
            filter_attack: Parameter::time("Filter Attack", DEFAULT_FILTER_ATTACK),
            filter_decay: Parameter::time("Filter Decay", DEFAULT_FILTER_DECAY),
            filter_envlope_mod: Parameter::percent("Filter EnvMod", DEFAULT_FILTER_ENVLOPE_MOD),
            filter_dry_wet: Parameter::percent("Filter DryWet", DEFAULT_FILTER_DRY_WET),
            filter_q: Parameter::with_units("Filter Q", "", DEFAULT_FILTER_Q),
            filter_type: Parameter::with_units("Filter Type", "", DEFAULT_FILTER_TYPE),
            filter_cutoff_freq: Parameter::with_units(
                "Filter Cutoff",
                "Hz",
                DEFAULT_FILTER_CUTOFF_FREQ,
            ),
            chorus_depth: Parameter::with_units("Chorus Rate", "", DEFAULT_CHORUS_DEPTH),
            chorus_distance: Parameter::with_units("Chorus Rate", "", DEFAULT_CHORUS_DISTANCE),
            chorus_rate: Parameter::with_units("Chorus Rate", "", DEFAULT_CHORUS_RATE),
            phase: Parameter::new("Phase", NameFormatter::Angle, DEFAULT_PHASE),
        }
    }

    pub fn master_vol(&self) -> Decibel {
        Decibel::ease_db(MASTER_VOL_MIN_DB, MASTER_VOL_MAX_DB).ease(self.gain.get())
    }

    pub fn phase(&self) -> f32 {
        self.phase.get()
    }

    pub fn noise_mix(&self) -> f32 {
        self.noise_mix.get()
    }

    pub fn portamento_time(&self) -> Seconds {
        todo!()
    }

    pub fn pitchbend_max(&self) -> usize {
        todo!()
    }

    pub fn polycat(&self) -> bool {
        self.polycat.get() > 0.5
    }

    pub fn vol_envelope(&self) -> VolumeEnvelopeParams {
        todo!()
    }

    pub fn filter(&self) -> FilterParams {
        todo!()
    }

    pub fn filter_envelope(&self) -> FilterEnvelopeParams {
        todo!()
    }

    pub fn chorus(&self) -> ChorusParams {
        todo!();
    }

    pub fn vibrato_lfo(&self) -> VibratoParams {
        todo!()
    }

    fn get(&self, index: i32) -> Option<&Parameter> {
        let param = match index {
            0 => &self.meow_attack,
            1 => &self.meow_decay,
            2 => &self.meow_sustain,
            3 => &self.meow_release,
            4 => &self.vibrato_amount,
            5 => &self.vibrato_attack,
            6 => &self.vibrato_rate,
            7 => &self.portamento_time,
            8 => &self.noise_mix,
            9 => &self.chorus_mix,
            10 => &self.pitch_bend,
            11 => &self.polycat,
            12 => &self.filter_attack,
            13 => &self.filter_decay,
            14 => &self.filter_envlope_mod,
            15 => &self.filter_dry_wet,
            16 => &self.filter_q,
            17 => &self.filter_type,
            18 => &self.filter_cutoff_freq,
            19 => &self.chorus_depth,
            20 => &self.chorus_distance,
            21 => &self.chorus_rate,
            22 => &self.phase,
            _ => return None,
        };
        Some(param)
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
            parameter.set_value(value)
        } else {
            log::error!(
                "Cannot set value for parameter index {} (expected value in range 0 to {})",
                index,
                MeowParameters::NUM_PARAMS
            )
        }
    }
}

#[derive(Debug)]
enum NameFormatter {
    Time,
    Percent,
    Decibel,
    Angle,
    Boolean,
    Other(&'static str),
}

impl NameFormatter {
    fn get_text(&self, value: f32) -> String {
        match self {
            NameFormatter::Time => time_formatter(value).0,
            NameFormatter::Percent => percent_formatter(value).0,
            NameFormatter::Other(_) => format!("{:.3}", value),
            NameFormatter::Decibel => {
                let decibel: Decibel = todo!();
                if decibel.get_db() <= Decibel::NEG_INF_DB_THRESHOLD {
                    "-inf".to_string()
                } else if decibel.get_db() < 0.0 {
                    format!("{:.2}", decibel.get_db())
                } else {
                    format!("+{:.2}", decibel.get_db())
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
        }
    }

    fn get_label(&self, value: f32) -> String {
        match self {
            NameFormatter::Time => time_formatter(value).1,
            NameFormatter::Percent => percent_formatter(value).1,
            NameFormatter::Other(units) => units.to_string(),
            NameFormatter::Decibel => "dB".to_string(),
            NameFormatter::Angle => "deg".to_string(),
            NameFormatter::Boolean => "".to_string(),
        }
    }
}

fn time_formatter(time_in_secs: f32) -> (String, String) {
    if time_in_secs < 1.0 {
        (format!("{:.1}", time_in_secs * 1000.0), " ms".to_string())
    } else {
        (format!("{:.2}", time_in_secs), " sec".to_string())
    }
}

fn percent_formatter(x: f32) -> (String, String) {
    (format!("{:.2}", x * 100.0), "%".to_string())
}

#[derive(Debug)]
struct Parameter {
    name: &'static str,
    /// The parameter text and label. The text is usually a number, such as "0.5" or "+7", and the
    /// label is usually a unit like "ms" or "semis".
    formatter: NameFormatter,
    value: AtomicFloat,
}

impl Parameter {
    fn get_text(&self) -> String {
        self.formatter.get_text(self.get())
    }

    fn get_label(&self) -> String {
        self.formatter.get_label(self.get())
    }

    fn get(&self) -> f32 {
        self.value.get()
    }

    fn set_value(&self, value: f32) {
        self.value.set(value)
    }

    fn time(name: &'static str, default: f32) -> Self {
        Self {
            name,
            formatter: NameFormatter::Time,
            value: default.into(),
        }
    }

    fn new(name: &'static str, formatter: NameFormatter, default: f32) -> Self {
        Self {
            name,
            formatter,
            value: default.into(),
        }
    }

    fn decibel(name: &'static str, default: f32) -> Self {
        Self {
            name,
            formatter: NameFormatter::Decibel,
            value: default.into(),
        }
    }
    fn percent(name: &'static str, default: f32) -> Self {
        Self {
            name,
            formatter: NameFormatter::Percent,
            value: default.into(),
        }
    }

    fn with_units(name: &'static str, units: &'static str, default: f32) -> Self {
        Self {
            name,
            formatter: NameFormatter::Other(units),
            value: default.into(),
        }
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
    sustain: f32,
    release: Seconds,
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
        self.sustain
    }

    fn release(&self) -> Seconds {
        self.release
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
