use vst::{plugin::PluginParameters, util::AtomicFloat};

use crate::common::{Decibel, I32Divable, Seconds};
use crate::common::{FilterType, Hertz};
use crate::ease::{DiscreteLinear, Easer, Easing};

const IDENTITY: Easing<f32> = Easing::Linear {
    start: 0.0,
    end: 1.0,
};

const FILTER_TYPES: [FilterType; 4] = [
    FilterType::LowPass,
    FilterType::HighPass,
    FilterType::BandPass,
    FilterType::Notch,
];

const VIBRATO_RATES: [VibratoRate; 8] = [
    VibratoRate::FourBar,
    VibratoRate::TwoBar,
    VibratoRate::OneBar,
    VibratoRate::Half,
    VibratoRate::Quarter,
    VibratoRate::Eighth,
    VibratoRate::Twelfth,
    VibratoRate::Sixteenth,
];

// Default values for master volume
const DEFAULT_MASTER_VOL: Decibel = Decibel::from_db(-3.0); // -3 dB

// Default values for volume envelope
const DEFAULT_MEOW_ATTACK: Seconds = Seconds::new(30.0 / 1000.0);
const DEFAULT_MEOW_DECAY: Seconds = Seconds::new(1.25);
const DEFAULT_MEOW_SUSTAIN: Decibel = Decibel::from_db(-15.0);
const DEFAULT_MEOW_RELEASE: Seconds = Seconds::new(400.0 / 1000.0);

const DEFAULT_VIBRATO_AMOUNT: f32 = 0.0;
const DEFAULT_VIBRATO_ATTACK: Seconds = Seconds::new(0.0);
const DEFAULT_VIBRATO_RATE: VibratoRate = VibratoRate::Eighth;

const DEFAULT_FILTER_ENVLOPE_MOD: Hertz = Hertz(7000.0);
const DEFAULT_FILTER_DRY_WET: f32 = 1.0; // 100% filter
const DEFAULT_FILTER_Q: f32 = 2.5;
const DEFAULT_FILTER_TYPE: FilterType = FilterType::LowPass; // Low Pass
const DEFAULT_FILTER_CUTOFF_FREQ: Hertz = Hertz(350.0); // this which will be around 7350 at max meow sustain on max velocity.

const DEFAULT_CHORUS_MIX: f32 = 0.0;
const DEFAULT_CHORUS_DEPTH: f32 = 0.0;
const DEFAULT_CHORUS_DISTANCE: f32 = 0.0;
const DEFAULT_CHORUS_RATE: Hertz = Hertz(1.0);

const DEFAULT_PHASE: f32 = 0.0;

const DEFAULT_NOISE_MIX: f32 = 0.0;

const DEFAULT_PITCHBEND: I32Divable = I32Divable(12); // +12 semis
const DEFAULT_PORTAMENTO: Seconds = Seconds::new(120.0 / 1000.0);
const DEFAULT_POLYCAT: f32 = 0.0; // Off

pub struct MeowParameters {
    // Public parameters (exposed in UI)
    meow_attack: Parameter<Seconds>,
    meow_decay: Parameter<Seconds>,
    meow_sustain: Parameter<Decibel>,
    meow_release: Parameter<Seconds>,
    vibrato_amount: Parameter<f32>,
    vibrato_attack: Parameter<Seconds>,
    vibrato_rate: Parameter<VibratoRate>,
    portamento_time: Parameter<Seconds>,
    noise_mix: Parameter<f32>,
    chorus_mix: Parameter<f32>,
    pitch_bend: Parameter<I32Divable>,
    polycat: Parameter<f32>,
    // Internal parametert not be exposed)
    gain: Parameter<Decibel>,
    filter_envlope_mod: Parameter<Hertz>,
    filter_dry_wet: Parameter<f32>,
    filter_q: Parameter<f32>,
    filter_type: Parameter<FilterType>,
    filter_cutoff_freq: Parameter<Hertz>,
    chorus_depth: Parameter<f32>,
    chorus_distance: Parameter<f32>,
    chorus_rate: Parameter<Hertz>,
    phase: Parameter<f32>,
}

impl MeowParameters {
    pub const NUM_PARAMS: usize = 22;

    pub fn new() -> MeowParameters {
        fn filter_type_formatter(value: FilterType) -> (String, String) {
            let value = match value {
                FilterType::SinglePoleLowPass => "Low Pass (Single Pole)",
                FilterType::LowPass => "Low Pass",
                FilterType::HighPass => "High Pass",
                FilterType::BandPass => "Band Pass",
                FilterType::Notch => "Notch",
            };
            (value.to_string(), "".to_string())
        }

        fn vibrato_formatter(value: VibratoRate) -> (String, String) {
            let value = match value {
                VibratoRate::FourBar => "4 bars",
                VibratoRate::TwoBar => "2 bars",
                VibratoRate::OneBar => "1 bars",
                VibratoRate::Half => "1/2",
                VibratoRate::Quarter => "1/4",
                VibratoRate::Eighth => "1/8",
                VibratoRate::Twelfth => "1/12",
                VibratoRate::Sixteenth => "1/16",
            };
            (value.to_string(), "".to_string())
        }

        fn semitone_formatter(value: I32Divable) -> (String, String) {
            (format!("{}", value.0), "semis".to_string())
        }

        fn polycat_formatter(value: f32) -> (String, String) {
            if value < 0.5 {
                ("Off".to_string(), "".to_string())
            } else {
                ("On".to_string(), "".to_string())
            }
        }

        fn angle_formatter(value: f32) -> (String, String) {
            (format!("{:.2}", value * 360.0), "deg".to_string())
        }

        fn unitless_formatter(value: f32) -> (String, String) {
            (format!("{:.3}", value), "".to_string())
        }

        let meow_sustain = Decibel::ease_db(-24.0, 0.0);
        let vibrato_rate = DiscreteLinear {
            values: VIBRATO_RATES,
        };
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
        let filter_type = DiscreteLinear {
            values: FILTER_TYPES,
        };
        let filter_envelope_mod = Hertz::ease_exp(0.0, 22100.0);
        let filter_cutoff_freq = Hertz::ease_exp(20.0, 22100.0);
        let filter_q = Easing::Linear {
            start: 0.01,
            end: 10.0,
        };
        let chorus_rate = Hertz::ease_exp(0.1, 10.0);

        MeowParameters {
            meow_attack: Parameter::time("Meow Attack", DEFAULT_MEOW_ATTACK, 0.001, 10.0),
            meow_decay: Parameter::time("Meow Decay", DEFAULT_MEOW_DECAY, 0.001, 5.0),
            meow_sustain: Parameter::decibel("Meow Sustain", DEFAULT_MEOW_SUSTAIN, meow_sustain),
            meow_release: Parameter::time("Meow Release", DEFAULT_MEOW_RELEASE, 0.001, 4.0),
            vibrato_amount: Parameter::percent("Vibrato Amount", DEFAULT_VIBRATO_AMOUNT),
            vibrato_attack: Parameter::time("Vibrato Attack", DEFAULT_VIBRATO_ATTACK, 0.001, 5.0),
            vibrato_rate: Parameter::new(
                "Vibrato Rate",
                DEFAULT_VIBRATO_RATE,
                vibrato_rate,
                vibrato_formatter,
            ),
            portamento_time: Parameter::time("Portamento", DEFAULT_PORTAMENTO, 0.0001, 5.0),
            noise_mix: Parameter::percent("Noise", DEFAULT_NOISE_MIX),
            chorus_mix: Parameter::percent("Chorus", DEFAULT_CHORUS_MIX),
            pitch_bend: Parameter::new(
                "Pitchbend",
                DEFAULT_PITCHBEND,
                pitch_bend,
                semitone_formatter,
            ),
            polycat: Parameter::new("Polycat", DEFAULT_POLYCAT, polycat, polycat_formatter),
            // Internal parameters (might not be exposed)
            gain: Parameter::decibel("Master Volume", DEFAULT_MASTER_VOL, gain),
            filter_envlope_mod: Parameter::freq(
                "Filter EnvMod",
                DEFAULT_FILTER_ENVLOPE_MOD,
                filter_envelope_mod,
            ),
            filter_dry_wet: Parameter::percent("Filter DryWet", DEFAULT_FILTER_DRY_WET),
            filter_q: Parameter::new("Filter Q", DEFAULT_FILTER_Q, filter_q, unitless_formatter),
            filter_type: Parameter::new(
                "Filter Type",
                DEFAULT_FILTER_TYPE,
                filter_type,
                filter_type_formatter,
            ),
            filter_cutoff_freq: Parameter::freq(
                "Filter Cutoff",
                DEFAULT_FILTER_CUTOFF_FREQ,
                filter_cutoff_freq,
            ),
            chorus_depth: Parameter::new(
                "Chorus Depth",
                DEFAULT_CHORUS_DEPTH,
                IDENTITY,
                unitless_formatter,
            ),
            chorus_distance: Parameter::new(
                "Chorus Distance",
                DEFAULT_CHORUS_DISTANCE,
                IDENTITY,
                unitless_formatter,
            ),
            chorus_rate: Parameter::freq("Chorus Rate", DEFAULT_CHORUS_RATE, chorus_rate),
            phase: Parameter::new("Phase", DEFAULT_PHASE, IDENTITY, angle_formatter),
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
        let cutoff_freq = self.filter_cutoff_freq.get();
        let q_value = self.filter_q.get();
        let dry_wet = self.filter_dry_wet.get();

        let filter_type = self.filter_type.get().into();
        FilterParams {
            cutoff_freq,
            q_value,
            filter_type,
            dry_wet,
        }
    }

    pub fn filter_envelope(&self) -> FilterEnvelopeParams {
        let attack = self.meow_attack.get();
        let decay = self.meow_decay.get();
        let sustain = self.meow_sustain.get_raw();
        let release = self.meow_release.get();
        let env_mod = self.filter_envlope_mod.get();
        FilterEnvelopeParams {
            attack,
            sustain,
            decay,
            env_mod,
            release,
        }
    }

    pub fn chorus(&self) -> ChorusParams {
        let rate = self.chorus_rate.get();
        let depth = self.chorus_depth.get();
        let distance = self.chorus_distance.get();
        let mix = self.chorus_mix.get();
        ChorusParams {
            rate,
            depth,
            distance,
            mix,
        }
    }

    pub fn vibrato_attack(&self) -> VibratoEnvelopeParams {
        let attack = self.vibrato_attack.get();
        VibratoEnvelopeParams { attack }
    }

    pub fn vibrato_lfo(&self, tempo: f32) -> VibratoLFOParams {
        let speed = self.vibrato_rate.get().as_hz(tempo);
        let amount = self.vibrato_amount.get();
        VibratoLFOParams { speed, amount }
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
            12 => self.filter_envlope_mod.view(),
            13 => self.filter_dry_wet.view(),
            14 => self.filter_q.view(),
            15 => self.filter_type.view(),
            16 => self.filter_cutoff_freq.view(),
            17 => self.chorus_depth.view(),
            18 => self.chorus_distance.view(),
            19 => self.chorus_rate.view(),
            20 => self.phase.view(),
            21 => self.gain.view(),
            _ => return None,
        };
        Some(view)
    }
}

impl PluginParameters for MeowParameters {
    fn get_parameter_label(&self, index: i32) -> String {
        if let Some(parameter) = self.get(index) {
            parameter.text_unit
        } else {
            "".to_string()
        }
    }

    fn get_parameter_text(&self, index: i32) -> String {
        if let Some(parameter) = self.get(index) {
            parameter.text_value
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

#[derive(Debug, Clone)]
struct ParameterView<'a> {
    name: &'a str,
    text_unit: String,
    text_value: String,
    value: &'a AtomicFloat,
}

impl<'a> ParameterView<'a> {
    fn get(&self) -> f32 {
        self.value.get()
    }

    fn set(&self, value: f32) {
        self.value.set(value)
    }
}

struct Parameter<T> {
    name: &'static str,
    value: AtomicFloat,
    easer: Box<dyn Easer<T> + Send + Sync>,
    formatter: fn(T) -> (String, String),
}

impl<T> Parameter<T> {
    fn get(&self) -> T {
        let value = self.get_raw();
        self.easer.ease(value)
    }

    fn get_raw(&self) -> f32 {
        self.value.get()
    }

    fn new(
        name: &'static str,
        default: T,
        easer: impl Easer<T> + 'static + Send + Sync,
        formatter: fn(T) -> (String, String),
    ) -> Parameter<T> {
        let default = easer.inv_ease(default);
        Parameter {
            name,
            value: default.into(),
            easer: Box::new(easer),
            formatter,
        }
    }

    fn view(&self) -> ParameterView {
        let value = self.get();
        let (text_value, text_unit) = (self.formatter)(value);
        ParameterView {
            text_unit,
            text_value,
            name: self.name,
            value: &self.value,
        }
    }
}

impl Parameter<Seconds> {
    fn time(name: &'static str, default: Seconds, min: f32, max: f32) -> Parameter<Seconds> {
        fn time_formatter(value: Seconds) -> (String, String) {
            let value = value.get();
            if value < 1.0 {
                (format!("{:.1}", value * 1000.0), "ms".to_string())
            } else {
                (format!("{:.2}", value), "sec".to_string())
            }
        }
        let easer = Easing::Exponential {
            start: min.into(),
            end: max.into(),
        };
        Parameter::new(name, default, easer, time_formatter)
    }
}

impl Parameter<Decibel> {
    fn decibel(name: &'static str, default: Decibel, easer: Easing<Decibel>) -> Parameter<Decibel> {
        fn decibel_formatter(decibel: Decibel) -> (String, String) {
            if decibel.get_db() <= Decibel::NEG_INF_DB_THRESHOLD {
                ("-inf".to_string(), "dB".to_string())
            } else if decibel.get_db() < 0.0 {
                (format!("{:.2}", decibel.get_db()), "dB".to_string())
            } else {
                (format!("+{:.2}", decibel.get_db()), "dB".to_string())
            }
        }

        Parameter::new(name, default, easer, decibel_formatter)
    }
}

impl Parameter<f32> {
    fn percent(name: &'static str, default: f32) -> Parameter<f32> {
        fn formatter(value: f32) -> (String, String) {
            (format!("{:.1}", value * 100.0), "%".to_string())
        }
        Parameter::new(name, default, IDENTITY, formatter)
    }
}

impl Parameter<Hertz> {
    pub fn freq(name: &'static str, default: Hertz, easer: Easing<Hertz>) -> Parameter<Hertz> {
        fn formatter(hz: Hertz) -> (String, String) {
            let hz = hz.get();
            if hz < 1000.0 {
                (format!("{:.2}", hz), "Hz".to_string())
            } else {
                (format!("{:.2}", hz / 1000.0), "kHz".to_string())
            }
        }
        Parameter::new(name, default, easer, formatter)
    }
}

pub struct ChorusParams {
    rate: Hertz,
    depth: f32,
    distance: f32,
    mix: f32,
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
    sustain: f32,
    decay: Seconds,
    release: Seconds,
    pub env_mod: Hertz,
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
    pub cutoff_freq: Hertz,
    pub q_value: f32,
    pub filter_type: biquad::Type<f32>,
    pub dry_wet: f32,
}

#[derive(Debug)]
pub struct VibratoLFOParams {
    pub speed: Hertz,
    pub amount: f32,
}

pub struct VibratoEnvelopeParams {
    attack: Seconds,
}

impl EnvelopeParams<f32> for VibratoEnvelopeParams {
    fn attack(&self) -> Seconds {
        self.attack
    }

    fn hold(&self) -> Seconds {
        Seconds::ZERO
    }

    fn decay(&self) -> Seconds {
        Seconds::new(0.001)
    }

    fn sustain(&self) -> f32 {
        1.0
    }

    fn release(&self) -> Seconds {
        Seconds::new(0.001)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VibratoRate {
    FourBar,
    TwoBar,
    OneBar,
    Half,
    Quarter,
    Eighth,
    Twelfth,
    Sixteenth,
}

impl VibratoRate {
    /// Converts the vibrato rate to herts, given a tempo in beats per minute
    pub fn as_hz(&self, tempo: f32) -> Hertz {
        let beats_per_seconds = tempo / 60.0;
        let multiplier = match self {
            VibratoRate::FourBar => 1.0 / 16.0,
            VibratoRate::TwoBar => 1.0 / 8.0,
            VibratoRate::OneBar => 1.0 / 4.0,
            VibratoRate::Half => 1.0 / 2.0,
            VibratoRate::Quarter => 1.0,
            VibratoRate::Eighth => 2.0,
            VibratoRate::Twelfth => 3.0,
            VibratoRate::Sixteenth => 4.0,
        };
        let hertz = beats_per_seconds * multiplier;
        Hertz::new(hertz)
    }
}
