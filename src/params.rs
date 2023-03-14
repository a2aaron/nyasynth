use std::sync::Arc;

use nih_plug::prelude::{
    BoolParam, Enum, EnumParam, FloatParam, FloatRange, IntParam, IntRange, Param, Params,
};

use crate::common::{self, Decibel, Seconds};
use crate::common::{FilterType, Hertz};
use crate::sound_gen::NoteShape;

// Default values for master volume
const DEFAULT_MASTER_VOL: Decibel = Decibel::from_db(-6.0);

// Default values for volume envelope
const DEFAULT_MEOW_ATTACK: Seconds = Seconds::new(30.0 / 1000.0);
const DEFAULT_MEOW_DECAY: Seconds = Seconds::new(1.25);
const DEFAULT_MEOW_SUSTAIN: Decibel = Decibel::from_db(-15.0);
const DEFAULT_MEOW_RELEASE: Seconds = Seconds::new(490.0 / 1000.0);

const DEFAULT_VIBRATO_AMOUNT: f32 = 0.0;
const DEFAULT_VIBRATO_ATTACK: Seconds = Seconds::new(0.0);
const DEFAULT_VIBRATO_RATE: VibratoRate = VibratoRate::Eighth;

const DEFAULT_FILTER_ENVLOPE_MOD: Hertz = Hertz(7000.0);
const DEFAULT_FILTER_DRY_WET: f32 = 1.0; // 100% filter
const DEFAULT_FILTER_Q: f32 = 2.5;
const DEFAULT_FILTER_TYPE: FilterType = FilterType::LowPass; // Low Pass
const DEFAULT_FILTER_CUTOFF_FREQ: Hertz = Hertz(350.0); // this which will be around 7350 at max meow sustain on max velocity.

const DEFAULT_CHORUS_MIX: f32 = 0.0;
const DEFAULT_CHORUS_DEPTH: f32 = 44.0;
const DEFAULT_CHORUS_DISTANCE: f32 = 450.0;
const DEFAULT_CHORUS_RATE: Hertz = Hertz(0.33);

const DEFAULT_NOISE_MIX: f32 = 0.0;

const DEFAULT_PITCHBEND: u8 = 12; // +12 semis
const DEFAULT_PORTAMENTO: Seconds = Seconds::new(120.0 / 1000.0);
const DEFAULT_POLYCAT: bool = false; // Off

pub const MAX_CHORUS_DEPTH: f32 = 100.0;
pub const MAX_CHORUS_DISTANCE: f32 = 1000.0;

/// The public facing parameters struct containing the computed values for each parameter value.
/// Avoid constructing too many of these--it is expensive to do so.
pub struct MeowParameters {
    pub master_vol: Decibel,
    pub noise_mix: f32,
    pub portamento_time: Seconds,
    pub pitchbend_max: u8,
    pub polycat: bool,
    pub vol_envelope: VolumeEnvelopeParams,
    pub filter: FilterParams,
    pub filter_envelope: FilterEnvelopeParams,
    pub chorus: ChorusParams,
    pub vibrato_attack: VibratoEnvelopeParams,
    pub vibrato_lfo: VibratoLFOParams,
    pub vibrato_note_shape: NoteShape,
    pub chorus_note_shape: NoteShape,
}

impl MeowParameters {
    /// Construct a MeowParameters from a normal Parameters. Doing this calls a lot of easing functions
    /// so avoid calling it too often (once per block, or ideally only once every time a parameter
    /// updates).
    pub fn new(parameters: &Parameters, tempo: f32) -> MeowParameters {
        fn seconds(param: &FloatParam) -> Seconds {
            Seconds::from(param.value())
        }

        fn hertz(param: &FloatParam) -> Hertz {
            Hertz::from(param.value())
        }

        fn decibel(param: &FloatParam) -> Decibel {
            Decibel::from_db(param.value())
        }

        // This exhaustive destructuring helps ensure that if you add a field to Parameters, that you
        // also need to add a field to MeowParameters.
        let Parameters {
            meow_attack,
            meow_decay,
            meow_sustain,
            meow_release,
            vibrato_amount,
            vibrato_attack,
            vibrato_rate,
            portamento_time,
            noise_mix,
            chorus_mix,
            pitch_bend,
            polycat,
            gain,
            filter_envlope_mod,
            filter_dry_wet,
            filter_q,
            filter_type,
            filter_cutoff_freq,
            chorus_depth,
            chorus_distance,
            chorus_rate,
            vibrato_note_shape,
            chorus_note_shape,
        } = parameters;
        MeowParameters {
            master_vol: decibel(gain),
            noise_mix: noise_mix.value(),
            portamento_time: seconds(portamento_time),
            pitchbend_max: pitch_bend.value() as u8,
            polycat: polycat.value(),
            vol_envelope: VolumeEnvelopeParams {
                attack: seconds(meow_attack),
                decay: seconds(meow_decay),
                sustain: meow_sustain.modulated_normalized_value(),
                release: seconds(meow_release),
            },
            filter: FilterParams {
                cutoff_freq: hertz(filter_cutoff_freq),
                q_value: filter_q.value(),
                filter_type: filter_type.value().into(),
                dry_wet: filter_dry_wet.value(),
            },
            filter_envelope: FilterEnvelopeParams {
                attack: seconds(meow_attack),
                decay: seconds(meow_decay),
                sustain: meow_sustain.modulated_normalized_value(),
                release: seconds(meow_release),
                env_mod: hertz(filter_envlope_mod),
            },
            chorus: ChorusParams {
                rate: Hertz(chorus_rate.value()),
                depth: chorus_depth.value(),
                min_distance: chorus_distance.value(),
                mix: chorus_mix.value(),
            },
            vibrato_attack: VibratoEnvelopeParams {
                attack: Seconds::from(vibrato_attack.value()),
            },
            vibrato_lfo: VibratoLFOParams {
                speed: vibrato_rate.value().as_hz(tempo),
                amount: vibrato_amount.value(),
            },
            vibrato_note_shape: vibrato_note_shape.value(),
            chorus_note_shape: chorus_note_shape.value(),
        }
    }
}

// This deny is triggered if you have a field that isn't read from. The places that you probably need
// to add code are in Parameters::get() and also a corresponding field in MeowParameters.
#[deny(dead_code)]
#[derive(Params)]
pub struct Parameters {
    // Public parameters (exposed in UI)
    #[id = "meow_attack"]
    pub meow_attack: FloatParam,
    #[id = "meow_decay"]
    pub meow_decay: FloatParam,
    #[id = "meow_sustain"]
    pub meow_sustain: FloatParam,
    #[id = "meow_release"]
    pub meow_release: FloatParam,
    #[id = "vibrato_amount"]
    pub vibrato_amount: FloatParam,
    #[id = "vibrato_attack"]
    pub vibrato_attack: FloatParam,
    #[id = "vibrato_rate"]
    pub vibrato_rate: EnumParam<VibratoRate>,
    #[id = "portamento_time"]
    pub portamento_time: FloatParam,
    #[id = "noise_mix"]
    pub noise_mix: FloatParam,
    #[id = "chorus_mix"]
    pub chorus_mix: FloatParam,
    #[id = "pitch_bend"]
    pub pitch_bend: IntParam,
    #[id = "polycat"]
    pub polycat: BoolParam,
    // Internal parameter (not exposed by the original Meowsynth)
    #[id = "gain"]
    gain: FloatParam,
    #[id = "filter_envlope_mod"]
    filter_envlope_mod: FloatParam,
    #[id = "filter_dry_wet"]
    filter_dry_wet: FloatParam,
    #[id = "filter_q"]
    filter_q: FloatParam,
    #[id = "filter_type"]
    filter_type: EnumParam<FilterType>,
    #[id = "filter_cutoff_freq"]
    filter_cutoff_freq: FloatParam,
    #[id = "chorus_depth"]
    chorus_depth: FloatParam,
    #[id = "chorus_distance"]
    chorus_distance: FloatParam,
    #[id = "chorus_rate"]
    chorus_rate: FloatParam,
    // "Debug" parameters (these might become not "debug" pretty soon)
    #[id = "vibrato_note_shape"]
    vibrato_note_shape: EnumParam<NoteShape>,
    #[id = "chorus_note_shape"]
    chorus_note_shape: EnumParam<NoteShape>,
}

impl Default for Parameters {
    fn default() -> Self {
        Parameters::new()
    }
}

impl Parameters {
    pub fn new() -> Parameters {
        fn polycat_formatter(value: bool) -> String {
            if value {
                "On".to_string()
            } else {
                "Off".to_string()
            }
        }

        fn time(name: &'static str, default: Seconds, min: f32, max: f32) -> FloatParam {
            fn formatter(value: f32) -> String {
                if value < 1.0 {
                    format!("{:.1} ms", value * 1000.0)
                } else {
                    format!("{:.2} sec", value)
                }
            }

            let range = FloatRange::Skewed {
                min,
                max,
                factor: FloatRange::skew_factor(-2.0),
            };
            FloatParam::new(name, default.get(), range).with_value_to_string(Arc::new(formatter))
        }

        fn decibel(name: &'static str, default: Decibel, min: f32, max: f32) -> FloatParam {
            fn formatter(decibel: f32) -> String {
                if decibel <= Decibel::NEG_INF_DB_THRESHOLD {
                    "-inf".to_string()
                } else if decibel < 0.0 {
                    format!("{:.2}", decibel)
                } else {
                    format!("+{:.2}", decibel)
                }
            }

            let range = FloatRange::Skewed {
                min,
                max,
                factor: FloatRange::gain_skew_factor(min, max),
            };
            FloatParam::new(name, default.get_db(), range)
                .with_unit(" db")
                .with_value_to_string(Arc::new(formatter))
        }

        fn percent(name: &'static str, default: f32) -> FloatParam {
            fn formatter(percent: f32) -> String {
                format!("{:.1}", percent * 100.0)
            }
            let range = FloatRange::Linear { min: 0.0, max: 1.0 };
            FloatParam::new(name, default, range)
                .with_unit(" %")
                .with_value_to_string(Arc::new(formatter))
        }

        pub fn freq(name: &'static str, default: Hertz, range: FloatRange) -> FloatParam {
            fn formatter(hz: f32) -> String {
                if hz < 1000.0 {
                    format!("{:.2} Hz", hz)
                } else {
                    format!("{:.2} kHz", hz / 1000.0)
                }
            }
            FloatParam::new(name, default.get(), range).with_value_to_string(Arc::new(formatter))
        }

        let filter_envelope_mod = Hertz::ease_exp(0.0, 22100.0);
        let filter_cutoff_freq = Hertz::ease_exp(20.0, 22100.0);
        let filter_q = common::ease_linear(0.01, 10.0);

        let chorus_rate = Hertz::ease_exp(0.1, 10.0);
        let chorus_depth = common::ease_linear(0.0, MAX_CHORUS_DEPTH);
        let chorus_distance = common::ease_linear(0.0, MAX_CHORUS_DISTANCE);

        Parameters {
            meow_attack: time("Meow Attack", DEFAULT_MEOW_ATTACK, 0.001, 10.0),
            meow_decay: time("Meow Decay", DEFAULT_MEOW_DECAY, 0.001, 5.0),
            meow_sustain: decibel("Meow Sustain", DEFAULT_MEOW_SUSTAIN, -24.0, 0.0),
            meow_release: time("Meow Release", DEFAULT_MEOW_RELEASE, 0.001, 4.0),
            vibrato_amount: percent("Vibrato Amount", DEFAULT_VIBRATO_AMOUNT),
            vibrato_attack: time("Vibrato Attack", DEFAULT_VIBRATO_ATTACK, 0.001, 5.0),
            vibrato_rate: EnumParam::new("Vibrato Rate", DEFAULT_VIBRATO_RATE),
            portamento_time: time("Portamento", DEFAULT_PORTAMENTO, 0.0001, 5.0),
            noise_mix: percent("Noise", DEFAULT_NOISE_MIX),
            chorus_mix: percent("Chorus", DEFAULT_CHORUS_MIX),
            pitch_bend: IntParam::new(
                "Pitchbend",
                DEFAULT_PITCHBEND as i32,
                IntRange::Linear { min: 1, max: 12 },
            )
            .with_unit(" semis"),
            polycat: BoolParam::new("Polycat", DEFAULT_POLYCAT)
                .with_value_to_string(Arc::new(polycat_formatter)),
            // Internal parameters (might not be exposed)
            gain: decibel("Master Volume", DEFAULT_MASTER_VOL, -36.0, 12.0),
            filter_envlope_mod: freq(
                "Filter EnvMod",
                DEFAULT_FILTER_ENVLOPE_MOD,
                filter_envelope_mod,
            ),
            filter_dry_wet: percent("Filter DryWet", DEFAULT_FILTER_DRY_WET),
            filter_q: FloatParam::new("Filter Q", DEFAULT_FILTER_Q, filter_q),
            filter_type: EnumParam::new("Filter Type", DEFAULT_FILTER_TYPE),
            filter_cutoff_freq: freq(
                "Filter Cutoff",
                DEFAULT_FILTER_CUTOFF_FREQ,
                filter_cutoff_freq,
            ),
            chorus_depth: FloatParam::new("Chorus Depth", DEFAULT_CHORUS_DEPTH, chorus_depth),
            chorus_distance: FloatParam::new(
                "Chorus Distance",
                DEFAULT_CHORUS_DISTANCE,
                chorus_distance,
            ),
            chorus_rate: freq("Chorus Rate", DEFAULT_CHORUS_RATE, chorus_rate),
            vibrato_note_shape: EnumParam::new("Vibrato Note Shape", NoteShape::Triangle),
            chorus_note_shape: EnumParam::new("Chorus Note Shape", NoteShape::Sine),
        }
    }
}

impl Parameters {
    pub fn dbg_polycat(&self) -> &BoolParam {
        &self.polycat
    }

    pub fn dbg_meow_decay(&self) -> &FloatParam {
        &self.meow_decay
    }

    pub fn dbg_meow_release(&self) -> &FloatParam {
        &self.meow_release
    }
}

pub struct ChorusParams {
    pub rate: Hertz,
    pub depth: f32,
    pub min_distance: f32,
    pub mix: f32,
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
    sustain: f32,
    release: Seconds,
}

impl EnvelopeParams<f32> for VolumeEnvelopeParams {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum VibratoRate {
    #[name = "4 bar"]
    FourBar,
    #[name = "2 bar"]
    TwoBar,
    #[name = "1 bar"]
    OneBar,
    #[name = "1/2"]
    Half,
    #[name = "1/4"]
    Quarter,
    #[name = "1/8"]
    Eighth,
    #[name = "1/12"]
    Twelfth,
    #[name = "1/16"]
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
