use std::ops::{Add, Mul, Sub};

use atomic_float::AtomicF32;
use biquad::ToHertz;
use derive_more::{Add, From, Into, Sub};
use nih_plug::{
    nih_debug_assert,
    prelude::{Enum, FloatRange, Smoothable},
    util::midi_note_to_freq,
};
use ordered_float::OrderedFloat;

use crate::sound_gen::EnvelopeType;

pub type SampleTime = usize;

/// A sample rate in Hz/seconds. Must be a positive value.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SampleRate(pub f32);

impl SampleRate {
    pub fn new(rate: f32) -> Option<SampleRate> {
        if rate <= 0.0 {
            None
        } else {
            Some(SampleRate(rate))
        }
    }

    pub fn get(&self) -> f32 {
        self.0
    }

    pub fn to_seconds(&self, samples: SampleTime) -> Seconds {
        let seconds = samples as f32 / self.get();
        Seconds::new(seconds)
    }

    pub fn hz(&self) -> biquad::Hertz<f32> {
        self.get().hz()
    }
}

impl From<f32> for SampleRate {
    fn from(value: f32) -> Self {
        SampleRate::new(value).unwrap()
    }
}

/// A normalized 0.0-1.0 float representation of a velocity value.
#[derive(Debug, Clone, Copy, From)]
pub struct Vel {
    pub raw: f32,
    pub eased: f32,
}

impl Vel {
    pub fn new(raw: f32) -> Vel {
        fn ease_in_expo(x: f32) -> f32 {
            if x <= 0.0 {
                0.0
            } else {
                (2.0f32.powf(10.0 * x) - 1.0) / (2.0f32.powf(10.0) - 1.0)
            }
        }

        // Easing sort of experimentally determined and is meant for use with the filter cutoff value.
        // See the following:
        // https://www.desmos.com/calculator/grjkm7iknd
        // https://docs.google.com/spreadsheets/d/174y4e5t8698O4-Wh9idkMVeZFb-L-7l8zLmDMGz-D-A/edit?usp=sharing
        Vel {
            raw,
            eased: ease_in_expo(raw),
        }
    }
}

/// A wrapper struct representing a duration of seconds. This struct implements [std::ops::Div], so
/// it's possible to divide a [Seconds] by another [Seconds] and get an [f32].
#[derive(Debug, Clone, Copy, Add, Sub, From, PartialEq, Eq, PartialOrd, Ord)]
pub struct Seconds(pub OrderedFloat<f32>);

impl Seconds {
    pub const ZERO: Seconds = Seconds(OrderedFloat(0.0));

    /// Construct a new [Seconds].
    pub const fn new(seconds: f32) -> Seconds {
        let value = OrderedFloat(seconds);
        Seconds(value)
    }

    /// Get the number of seconds as an [f32].
    pub const fn get(&self) -> f32 {
        self.0 .0
    }

    /// Interprets the seconds value as a period and converts it to Hz.
    pub fn as_hz(&self) -> biquad::Hertz<f32> {
        let hz = 1.0 / self.get();
        hz.hz()
    }
}

impl std::ops::Mul<f32> for Seconds {
    type Output = Seconds;

    fn mul(self, rhs: f32) -> Self::Output {
        Seconds::new(self.get() * rhs)
    }
}

impl std::ops::Div for Seconds {
    type Output = f32;

    fn div(self, rhs: Self) -> Self::Output {
        self.get() / rhs.get()
    }
}

impl From<Seconds> for f32 {
    fn from(value: Seconds) -> Self {
        value.get()
    }
}

impl From<f32> for Seconds {
    fn from(value: f32) -> Self {
        Seconds(value.into())
    }
}

/// A MIDI note
#[derive(Debug, Clone, Copy, PartialEq, From, Into)]
pub struct Note(pub u8);

/// A struct representing linear pitch space. This exists so that portamento and filter cutoff
/// sweep do not need to recompute their start and end frequencies every sample.
#[derive(Debug, Clone, Copy, PartialEq, Add, Sub, From, Into)]
pub struct Pitch(pub f32);

impl Pitch {
    pub fn from_note(note: Note) -> Self {
        Pitch(midi_note_to_freq(note.0).log2())
    }

    pub fn from_hertz(hertz: Hertz) -> Self {
        Pitch(hertz.get().log2())
    }

    pub fn into_hertz(&self) -> Hertz {
        Hertz(self.0.exp2())
    }
}

impl std::ops::Mul<f32> for Pitch {
    type Output = Pitch;

    fn mul(self, rhs: f32) -> Self::Output {
        Pitch(rhs * self.0)
    }
}

/// A struct representing Hertz.
#[derive(Debug, Clone, Copy, PartialEq, Add, Sub, From, Into)]
pub struct Hertz(pub f32);

impl Hertz {
    pub fn new(hz: f32) -> Hertz {
        Hertz(hz)
    }

    pub fn get(&self) -> f32 {
        self.0
    }

    // Lerp linearly in octave-space
    pub fn lerp_octave(start: Hertz, end: Hertz, t: f32) -> Hertz {
        let start = start.get().log2();
        let end = end.get().log2();
        let interpolated = lerp(start, end, t);
        let hz = interpolated.exp2();
        Hertz::new(hz)
    }

    pub fn ease_exp(start: f32, end: f32) -> FloatRange {
        FloatRange::Skewed {
            min: start,
            max: end,
            factor: 6.0,
        }
    }

    pub fn clamp(&self, min: f32, max: f32) -> Hertz {
        Hertz(self.get().clamp(min, max))
    }
}

impl From<biquad::Hertz<f32>> for Hertz {
    fn from(value: biquad::Hertz<f32>) -> Self {
        Hertz(value.hz())
    }
}

impl From<Hertz> for biquad::Hertz<f32> {
    fn from(value: Hertz) -> Self {
        // biquad::Hertz does not accept negative Hertz.
        let hz = value.get().max(0.0);
        biquad::Hertz::<f32>::from_hz(hz).unwrap()
    }
}

impl std::ops::Mul<f32> for Hertz {
    type Output = Hertz;

    fn mul(self, rhs: f32) -> Self::Output {
        Hertz(self.0 * rhs)
    }
}

impl std::ops::Div<Hertz> for Hertz {
    type Output = f32;
    fn div(self, rhs: Hertz) -> Self::Output {
        self.0 / rhs.0
    }
}

/// A pitchbend value in [-1.0, +1.0] range, where +1.0 means "max upward bend"
/// and -1.0 means "max downward bend"
#[derive(Debug, Clone, Copy, From, Into)]
pub struct Pitchbend(f32);

impl Pitchbend {
    pub fn new(value: f32) -> Pitchbend {
        Pitchbend(value)
    }

    pub fn get(&self) -> f32 {
        self.0
    }

    /// Convert a pitchbend value that is in the range [0.0, 1.0] to a NormalizedPitchbend.
    /// The pitchbend value is assumed such that 0.5 is considered to be "no bend", 0.0 is "max
    /// downward bend", and 1.0 is "max upward bend".
    pub fn from_zero_one_range(value: f32) -> Pitchbend {
        nih_debug_assert!(0.0 <= value && value <= 1.0);
        Pitchbend((value * 2.0) - 1.0)
    }
}

impl Default for Pitchbend {
    fn default() -> Self {
        Self(0.0)
    }
}

impl Smoothable for Pitchbend {
    type Atomic = AtomicPitchbend;

    fn to_f32(self) -> f32 {
        self.get()
    }

    fn from_f32(value: f32) -> Self {
        Pitchbend(value)
    }

    fn atomic_new(value: Self) -> Self::Atomic {
        AtomicPitchbend(AtomicF32::from(value.get()))
    }

    fn atomic_load(this: &Self::Atomic) -> Self {
        Pitchbend(this.0.load(std::sync::atomic::Ordering::Relaxed))
    }

    fn atomic_store(this: &Self::Atomic, value: Self) {
        this.0
            .store(value.get(), std::sync::atomic::Ordering::Relaxed)
    }
}

pub struct AtomicPitchbend(AtomicF32);

impl Default for AtomicPitchbend {
    fn default() -> Self {
        Self(AtomicF32::from(0.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Add, Sub)]
/// A struct representing Decibels. This struct can be used with [Easing] and [EnvelopeType]. Note
/// that the specific implementation for EnvelopeType has decibels lerped in amplitude space for the
/// attack of a note, and then lerped in dB space for the rest of the note. This is to ensure that
/// sharp attacks remain sharp. Additionally, note that decibel values below `NEG_INF_DB_THRESHOLD`
/// are treated as zero. This is done so that it is possible to lerp from negative infinity decibels
/// (that is, an amplitude of zero) to positive amounts in a reasonable fashion (in particular, this
/// means extremely quiet sounds will instead become silence).
pub struct Decibel(pub f32);

impl Decibel {
    /// The threshold for which Decibel values below it will be treated as negative
    /// infinity dB.
    pub const NEG_INF_DB_THRESHOLD: f32 = -70.0;

    pub const fn from_db(db: f32) -> Decibel {
        Decibel(db)
    }

    pub const fn neg_inf_db() -> Decibel {
        Decibel::from_db(Decibel::NEG_INF_DB_THRESHOLD)
    }

    pub const fn zero_db() -> Decibel {
        Decibel::from_db(0.0)
    }

    pub fn from_amp(amp: f32) -> Decibel {
        Decibel::from_db(f32::log10(amp) * 10.0)
    }

    // Linearly interpolate in amplitude space.
    pub fn lerp_amp(start: Decibel, end: Decibel, t: f32) -> Decibel {
        let amp = lerp(start.get_amp(), end.get_amp(), t);
        Decibel::from_amp(amp)
    }

    // Linearly interpolate in Decibel space.
    pub fn lerp_db(start: f32, end: f32, t: f32) -> Decibel {
        let db = lerp(start, end, t);
        Decibel::from_db(db)
    }

    /// Get the peak amplitude a signal played at the given Decibel amount would produce.
    pub fn get_amp(&self) -> f32 {
        if self.get_db() <= Decibel::NEG_INF_DB_THRESHOLD {
            0.0
        } else {
            10.0f32.powf(self.get_db() / 10.0)
        }
    }

    // Get the decibel value as an f32.
    pub fn get_db(&self) -> f32 {
        self.0
    }
}

impl EnvelopeType for Decibel {
    fn lerp_attack(start: Self, end: Self, t: f32) -> Self {
        // Lerp in amplitude space during the attack phase. This is useful
        // long attacks usually need linear amplitude ramp ups.
        Decibel::lerp_amp(start, end, t)
    }
    fn lerp_decay(start: Self, end: Self, t: f32) -> Self {
        Decibel::lerp_db(start.get_db(), end.get_db(), t)
    }
    fn lerp_release(start: Self, end: Self, t: f32) -> Self {
        Decibel::lerp_db(start.get_db(), end.get_db(), t)
    }
    fn lerp_retrigger(start: Self, end: Self, t: f32) -> Self {
        Decibel::lerp_amp(start, end, t)
    }
    fn one() -> Self {
        Decibel::zero_db()
    }
    fn zero() -> Self {
        Decibel::neg_inf_db()
    }
}

impl std::ops::Mul<f32> for Decibel {
    type Output = Decibel;

    fn mul(self, rhs: f32) -> Self::Output {
        Decibel::from_db(self.0 * rhs)
    }
}

impl std::ops::Div<Decibel> for Decibel {
    type Output = f32;
    fn div(self, rhs: Decibel) -> Self::Output {
        self.get_db() / rhs.get_db()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Enum)]
pub enum FilterType {
    #[name = "Low Pass (Single Pole)"]
    SinglePoleLowPass,
    #[name = "Low Pass"]
    LowPass,
    #[name = "High Pass"]
    HighPass,
    #[name = "Band Pass"]
    BandPass,
    #[name = "Notch"]
    Notch,
}

impl From<biquad::Type<f32>> for FilterType {
    fn from(value: biquad::Type<f32>) -> Self {
        match value {
            biquad::Type::SinglePoleLowPass => FilterType::SinglePoleLowPass,
            biquad::Type::LowPass => FilterType::LowPass,
            biquad::Type::HighPass => FilterType::HighPass,
            biquad::Type::BandPass => FilterType::BandPass,
            biquad::Type::Notch => FilterType::Notch,
            biquad::Type::SinglePoleLowPassApprox => todo!(),
            biquad::Type::AllPass => todo!(),
            biquad::Type::LowShelf(_) => todo!(),
            biquad::Type::HighShelf(_) => todo!(),
            biquad::Type::PeakingEQ(_) => todo!(),
        }
    }
}

impl From<FilterType> for biquad::Type<f32> {
    fn from(value: FilterType) -> Self {
        match value {
            FilterType::SinglePoleLowPass => biquad::Type::SinglePoleLowPass,
            FilterType::LowPass => biquad::Type::LowPass,
            FilterType::HighPass => biquad::Type::HighPass,
            FilterType::BandPass => biquad::Type::BandPass,
            FilterType::Notch => biquad::Type::Notch,
        }
    }
}

// TODO: you could reduce the Copy bound to just Clone although it'd be more annoying to work with.
pub trait Lerpable =
    Add<Self, Output = Self> + Sub<Self, Output = Self> + Mul<f32, Output = Self> + Copy;

/// Lerp between two values. This function is clamped.
pub fn lerp<T: Lerpable>(start: T, end: T, t: f32) -> T {
    (end - start) * t.clamp(0.0, 1.0) + start
}
