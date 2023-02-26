use crate::{
    common::{Decibel, SampleRate, SampleTime},
    ease::lerp,
    neighbor_pairs::NeighborPairsIter,
    params::{EnvelopeParams, MeowParameters},
};

use biquad::{Biquad, DirectForm1, Hertz, ToHertz, Q_BUTTERWORTH_F32};
use variant_count::VariantCount;
use wmidi::{PitchBend, U14, U7};

const TAU: f32 = std::f32::consts::TAU;

// The time, in samples, for how long retrigger phase is.
const RETRIGGER_TIME: SampleTime = 88; // 88 samples is about 2 miliseconds.

/// An offset, in samples, from the start of the frame.
type FrameDelta = usize;

/// A value in range [0.0, 1.0] which denotes the position wihtin a wave cycle.
type Angle = f32;
/// A pitchbend value in [-1.0, +1.0] range, where +1.0 means "max upward bend"
/// and -1.0 means "max downward bend"
pub type NormalizedPitchbend = f32;

// A type that an Envelope and EnvelopeParameter can work with. This type must
// support addition and subtraction and scalar multiplication with f32. It can also
// specify the easing used for the attack, decay, release, and retrigger phases
// of the envelope (by default, it will use `lerp<T>` and uses whatever the defined
// Add/Sub/Mul operations do)
pub trait EnvelopeType: std::ops::Mul<f32, Output = Self> + Copy + Clone + std::fmt::Debug
where
    Self: std::marker::Sized,
{
    fn lerp_attack(start: Self, end: Self, t: f32) -> Self;
    fn lerp_decay(start: Self, end: Self, t: f32) -> Self;
    fn lerp_release(start: Self, end: Self, t: f32) -> Self;
    fn lerp_retrigger(start: Self, end: Self, t: f32) -> Self;
    // The value to ease from during attack phase and to ease to during release phase
    fn zero() -> Self;
    // The value to ease to during decay phase and to ease from during decay phase
    fn one() -> Self;
}

impl EnvelopeType for f32 {
    fn lerp_attack(start: Self, end: Self, t: f32) -> Self {
        lerp::<Self>(start, end, t)
    }
    fn lerp_decay(start: Self, end: Self, t: f32) -> Self {
        lerp::<Self>(start, end, t)
    }
    fn lerp_release(start: Self, end: Self, t: f32) -> Self {
        lerp::<Self>(start, end, t)
    }
    fn lerp_retrigger(start: Self, end: Self, t: f32) -> Self {
        lerp::<Self>(start, end, t)
    }

    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

pub struct SoundGenerator {
    osc_1: OSCGroup,
    pub note: wmidi::Note,
    // The pitch of the note this SoundGenerator is playing, ignoring all coarse
    // detune and pitch bend effects. This is in hertz.
    note_pitch: Hertz<f32>,
    // The velocity of the note that this SoundGenerator is playing, ignoring all
    // amplitude modulation effects. This is a 0.0 - 1.0 normalized value.
    vel: f32,
    // The time, in samples, that this SoundGenerator has run since the last note on
    // event. This is NOT an interframe sample number!
    samples_since_note_on: SampleTime,
    // The current state of the SoundGenerator (held, released, etc)
    note_state: NoteState,
    // The per-sample filter applied to the output
    // filter: DirectForm1<f32>,
    // If Some(frame_delta, vel), then a note on event occurs in the next frame
    // at sample `frame_delta` samples into the the frame, and the note has a
    // note velocity of `vel`
    next_note_on: Option<(FrameDelta, f32)>,
    // If Some(frame_delta), then the next note off event occurs in the next frame
    // at `frame_delta` samples into the frame
    next_note_off: Option<FrameDelta>,
}

impl SoundGenerator {
    pub fn new(note: wmidi::Note, vel: f32, sample_rate: SampleRate) -> SoundGenerator {
        SoundGenerator {
            note,
            note_pitch: wmidi::Note::to_freq_f32(note).hz(),
            vel,
            samples_since_note_on: 0,
            note_state: NoteState::None,
            osc_1: OSCGroup::new(sample_rate),
            next_note_on: None,
            next_note_off: None,
        }
    }

    /// Returns true if the note is "alive" (playing audio). A note is dead if
    /// it is in the release state and it is after the total release time.
    pub fn is_alive(&self, sample_rate: SampleRate, params: &MeowParameters) -> bool {
        match self.note_state {
            NoteState::None | NoteState::Held | NoteState::Retrigger(_) => true,
            NoteState::Released(release_time) => {
                // The number of seconds it has been since release
                let time = sample_rate.to_seconds(self.samples_since_note_on - release_time);
                time < params.vol_envelope().release()
            }
        }
    }

    pub fn next_sample(
        &mut self,
        params: &MeowParameters,
        i: FrameDelta,
        sample_rate: SampleRate,
        pitch_bend: f32,
    ) -> (f32, f32) {
        let context = NoteContext {
            note_state: self.note_state,
            sample_rate,
            samples_since_note_on: self.samples_since_note_on,
        };

        // Only advance time if the note is being held down.
        match self.note_state {
            NoteState::None => (),
            _ => self.samples_since_note_on += 1,
        }

        // Handle note on event. If the note was previously not triggered (aka:
        // this is the first time the note has been triggered), then the note
        // transitions to the hold state. If this is a retrigger, then the note
        // transitions to the retrigger state, with volume `vol`.
        // Also, we set the note velocity to the appropriate new note velocity.
        match self.next_note_on {
            Some((note_on, note_on_vel)) if note_on == i => {
                let edge = match self.note_state {
                    NoteState::None => NoteStateEdge::InitialTrigger,
                    _ => NoteStateEdge::NoteRetriggered,
                };

                self.osc_1.note_state_changed(edge);

                // Update the note state
                self.note_state = match self.note_state {
                    NoteState::None => NoteState::Held,
                    _ => NoteState::Retrigger(self.samples_since_note_on),
                };
                self.vel = note_on_vel;
                self.next_note_on = None;
            }
            _ => (),
        }

        // Trigger note off events
        match self.next_note_off {
            Some(note_off) if note_off == i => {
                self.osc_1.note_state_changed(NoteStateEdge::NoteReleased);

                self.note_state = NoteState::Released(self.samples_since_note_on);
                self.next_note_off = None;
            }
            _ => (),
        }

        // If it has been 10 samples in the retrigger state, switch back to
        // the held state. This also resets the time.
        if let NoteState::Retrigger(retrigger_time) = self.note_state {
            if self.samples_since_note_on - retrigger_time > RETRIGGER_TIME {
                self.note_state = NoteState::Held;
                self.samples_since_note_on = 0;
            }
        }

        let osc_1 = self
            .osc_1
            .next_sample(&params, context, self.vel, self.note_pitch, pitch_bend);

        (osc_1, osc_1)
    }

    pub fn note_on(&mut self, frame_delta: i32, vel: f32) {
        self.next_note_on = Some((frame_delta as usize, vel));
    }

    pub fn note_off(&mut self, frame_delta: i32) {
        self.next_note_off = Some(frame_delta as usize);
    }
}

struct OSCGroup {
    osc: Oscillator,
    vol_env: Envelope<Decibel>,
    vibrato_env: Envelope<f32>,
    vibrato_lfo: Oscillator,
    // The state for the EQ/filters, applied after the signal is generated
    filter: DirectForm1<f32>,
    filter_env: Envelope<f32>,
}

impl OSCGroup {
    fn new(sample_rate: SampleRate) -> OSCGroup {
        OSCGroup {
            osc: Oscillator::new(),
            vol_env: Envelope::<Decibel>::new(),
            vibrato_env: Envelope::<f32>::new(),
            vibrato_lfo: Oscillator::new(),
            filter_env: Envelope::<f32>::new(),
            filter: DirectForm1::<f32>::new(
                biquad::Coefficients::<f32>::from_params(
                    biquad::Type::LowPass,
                    sample_rate.hz(),
                    (10000).hz(),
                    Q_BUTTERWORTH_F32,
                )
                .unwrap(),
            ),
        }
    }

    /// Get the next sample from the osc group, applying modulation parameters
    /// as well.
    /// base_vel - The velocity of the note. This is affected by volume
    ///            modulation.
    /// base_note - The base pitch, in Hz, of the note
    /// pitch_bend - A [-1.0, 1.0] range value
    /// (mod_type, modulation) - Indicates what modulation type, if any, to
    ///                          apply to the signal. This is from OSC 2
    /// mod_bank - the various mod_bank envelopes and LFOs that also modulate
    ///            the signal
    /// apply_filter - if true, apply the current filter.
    fn next_sample(
        &mut self,
        params: &MeowParameters,
        context: NoteContext,
        base_vel: f32,
        base_note: Hertz<f32>,
        pitch_bend: f32,
    ) -> f32 {
        let sample_rate = context.sample_rate;

        // Compute volume from parameters
        let vol_env = self.vol_env.get(&params.vol_envelope(), context);

        // Apply parameter, ADSR, LFO, and AmpMod for total volume
        // We clamp the LFO to positive values because negative values cause the
        // signal to be inverted, which isn't what we want (instead it should
        // just have zero volume). We don't do this for the AmpMod because inverting
        // the signal allows for more interesting audio.
        let total_volume = base_vel * vol_env.get_amp().max(0.0) * params.master_vol();

        let vibrato_env = self.vibrato_env.get(&params.vibrato_lfo(), context);
        // Compute note pitch multiplier
        let vibrato_lfo = self.vibrato_lfo.next_sample(
            sample_rate,
            NoteShape::Sine,
            params.vibrato_lfo().freq(),
            1.0,
        ) * vibrato_env;
        let pitch_bend = to_pitch_multiplier(pitch_bend, params.pitchbend_max() as i32);
        let pitch_mods = to_pitch_multiplier(vibrato_lfo, 24);

        // The final pitch multiplier, post-FM
        // Base note is the base note frequency, in hz
        // Pitch mods consists of the applied pitch bend, pitch ADSR, pitch LFOs
        // applied to it, with a max range of 12 semis.
        // Fine and course pitchbend come from the parameters.
        // The FM Mod comes from the modulation value.
        // Mod bank pitch comes from the mod bank.
        let pitch = base_note.hz() * pitch_mods * pitch_bend;

        let shape = NoteShape::Skewtooth(1.0);

        // Get next sample
        let value = self
            .osc
            .next_sample(sample_rate, shape, pitch.hz(), params.phase());

        // Apply noise
        let noise = NoteShape::Noise.get(0.0);
        let value = value + noise * params.noise_mix();
        // TODO: check if the noise is applied before or after the filter!

        // Apply filter
        let value = {
            let filter = params.filter();
            let filter_env = self.filter_env.get(&params.filter_envelope(), context);
            let cutoff_freq = (filter.cutoff_freq.hz() * filter_env).hz();

            let coefficents = biquad::Coefficients::<f32>::from_params(
                filter.filter_type,
                sample_rate.hz(),
                cutoff_freq,
                filter.q_value.max(0.0),
            )
            .unwrap();

            self.filter.update_coefficients(coefficents);
            let output = self.filter.run(value);
            if output.is_finite() {
                lerp(value, output, params.filter().dry_wet)
            } else {
                // If the output happens to be NaN or Infinity, output the
                // original  signal instead. Hopefully, this will "reset"
                // the filter on the next sample, instead of being filled
                // with garbage values.
                value
            }
        };

        value * total_volume
    }

    /// Handle hold-to-release and release-to-retrigger state transitions
    fn note_state_changed(&mut self, edge: NoteStateEdge) {
        match edge {
            NoteStateEdge::NoteReleased | NoteStateEdge::NoteRetriggered => {
                self.vol_env.remember();
            }
            _ => {}
        }
    }
}

#[derive(Debug)]
struct Oscillator {
    angle: Angle,
}

impl Oscillator {
    fn new() -> Oscillator {
        Oscillator { angle: 0.0 }
    }

    /// Return the next sample from the oscillator
    /// sample_rate - the sample rate of the note. This is used to ensure that
    ///               the pitch of a note stays the same across sample rates
    /// shape - what noteshape to use for the signal
    /// pitch - the pitch multiplier to be applied to the base frequency of the
    ///         oscillator.
    /// phase_mod - how much to add to the current angle value to produce a
    ///             a phase offset. Units are 0.0-1.0 normalized angles (so
    ///             0.0 is zero radians, 1.0 is 2pi radians.)
    fn next_sample(
        &mut self,
        sample_rate: SampleRate,
        shape: NoteShape,
        pitch: Hertz<f32>,
        phase_mod: f32,
    ) -> f32 {
        // Get the raw signal (we use rem_euclid here to constrain the angle
        // between 0.0-1.0 (the normal % operator would allow for negative angles
        // which we do not want!))
        // NOTE: fract also does not do what we want, since that also allows for negative angles.
        let value = shape.get((self.angle + phase_mod).rem_euclid(1.0));

        // Update the angle. Each sample is 1.0 / sample_rate apart for a
        // complete waveform. We also multiply by pitch to advance the right amount
        // We also constrain the angle between 0 and 1, as this reduces
        // roundoff error.
        let angle_delta = pitch.hz() / sample_rate.get();
        self.angle = (self.angle + angle_delta) % 1.0;

        value
    }
}

/// Convience struct for holding the external state a particular note (when it was
/// triggered, what state it is in, etc)
/// This is mostly needed for doing envelope calculations.
#[derive(Debug, Clone, Copy)]
struct NoteContext {
    /// The number of samples since the most recent note on event.
    /// This value is expected to reset on Retrigger-Held state transitions.
    samples_since_note_on: SampleTime,
    /// The current state of the note being played.
    note_state: NoteState,
    /// The sample rate, in Hz/sec.
    sample_rate: SampleRate,
}

#[derive(Debug)]
pub struct Envelope<T> {
    // The value to lerp from when in Retrigger or Release state
    ease_from: T,
    // The previous computed envelope value, updated every time get() is called
    last_env_value: T,
}

impl<T: EnvelopeType> Envelope<T> {
    pub fn new() -> Envelope<T> {
        Envelope {
            ease_from: T::zero(),
            last_env_value: T::zero(),
        }
    }

    /// Get the current envelope value. `time` is how many samples it has been
    /// since the start of the note
    fn get(&mut self, params: &impl EnvelopeParams<T>, context: NoteContext) -> T {
        let time = context.samples_since_note_on;
        let note_state = context.note_state;
        let sample_rate = context.sample_rate;

        let value = match note_state {
            NoteState::None => T::zero(),
            NoteState::Held => {
                let time = sample_rate.to_seconds(time);
                let attack = params.attack();
                let hold = params.hold();
                let decay = params.decay();
                let sustain = params.sustain();
                if time < attack {
                    // Attack
                    T::lerp_attack(T::zero(), T::one(), time / attack)
                } else if time < attack + hold {
                    // Hold
                    T::one()
                } else if time < attack + hold + decay {
                    // Decay
                    let time = time - attack - hold;
                    T::lerp_decay(T::one(), sustain, time / decay)
                } else {
                    // Sustain
                    sustain
                }
            }
            NoteState::Released(rel_time) => {
                let time = sample_rate.to_seconds(time - rel_time);
                T::lerp_release(self.ease_from, T::zero(), time / params.release())
            }
            NoteState::Retrigger(retrig_time) => {
                // Forcibly decay over RETRIGGER_TIME.
                let time = (time - retrig_time) as f32 / RETRIGGER_TIME as f32;
                T::lerp_retrigger(self.ease_from, T::zero(), time)
            }
        };
        // Store the premultiplied value. This is because using the post-multiplied
        // value will cause us to apply the multiply value again in release phase
        // which will cause unwanted clicks.
        self.last_env_value = value;
        value * params.multiply()
    }

    /// Set self.ease_from to the value computed by the most recent `get` call.
    /// This needs to be called JUST BEFORE transitioning from a Held to Released
    /// state or from Released to Retrigger state.
    /// In particular, if going from Held to Released, then `note_state` should
    /// be Held and `time` should be the last sample of the Hold state.
    /// And if going from Released to Retrigger, then `note_state` should be
    /// Released and `time` should be the last sample of the Released state.
    fn remember(&mut self) -> T {
        self.ease_from = self.last_env_value;
        self.ease_from
    }
}

/// The state of a note, along with the time and velocity that note has, if
/// relevant. The typical life cycle of a note is as follows:
/// None -> Held -> Released -> [removed] or Retrigger -> Held
/// Notes start at None, then become Held. When they are released, they become
/// Released, and are either removed if the release time on the note expires or
/// become Retrigger if the note is retriggered during the release time.
/// Retriggered notes become Held after a few samples automatically.
/// TODO: Note states should really not track the volume of the note. That should
/// be tracked on a per envelope basis, I think.
#[derive(Debug, Clone, Copy)]
enum NoteState {
    /// The note is not being held down, but no previous NoteOn or NoteOff exists
    /// for the note. This state indicates that a note was triggered this frame
    /// but the sample for when the note was triggered has not yet been reached.
    None,
    /// The note is being held down
    Held,
    /// The note has just been released. The field is in samples and denotes how many
    /// samples since the oscillator has started.
    Released(SampleTime),
    /// The note has just be retriggered during a release. Time is in samples
    /// since the oscillator has retriggered.
    Retrigger(SampleTime),
}

/// A state transition for a note.
#[derive(Debug, Copy, Clone)]
enum NoteStateEdge {
    InitialTrigger,  // The note is being pressed for the first time
    NoteReleased,    // The note has just been released
    NoteRetriggered, // The note is being pressed, but not the first time
}

/// The shape of a note. The associated f32 indicates the "warp" of the note.
/// The warp is a value between 0.0 and 1.0.
#[derive(Debug, Clone, Copy, VariantCount)]
pub enum NoteShape {
    /// A sine wave
    Sine,
    /// A duty-cycle wave. The note is a square wave when the warp is 0.5.
    /// The warp for this NoteShape is clamped between 0.001 and 0.999.
    Square(f32),
    /// A NoteShape which warps between a sawtooth and triangle wave.
    /// Sawtooths: 0.0 and 1.0
    /// Triangle: 0.5
    Skewtooth(f32),
    /// White noise
    Noise,
}

impl NoteShape {
    /// Return the raw waveform using the given angle
    fn get(&self, angle: Angle) -> f32 {
        match self {
            // See https://www.desmos.com/calculator/dqg8kdvung for visuals
            // and https://www.desmos.com/calculator/hs8zd0sfkh for more visuals
            NoteShape::Sine => (angle * TAU).sin(),
            NoteShape::Square(warp) => {
                // This clamp is used to prevent the note from being completely
                // silent, which would occur at 0.0 and 1.0.
                if angle < (*warp).clamp(0.001, 0.999) {
                    -1.0
                } else {
                    1.0
                }
            }
            NoteShape::Skewtooth(warp) => {
                let warp = *warp;
                // Check if the warp makes the note a sawtooth and directly calculate
                // it. This avoids potential divide by zero issues.
                // Clippy lint complains about floating point compares but this
                // is ok to do since 1.0 is exactly representible in floating
                // point and also warp is always in range [0.0, 1.0].
                #[allow(clippy::float_cmp)]
                if warp == 0.0 {
                    return -2.0 * angle + 1.0;
                } else if warp == 1.0 {
                    return 2.0 * angle - 1.0;
                }

                // Otherwise, compute a triangle/skewed triangle shape.
                if angle < warp {
                    (2.0 * angle / warp) - 1.0
                } else {
                    -(2.0 * (angle - warp)) / (1.0 - warp) + 1.0
                }
            }
            NoteShape::Noise => rand::Rng::gen_range(&mut rand::thread_rng(), -1.0..1.0),
        }
    }
}

impl std::fmt::Display for NoteShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use NoteShape::*;
        let string = match self {
            Sine => "Sine",
            Square(warp) => {
                if (warp - 0.5).abs() < 0.1 {
                    "Square"
                } else {
                    "Pulse"
                }
            }
            Skewtooth(warp) => {
                if (warp - 0.0).abs() < 0.1 || (warp - 1.0).abs() < 0.1 {
                    "Sawtooth"
                } else if (warp - 0.5).abs() < 0.1 {
                    "Triangle"
                } else {
                    "Skewtooth"
                }
            }
            Noise => "Noise",
        };

        write!(f, "{}", string)
    }
}

/// Returns an iterator of size num_samples which linearly interpolates between the
/// points specified by pitch_bend. last_pitch_bend is assumed to be the "-1th"
/// value and is used as the starting point.
/// Thank you to Cassie for this code!
pub fn to_pitch_envelope(
    pitch_bend: &[(f32, i32)],
    prev_pitch_bend: f32,
    num_samples: usize,
) -> (impl Iterator<Item = f32> + '_, f32) {
    // Linearly interpolate over num values
    fn interpolate_n(start: f32, end: f32, num: usize) -> impl Iterator<Item = f32> {
        (0..num).map(move |i| lerp(start, end, i as f32 / num as f32))
    }

    // We first make the first and last points to interpolate over. The first
    // point is just prev_pitch_bend, and the last point either gets the value
    // of the last point in pitch_bend, or just prev_pitch_bend if pitch_bend
    // is empty. If pitch_bend is nonempty, this means that the last "segment"
    // is constant value, which is okay since we can't see into the future
    // TODO: Use linear extrapolation for the last segment.
    let first = Some((prev_pitch_bend, 0));

    let last_bend = pitch_bend
        .last()
        .map(|&(bend, _)| bend)
        .unwrap_or(prev_pitch_bend);
    let last = Some((last_bend, num_samples as i32));

    // Now we make a list of points, starting with the first point, then all of
    // pitch_bend, then the last point
    let iter = first
        .into_iter()
        .chain(pitch_bend.iter().copied())
        .chain(last)
        // Make it a NeighborPairs so we can get the current point and the next point
        .neighbor_pairs()
        // Then interpolate the elements.
        .flat_map(|((start, a), (end, b))| {
            let num = b - a;
            interpolate_n(start, end, num as usize)
        });

    (iter, last_bend)
}

// rustc doesn't think "U7" is good snake case style, but its also the name of
// the type, so oh well.
#[allow(non_snake_case)]
/// Convert a U7 value into a normalized [0.0, 1.0] float.
pub fn normalize_u7(num: U7) -> f32 {
    // A U7 in is in range [0, 127]
    let num = U7::data_to_bytes(&[num])[0];
    // convert to f32 - range [0.0, 1.0]
    num as f32 / 127.0
}

/// Convert a PitchBend U14 value into a normalized [-1.0, 1.0] float
pub fn normalize_pitch_bend(pitch_bend: PitchBend) -> NormalizedPitchbend {
    // A pitchbend is a U14 in range [0, 0x3FFF] with 0x2000 meaning "no bend",
    // 0x0 meaning "max down bend" and 0x3FFF meaning "max up bend".
    // convert to u16 - range [0, 0x3FFF]
    let pitch_bend = U14::data_to_slice(&[pitch_bend])[0];
    // convert to i16 - range [-0x2000, 0x1FFF]
    let pitch_bend = pitch_bend as i16 - 0x2000;
    // convert to f32 - range [-1.0, 1.0]
    pitch_bend as f32 * (1.0 / 0x2000 as f32)
}

/// Convert a NormalizedPitchbend into a pitch multiplier. The multiplier is such
/// that a `pitch_bend` of +1.0 will bend up by `semitones` semitones and a value
/// of -1.0 will bend down by `semitones` semitones.
pub fn to_pitch_multiplier(pitch_bend: NormalizedPitchbend, semitones: i32) -> f32 {
    // Given any note, the note a single semitone away is 2^1/12 times the original note
    // So (2^1/12)^n is n semitones away
    let exponent = 2.0f32.powf(semitones as f32 / 12.0);
    // We take an exponential here because frequency is exponential with respect
    // to note value
    exponent.powf(pitch_bend)
}
