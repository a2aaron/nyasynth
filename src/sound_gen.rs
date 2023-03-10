use crate::{
    common::{Hertz, Note, Pitch, Pitchbend, SampleRate, SampleTime, Seconds, Vel},
    ease::lerp,
    params::{EnvelopeParams, MeowParameters},
};

use biquad::{Biquad, DirectForm1, ToHertz, Q_BUTTERWORTH_F32};
use nih_plug::prelude::Enum;

const TAU: f32 = std::f32::consts::TAU;

// The time, in samples, for how long retrigger phase is.
pub const RETRIGGER_TIME: SampleTime = 88; // 88 samples is about 2 miliseconds.

/// An offset, in samples, from the start of the frame.
type FrameDelta = usize;

/// A value in range [0.0, 1.0] which denotes the position wihtin a wave cycle.
type Angle = f32;

/// A small noise generator using xorshift.
pub struct NoiseGenerator {
    state: u32,
}

impl NoiseGenerator {
    pub fn new() -> NoiseGenerator {
        let mut bytes = [0, 0, 0, 0];
        // If this fails, then we just default to the random seed of 413. Any non-zero seed is acceptable
        // for our white noise generating purposes. Also, this almost certainly won't fail.
        let _ = getrandom::getrandom(&mut bytes);
        let mut seed = u32::from_be_bytes(bytes);
        if seed == 0 {
            seed = 413
        }
        NoiseGenerator { state: seed }
    }

    fn next(&mut self) -> f32 {
        // RNG algorithm used here is Xorshift, specifically the one listed at Wikipedia
        // https://en.wikipedia.org/wiki/Xorshift
        let x = self.state;
        let x = x ^ (x << 13);
        let x = x ^ (x >> 17);
        let x = x ^ (x << 5);
        self.state = x;

        // Mantissa trick: Every float in [2.0 - 4.0] is evenly spaced
        // so if you want evenly distributed floats, just jam random bits in the mantissa
        // and then convert to the appropriate range by subtraciting.

        // set exponent + sign bit to zero
        let x = x & 0b0_00000000_11111111111111111111111;
        // set exponent to 1000000
        let x = x | 0b0_10000000_00000000000000000000000;
        // This ensures x has the following value:
        // 0 10000000 XXXXXXXXXXXXXXXXXXXXXXX
        // ^ ^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^
        // | |        mantissa
        // | exponent
        // sign
        // Where X is a random bit, and 0 or 1 are constant. This ensures that x, interpreted as a
        // float, is a randomly chosen float in range [2.0 - 4.0]
        // Finally, to get the [-1.0, 1.0] range, we just subtract by 3.0.
        f32::from_bits(x) - 3.0
    }
}
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

#[derive(Debug)]
pub struct SoundGenerator {
    osc_1: OSCGroup,
    pub note: Note,
    // The ending pitch from which portamento ends up at. This and `start_pitch` are unaffected by
    // by pitch bend and pitch modifiers.
    end_pitch: Pitch,
    // The starting pitch from which portamento bends from.
    start_pitch: Pitch,
    // The velocity of the note that this SoundGenerator is playing, ignoring all
    // amplitude modulation effects. This is a 0.0 - 1.0 normalized value.
    vel: Vel,
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
    next_note_on: Option<(FrameDelta, NoteOnEvent)>,
    // If Some(frame_delta), then the next note off event occurs in the next frame
    // at `frame_delta` samples into the frame
    next_note_off: Option<FrameDelta>,
    // The computed filter sweep values. This is updated on NoteOn (or any note velocity change) as
    // well as once per block.
    filter_sweep: FilterSweeper,
}

impl SoundGenerator {
    pub fn new(
        params: &MeowParameters,
        note: Note,
        vel: Vel,
        sample_rate: SampleRate,
    ) -> SoundGenerator {
        let end_pitch = Pitch::from_note(note);
        let start_pitch = end_pitch;
        SoundGenerator {
            note,
            start_pitch,
            end_pitch,
            vel,
            samples_since_note_on: 0,
            note_state: NoteState::None,
            osc_1: OSCGroup::new(sample_rate),
            next_note_on: None,
            next_note_off: None,
            filter_sweep: FilterSweeper::new(params, vel),
        }
    }

    /// Returns true if the note is "alive" (playing audio). A note is dead if
    /// it is in the release state and it is after the total release time.
    pub fn is_alive(&self, sample_rate: SampleRate, params: &MeowParameters) -> bool {
        match self.note_state {
            NoteState::None | NoteState::Held | NoteState::Retrigger { .. } => true,
            NoteState::Released(release_time) => {
                // The number of seconds it has been since release
                let time = sample_rate.to_seconds(self.samples_since_note_on - release_time);
                time < params.vol_envelope.release()
            }
        }
    }

    pub fn next_sample(
        &mut self,
        params: &MeowParameters,
        noise_generator: &mut NoiseGenerator,
        i: FrameDelta,
        sample_rate: SampleRate,
        pitch_bend: Pitchbend,
        vibrato_mod: f32,
    ) -> (f32, f32) {
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
            Some((note_on, event)) if note_on == i => {
                let edge = match self.note_state {
                    NoteState::None => NoteStateEdge::InitialTrigger,
                    _ => NoteStateEdge::NoteRetriggered,
                };

                self.osc_1.note_state_changed(edge);

                // Update the note state
                match self.note_state {
                    NoteState::None => {
                        self.note_state = NoteState::Held;

                        self.apply_note_on_event(params, event);
                    }
                    _ => {
                        self.note_state = NoteState::Retrigger {
                            start_time: self.samples_since_note_on,
                            event,
                        };

                        // On a Retrigger, save the NoteOn event to be applied after the Retrigger
                        // state is finished. If we apply the NoteOn event now, a click will be caused
                        // due to the note pitch changing suddenly
                    }
                };
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
        if let NoteState::Retrigger {
            start_time: retrigger_time,
            event,
        } = self.note_state
        {
            if self.samples_since_note_on - retrigger_time > RETRIGGER_TIME {
                self.apply_note_on_event(params, event);

                self.note_state = NoteState::Held;
                self.samples_since_note_on = 0;
            }
        }

        // Note: Note context will have changed from above state transitions, so we must get a fresh copy.
        // Failing to do so means that we persist in the release state for one sample too long (which
        // cause envelope discontiunities due to the fact that `note_state_changed` calls `remember`, which
        // in turn results in the `ease_from` values for envelopes changing.)
        let osc_1 = self.osc_1.next_sample(
            &params,
            self.filter_sweep,
            self.get_note_context(sample_rate),
            noise_generator,
            self.vel,
            self.get_current_pitch(sample_rate, params.portamento_time),
            pitch_bend,
            vibrato_mod,
        );

        (osc_1, osc_1)
    }

    pub fn note_on(&mut self, frame_delta: i32, vel: Vel, bend_note: Option<Note>) {
        let start_pitch = bend_note.map(Pitch::from_note);
        let note_on = NoteOnEvent::new(self.note, vel, start_pitch);
        self.next_note_on = Some((frame_delta as usize, note_on));
    }

    pub fn note_off(&mut self, frame_delta: i32) {
        self.next_note_off = Some(frame_delta as usize);
    }

    pub fn is_released(&self) -> bool {
        match self.note_state {
            NoteState::Released(_) => true,
            NoteState::None => false,
            NoteState::Held => false,
            NoteState::Retrigger { .. } => false,
        }
    }

    pub fn retrigger(
        &mut self,
        sample_rate: SampleRate,
        portamento_time: Seconds,
        bend_from_current: bool,
        new_note: Note,
        new_vel: Vel,
        frame_delta: i32,
    ) {
        let start_pitch = if bend_from_current {
            Some(self.get_current_pitch(sample_rate, portamento_time))
        } else {
            None
        };

        let note_on = NoteOnEvent::new(new_note, new_vel, start_pitch);
        self.next_note_on = Some((frame_delta as usize, note_on));
        self.next_note_off = None;
    }

    fn get_note_context(&self, sample_rate: SampleRate) -> NoteContext {
        NoteContext {
            note_state: self.note_state,
            sample_rate,
            samples_since_note_on: self.samples_since_note_on,
        }
    }

    fn get_current_pitch(&self, sample_rate: SampleRate, portamento_time: Seconds) -> Pitch {
        let context = self.get_note_context(sample_rate);
        let time = context
            .sample_rate
            .to_seconds(context.samples_since_note_on);
        let t = (time / portamento_time).clamp(0.0, 1.0);
        lerp(self.start_pitch, self.end_pitch, t)
    }

    /// Apply the values contained within a NoteOnEvent. This is used to set the velocity and pitch
    /// values that occur at the start of a note event (that is, this function should be called
    /// when entering the Held state).
    fn apply_note_on_event(&mut self, params: &MeowParameters, event: NoteOnEvent) {
        self.note = event.note;
        self.vel = event.vel;
        self.start_pitch = event.start_pitch;
        self.end_pitch = event.end_pitch;
        self.filter_sweep = FilterSweeper::new(params, event.vel);
    }
}

#[derive(Debug)]
struct OSCGroup {
    osc: Oscillator,
    vol_env: Envelope<f32>,
    vibrato_env: Envelope<f32>,
    // The state for the EQ/filters, applied after the signal is generated
    filter: DirectForm1<f32>,
    filter_env: Envelope<f32>,
    sample_counter: usize,
}

impl OSCGroup {
    fn new(sample_rate: SampleRate) -> OSCGroup {
        OSCGroup {
            osc: Oscillator::new(),
            vol_env: Envelope::<f32>::new(),
            vibrato_env: Envelope::<f32>::new(),
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
            sample_counter: 0,
        }
    }

    /// Get the next sample from the osc group, applying modulation parameters
    /// as well.
    /// base_vel - The velocity of the note. This is affected by volume
    ///            modulation. This is a 0.0-1.0 normalized value.
    /// base_note - The base pitch of the note
    /// pitch_bend - A [-1.0, 1.0] range value
    /// (mod_type, modulation) - Indicates what modulation type, if any, to
    ///                          apply to the signal. This is from OSC 2
    /// mod_bank - the various mod_bank envelopes and LFOs that also modulate
    ///            the signal
    /// apply_filter - if true, apply the current filter.
    fn next_sample(
        &mut self,
        params: &MeowParameters,
        filter_sweep: FilterSweeper,
        context: NoteContext,
        noise_generator: &mut NoiseGenerator,
        base_vel: Vel,
        base_note: Pitch,
        pitch_bend: Pitchbend,
        vibrato_mod: f32,
    ) -> f32 {
        let sample_rate = context.sample_rate;

        // Compute volume from parameters
        let vol_env = {
            // Easing computed somewhat empirically.
            // See https://www.desmos.com/calculator/r7k5ee8k5j for details.
            let x = self.vol_env.get(&params.vol_envelope, context);
            (x * x * x + x) / 2.0
        };
        let total_volume = base_vel.raw * vol_env.max(0.0);

        let pitch_mod = {
            let pitch_bend_mod = pitch_bend.get() * (params.pitchbend_max as f32);

            // Both vibrato_mod and vibrato_env are in the 0.0-1.0 range. We multiply by two here to
            // allow the vibrato to modulate the pitch by up to two semitones.
            let vibrato_env = self.vibrato_env.get(&params.vibrato_attack, context);
            let vibrato_mod = vibrato_mod * vibrato_env * 2.0;

            // Given any note, the note a single semitone away is 2^1/12 times the original note
            // So (2^1/12)^n = 2^(n/12) is n semitones away.
            Pitch((vibrato_mod + pitch_bend_mod) / 12.0)
        };

        // Note that we can just add these values together. This is because base_note and pitch_mod
        // are in the same linear space (specifically: +1.0 maps to one octave, which happens because
        // converting to and from Hertz uses exp2 and log2).
        let pitch = (base_note + pitch_mod).into_hertz();

        let shape = NoteShape::Sawtooth;

        // Get next sample
        let value = self
            .osc
            .next_sample(sample_rate, shape, pitch, params.phase);

        // Apply noise, if the noise is turned on.
        let value = if params.noise_mix > 0.01 {
            let noise = noise_generator.next();
            value + noise * params.noise_mix
        } else {
            value
        };

        // Apply filter
        let value = {
            let filter = &params.filter;
            // TODO: investigate if this is correct
            let filter_env = self.filter_env.get(&params.filter_envelope, context);

            let cutoff_freq = filter_sweep.lerp(filter_env);

            // avoid numerical instability encountered at very low
            // or high frequencies. Clamping at around 20 Hz also
            // avoids blowing out the speakers.
            let cutoff_freq = cutoff_freq.clamp(20.0, sample_rate.0 * 0.99 / 2.0);

            if self.sample_counter % 16 == 0 {
                let coefficents = biquad::Coefficients::<f32>::from_params(
                    filter.filter_type,
                    sample_rate.hz(),
                    cutoff_freq.into(),
                    filter.q_value.max(0.0),
                )
                .unwrap();
                self.filter.update_coefficients(coefficents);
            }

            let output = self.filter.run(value);
            if output.is_finite() {
                lerp(value, output, params.filter.dry_wet)
            } else {
                // If the output happens to be NaN or Infinity, output the
                // original  signal instead. Hopefully, this will "reset"
                // the filter on the next sample, instead of being filled
                // with garbage values.
                value
            }
        };
        self.sample_counter += 1;
        value * total_volume
    }

    /// Handle hold-to-release and release-to-retrigger state transitions
    fn note_state_changed(&mut self, edge: NoteStateEdge) {
        match edge {
            NoteStateEdge::NoteReleased | NoteStateEdge::NoteRetriggered => {
                self.vol_env.remember();
                self.filter_env.remember();
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FilterSweeper {
    start_pitch: Pitch,
    end_pitch: Pitch,
}

impl FilterSweeper {
    fn new(params: &MeowParameters, base_vel: Vel) -> FilterSweeper {
        let start_freq = params.filter.cutoff_freq;
        let end_freq = params.filter.cutoff_freq + params.filter_envelope.env_mod * base_vel.eased;
        FilterSweeper {
            start_pitch: Pitch::from_hertz(start_freq),
            end_pitch: Pitch::from_hertz(end_freq),
        }
    }

    fn lerp(&self, t: f32) -> Hertz {
        let pitch = lerp(self.start_pitch, self.end_pitch, t);
        pitch.into_hertz()
    }
}

#[derive(Debug)]
pub struct Oscillator {
    angle: Angle,
}

impl Oscillator {
    pub fn new() -> Oscillator {
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
    pub fn next_sample(
        &mut self,
        sample_rate: SampleRate,
        shape: NoteShape,
        pitch: Hertz,
        phase_mod: f32,
    ) -> f32 {
        // Get the raw signal. Note that we constrain the angle between 0.0-1.0. Since phase_mod is
        // required to be positive and so is self.angle, then `fract` itself is also always positive
        // and hence in 0.0-1.0 range. Note that we use `fract` instead of just doing `% 1.0` since
        // fmod is slow.
        let angle = (self.angle + phase_mod).fract();
        let value = shape.get(angle);

        // Update the angle. Each sample is 1.0 / sample_rate apart for a
        // complete waveform. We also multiply by pitch to advance the right amount
        // We also constrain the angle between 0 and 1, as this reduces
        // roundoff error.
        let angle_delta = pitch.get() / sample_rate.get();
        // Similary, compute (self.angle + angle_delta) % 1.0 without actually calling fmod
        self.angle = (self.angle + angle_delta).fract();

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

/// Convience struct for the next_note_on flag.
#[derive(Debug, Clone, Copy)]
struct NoteOnEvent {
    vel: Vel,
    note: Note,
    start_pitch: Pitch,
    end_pitch: Pitch,
}

impl NoteOnEvent {
    fn new(note: Note, vel: Vel, start_pitch: Option<Pitch>) -> NoteOnEvent {
        let end_pitch = Pitch::from_note(note);
        let start_pitch = start_pitch.unwrap_or(end_pitch);
        NoteOnEvent {
            vel,
            note,
            start_pitch,
            end_pitch,
        }
    }
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
                // We check if the attack time is zero. If so, we skip the attack phase.
                if time < attack && attack.get() != 0.0 {
                    // Attack
                    T::lerp_attack(T::zero(), T::one(), time / attack)
                } else if time < attack + hold {
                    // Hold
                    T::one()
                } else if time < attack + hold + decay && decay.get() != 0.0 {
                    // Similarly, we check if decay is zero. If so, skikp right to sustain.
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
                // If release is zero, then skip release and drop instantly to zero.
                if params.release().get() != 0.0 {
                    T::lerp_release(self.ease_from, T::zero(), time / params.release())
                } else {
                    T::zero()
                }
            }
            NoteState::Retrigger {
                start_time: retrigger_time,
                ..
            } => {
                // Forcibly decay over RETRIGGER_TIME.
                let time = (time - retrigger_time) as f32 / RETRIGGER_TIME as f32;
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
    /// The note has just be retriggered during a release. `start_time` denotes when this state was
    /// entered (specifically: it is a timestamp of how many samples since the `Held` state was entered.)
    /// The `event` field is the `NoteOnEvent` that should be applied once the Retrigger state ends.
    Retrigger {
        start_time: SampleTime,
        event: NoteOnEvent,
    },
}

/// A state transition for a note.
#[derive(Debug, Copy, Clone)]
enum NoteStateEdge {
    InitialTrigger,  // The note is being pressed for the first time
    NoteReleased,    // The note has just been released
    NoteRetriggered, // The note is being pressed, but not the first time
}

#[derive(Debug, Clone, Copy, PartialEq, Enum)]
pub enum NoteShape {
    /// A sine wave
    Sine,
    /// A sawtooth wave
    Sawtooth,
    /// A triangle wave, with a warp parameter.
    Triangle,
}

impl NoteShape {
    /// Return the raw waveform using the given angle
    fn get(&self, angle: Angle) -> f32 {
        // See https://www.desmos.com/calculator/dqg8kdvung for visuals
        // and https://www.desmos.com/calculator/hs8zd0sfkh for more visuals
        match self {
            NoteShape::Sine => (angle * TAU).sin(),
            NoteShape::Sawtooth => 2.0 * angle - 1.0,
            NoteShape::Triangle => {
                // Otherwise, compute a triangle/skewed triangle shape.
                if angle < 0.5 {
                    (angle) - 1.0
                } else {
                    -(2.0 * angle - 1.0) / (0.5) + 1.0
                }
            }
        }
    }
}
