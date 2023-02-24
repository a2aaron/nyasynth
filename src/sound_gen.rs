use crate::{
    ease::{ease_in_poly, lerp, Easing},
    neighbor_pairs::NeighborPairsIter,
    params::{EnvelopeParams, OSCType},
};
use crate::{
    params::OSCParams, ModBankEnvs, ModulationBank, ModulationSend, ModulationType, Parameters,
};

use biquad::{Biquad, DirectForm1, ToHertz, Q_BUTTERWORTH_F32};
use derive_more::{Add, Sub};
use serde::{Deserialize, Serialize};
use variant_count::VariantCount;
use wmidi::{PitchBend, U14, U7};

const TAU: f32 = std::f32::consts::TAU;

// The time, in samples, for how long retrigger phase is.
const RETRIGGER_TIME: SampleTime = 88; // 88 samples is about 2 miliseconds.

// The threshold for which Decibel values below it will be treated as negative
// infinity dB.
pub const NEG_INF_DB_THRESHOLD: f32 = -70.0;

/// An offset, in samples, from the start of the frame.
type FrameDelta = usize;
type SampleTime = usize;
/// A value in range [0.0, 1.0] which denotes the position wihtin a wave cycle.
type Angle = f32;
/// The sample rate in Hz/seconds.
pub type SampleRate = f32;
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
    osc_2: OSCGroup,
    pub note: wmidi::Note,
    // The pitch of the note this SoundGenerator is playing, ignoring all coarse
    // detune and pitch bend effects. This is in hertz.
    note_pitch: f32,
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
            note_pitch: wmidi::Note::to_freq_f32(note),
            vel,
            samples_since_note_on: 0,
            note_state: NoteState::None,
            osc_1: OSCGroup::new(sample_rate, OSCType::OSC1),
            osc_2: OSCGroup::new(sample_rate, OSCType::OSC2),
            next_note_on: None,
            next_note_off: None,
        }
    }

    /// Returns true if the note is "alive" (playing audio). A note is dead if
    /// it is in the release state and it is after the total release time.
    pub fn is_alive(&self, sample_rate: SampleRate, params: &Parameters) -> bool {
        match self.note_state {
            NoteState::None | NoteState::Held | NoteState::Retrigger(_) => true,
            NoteState::Released(release_time) => {
                // The number of seconds it has been since release
                let time = (self.samples_since_note_on - release_time) as f32 / sample_rate;
                match params.osc_2_mod {
                    // If we mix together the two sounds then we should only kill the note
                    // after both oscillators have died.
                    ModulationType::Mix => {
                        time < params.osc_1.vol_adsr.release || time < params.osc_2.vol_adsr.release
                    }
                    _ => time < params.osc_1.vol_adsr.release,
                }
            }
        }
    }

    pub fn next_sample(
        &mut self,
        params: &Parameters,
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
                self.osc_2.note_state_changed(edge);

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
                self.osc_2.note_state_changed(NoteStateEdge::NoteReleased);

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

        // When osc_2 is not in mix mode, the note velocity is ignored (to make
        // it easier to get a consistent sound)
        let osc_2_is_mix = params.osc_2_mod == ModulationType::Mix;
        let osc_2 = self.osc_2.next_sample(
            &params.osc_2,
            context,
            if osc_2_is_mix { self.vel } else { 1.0 },
            self.note_pitch,
            pitch_bend,
            (ModulationType::Mix, 0.0),
            &params.mod_bank,
            osc_2_is_mix,
        );

        let osc_1 = self.osc_1.next_sample(
            &params.osc_1,
            context,
            self.vel,
            self.note_pitch,
            pitch_bend,
            (params.osc_2_mod, osc_2),
            &params.mod_bank,
            true,
        );

        fn pan_split(pan: f32) -> (f32, f32) {
            let radians = (pan + 1.0) * std::f32::consts::PI / 4.0;
            (radians.cos(), radians.sin())
        }

        if params.osc_2_mod == ModulationType::Mix {
            let (osc_1_left, osc_1_right) = pan_split(params.osc_1.pan);
            let (osc_2_left, osc_2_right) = pan_split(params.osc_2.pan);
            let left = osc_1 * osc_1_left + osc_2 * osc_2_left;
            let right = osc_1 * osc_1_right + osc_2 * osc_2_right;
            (left, right)
        } else {
            let (pan_left, pan_right) = pan_split(params.osc_1.pan);
            (osc_1 * pan_left, osc_1 * pan_right)
        }
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
    pitch_env: Envelope<f32>,
    mod_bank_envs: ModBankEnvs,
    volume_lfo: Oscillator,
    pitch_lfo: Oscillator,
    // The OSC that this OSCGroup belongs to.
    osc_type: OSCType,
    // The state for the EQ/filters, applied after the signal is generated
    filter: DirectForm1<f32>,
}

impl OSCGroup {
    fn new(sample_rate: f32, osc_type: OSCType) -> OSCGroup {
        OSCGroup {
            osc: Oscillator::new(),
            vol_env: Envelope::<Decibel>::new(),
            pitch_env: Envelope::<f32>::new(),
            mod_bank_envs: ModBankEnvs::new(),
            volume_lfo: Oscillator::new(),
            pitch_lfo: Oscillator::new(),
            filter: DirectForm1::<f32>::new(
                biquad::Coefficients::<f32>::from_params(
                    biquad::Type::LowPass,
                    sample_rate.hz(),
                    (10000).hz(),
                    Q_BUTTERWORTH_F32,
                )
                .unwrap(),
            ),
            osc_type,
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
        params: &OSCParams,
        context: NoteContext,
        base_vel: f32,
        base_note: f32,
        pitch_bend: f32,
        (mod_type, modulation): (ModulationType, f32),
        mod_bank: &ModulationBank,
        apply_filter: bool,
    ) -> f32 {
        let sample_rate = context.sample_rate;

        // TODO: consider merging ModulationValues with the rest of the modulation
        // calculations in this block of code. Some notes
        // 1. You probably need to commit to storing either the post-multiplied
        //    or the pre-multiplied values (WITH the semi-tone amount) in the
        //    pitch variable. This probably means making more pitch field?
        //    Also, some modulations have additional weird offsets applied
        //    EX: AmpMod and VolLFO both are plus one'd and VolLFO is clamped at
        //    zero. Tis is easy to do but will be annoying to generalize.
        //    Also, how do we handle LFOs in the mod bank? (i think this should
        //    handled at from_mod_bank time)
        //    You need to probably store pre-multiplied values for each of the
        //    various modulation values.

        let mod_bank = ModulationValues::from_mod_bank(
            &mut self.mod_bank_envs,
            mod_bank,
            context,
            self.osc_type,
        );

        // Compute volume from parameters, ADSR, LFO, and AmpMod
        let vol_env = self.vol_env.get(&params.vol_adsr, context);

        let vol_lfo = self.volume_lfo.next_sample(
            context.sample_rate,
            NoteShape::Sine,
            1.0 / params.vol_lfo.period,
            0.0, // no phase mod
        ) * params.vol_lfo.amplitude;

        let vol_mod = if mod_type == ModulationType::AmpMod {
            modulation
        } else {
            0.0
        };

        // Apply parameter, ADSR, LFO, and AmpMod for total volume
        // We clamp the LFO to positive values because negative values cause the
        // signal to be inverted, which isn't what we want (instead it should
        // just have zero volume). We don't do this for the AmpMod because inverting
        // the signal allows for more interesting audio.
        let total_volume = base_vel
            * (params.volume + vol_env).get_amp()
            * (1.0 + vol_lfo).max(0.0)
            * (1.0 + vol_mod)
            * (1.0 - mod_bank.amplitude);

        // Compute note pitch multiplier from ADSR and envelope
        let pitch_env = self.pitch_env.get(&params.pitch_adsr, context);
        let pitch_lfo = self.pitch_lfo.next_sample(
            sample_rate,
            NoteShape::Sine,
            1.0 / params.pitch_lfo.period,
            1.0,
        ) * params.pitch_lfo.amplitude;
        let pitch_coarse = to_pitch_multiplier(1.0, params.coarse_tune);
        let pitch_fine = to_pitch_multiplier(params.fine_tune / 100.0, 1);
        let pitch_bend = to_pitch_multiplier(pitch_bend, 12);
        let pitch_mods = to_pitch_multiplier(pitch_env + pitch_lfo, 24);
        let mod_bank_pitch = to_pitch_multiplier(mod_bank.pitch, 24);

        let fm_mod = if mod_type == ModulationType::FreqMod {
            to_pitch_multiplier(modulation, 24)
        } else {
            1.0
        };

        // The final pitch multiplier, post-FM
        // Base note is the base note frequency, in hz
        // Pitch mods consists of the applied pitch bend, pitch ADSR, pitch LFOs
        // applied to it, with a max range of 12 semis.
        // Fine and course pitchbend come from the parameters.
        // The FM Mod comes from the modulation value.
        // Mod bank pitch comes from the mod bank.
        let pitch = base_note
            * pitch_mods
            * pitch_bend
            * pitch_coarse
            * pitch_fine
            * fm_mod
            * mod_bank_pitch;

        let warp_mod = if mod_type == ModulationType::WarpMod {
            modulation
        } else {
            0.0
        };

        let phase_mod = if mod_type == ModulationType::PhaseMod {
            modulation
        } else {
            0.0
        };

        // Disable the filter when doing modulation (filtering the signal makes
        // it nearly impossible to get a nice modulation signal since it messes
        // with the phase a lot)
        let filter_params = if apply_filter {
            Some(params.filter_params)
        } else {
            None
        };

        // Get next sample
        let value = self.osc.next_sample(
            sample_rate,
            params.shape.add(warp_mod).add(mod_bank.warp),
            pitch,
            phase_mod + params.phase + mod_bank.phase,
        );

        // Apply filter (if desired)
        let value = match filter_params {
            Some(mut params) => {
                params.freq += mod_bank.filter_freq;
                let coefficents = FilterParams::into_coefficients(params, sample_rate);
                self.filter.update_coefficients(coefficents);
                let output = self.filter.run(value);
                if output.is_finite() {
                    output
                } else {
                    // If the output happens to be NaN or Infinity, output the
                    // original  signal instead. Hopefully, this will "reset"
                    // the filter on the next sample, instead of being filled
                    // with garbage values.
                    value
                }
            }
            None => value,
        };
        value * total_volume
    }

    /// Handle hold-to-release and release-to-retrigger state transitions
    fn note_state_changed(&mut self, edge: NoteStateEdge) {
        match edge {
            NoteStateEdge::NoteReleased | NoteStateEdge::NoteRetriggered => {
                self.vol_env.remember();
                self.mod_bank_envs.env_1.remember();
                self.mod_bank_envs.env_2.remember();
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
    ///         oscillator. This is a unitless value.
    /// phase_mod - how much to add to the current angle value to produce a
    ///             a phase offset. Units are 0.0-1.0 normalized angles (so
    ///             0.0 is zero radians, 1.0 is 2pi radians.)
    fn next_sample(
        &mut self,
        sample_rate: SampleRate,
        shape: NoteShape,
        pitch: f32,
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
        let angle_delta = pitch / sample_rate;
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
                let time = time as f32 / sample_rate;
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
                let time = (time - rel_time) as f32 / sample_rate;
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

/// The modulation values for each of the various parameters that can be modulated
/// These values need to be in normalized float format, where 0.0 means "no modulation"
/// +1.0 means "max positive modulation", and -1.0 means "max negative modulation"
/// It is acceptable for values to be outside the -1.0 - +1.0 range.
#[derive(Debug, Default, Add)]
struct ModulationValues {
    amplitude: f32,
    pitch: f32, // pre-multiplied pitch bend value
    phase: f32,
    warp: f32,
    filter_freq: f32,
}

impl ModulationValues {
    fn from_value(value: f32, send: ModulationSend) -> ModulationValues {
        let (mut amplitude, mut pitch, mut phase, mut warp, mut filter_freq) =
            (0.0, 0.0, 0.0, 0.0, 0.0);
        match send {
            ModulationSend::Amplitude => amplitude = value,
            ModulationSend::Phase => phase = value,
            ModulationSend::Pitch => pitch = value,
            ModulationSend::Warp => warp = value,
            ModulationSend::FilterFreq => filter_freq = value,
        }
        ModulationValues {
            amplitude,
            pitch,
            phase,
            warp,
            filter_freq,
        }
    }

    fn from_mod_bank(
        mod_bank_envs: &mut ModBankEnvs,
        mod_bank: &ModulationBank,
        context: NoteContext,
        osc_type: OSCType,
    ) -> ModulationValues {
        let env_1 = if mod_bank.env_1_send.osc == osc_type {
            mod_bank_envs.env_1.get(&mod_bank.env_1, context)
        } else {
            0.0
        };

        let env_2 = if mod_bank.env_2_send.osc == osc_type {
            mod_bank_envs.env_2.get(&mod_bank.env_2, context)
        } else {
            0.0
        };

        ModulationValues::from_value(env_1, mod_bank.env_1_send.mod_type)
            + ModulationValues::from_value(env_2, mod_bank.env_2_send.mod_type)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FilterParams {
    pub filter: biquad::Type<f32>,
    /// in [0.0 - 1.0] float range. When turning into an actual Hertz value,
    /// this value is clamped between 20 and 99% of the Nyquist frequency
    /// in order to prevent numerical instability at extremely high or low values
    /// and/or blowing out the speakers.
    pub freq: f32,
    /// Must be non-negative. If it is negative, it will be clamped to zero
    pub q_value: f32,
}

impl FilterParams {
    fn into_coefficients(
        params: FilterParams,
        sample_rate: SampleRate,
    ) -> biquad::Coefficients<f32> {
        // convert normalized 0.0-1.0 to a frequency in Hertz
        let freq = ease_in_poly(params.freq, 4).clamp(0.0, 1.0) * 22100.0;
        // avoid numerical instability encountered at very low
        // or high frequencies. Clamping at around 20 Hz also
        // avoids blowing out the speakers.
        let freq = freq.clamp(20.0, sample_rate * 0.99 / 2.0).hz();
        let q_value = params.q_value.max(0.0);
        biquad::Coefficients::<f32>::from_params(params.filter, sample_rate.hz(), freq, q_value)
            .unwrap()
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
    pub fn new(shape: NoteShapeDiscrim, warp: f32) -> NoteShape {
        let warp = warp.clamp(0.0, 1.0);
        match shape {
            NoteShapeDiscrim::Sine => NoteShape::Sine,
            NoteShapeDiscrim::Square => NoteShape::Square(warp),
            NoteShapeDiscrim::Skewtooth => NoteShape::Skewtooth(warp),
            NoteShapeDiscrim::Noise => NoteShape::Noise,
        }
    }

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

    /// Create a NoteShape using the given shape and warp. This is used for
    /// RawParameters mainly.
    pub fn from_f32s(shape: f32, warp: f32) -> Self {
        NoteShape::new(
            crate::params::EASER.shape.ease(shape),
            crate::params::EASER.warp.ease(warp),
        )
    }

    /// Add the warp of the given NoteShape with the modulation parameter. This
    /// is used for note shape modulation.
    fn add(&self, modulate: f32) -> Self {
        use NoteShape::*;
        match self {
            Sine => Sine,
            Square(warp) => Square((warp + modulate).clamp(0.0, 1.0)),
            Skewtooth(warp) => Skewtooth((warp + modulate).clamp(0.0, 1.0)),
            Noise => Noise,
        }
    }

    pub fn get_warp(&self) -> f32 {
        match self {
            NoteShape::Square(warp) | NoteShape::Skewtooth(warp) => *warp,
            // TODO: is it really okay to return 0.5 here?
            NoteShape::Sine | NoteShape::Noise => 0.5,
        }
    }

    pub fn get_shape(&self) -> NoteShapeDiscrim {
        match self {
            NoteShape::Sine => NoteShapeDiscrim::Sine,
            NoteShape::Square(_) => NoteShapeDiscrim::Square,
            NoteShape::Skewtooth(_) => NoteShapeDiscrim::Skewtooth,
            NoteShape::Noise => NoteShapeDiscrim::Noise,
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

#[derive(Debug, Clone, Copy, VariantCount, Serialize, Deserialize, PartialEq, Eq)]
pub enum NoteShapeDiscrim {
    Sine,
    Square,
    Skewtooth,
    Noise,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Add, Sub)]
pub struct Decibel(f32);

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

impl crate::sound_gen::EnvelopeType for Decibel {
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

impl Decibel {
    pub const fn from_db(db: f32) -> Decibel {
        Decibel(db)
    }

    pub const fn neg_inf_db() -> Decibel {
        Decibel::from_db(NEG_INF_DB_THRESHOLD)
    }

    pub const fn zero_db() -> Decibel {
        Decibel::from_db(0.0)
    }

    pub fn from_amp(amp: f32) -> Decibel {
        Decibel::from_db(f32::log10(amp) * 10.0)
    }

    // Linearly interpolate in amplitude space.
    pub fn lerp_amp(start: Decibel, end: Decibel, t: f32) -> Decibel {
        let amp = crate::sound_gen::lerp(start.get_amp(), end.get_amp(), t);
        Decibel::from_amp(amp)
    }

    // Linearly interpolate in Decibel space.
    pub fn lerp_db(start: f32, end: f32, t: f32) -> Decibel {
        let db = crate::sound_gen::lerp(start, end, t);
        Decibel::from_db(db)
    }

    // Linearly interpolate in Decibel space, but values of t below 0.125 will
    // lerp from `start` to `Decibel::zero()`. This function is meant for use
    // with user-facing parameter knobs.
    pub const fn ease_db(start: f32, end: f32) -> Easing<Decibel> {
        Easing::SplitLinear {
            start: Decibel::neg_inf_db(),
            mid: Decibel::from_db(start),
            end: Decibel::from_db(end),
            split_at: 0.125,
        }
    }

    pub fn get_amp(&self) -> f32 {
        if self.get_db() <= NEG_INF_DB_THRESHOLD {
            0.0
        } else {
            10.0f32.powf(self.get_db() / 10.0)
        }
    }

    pub fn get_db(&self) -> f32 {
        self.0
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
pub fn normalize_U7(num: U7) -> f32 {
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
