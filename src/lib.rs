#![feature(trait_alias)]
#![feature(anonymous_lifetime_in_impl_trait)]
#![feature(portable_simd)]
#![feature(let_chains)]

mod chorus;
pub mod common;
pub mod ease;
mod keys;
mod neighbor_pairs;
mod params;
mod sound_gen;

use std::sync::Arc;

use chorus::Chorus;
use common::{Note, Pitch, Pitchbend, SampleRate, Vel};
use ease::lerp;
use keys::KeyTracker;
use nih_plug::{nih_export_vst3, prelude::*};
use params::{MeowParameters, Parameters};

use sound_gen::{NoiseGenerator, Oscillator, SoundGenerator, RETRIGGER_TIME, SIMD_SIZE};

/// The main plugin struct.
pub struct Nyasynth {
    /// All the notes to be played.
    notes: Vec<SoundGenerator>,
    /// The parameters which are shared with the VST host
    params: Arc<Parameters>,
    pitch_bend_smoother: Smoother<Pitchbend>,
    key_tracker: KeyTracker,
    // The vibrato LFO is global--the vibrato amount is shared across all generators, although each
    // generator gets it's own vibrato envelope.
    vibrato_lfo: Oscillator,
    // The chorus effect is also global.
    chorus: Chorus,
    /// The global noise generator
    noise_generator: NoiseGenerator,
    sample_rate: SampleRate,
}

impl Plugin for Nyasynth {
    type SysExMessage = ();
    type BackgroundTask = ();

    const NAME: &'static str = "Nyasynth";
    const VENDOR: &'static str = "a2aaron";
    const URL: &'static str = "https://a2aaron.github.io/";
    const EMAIL: &'static str = "aaronko@umich.edu";
    const VERSION: &'static str = "1.0";

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: None,
        main_output_channels: NonZeroU32::new(2),
        aux_input_ports: &[],
        aux_output_ports: &[],
        names: PortNames::const_default(),
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::MidiCCs;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = false;
    const HARD_REALTIME_ONLY: bool = false;

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        nih_plug::wrapper::setup_logger();
        std::env::set_var("NIH_LOG", "/Users/aaron/dev/Rust/nyasynth/nyasynth_nih.log");
        nih_log!("Initalizing VST...");
        // On a retrigger, the next note is delayed by RETRIGGER_TIME. Hence, there is a latency
        // of RETRIGGER_TIME. Note that this latency doesn't exist for non-retriggered notes.
        context.set_latency_samples(RETRIGGER_TIME as u32);
        self.set_sample_rate(SampleRate(buffer_config.sample_rate));
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let sample_rate = SampleRate(context.transport().sample_rate);
        if sample_rate != self.sample_rate {
            self.set_sample_rate(sample_rate);
        }

        let num_samples = buffer.samples();
        let tempo = context.transport().tempo.unwrap_or(120.0) as f32;

        let params = MeowParameters::new(&self.params, tempo);

        // remove "dead" notes
        // we do this _before_ processing any events
        // because this is the start of a new frame, and we want to make sure
        // that midi messages do not apply to dead notes
        // ex: if we do this after processing midi messages, a bug occurs where
        // - frame 0 - note is in release state and is dead by end of frame
        // - frame 1 - process events send midi messages to dead note
        // - frame 1 - process removes dead note
        // - frame 1 - user is confused to why note does not play despite holding
        //             down key (the KeyOn event was "eaten" by the dead note!)
        {
            // this weird block is required because the closure in `retain` captures all of `self`
            // if you pass it `self.sample_rate` or `self.params`. Doing it like this allows it to
            // only capture the `params` field, which avoids the issue of cannot borrow while
            // mutably borrowed
            self.notes.retain(|gen| gen.is_alive(sample_rate, &params));
        }

        let (left_out, right_out) = {
            let outputs = buffer.as_slice();
            let (left_out, rest) = outputs.split_first_mut().unwrap();
            let right_out = &mut rest[0];
            (left_out, right_out)
        };

        let mut block_start = 0;
        while block_start < num_samples {
            // Initially set the block size to SIMD_SIZE (or, if the number of samples in the buffer
            // is smaller than SIMD_SIZE, to just that value)
            let mut block_len = (num_samples - block_start).min(SIMD_SIZE);
            // Consume all events from the context which happen before or at the start
            // of the block. This also shrinks the current block if there would be an event within
            // the block.
            while let Some(next_event) = context.peek_event() {
                let timing = next_event.timing() as usize;
                // If the event occurs before or at the start of this block, then process the event
                if timing <= block_start {
                    self.process_event(&params, sample_rate, context.next_event().unwrap())
                } else if timing < block_start + block_len {
                    // If the event would occur in the middle of the block, then do not process the
                    // event and cut this block short such that the event occurs on the first
                    // sample of the next block.
                    block_len = timing - block_start;
                } else {
                    break;
                }
            }

            let block_end = block_start + block_len;

            // Fill each block with zeros
            left_out[block_start..block_end].fill(0.0);
            right_out[block_start..block_end].fill(0.0);

            let vibrato_params = &params.vibrato_lfo;

            for i in 0..block_len {
                // Get the vibrato modifier, which is global across all of the voices. (Note that each
                // generator gets it's own vibrato envelope).
                let vibrato_mod = self.vibrato_lfo.next_sample(
                    sample_rate,
                    params.vibrato_note_shape,
                    vibrato_params.speed,
                    1.0,
                ) * vibrato_params.amount;

                let pitch_bend = self.pitch_bend_smoother.next();

                for voice in &mut self.notes {
                    let (left, right) = voice.next_sample(
                        &params,
                        &mut self.noise_generator,
                        sample_rate,
                        pitch_bend,
                        vibrato_mod,
                    );

                    left_out[block_start + i] += left;
                    right_out[block_start + i] += right;
                }
            }

            block_start = block_end;
        }

        let chorus_params = &params.chorus;
        // Chorus  and other post processing effects
        for i in 0..num_samples {
            let left = left_out[i];
            let right = right_out[i];

            // Get the chorus effect
            let chorus = self.chorus.next_sample(
                left,
                sample_rate,
                &chorus_params,
                params.chorus_note_shape,
            );

            let left = lerp(left, chorus, chorus_params.mix);
            let right = lerp(right, chorus, chorus_params.mix);

            left_out[i] = left * params.master_vol.get_amp();
            right_out[i] = right * params.master_vol.get_amp();
        }
        ProcessStatus::Normal
    }

    fn filter_state(_state: &mut PluginState) {}

    fn reset(&mut self) {}

    fn deactivate(&mut self) {
        // Turn all notes off (this is done so that notes do not "dangle", since
        // its possible that noteoff won't ever be recieved).
        for note in &mut self.notes {
            note.note_off();
        }
    }

    fn params(&self) -> Arc<dyn Params> {
        Arc::clone(&self.params) as Arc<dyn Params>
    }

    fn task_executor(&self) -> TaskExecutor<Self> {
        Box::new(|_| ())
    }

    fn editor(&self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        None
    }
}
impl Default for Nyasynth {
    fn default() -> Self {
        let sample_rate = SampleRate::from(44100.0);
        Nyasynth {
            params: Arc::new(Parameters::new()),
            notes: Vec::with_capacity(16),
            key_tracker: KeyTracker::new(),
            vibrato_lfo: Oscillator::new(),
            chorus: Chorus::new(sample_rate),
            noise_generator: NoiseGenerator::new(),
            sample_rate: SampleRate(44100.0),
            pitch_bend_smoother: Smoother::new(SmoothingStyle::Linear(0.1)),
        }
    }
}

impl Vst3Plugin for Nyasynth {
    const VST3_CLASS_ID: [u8; 16] = *b"nyasynth.a2aaron";

    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[Vst3SubCategory::Synth];
}

impl Nyasynth {
    fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
        self.chorus.set_sample_rate(sample_rate);
    }

    fn process_event(
        &mut self,
        params: &MeowParameters,
        sample_rate: SampleRate,
        event: NoteEvent<()>,
    ) {
        match event {
            NoteEvent::NoteOn { note, velocity, .. } => {
                let vel = Vel::new(velocity);
                let note = Note(note);
                let polycat = params.polycat;
                let bend_note = self.key_tracker.note_on(note, vel, polycat);
                if polycat {
                    // In polycat mode, we simply add the new note.
                    let start_pitch = bend_note.map(Pitch::from_note);
                    let gen = SoundGenerator::new(&params, start_pitch, note, vel, sample_rate);
                    self.notes.push(gen);
                } else {
                    // Monocat mode.

                    // If there are no generators playing, start a new note
                    if self.notes.len() == 0 {
                        let gen = SoundGenerator::new(&params, None, note, vel, sample_rate);
                        self.notes.push(gen);
                    } else {
                        // If there is a generator playing, retrigger it. If the generator is release state
                        // then also do portamento.
                        let bend_from_current = !self.notes.last().unwrap().is_released();
                        let new_gen = self.notes.last_mut().unwrap().retrigger(
                            params,
                            sample_rate,
                            params.portamento_time,
                            bend_from_current,
                            note,
                            vel,
                        );
                        self.notes.push(new_gen);
                    }
                };
            }
            NoteEvent::NoteOff { note, .. } => {
                let polycat = params.polycat;
                let note = Note(note);
                let top_of_stack = self.key_tracker.note_off(note);

                if polycat {
                    // On note off, send note off to all sound generators matching the note
                    // This is done only to notes which are not yet released
                    for gen in self
                        .notes
                        .iter_mut()
                        .filter(|gen| !gen.is_released() && gen.note == note)
                    {
                        gen.note_off();
                    }
                } else {
                    // Monocat mode.

                    if self.key_tracker.held_keys.len() == 0 {
                        // If there aren't any notes currently being held anymore, just send note off
                        self.notes.iter_mut().for_each(|x| x.note_off());
                    } else {
                        // If there is a sound playing and the key tracker has a new top-of-stack note,
                        // then ask the generator retrigger.
                        match (self.notes.last_mut(), top_of_stack) {
                            (None, None) => (),
                            (None, Some(_)) => (),
                            (Some(_), None) => (),
                            (Some(gen), Some((new_note, new_vel))) => {
                                let new_gen = gen.retrigger(
                                    params,
                                    sample_rate,
                                    params.portamento_time,
                                    true,
                                    new_note,
                                    new_vel,
                                );
                                self.notes.push(new_gen)
                            }
                        }
                    }
                }
            }
            NoteEvent::MidiPitchBend { value, .. } => {
                let pitch_bend = Pitchbend::from_zero_one_range(value);
                self.pitch_bend_smoother
                    .set_target(sample_rate.get(), pitch_bend);
            }
            _ => (),
        }
    }
}

impl Nyasynth {
    pub fn debug_params(&mut self) -> &mut Arc<Parameters> {
        &mut self.params
    }
}

// Export symbols for main
nih_export_vst3!(Nyasynth);
