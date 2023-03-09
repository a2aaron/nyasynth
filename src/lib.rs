#![feature(trait_alias)]
#![feature(anonymous_lifetime_in_impl_trait)]

mod chorus;
pub mod common;
pub mod ease;
mod keys;
mod neighbor_pairs;
mod params;
mod sound_gen;

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use chorus::Chorus;
use common::{Note, SampleRate, Vel};
use ease::lerp;
use keys::KeyTracker;
use nih_plug::{nih_export_vst3, prelude::*};
use once_cell::sync::Lazy;
use params::{MeowParameters, Parameters};

use sound_gen::{
    to_pitch_envelope, NoiseGenerator, NormalizedPitchbend, Oscillator, SoundGenerator,
    RETRIGGER_TIME,
};

static PROJECT_DIRS: Lazy<Option<directories::ProjectDirs>> =
    Lazy::new(|| directories::ProjectDirs::from("", "", "Nyasynth VST"));

pub static LOG_DIR: Lazy<PathBuf> = Lazy::new(|| {
    let mut log_dir = match PROJECT_DIRS.as_ref() {
        Some(project_dirs) => project_dirs.cache_dir().to_path_buf(),
        None => FALLBACK_LOG_DIR.to_path_buf(),
    };
    if !log_dir.exists() {
        if let Err(err) = std::fs::create_dir_all(&log_dir) {
            log::info!(
                "Couldn't create log dir {}: Reason: {:?}",
                log_dir.display(),
                err
            );
        }
    }
    log_dir.push("nyasynth.log");
    log_dir
});

pub static DATA_DIR: Lazy<PathBuf> = Lazy::new(|| {
    let data_dir = match PROJECT_DIRS.as_ref() {
        Some(project_dirs) => project_dirs.data_dir().to_path_buf(),
        None => FALLBACK_DATA_DIR.to_path_buf(),
    };
    if !data_dir.exists() {
        if let Err(err) = std::fs::create_dir_all(&data_dir) {
            log::info!(
                "Couldn't create data dir {}: Reason: {:?}",
                data_dir.display(),
                err
            );
        }
    }
    data_dir
});

static FALLBACK_DATA_DIR: Lazy<&'static Path> = Lazy::new(|| Path::new("./nyasynth_VST/data/"));

static FALLBACK_LOG_DIR: Lazy<&'static Path> = Lazy::new(|| Path::new("./nyasynth_VST/log"));

/// The main plugin struct.
pub struct Nyasynth {
    /// All the notes to be played.
    notes: Vec<SoundGenerator>,
    /// The parameters which are shared with the VST host
    params: Arc<Parameters>,
    /// Pitchbend messages. Format is (value, frame_delta) where
    /// value is a normalized f32 and frame_delta is the offset into the current
    /// frame for which the pitchbend value occurs
    pitch_bend: Vec<(NormalizedPitchbend, i32)>,
    /// The last pitch bend value from the previous frame.
    last_pitch_bend: NormalizedPitchbend,
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

    const MIDI_INPUT: MidiConfig = MidiConfig::Basic;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = false;
    const HARD_REALTIME_ONLY: bool = false;

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let result = simple_logging::log_to_file(&*LOG_DIR, log::LevelFilter::Info);
        // let result = simple_logging::log_to_file(
        //     "D:\\dev\\Rust\\nyasynth\\nyasynth.log",
        //     log::LevelFilter::Info,
        // );

        std::env::set_var("NIH_LOG", "/Users/aaron/dev/Rust/nyasynth/nyasynth.log");

        if let Err(err) = result {
            println!("Couldn't start logging! {}", err);
        } else {
            if PROJECT_DIRS.is_none() {
                log::info!("Couldn't obtain project dirs folder!");
            }
            log::info!("Logging to {}", LOG_DIR.display());
        }

        std::panic::set_hook(Box::new(|panic_info| {
            log::info!("PANICKED!! Reason: {:#?}", panic_info);
            let bt = backtrace::Backtrace::new();
            log::info!("Backtrace: {:#?}", bt);
        }));

        log::info!("Begin VST log...");

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
        self.process_events(context);
        let sample_rate = SampleRate(context.transport().sample_rate);
        if sample_rate != self.sample_rate {
            self.set_sample_rate(sample_rate);
        }

        let tempo = context.transport().tempo.unwrap_or(120.0) as f32;
        self.process(buffer, sample_rate, tempo);
        ProcessStatus::Normal
    }

    fn filter_state(_state: &mut PluginState) {}

    fn reset(&mut self) {}

    fn deactivate(&mut self) {
        // Turn all notes off (this is done so that notes do not "dangle", since
        // its possible that noteoff won't ever be recieved).
        for note in &mut self.notes {
            note.note_off(0);
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
            pitch_bend: Vec::with_capacity(16),
            key_tracker: KeyTracker::new(),
            last_pitch_bend: 0.0,
            vibrato_lfo: Oscillator::new(),
            chorus: Chorus::new(sample_rate),
            noise_generator: NoiseGenerator::new(),
            sample_rate: SampleRate(44100.0),
        }
    }
}

impl Vst3Plugin for Nyasynth {
    const VST3_CLASS_ID: [u8; 16] = *b"nyasynth.a2aaron";

    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[Vst3SubCategory::Synth];
}

impl Nyasynth {
    // Output audio given the current state of the VST
    fn process(&mut self, buffer: &mut Buffer, sample_rate: SampleRate, tempo: f32) {
        let params = MeowParameters::new(&self.params, tempo);

        let num_samples = buffer.samples();

        // Get the envelope from MIDI pitch bend
        let (pitch_bends, last_bend) =
            to_pitch_envelope(&self.pitch_bend, self.last_pitch_bend, num_samples);
        self.last_pitch_bend = last_bend;

        let pitch_bends: Vec<f32> = pitch_bends.collect();

        // Get sound for each note
        let (left_out, right_out) = &mut buffer.as_slice().split_at_mut(1);
        let left_out = &mut left_out[0];
        let right_out = &mut right_out[0];
        left_out.fill(0.0);
        right_out.fill(0.0);

        let vibrato_params = &params.vibrato_lfo;
        let chorus_params = &params.chorus;

        for gen in &mut self.notes {
            for i in 0..num_samples {
                // Get the vibrato modifier, which is global across all of the generators. (Note that each
                // generator gets it's own vibrato envelope).
                let vibrato_mod = self.vibrato_lfo.next_sample(
                    sample_rate,
                    params.vibrato_note_shape,
                    vibrato_params.speed,
                    1.0,
                ) * vibrato_params.amount;

                let (left, right) = gen.next_sample(
                    &params,
                    &mut self.noise_generator,
                    i,
                    sample_rate,
                    pitch_bends[i],
                    vibrato_mod,
                );

                left_out[i] += left;
                right_out[i] += right;
            }
        }

        // Write sound
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
    }

    fn process_events(&mut self, events: &mut impl ProcessContext<Self>) {
        let sample_rate = SampleRate(events.transport().sample_rate);
        let tempo = events.transport().tempo.unwrap_or(120.0) as f32;
        let params = MeowParameters::new(&self.params, tempo);
        // remove "dead" notes
        // we do this in process_events _before_ processing any midi messages
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

        // Clear pitch bend to get new messages
        self.pitch_bend.clear();
        while let Some(event) = events.next_event() {
            let frame_delta = event.timing() as i32;
            match event {
                NoteEvent::NoteOn { note, velocity, .. } => {
                    let vel = Vel::new(velocity);
                    let note = Note(note);
                    let polycat = params.polycat;
                    let bend_note = self.key_tracker.note_on(note, vel, polycat);
                    if polycat {
                        // In polycat mode, we simply add the new note.
                        let mut gen = SoundGenerator::new(&params, note, vel, sample_rate);
                        gen.note_on(frame_delta, vel, bend_note);
                        self.notes.push(gen);
                    } else {
                        // Monocat mode.
                        if self.notes.len() > 1 {
                            log::warn!(
                                "More than one note playing in monocat mode? (noteon) {:?}",
                                self.notes
                            );
                        }

                        // If there are no generators playing, start a new note
                        if self.notes.len() == 0 {
                            let mut gen = SoundGenerator::new(&params, note, vel, sample_rate);
                            gen.note_on(frame_delta, vel, None);
                            self.notes.push(gen);
                        } else {
                            // If there is a generator playing, retrigger it. If the generator is release state
                            // then also do portamento.
                            let bend_from_current = !self.notes[0].is_released();
                            self.notes[0].retrigger(
                                sample_rate,
                                params.portamento_time,
                                bend_from_current,
                                note,
                                vel,
                                frame_delta,
                            );
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
                            gen.note_off(frame_delta);
                        }
                    } else {
                        // Monocat mode.
                        if self.notes.len() > 1 {
                            log::warn!(
                                "More than one note playing in monocat mode? (noteoff) {:?}",
                                self.notes
                            );
                        }

                        if self.key_tracker.held_keys.len() == 0 {
                            // If there aren't any notes currently being held anymore, just send note off
                            self.notes.iter_mut().for_each(|x| x.note_off(frame_delta));
                        } else {
                            // If there is a sound playing and the key tracker has a new top-of-stack note,
                            // then ask the generator retrigger.
                            match (self.notes.first_mut(), top_of_stack) {
                                (None, None) => (),
                                (None, Some(_)) => (),
                                (Some(_), None) => (),
                                (Some(gen), Some((new_note, new_vel))) => gen.retrigger(
                                    sample_rate,
                                    params.portamento_time,
                                    true,
                                    new_note,
                                    new_vel,
                                    frame_delta,
                                ),
                            }
                        }
                    }
                }
                NoteEvent::MidiPitchBend { value, .. } => {
                    self.pitch_bend.push((value, frame_delta));
                }
                _ => todo!(),
            }
        }

        // Sort pitch bend changes by delta_frame.
        self.pitch_bend.sort_unstable_by(|a, b| a.1.cmp(&b.1));
    }

    fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
        self.chorus.set_sample_rate(sample_rate);
    }
}

impl Nyasynth {
    pub fn debug_params(&mut self) -> &mut Arc<Parameters> {
        &mut self.params
    }
}

// Export symbols for main
nih_export_vst3!(Nyasynth);
