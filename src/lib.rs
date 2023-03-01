#![feature(trait_alias)]

#[macro_use]
extern crate vst;

mod common;
mod ease;
mod keys;
mod neighbor_pairs;
mod params;
mod sound_gen;

use std::{
    convert::TryFrom,
    path::{Path, PathBuf},
    sync::Arc,
};

use backtrace::Backtrace;
use common::{SampleRate, Vel};
use keys::KeyTracker;
use once_cell::sync::Lazy;
use params::MeowParameters;
use vst::{
    api::{Events, Supported, TimeInfoFlags},
    buffer::AudioBuffer,
    editor::Editor,
    host::Host,
    plugin::{CanDo, Category, HostCallback, Info, Plugin, PluginParameters},
};
use wmidi::MidiMessage;

use sound_gen::{
    normalize_pitch_bend, to_pitch_envelope, NormalizedPitchbend, SoundGenerator, RETRIGGER_TIME,
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
struct Nyasynth {
    /// All the notes to be played.
    notes: Vec<SoundGenerator>,
    /// The sample rate in Hz/sec (usually 44,100)
    sample_rate: SampleRate,
    /// The parameters which are shared with the VST host
    params: Arc<MeowParameters>,
    /// Pitchbend messages. Format is (value, frame_delta) where
    /// value is a normalized f32 and frame_delta is the offset into the current
    /// frame for which the pitchbend value occurs
    pitch_bend: Vec<(NormalizedPitchbend, i32)>,
    /// The last pitch bend value from the previous frame.
    last_pitch_bend: NormalizedPitchbend,
    /// The tempo, in beats per minute
    tempo: f64,
    key_tracker: KeyTracker,
    /// The host callback
    host: HostCallback,
}

impl Plugin for Nyasynth {
    fn new(host: HostCallback) -> Self {
        Nyasynth {
            params: Arc::new(MeowParameters::new()),
            notes: Vec::with_capacity(16),
            sample_rate: SampleRate::from(44100.0),
            pitch_bend: Vec::with_capacity(16),
            key_tracker: KeyTracker::new(),
            last_pitch_bend: 0.0,
            tempo: 120.0,
            host,
        }
    }

    fn init(&mut self) {
        let result = simple_logging::log_to_file(&*LOG_DIR, log::LevelFilter::Info);
        // let result = simple_logging::log_to_file(
        //     "D:\\dev\\Rust\\nyasynth\\nyasynth.log",
        //     log::LevelFilter::Info,
        // );

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
            let bt = Backtrace::new();
            log::info!("Backtrace: {:#?}", bt);
        }));

        log::info!("Begin VST log...");
    }

    fn get_info(&self) -> Info {
        Info {
            name: "Nyasynth".to_string(),
            vendor: "a2aaron".to_string(),
            // Used by hosts to differentiate between plugins.
            // Don't worry much about this now - just fill in a random number.
            unique_id: i32::from_be_bytes([13, 5, 15, 23]), // "MEOW"
            version: 1,
            category: Category::Synth,
            parameters: MeowParameters::NUM_PARAMS as i32,
            // No audio inputs
            inputs: 0,
            // Two channel audio!
            outputs: 2,
            // On a retrigger, the next note is delayed by RETRIGGER_TIME. Hence, there is a latency
            // of RETRIGGER_TIME. Note that this latency doesn't exist for non-retriggered notes.
            initial_delay: RETRIGGER_TIME as i32,
            // For now, fill in the rest of our fields with `Default` info.
            ..Default::default()
        }
    }

    fn can_do(&self, can_do: CanDo) -> Supported {
        match can_do {
            CanDo::ReceiveMidiEvent => Supported::Yes,
            CanDo::ReceiveTimeInfo => Supported::Yes,
            _ => Supported::No,
        }
    }

    // Output audio given the current state of the VST
    fn process(&mut self, buffer: &mut AudioBuffer<f32>) {
        let num_samples = buffer.samples();

        // Get the envelope from MIDI pitch bend
        let (pitch_bends, last_bend) =
            to_pitch_envelope(&self.pitch_bend, self.last_pitch_bend, num_samples);
        self.last_pitch_bend = last_bend;

        let pitch_bends: Vec<f32> = pitch_bends.collect();

        // Get sound for each note
        let (_, mut output_buffer) = buffer.split();

        let mut left_out = vec![0.0; num_samples];
        let mut right_out = vec![0.0; num_samples];

        for gen in &mut self.notes {
            for i in 0..num_samples {
                let (left, right) = gen.next_sample(
                    &self.params,
                    i,
                    self.sample_rate,
                    pitch_bends[i],
                    self.tempo as f32,
                );
                left_out[i] += left;
                right_out[i] += right;
            }
        }

        // Write sound
        for i in 0..num_samples {
            output_buffer[0][i] = left_out[i];
            output_buffer[1][i] = right_out[i];
        }
    }

    fn process_events(&mut self, events: &Events) {
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
            let sample_rate = self.sample_rate;
            let params = &self.params;
            self.notes.retain(|gen| gen.is_alive(sample_rate, &params));
        }

        // Clear pitch bend to get new messages
        self.pitch_bend.clear();
        for event in events.events() {
            match event {
                vst::event::Event::Midi(event) => {
                    let message = MidiMessage::try_from(&event.data as &[u8]);
                    if let Ok(message) = message {
                        match message {
                            MidiMessage::NoteOn(_, note, vel) => {
                                let vel = Vel::from(vel);
                                let polycat = self.params.polycat();
                                let bend_note = self.key_tracker.note_on(note, vel, polycat);
                                if polycat {
                                    // In polycat mode, we simply add the new note.
                                    let mut gen = SoundGenerator::new(note, vel, self.sample_rate);
                                    gen.note_on(event.delta_frames, vel, bend_note);
                                    self.notes.push(gen);
                                } else {
                                    // Monocat mode.
                                    if self.notes.len() > 1 {
                                        log::warn!("More than one note playing in monocat mode? (noteon) {:?}", self.notes);
                                    }

                                    // If there are no generators playing, start a new note
                                    if self.notes.len() == 0 {
                                        let mut gen =
                                            SoundGenerator::new(note, vel, self.sample_rate);
                                        gen.note_on(event.delta_frames, vel, None);
                                        self.notes.push(gen);
                                    } else {
                                        // If there is a generator playing, retrigger it. If the generator is release state
                                        // then also do portamento.
                                        let bend_from_current = !self.notes[0].is_released();
                                        self.notes[0].retrigger(
                                            self.sample_rate,
                                            self.params.portamento_time(),
                                            bend_from_current,
                                            note,
                                            vel,
                                            event.delta_frames,
                                        );
                                    }
                                };
                            }
                            MidiMessage::NoteOff(_, note, _) => {
                                let polycat = self.params.polycat();
                                let top_of_stack = self.key_tracker.note_off(note);

                                if polycat {
                                    // On note off, send note off to all sound generators matching the note
                                    // This is done only to notes which are not yet released
                                    for gen in self
                                        .notes
                                        .iter_mut()
                                        .filter(|gen| !gen.is_released() && gen.note == note)
                                    {
                                        gen.note_off(event.delta_frames);
                                    }
                                } else {
                                    // Monocat mode.
                                    if self.notes.len() > 1 {
                                        log::warn!("More than one note playing in monocat mode? (noteoff) {:?}", self.notes);
                                    }

                                    if self.key_tracker.held_keys.len() == 0 {
                                        // If there aren't any notes currently being held anymore, just send note off
                                        self.notes
                                            .iter_mut()
                                            .for_each(|x| x.note_off(event.delta_frames));
                                        log::info!("Note off");
                                    } else {
                                        // If there is a sound playing and the key tracker has a new top-of-stack note,
                                        // then ask the generator retrigger.
                                        log::info!(
                                            "maybe retrigger: {:?} {:?}",
                                            self.notes.first(),
                                            top_of_stack
                                        );
                                        match (self.notes.first_mut(), top_of_stack) {
                                            (None, None) => (),
                                            (None, Some(_)) => (),
                                            (Some(_), None) => (),
                                            (Some(gen), Some((new_note, new_vel))) => gen
                                                .retrigger(
                                                    self.sample_rate,
                                                    self.params.portamento_time(),
                                                    true,
                                                    new_note,
                                                    new_vel,
                                                    event.delta_frames,
                                                ),
                                        }
                                    }
                                }
                            }
                            MidiMessage::PitchBendChange(_, pitch_bend) => {
                                self.pitch_bend
                                    .push((normalize_pitch_bend(pitch_bend), event.delta_frames));
                            }
                            _ => (),
                        }
                    }
                }

                vst::event::Event::SysEx(_) => (),
                vst::event::Event::Deprecated(_) => (),
            }
        }

        // Sort pitch bend changes by delta_frame.
        self.pitch_bend.sort_unstable_by(|a, b| a.1.cmp(&b.1));

        // Set tempo, if available.
        if let Some(time_info) = self.host.get_time_info(TimeInfoFlags::TEMPO_VALID.bits()) {
            if TimeInfoFlags::TEMPO_VALID.bits() & time_info.flags != 0 {
                self.tempo = time_info.tempo;
            }
        }
    }

    fn suspend(&mut self) {
        // Turn all notes off (this is done so that notes do not "dangle", since
        // its possible that noteoff won't ever be recieved).
        for note in &mut self.notes {
            note.note_off(0);
        }
        log::info!("Suspending VST...");
    }

    fn stop_process(&mut self) {
        log::info!("Stopping VST...");
    }

    fn set_sample_rate(&mut self, rate: f32) {
        if let Some(rate) = SampleRate::new(rate) {
            self.sample_rate = rate;
        } else {
            log::error!(
                "Cannot set sample rate to {} (expected a positive value)",
                rate
            );
        }
    }

    // The raw parameters exposed to the host
    fn get_parameter_object(&mut self) -> Arc<dyn PluginParameters> {
        Arc::clone(&self.params) as Arc<dyn PluginParameters>
    }

    // The GUI exposed to the host
    fn get_editor(&mut self) -> Option<Box<dyn Editor>> {
        None
    }
}

// Export symbols for main
plugin_main!(Nyasynth);
