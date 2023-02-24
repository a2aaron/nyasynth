#![feature(trait_alias)]

#[macro_use]
extern crate vst;

mod ease;
mod macros;
mod neighbor_pairs;
mod params;
mod presets;
mod sound_gen;
mod ui;
mod ui_tabs;

use std::{
    convert::TryFrom,
    path::{Path, PathBuf},
    sync::Arc,
};

use backtrace::Backtrace;
use iced_baseview::Application;
use once_cell::sync::Lazy;
use vst::{
    api::{Events, Supported},
    buffer::AudioBuffer,
    editor::Editor,
    plugin::{CanDo, Category, HostCallback, Info, Plugin, PluginParameters},
};
use wmidi::MidiMessage;

use params::{
    ModBankEnvs, ModulationBank, ModulationSend, ModulationType, ParameterType, Parameters,
    RawParameters,
};
use sound_gen::{
    normalize_U7, normalize_pitch_bend, to_pitch_envelope, NormalizedPitchbend, SampleRate,
    SoundGenerator,
};
use ui::UIFrontEnd;

static PROJECT_DIRS: Lazy<Option<directories::ProjectDirs>> =
    Lazy::new(|| directories::ProjectDirs::from("", "", "Revisit VST"));

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
    log_dir.push("revisit.log");
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

static FALLBACK_DATA_DIR: Lazy<&'static Path> = Lazy::new(|| Path::new("./revisit_VST/data/"));

static FALLBACK_LOG_DIR: Lazy<&'static Path> = Lazy::new(|| Path::new("./revisit_VST/log"));

/// The main plugin struct.
struct Revisit {
    /// All the notes to be played.
    notes: Vec<SoundGenerator>,
    /// The sample rate in Hz/sec (usually 44,100)
    sample_rate: SampleRate,
    /// The parameters which are shared with the VST host
    params: Arc<RawParameters>,
    /// Pitchbend messages. Format is (value, frame_delta) where
    /// value is a normalized f32 and frame_delta is the offset into the current
    /// frame for which the pitchbend value occurs
    pitch_bend: Vec<(NormalizedPitchbend, i32)>,
    /// The last pitch bend value from the previous frame.
    last_pitch_bend: NormalizedPitchbend,
    /// If true, then the GUI has been initalized and `get_editor()` will return
    /// None.
    gui_initialized: bool,
}

impl Plugin for Revisit {
    fn new(host: HostCallback) -> Self {
        Revisit {
            params: Arc::new(RawParameters::new(host)),
            notes: Vec::with_capacity(16),
            sample_rate: 44100.0,
            pitch_bend: Vec::with_capacity(16),
            last_pitch_bend: 0.0,
            gui_initialized: false,
        }
    }

    fn init(&mut self) {
        let result = simple_logging::log_to_file(&*LOG_DIR, log::LevelFilter::Info);
        // let result = simple_logging::log_to_file(
        //     "D:\\dev\\Rust\\revist\\revisit.log",
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
            name: "Revisit".to_string(),
            vendor: "a2aaron".to_string(),
            // Used by hosts to differentiate between plugins.
            // Don't worry much about this now - just fill in a random number.
            unique_id: 413612,
            version: 1,
            category: Category::Synth,
            parameters: ParameterType::COUNT as i32,
            // No audio inputs
            inputs: 0,
            // Two channel audio!
            outputs: 2,
            // For now, fill in the rest of our fields with `Default` info.
            ..Default::default()
        }
    }

    fn can_do(&self, can_do: CanDo) -> Supported {
        match can_do {
            CanDo::ReceiveMidiEvent => Supported::Yes,
            _ => Supported::No,
        }
    }

    // Output audio given the current state of the VST
    fn process(&mut self, buffer: &mut AudioBuffer<f32>) {
        let num_samples = buffer.samples();

        let params = Parameters::from(self.params.as_ref());

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
                let (left, right) = gen.next_sample(&params, i, self.sample_rate, pitch_bends[i]);
                left_out[i] += left;
                right_out[i] += right;
            }
        }

        // Write sound
        for i in 0..num_samples {
            output_buffer[0][i] = left_out[i] * params.master_vol.get_amp();
            output_buffer[1][i] = right_out[i] * params.master_vol.get_amp();
        }
    }

    fn process_events(&mut self, events: &Events) {
        let sample_rate = self.sample_rate;
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
        let params = Parameters::from(self.params.as_ref());
        self.notes.retain(|gen| gen.is_alive(sample_rate, &params));

        // Clear pitch bend to get new messages
        self.pitch_bend.clear();
        for event in events.events() {
            match event {
                vst::event::Event::Midi(event) => {
                    let message = MidiMessage::try_from(&event.data as &[u8]);
                    if let Ok(message) = message {
                        match message {
                            MidiMessage::NoteOn(_, note, vel) => {
                                // On note on, either add or retrigger the note
                                let vel = normalize_U7(vel);
                                let gen = if let Some(osc) =
                                    self.notes.iter_mut().find(|gen| gen.note == note)
                                {
                                    osc
                                } else {
                                    self.notes.push(SoundGenerator::new(note, vel, sample_rate));
                                    self.notes.last_mut().unwrap()
                                };

                                gen.note_on(event.delta_frames, vel);
                            }
                            MidiMessage::NoteOff(_, note, _) => {
                                // On note off, send note off
                                if let Some(i) = self.notes.iter().position(|gen| gen.note == note)
                                {
                                    let gen = &mut self.notes[i];
                                    gen.note_off(event.delta_frames);
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
        self.sample_rate = rate;
    }

    // The raw parameters exposed to the host
    fn get_parameter_object(&mut self) -> Arc<dyn PluginParameters> {
        Arc::clone(&self.params) as Arc<dyn PluginParameters>
    }

    // The GUI exposed to the host
    fn get_editor(&mut self) -> Option<Box<dyn Editor>> {
        if self.gui_initialized {
            None
        } else {
            self.gui_initialized = true;
            let (editor, _) = UIFrontEnd::new(self.params.clone());
            Some(Box::new(editor))
        }
    }
}

// Export symbols for main
plugin_main!(Revisit);
