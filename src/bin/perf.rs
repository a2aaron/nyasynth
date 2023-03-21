use std::{collections::BTreeMap, error::Error, path::PathBuf};

use clap::Parser;
use derive_more::{Add, AddAssign, From, Into, Sub, SubAssign};
use nih_plug::context::process::Transport;
use nih_plug::context::PluginApi;
use nih_plug::prelude::*;
use nyasynth::Nyasynth;
use nyasynth::{
    self,
    common::{SampleRate, SampleTime},
};

type VstEvent = NoteEvent<<Nyasynth as Plugin>::SysExMessage>;

struct MidiBlocks {
    event_blocks: BTreeMap<usize, Vec<VstEvent>>,
}

impl MidiBlocks {
    fn new(
        smf: midly::Smf<'_>,
        sample_rate: SampleRate,
        block_size: usize,
        tempo_info: TempoInfo,
    ) -> MidiBlocks {
        let events = to_vst_events(&smf, sample_rate, tempo_info);
        let event_blocks = split_blocks(events, block_size);
        MidiBlocks { event_blocks }
    }

    fn get(&self, i: usize) -> Vec<VstEvent> {
        self.event_blocks.get(&i).unwrap_or(&vec![]).clone()
    }

    fn max_block(&self) -> usize {
        *self.event_blocks.keys().max().unwrap_or(&0)
    }
}

#[derive(Debug, Clone, Copy)]
struct TempoInfo {
    header_timing: midly::Timing,
    microseconds_per_beat: u32,
}

impl TempoInfo {
    fn new(smf: &midly::Smf) -> TempoInfo {
        let mut first_tempo_event = None;
        for track in &smf.tracks {
            for event in track {
                if let midly::TrackEventKind::Meta(meta) = event.kind {
                    if let midly::MetaMessage::Tempo(microseconds_per_beat) = meta {
                        first_tempo_event = Some(microseconds_per_beat.as_int())
                    }
                }
            }
        }
        TempoInfo {
            header_timing: smf.header.timing,
            microseconds_per_beat: first_tempo_event.unwrap_or(500000), // 120.0 bpm
        }
    }

    fn beats_per_minute(&self) -> f64 {
        let seconds_per_beat = self.microseconds_per_beat as f64 / 1_000_000.0;
        let minutes_per_beat = seconds_per_beat / 60.0;
        let beats_per_minute = 1.0 / minutes_per_beat;
        beats_per_minute
    }

    fn ticks_per_second(&self) -> f64 {
        match self.header_timing {
            midly::Timing::Metrical(ticks_per_beat) => {
                let seconds_per_beat = self.microseconds_per_beat as f64 / 1_000_000.0;
                let beats_per_second = 1.0 / seconds_per_beat;
                ticks_per_beat.as_int() as f64 * beats_per_second
            }
            midly::Timing::Timecode(frames_per_second, ticks_per_frame) => {
                ticks_per_frame as f64 * frames_per_second.as_f32() as f64
            }
        }
    }

    fn ticks_to_samples(&self, ticks: MIDITick, sample_rate: SampleRate) -> usize {
        let seconds = ticks.as_f64() / self.ticks_per_second();
        let samples = sample_rate.0 as f64 * seconds;
        samples as usize
    }
}

// Split events into blocks such that no block exactly `block_size` samples large. Events must be sorted by the sample time.
// This also sets the `delta_frames` for the events to the appropriate value.
fn split_blocks(
    events: Vec<(VstEvent, SampleTime)>,
    block_size: usize,
) -> BTreeMap<usize, Vec<VstEvent>> {
    let mut blocks = BTreeMap::new();

    for (mut event, samples) in events {
        let block_number = samples / block_size;
        let interblock_sample_number = samples % block_size;
        let block = blocks.entry(block_number).or_insert(vec![]);

        let timing = match &mut event {
            NoteEvent::NoteOn { timing, .. } => timing,
            NoteEvent::NoteOff { timing, .. } => timing,
            NoteEvent::Choke { timing, .. } => timing,
            NoteEvent::VoiceTerminated { timing, .. } => timing,
            NoteEvent::PolyModulation { timing, .. } => timing,
            NoteEvent::MonoAutomation { timing, .. } => timing,
            NoteEvent::PolyPressure { timing, .. } => timing,
            NoteEvent::PolyVolume { timing, .. } => timing,
            NoteEvent::PolyPan { timing, .. } => timing,
            NoteEvent::PolyTuning { timing, .. } => timing,
            NoteEvent::PolyVibrato { timing, .. } => timing,
            NoteEvent::PolyExpression { timing, .. } => timing,
            NoteEvent::PolyBrightness { timing, .. } => timing,
            NoteEvent::MidiChannelPressure { timing, .. } => timing,
            NoteEvent::MidiPitchBend { timing, .. } => timing,
            NoteEvent::MidiCC { timing, .. } => timing,
            NoteEvent::MidiProgramChange { timing, .. } => timing,
            NoteEvent::MidiSysEx { timing, .. } => timing,
            _ => unreachable!(),
        };
        *timing = interblock_sample_number as u32;
        block.push(event)
    }
    blocks
}

/// Convert the tracks in a [midly::Smf] object into [VstEvent] events. Additionally, the
/// time for when this event occurs, given in [SampleTime] is also provided. Note that the `delta_frames`
/// value on the [VstEvent]s is always zero, and [VstMidiEvent] values are not set other than `data`.
fn to_vst_events(
    smf: &midly::Smf,
    sample_rate: SampleRate,
    tempo_info: TempoInfo,
) -> Vec<(VstEvent, SampleTime)> {
    let mut vst_events = vec![];
    for track in &smf.tracks {
        let mut delta_ticks = MIDITick(0);
        for track_event in track {
            delta_ticks += track_event.delta.into();
            let vst_event = match track_event.kind {
                midly::TrackEventKind::Midi { channel, message } => {
                    Some(to_vst_event(channel, message))
                }
                midly::TrackEventKind::SysEx(_) => None,
                midly::TrackEventKind::Escape(_) => None,
                midly::TrackEventKind::Meta(_) => None,
            };
            if let Some(vst_event) = vst_event {
                let samples = tempo_info.ticks_to_samples(delta_ticks, sample_rate);
                vst_events.push((vst_event, samples))
            }
        }
    }
    // Sort by sample times.
    vst_events.sort_by(|(_, a), (_, b)| a.cmp(b));
    vst_events
}

fn to_vst_event(channel: midly::num::u4, midi_event: midly::MidiMessage) -> VstEvent {
    fn normalize_u7(u7: midly::num::u7) -> f32 {
        u7.as_int() as f32 / 127.0
    }

    let channel = channel.as_int();
    match midi_event {
        midly::MidiMessage::NoteOff { key, vel } => NoteEvent::NoteOff {
            timing: 0,
            voice_id: None,
            channel,
            note: key.as_int(),
            velocity: normalize_u7(vel),
        },
        midly::MidiMessage::NoteOn { key, vel } => NoteEvent::NoteOn {
            timing: 0,
            voice_id: None,
            channel,
            note: key.as_int(),
            velocity: normalize_u7(vel),
        },
        midly::MidiMessage::Aftertouch { key, vel } => NoteEvent::PolyPressure {
            timing: 0,
            voice_id: None,
            channel,
            note: key.as_int(),
            pressure: normalize_u7(vel),
        },
        midly::MidiMessage::Controller { controller, value } => NoteEvent::MidiCC {
            timing: 0,
            channel,
            cc: controller.as_int(),
            value: normalize_u7(value),
        },
        midly::MidiMessage::ProgramChange { program } => NoteEvent::MidiProgramChange {
            timing: 0,
            channel,
            program: program.as_int(),
        },
        midly::MidiMessage::ChannelAftertouch { vel } => NoteEvent::MidiChannelPressure {
            timing: 0,
            channel,
            pressure: normalize_u7(vel),
        },
        midly::MidiMessage::PitchBend { bend } => NoteEvent::MidiPitchBend {
            timing: 0,
            channel,
            // Note: midly pitchbend values are in the range [-1.0, 1.0]
            // but nih-plug's pitchbend values are in the range [0.0, 1.0]
            value: (bend.as_f32() + 1.0) / 2.0,
        },
    }
}

#[derive(Debug, Clone, Copy, Add, AddAssign, Sub, SubAssign, From, Into)]
/// A MIDI tick. The number of beats or seconds a tick is equal to depends on the particular MIDI file being played.
struct MIDITick(u32);
impl MIDITick {
    fn as_f64(&self) -> f64 {
        self.0 as f64
    }
}
impl From<midly::num::u28> for MIDITick {
    fn from(value: midly::num::u28) -> Self {
        MIDITick(value.as_int())
    }
}

impl From<MIDITick> for f64 {
    fn from(value: MIDITick) -> Self {
        value.as_f64()
    }
}

struct DebugContext;

impl InitContext<Nyasynth> for DebugContext {
    fn plugin_api(&self) -> PluginApi {
        PluginApi::Standalone
    }

    fn execute(&self, _task: ()) {}

    fn set_latency_samples(&self, _samples: u32) {}

    fn set_current_voice_capacity(&self, _capacity: u32) {}
}

impl GuiContext for DebugContext {
    fn plugin_api(&self) -> PluginApi {
        PluginApi::Standalone
    }

    fn request_resize(&self) -> bool {
        false
    }

    unsafe fn raw_begin_set_parameter(&self, _param: ParamPtr) {}

    unsafe fn raw_set_parameter_normalized(&self, param: ParamPtr, normalized: f32) {
        param.set_normalized_value(normalized);
    }

    unsafe fn raw_end_set_parameter(&self, _param: ParamPtr) {}

    fn get_state(&self) -> PluginState {
        todo!()
    }

    fn set_state(&self, _state: PluginState) {
        todo!()
    }
}

struct DebugProcessContext {
    events: Vec<VstEvent>,
    event_index: usize,
    transport: Transport,
}

impl DebugProcessContext {
    fn new(
        events: Vec<VstEvent>,
        tempo_info: &TempoInfo,
        sample_rate: SampleRate,
    ) -> DebugProcessContext {
        let tempo = tempo_info.beats_per_minute();
        let sample_rate = sample_rate.get();
        let mut transport: Transport = Transport::new(sample_rate);
        transport.tempo = Some(tempo);
        DebugProcessContext {
            events,
            event_index: 0,
            transport,
        }
    }
}

impl ProcessContext<Nyasynth> for DebugProcessContext {
    fn plugin_api(&self) -> PluginApi {
        PluginApi::Standalone
    }

    fn execute_background(&self, _task: ()) {}

    fn execute_gui(&self, _task: ()) {}

    fn transport(&self) -> &Transport {
        &self.transport
    }

    fn next_event(&mut self) -> Option<PluginNoteEvent<Nyasynth>> {
        let event = self.events.get(self.event_index);
        self.event_index += 1;
        event.copied()
    }

    fn peek_event(&self) -> Option<&PluginNoteEvent<Nyasynth>> {
        self.events.get(self.event_index)
    }

    fn send_event(&mut self, _event: PluginNoteEvent<Nyasynth>) {}

    fn set_latency_samples(&self, _samples: u32) {}

    fn set_current_voice_capacity(&self, _capacity: u32) {}
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long = "in")]
    in_file: PathBuf,
    #[arg(short, long = "out")]
    out_file: PathBuf,
    #[arg(short, long)]
    polycat: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let block_size = 1024;
    let sample_rate = SampleRate(44100.0);

    let raw = std::fs::read(args.in_file)?;
    let smf = midly::Smf::parse(&raw)?;

    let tempo_info = TempoInfo::new(&smf);
    let blocks = MidiBlocks::new(smf, sample_rate, block_size, tempo_info);

    let mut nyasynth = Nyasynth::default();
    let mut context = DebugContext;

    let audio_io_layout = Nyasynth::AUDIO_IO_LAYOUTS[0];
    let buffer_config = BufferConfig {
        sample_rate: sample_rate.get(),
        min_buffer_size: None,
        max_buffer_size: block_size as u32,
        process_mode: ProcessMode::Offline,
    };
    nyasynth.initialize(&audio_io_layout, &buffer_config, &mut context);
    {
        let params = nyasynth.debug_params();
        let param_setter = ParamSetter::new(&context);
        {
            param_setter.set_parameter(&params.polycat, args.polycat);
            // set to 0.5s
            param_setter.set_parameter(&params.meow_decay, 0.5);
            // set to 40ms
            param_setter.set_parameter(&params.meow_release, 40.0 / 1000.0);
            // set vibrato amount
            param_setter.set_parameter(&params.vibrato_amount, 0.5);
            // set chorus on
            param_setter.set_parameter(&params.chorus_mix, 0.5);
            // set noise on
            param_setter.set_parameter(&params.noise_mix, 0.5);
        }
    }

    // // Set noise on.
    // // params.set_parameter(8, 1.0);

    // // Set vibrato amount
    // // params.set_parameter(4, 1.0);

    // // Set vibrato rate
    // // params.set_parameter(6, 0.5);

    // // Set chorus amount
    // // params.set_parameter(9, 0.5);

    nyasynth.reset();

    let mut outputs: Vec<f32> = Vec::with_capacity(8_000_000);

    fn new_buffer<'a>(backing_buffer: &'a mut [Vec<f32>]) -> Buffer<'a> {
        let num_samples = backing_buffer[0].len();
        let mut buffer = Buffer::default();
        unsafe {
            buffer.set_slices(num_samples, move |output_slices| {
                let (first_channel, other_channels) = backing_buffer.split_at_mut(1);
                *output_slices = vec![&mut first_channel[0], &mut other_channels[0]];
            });
        }
        buffer
    }

    let mut backing_buffer = vec![vec![0.0; block_size]; 2];
    for i in 0..(blocks.max_block() + 100) {
        let block = blocks.get(i);
        let mut context = DebugProcessContext::new(block, &tempo_info, sample_rate);
        let mut buffer = new_buffer(&mut backing_buffer);
        let mut aux = AuxiliaryBuffers {
            inputs: &mut [],
            outputs: &mut [],
        };
        // nyasynth.process_events(events_buffer.events());
        nyasynth.process(&mut buffer, &mut aux, &mut context);

        let output_left = &buffer.as_slice()[0];
        outputs.extend_from_slice(output_left);
    }

    let mut out_file = std::fs::File::create(args.out_file)?;
    let header = wav::Header::new(wav::WAV_FORMAT_IEEE_FLOAT, 1, 44100, 32);
    wav::write(
        header,
        &wav::BitDepth::ThirtyTwoFloat(outputs),
        &mut out_file,
    )?;
    Ok(())
}
