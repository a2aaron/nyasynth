use std::{
    collections::BTreeMap,
    error::Error,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use clap::Parser;
use derive_more::{Add, AddAssign, From, Into, Sub, SubAssign};
use nyasynth::{
    self,
    common::{SampleRate, SampleTime},
};
use vst::{
    api::{SmpteFrameRate, TimeInfo, TimeInfoFlags},
    host::{Host, HostBuffer, PluginLoader},
    prelude::*,
};

use vst::event::Event as VstEvent;
use vst::event::SysExEvent as VstSysExEvent;

struct MidiBlocks<'a> {
    event_blocks: BTreeMap<usize, Vec<VstEvent<'a>>>,
}

impl<'a> MidiBlocks<'a> {
    fn new(
        smf: midly::Smf<'a>,
        sample_rate: SampleRate,
        block_size: usize,
        tempo_info: TempoInfo,
    ) -> MidiBlocks<'a> {
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

struct SimpleHost {
    block_size: usize,
    tempo: f64,
}

impl SimpleHost {
    fn new(block_size: usize, tempo: f64) -> SimpleHost {
        SimpleHost { block_size, tempo }
    }
}

impl Host for SimpleHost {
    fn get_time_info(&self, _mask: i32) -> Option<TimeInfo> {
        let time_info = TimeInfo {
            sample_pos: 0.0,
            sample_rate: 44100.0,
            nanoseconds: 100.0,
            ppq_pos: 0.0,
            tempo: self.tempo,
            bar_start_pos: 0.0,
            cycle_start_pos: 0.0,
            cycle_end_pos: 0.0,
            time_sig_numerator: 0,
            time_sig_denominator: 0,
            smpte_offset: 0,
            smpte_frame_rate: SmpteFrameRate::Smpte60fps,
            samples_to_next_clock: 0,
            flags: TimeInfoFlags::TEMPO_VALID.bits(),
        };
        Some(time_info)
    }

    fn get_block_size(&self) -> isize {
        self.block_size as isize
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
fn split_blocks<'a>(
    events: Vec<(VstEvent<'a>, SampleTime)>,
    block_size: usize,
) -> BTreeMap<usize, Vec<VstEvent<'a>>> {
    let mut blocks = BTreeMap::new();

    for (event, samples) in events {
        let block_number = samples / block_size;
        let interblock_sample_number = samples % block_size;
        let block = blocks.entry(block_number).or_insert(vec![]);
        match event {
            VstEvent::Midi(mut event) => event.delta_frames = interblock_sample_number as i32,
            VstEvent::SysEx(mut event) => event.delta_frames = interblock_sample_number as i32,
            VstEvent::Deprecated(mut event) => event.delta_frames = interblock_sample_number as i32,
        }
        block.push(event)
    }
    blocks
}

/// Convert the tracks in a [midly::Smf] object into [VstEvent] events. Additionally, the
/// time for when this event occurs, given in [SampleTime] is also provided. Note that the `delta_frames`
/// value on the [VstEvent]s is always zero, and [VstMidiEvent] values are not set other than `data`.
fn to_vst_events<'a>(
    smf: &midly::Smf<'a>,
    sample_rate: SampleRate,
    tempo_info: TempoInfo,
) -> Vec<(VstEvent<'a>, SampleTime)> {
    let mut vst_events = vec![];
    for track in &smf.tracks {
        let mut delta_ticks = MIDITick(0);
        for track_event in track {
            delta_ticks += track_event.delta.into();
            let vst_event = match track_event.kind {
                midly::TrackEventKind::Midi { channel, message } => {
                    let event = to_wmidi_event(channel, message);
                    let mut data = [0; 3];
                    event.copy_to_slice(&mut data).unwrap();
                    let event = MidiEvent {
                        data,
                        delta_frames: 0,
                        live: false,
                        note_length: None,
                        note_offset: None,
                        detune: 0,
                        note_off_velocity: 0,
                    };
                    Some(VstEvent::Midi(event))
                }
                midly::TrackEventKind::SysEx(data) => {
                    let event = VstSysExEvent {
                        payload: data,
                        delta_frames: 0,
                    };
                    Some(VstEvent::SysEx(event))
                }
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

fn to_wmidi_event(
    channel: midly::num::u4,
    message: midly::MidiMessage,
) -> wmidi::MidiMessage<'static> {
    fn to_wmidi_u7(u7: midly::num::u7) -> wmidi::U7 {
        wmidi::U7::try_from(u7.as_int()).unwrap()
    }

    fn to_wmidi_u14(bend: midly::PitchBend) -> wmidi::U14 {
        wmidi::U14::try_from((bend.as_int() + 0x2000) as u16).unwrap()
    }

    fn to_channel(channel: midly::num::u4) -> wmidi::Channel {
        wmidi::Channel::from_index(channel.as_int()).unwrap()
    }

    fn to_note(note: midly::num::u7) -> wmidi::Note {
        let note = wmidi::U7::try_from(note.as_int()).unwrap();
        wmidi::Note::from(note)
    }

    fn to_control_function(controller: midly::num::u7) -> wmidi::ControlFunction {
        wmidi::ControlFunction(to_wmidi_u7(controller))
    }

    let channel = to_channel(channel);
    match message {
        midly::MidiMessage::NoteOff { key, vel } => {
            let key = to_note(key);
            let vel = to_wmidi_u7(vel);
            wmidi::MidiMessage::NoteOff(channel, key, vel)
        }
        midly::MidiMessage::NoteOn { key, vel } => {
            let key = to_note(key);
            let vel = to_wmidi_u7(vel);
            wmidi::MidiMessage::NoteOn(channel, key, vel)
        }
        midly::MidiMessage::PitchBend { bend } => {
            let bend = to_wmidi_u14(bend);
            wmidi::MidiMessage::PitchBendChange(channel, bend)
        }
        midly::MidiMessage::Aftertouch { key, vel } => {
            let key = to_note(key);
            let vel = to_wmidi_u7(vel);
            wmidi::MidiMessage::PolyphonicKeyPressure(channel, key, vel)
        }
        midly::MidiMessage::ChannelAftertouch { vel } => {
            let vel = to_wmidi_u7(vel);
            wmidi::MidiMessage::ChannelPressure(channel, vel)
        }
        midly::MidiMessage::Controller { controller, value } => {
            let controller = to_control_function(controller);
            let value = to_wmidi_u7(value);
            wmidi::MidiMessage::ControlChange(channel, controller, value)
        }
        midly::MidiMessage::ProgramChange { program } => {
            let program = to_wmidi_u7(program);
            wmidi::MidiMessage::ProgramChange(channel, program)
        }
    }
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long = "vst")]
    vst_path: PathBuf,
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

    let host = SimpleHost::new(block_size, tempo_info.beats_per_minute());
    let host = Arc::new(Mutex::new(host));

    let path = Path::new(&args.vst_path);
    let mut loader = PluginLoader::load(path, host)?;
    let mut nyasynth = loader.instance()?;

    nyasynth.init();

    nyasynth.set_sample_rate(sample_rate.0);
    nyasynth.set_block_size(block_size as i64);

    nyasynth.resume();
    nyasynth
        .get_parameter_object()
        .set_parameter(11, if args.polycat { 1.0 } else { 0.0 });

    // Set noise on.
    // nyasynth.get_parameter_object().set_parameter(8, 1.0);

    let mut outputs: Vec<f32> = Vec::with_capacity(8_000_000);

    let mut host_buffer = HostBuffer::new(0, 2);
    let mut output_arrays = vec![vec![0.0; block_size]; 2];
    let mut audio_buffer = host_buffer.bind::<Vec<f32>, Vec<f32>>(&vec![], &mut output_arrays);

    for i in 0..(blocks.max_block() + 100) {
        let block = blocks.get(i);

        let mut events_buffer = SendEventBuffer::new(64);
        events_buffer.store_events(block.iter());

        nyasynth.process_events(events_buffer.events());
        nyasynth.process(&mut audio_buffer);

        let output_left = &audio_buffer.split().1[0];
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
