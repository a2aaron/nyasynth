use nyasynth::Nyasynth;

fn main() {
    nih_plug::wrapper::standalone::nih_export_standalone::<Nyasynth>();
}
