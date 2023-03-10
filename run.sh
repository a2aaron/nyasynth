# pkill "vPlayer 3"
# trash Nyasynth.vst
# cargo build --release
# bash ./osx_bundler.sh Nyasynth target/release/libnyasynth.dylib
# open Nyasynth.vst
# ps aux | rg "vPlayer 3"

trash $1.trace
trash $2.wav
cargo build --release
xctrace record --template 'Time Profiler' --output $1.trace --launch -- target/release/perf --in megalovania.mid --out $2.wav --polycat