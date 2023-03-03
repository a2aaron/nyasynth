# pkill "vPlayer 3"
# trash Nyasynth.vst
# cargo build --release
# bash ./osx_bundler.sh Nyasynth target/release/libnyasynth.dylib
# open Nyasynth.vst
# ps aux | rg "vPlayer 3"


trash Nyasynth86.vst
cargo build --release --target x86_64-apple-darwin
bash ./osx_bundler.sh Nyasynth86 target/x86_64-apple-darwin/release/libnyasynth.dylib