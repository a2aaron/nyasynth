hello:
    clang -nostdlib hello.s -l system -o target/hello -g
    ./target/hello


perf name:
    trash {{name}}.trace
    trash {{name}}.wav
    cargo build --release
    cargo xtask bundle nyasynth --release
    xctrace record --template 'Time Profiler' --output {{name}}.trace --launch -- target/release/perf --in megalovania.mid --out {{name}}.wav --polycat

vplayer:
    trash nyasynth_stdout.log
    trash nyasynth_stderr.log
    pkill "vPlayer 3"
    cargo xtask bundle nyasynth --release
    open /Applications/vPlayer\ 3.app/ --stdout ./nyasynth_stdout.log --stderr ./nyasynth_stderr.log

run:
    cargo run --release --bin standalone -- --midi-input "USB Axiom 49 Port 1" --backend auto --sample-rate 44100

release-small:
    cargo build --release
    ls -lh target/release/libnyasynth.dylib
    strip target/release/libnyasynth.dylib -Sx
    ls -lh target/release/libnyasynth.dylib