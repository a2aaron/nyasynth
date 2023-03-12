trash $1.trace
trash $2.wav
cargo build --release
cargo xtask bundle nyasynth --release
xctrace record --template 'Time Profiler' --output $1.trace --launch -- target/release/perf --in megalovania.mid --out $2.wav --polycat