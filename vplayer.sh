trash nyasynth_stdout.log
trash nyasynth_stderr.log
pkill "vPlayer 3"
cargo xtask bundle nyasynth --release
open /Applications/vPlayer\ 3.app/ --stdout ./nyasynth_stdout.log --stderr ./nyasynth_stderr.log