# Nyasynth - The World's Second Meowizer

Do you remember [Meowsynth](https://www.youtube.com/watch?v=_VYtQ9jP73s)? So do I. Unfortunately, it
seems that Meowsynth is only 32-bit and hence isn't compatible with many 64-bit DAWs (in particular,
it's not compatible with Ableton, which doesn't support 32-bit VSTs any more). Additionally, there
isn't a Mac or Linux version.

# Build instruction
To build the plugin as a vst3 bundle, run the following command:

```
cargo xtask bundle nyasynth --release
```

This will create a `nyasynth.vst3` bundle in `/target/bundled/`. Install this into any vst of your choice.


You can also create a standalone binary by running the following command:

```
cargo build --release --bin standalone
```

This will create a `standalone` binary in `/target/release/`. You can see the arguments it uses with `standalone -h`. See [nih-plug](https://github.com/robbert-vdh/nih-plug) for more information.