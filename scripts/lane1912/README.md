# Random-input comparison for #1912

This tool dispatches the EXEC-emulated HLSL and per-lane DXIL on identically seeded inputs. The manifest pins the 111 admissions to #1912 head `5d0c1f0b8a802712e9542f0ffa269d9e36b6747f`. The parent comparison used `438d4203`. The runner checks the SHA-256 digests of the emitted HLSL and per-lane DXIL against the manifest before building a PSO.

## Run the native wave64 comparison

Use Windows, a hardware D3D12 adapter supporting native 64-lane waves and tiled resources tier 2, the Rust MSVC toolchain for this repository, Python 3.9+ and an x64 DXC executable. The local wave32 runs used DXC 1.9.2609. Choose an empty private output folder. The output contains shader binaries and generated HLSL.

From the checkout containing this kit, set `$pack`, `$dxc` and `$out` to your local paths, then run the following commands.

~~~powershell
cargo build --release -p xcl --example lane1912_split --example lane1912_replay
$target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { "target" }
python scripts/lane1912/replay.py --pack "$pack" --dxc "$dxc" --out "$out" --bin-dir "$target/release/examples" --wave 64
~~~

This selects the 67 native-wave64 entries. Omit `--wave 64` for all 111. Use `--hash HASH` for one entry or `--split` to request split wave64 on wave32 hardware. The 13 split refusals are recorded in the manifest. The export uses four switch-selected windows and each program's own thread-group dimensions and sampler declarations. `XCL_ADAPTER` selects the hardware adapter when there are several.

## Inputs and readback

The default seeds are `191201 191203 191211`. Each uses one full workgroup. The diagnostic pass discovers the window pages and bindless texture slots the HLSL accesses on that seed. Every pass starts with freshly initialized resources. New pages receive independent seeded words. Newly discovered slots receive typed textures with seeded texels. A fixture proceeds when its page and descriptor layout stabilizes. Conflicting descriptors, unsupported indices or allocations exceeding the fixture cap produce an unavailable row.

For each executable, the runner reads back all ABI and result words, every resident tile in four independent sparse 4 GiB windows, and every allocated texture. The seeded I/O uses small aligned low words and bounded high words shaped for pointers, with UMA base high word zero. Samplers use fixed linear filtering. Unmapped tiles have D3D12 sparse-resource behavior. The result describes only the tested fixture. Set `--seeds` to use other seed values. The public runner uses one whole workgroup for every seed.

## Results

`results.csv` has one row per requested program and seed. `hlsl_changed` and `lane_changed` count output words differing from the corresponding initial state. `differing_words` compares the two final outputs. For a mismatch the row gives the first resource, zero-based word index, and both hexadecimal values. The mismatch columns also count sign-only differences between zero values, differing words that both decode to NaNs as binary32, and all remaining patterns. A second fresh dispatch of each form determines whether the difference repeats. `probe_diff` compares the diagnostic pass's application-visible output with the unmodified HLSL result. A nonzero value qualifies the fixture even when the unmodified forms agree.

An equal result with zero changed words gives limited coverage. A PSO failure, timeout, missing result or incompatible fixture remains unavailable. Exit code 1 means an output difference occurred. Exit code 2 means a required comparison was unavailable or its diagnostic output differed. Exit code 0 means all requested comparisons completed with equal readbacks and agreeing diagnostic outputs. Report the GPU, driver, DXC version, seeds, per-form counts and any first differing words. Keep the generated shader artifacts local.
