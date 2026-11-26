# integration branch manifest

Worktree `upstream/xcl-integration`, local branch `integration`, target dir `.targets/xcl-integration`.
Rebuilt from xcl-dev/xcl main plus PR heads (`refs/pull/<n>/head`), merged `--no-ff` in this order
(coordinator's order: VRAM shadow, then #2551 / #2613 / journal, then ours, then the rest).
Helper scripts: `scratch/integration/merge-next.sh` (merge in order, stop at a conflict), `both.py`
(keep both sides). rerere is on in the shared repo, so the same resolutions replay tomorrow.

## Build 2026-11-26

Base: main `b49be183` (Merge #2371). States read from the API 2026-11-26 ~05:00Z: every PR below open.

| # | PR | author | head | conflicts and resolution |
|---|----|--------|------|--------------------------|
| 1 | #1329 (+#1326, #1327) VRAM copy of the committed window | vakaobr | ac4b9978 | #1329's branch carries rebased copies of #1326 (3dd994ad) and #1327 (39f25043), so only #1329 was merged. `sparse.rs`: main added `State::quarantined` / `queue_fences`, the PR moved State's construction into `new_state()`: took `new_state()` and added the two fields (same as hang-dred's def2e875). |
| 2 | #2551 chain whole forms in --pipelines | DTsordas | 3567b2ef | clean |
| 3 | #2613 graphics precompile, runtime variants | Decodere | a3312c05 | clean |
| 4 | #1826 (+#759, #758, #757) pipeline journal and replay, --shard | VolenNakov / oishigeso | 7ae2bb71 (759 dd66d209, 758 c02d0592, 757 4e8aaf4b) | clean; #757-#759 are ancestors of #1826 |
| 5 | #2633 (+#2404) root CBV slabs on chained ExecuteIndirect | Kalmarv | acba153c (2404 97a70a73) | clean; #2404 is an ancestor of #2633 |
| 6 | #2428 (+#2427) flat heap windows, NVIDIA compute default | Kalmarv | c6e5af86 (2427 c8c2b7a6) | 3 files vs #759/#2613. `pso.rs`: kept the journal's `set_mem_windows` override in the new per-stage functions: `mem_windows()` = override count with the run's heap bit through `with_heap` (adds HEAP_FLAT as this process would), `compute_mem_windows()` = override count in this process's compute form. `real.rs`: kept #759's `GraphicsPrep`/`prep_graphics`; `translated_graphics_pso` takes #2428's `heap` arg; journal `warm` passes heap by entry kind (compute: `compute_mem_windows`, else `mem_windows`); `note_heap_window_oom` reads `HEAP_WINDOW_BUILD` like #2428's `driver_pso`. `shaders.rs`: precompile's compute/mesh HLSL loop and #2613's new VS/PS warm use `graphics_windows`. |
| 7 | #2567 scalar-load outlining (NVIDIA) | Kalmarv | 89e1a91d | `real::init` vendor: one mechanism, `adapter::TITLE_VENDOR` / `shader_vendor()`. Removed #2428's `pso::ADAPTER_VENDOR` / `set_adapter_vendor`, `adapter::vendor()` and `shaders::note_adapter_vendor()`; `heap_windows()` reads `shader_vendor()`; #2428's test sets `TITLE_VENDOR` in its child process. Side effect: `xcl pipelines` now gets the NVIDIA compute default too (it never called `set_adapter_vendor`). |
| 8 | #2576 (+#1906) DXIL EXEC-select / pair folds | Kalmarv (1906: Decodere) | 71b76742 (1906 894e29d6) | `core_tests.rs`: both test sets kept. `kernel.rs`: #2576's thread closure with #2567's `outline` argument. |
| 9 | #2097 wave64 groups split across dispatches | Kalmarv | d98d2d46 | `split.rs`: both sets of match arms kept (#2567's return-slot arms, then #2097's multi-wave arms). |
| 10 | #2098 compile slots scale with CPUs | Kalmarv | f22f2396 | clean |
| 11 | #2571 direct PS compares skip per-quad mask | Kalmarv | ad0ded91 | clean |
| 12 | #350 comparison samples use supplied gradients | Kalmarv | 9249b7af | clean |
| 13 | #245 compute groups not whole waves | Kalmarv | 2fcb0950 | `recomp/mod.rs`: both `pub mod` lines kept |
| 14 | #247 image_sample_b/d/cd/c_o offsets | Kalmarv | 227a60a9 | clean |
| 15 | #253 xcl run thread dumps | Kalmarv | f67e35ae | clean |
| 16 | #266 rect lists CPU expansion | Kalmarv | 6bf3305a | clean |
| 17 | #349 pixel comparison samples LOD bias | Kalmarv | 7d1c008d | clean |
| 18 | #431 TLS image install | Kalmarv | 940f7cb7 | clean |
| 19 | #603 tests: wait for later signal on another queue | Kalmarv | fc7336c3 | clean |
| 20 | #612 XCL_NETWORK=0 WSASendMsg | Kalmarv | bbf31746 | clean |
| 21 | #619 XCL_NETWORK=0 IPv4-mapped loopback | Kalmarv | 29706267 | clean |
| 22 | #1911 second atomic operand | Kalmarv | 39601b48 | clean |
| 23 | #1912 scalar state across per-lane paths | Kalmarv | c4093f4c | clean |
| 24 | #2123 split compute programs with global atomics | Kalmarv | 655885ec | `split.rs` `dispatch_plan`: #2097 moved the packing out of the component loop; #2123's polling-loop refusal put at the end of that loop, before `stages`/`components.push`. |
| 25 | #2484 per-page hashes of title textures | ChalanaGimhanaX | 1c9e8396 | clean |
| 26 | #2628 big-compile waiter leaves the queue | KOLESNiii | 3693d9ef | clean |
| 27 | #2629 failed whole build not retried | Royalflamejlh | bf200e16 | `dispatch_chain.rs` vs #2404/#2633: `whole: OnceLock<Option<_>>`; kept #2633's `build_whole` mutex, `waiting` counter and `build_whole_form` (its test uses `build_whole.try_lock`), which now `get_or_init`s the Option; `execute_indirect` reads `.cloned().flatten()`; #2629's test gets #2633's fields. (chain-ei2's patch was for the older 1202ac10 shape.) |
| 28 | #2640 prune unused module constants | Decodere | b7cfe9b2 | `module.rs`: took #2640's `live_values()`, extended to walk the emitted helpers #2567 calls (they share the module's constants, `finish_helper`); otherwise a constant used only in an outlined helper would be dropped. |
| 29 | #2641 frexp/alignbit simplifications | Decodere | 84ae02e8 | clean |
| 30 | #2635 batch used-pipelines.txt writes | Decodere | bd9b4008 | clean |
| 31 | frame-perf repeat-upload skip | (local) | c32d87d4 cherry-picked as 32fa412f | clean (on top of #2484) |

Skipped: none. Repo-watch's list of other ready Kalmarv PRs (13-24) was added as task.md allows
("any other open Kalmarv PR that's ready"). Not in it by rule: stand-ins, VIctory / GPL-3 code.

| 32 | integration fix aed7013e | - | - | build fix after all merges: #2098 (f22f2396) gave `big_compile_turn` a `need` argument and a `commit_free(need)`; #759's journal `warm` and #2633's two tests called the old form. Also dropped the `pso` import #2428's test no longer uses. |

### Check

`cargo check -p xcl --lib --tests` (target `.targets/xcl-integration`): first run 10 errors, all the
`big_compile_turn` signature (#2098 vs #759 / #2633) plus one unused-import warning; after aed7013e:
passes, 3 warnings (two deprecated `fetch_update` and `RetiringBatch::heaps` never read, in vram.rs, not from a resolution).
