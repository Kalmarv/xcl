# Splitting big compute programs into consecutive dispatches

Status: on by default for direct-route compute programs (`gpu/dxil/module/split.rs`, run from
`gpu/dxil/kernel.rs`; recorded by `hle/d3d12x/real/dispatch_chain.rs` and
`gpu/recomp/pso.rs` `record_dispatch_chain`). This note covers why the cuts are where they are,
how values cross a cut, and what the payload slots guarantee. A change to `split.rs` should keep
these three properties or say which one it gives up.

A driver's compile time grows faster than linearly with the size of one program. A program over
the piece limit (`kernel::dispatch_piece_limit()`: 12,500 instructions, or `XCL_DIRECT_SPLIT_INSTRUCTIONS`) is cut into pieces.
Each piece is a complete compute shader, and the pieces run as consecutive dispatches of the same
grid. Each workgroup carries its state from one piece to the next through a scratch buffer.

## Why cuts go at CFG edges

- **The state at an edge is exactly the SSA values live on it.** A cut inside a block would also
  need the block's partial results, ordering against side effects in the same block, and a
  resume point in straight-line code. At an edge, a piece leaves by writing the live values and
  a tag, and the next piece resumes by branching to the edge's target block.
- **A strongly connected component is indivisible.** Dispatches only go forward: a later piece
  cannot jump back into an earlier one. So a loop, or any cycle in the CFG, stays inside one piece.
  `dispatch_plan` finds the components with Kosaraju's algorithm over the reverse postorder. This
  yields them in topological order, and they are packed greedily into pieces of up to `limit`
  instructions. A single component over the limit takes a piece of its own. A component over
  `4 × limit` declines the split, so a very large loop leaves the whole program unsplit.
- **Every crossing edge goes forward**, from an earlier piece to a later one (`debug_assert`
  in the planner). A piece can be skipped entirely when control goes from piece 1 straight to
  piece 3.
- **What can't cross a cut declines the split.** Groupshared memory (other than the wave64
  exchange arrays `xcl_xw` / `xcl_xs`), private arrays (`alloca`), atomics whose result is read,
  and barriers outside the exchange pattern don't survive a dispatch boundary. `split_compute`
  returns `None` for any of them, so the program stays whole. So does any value of a type that
  can't be transported (below). So does a piece that fails to build or fails validation (warning
  `compute split: piece i of n refused`).

## How values cross a cut

- **Live sets.** For each crossing edge, `Edge::live` holds the phi inputs read on that edge plus
  the target block's live-in values.
- **Transport, per value** (`transport_words`):
  - i1..i32 and f32 take one word; i64 takes two.
  - A handle built from a constant binding takes zero words: the receiving piece rebuilds it
    (`rebuild_static`).
  - A heap handle with constant properties, and a pointer into `xcl_xs`, carry only their index,
    in one word (`indexed`).
  - Anything else declines the split, f16 and f64 included.
- **Leaving a piece.** An exit block stores the values to their slots, stores the edge's tag
  (edge index + 1) in slot 0, and returns. When a non-final piece reaches the program's own `ret`,
  it stores tag 0 instead, so every later piece sees "done".
- **Entering a piece.** Every piece after the first starts with a `switch` on the tag in slot 0.
  Each incoming edge's tag goes through a hop block to that edge's target. The default, which
  includes tag 0, returns at once.
- **Restores.** A restored value starts as a placeholder. `place_restores` then puts its load at
  the nearest common dominator of its uses (after any leading phis), not at the piece's entry. So
  a value only one branch needs is loaded only on that branch, and a value that passes through a
  piece unused isn't loaded there at all.
- **Scratch layout** (`Scratch`):
  - word index = `group × stride × threads + slot × threads + lane`, where `group` is the
    linear group id (width and height come from the chain's root constants).
  - For one slot, consecutive lanes have consecutive words. Each workgroup owns its own
    `stride × threads` words.
  - The runtime allocates `dwords_per_group × 4 × groups` bytes. Above 4 GiB, in a bundle, or for
    ExecuteIndirect (whose group count the CPU doesn't know), it binds the whole program instead.
    It puts a UAV barrier between pieces.

## What payload-slot stability guarantees

Slot 0 is the tag. Every other transported value gets its own slot range, fixed for the whole
chain and assigned once in `split_compute`. No two values share a slot.

- **Only the defining piece writes a value's slot.** It writes it at each of its exits where the
  value is live. A later piece that only passes the value along doesn't copy it.
- **A slot keeps its value until the chain ends.** So a piece can read any value defined in any
  earlier piece, whichever pieces in between were skipped or which exit they took.
- **A piece's outgoing stores never overwrite a value it still has to restore.** This holds even
  when the restore is sunk past a branch, below the store.
- **The cost is scratch size.** `stride` is 1 plus the words of every value that crosses any
  cut, not the largest live set of a single edge. Programs with many short-lived values crossing
  different cuts pay for all of them at once. Reusing a slot for values whose live ranges (in
  pieces) don't overlap would shrink this, but it must keep the three guarantees above.

Tests: `split.rs` `mod tests`. GPU tests compare each split program's output with the whole
program's: wave32 and wave64 live loops, divergent forward edges in 3-D grids, payloads that skip
a middle piece, a reused edge slot, and an oversized loop. The planner tests cover the 4× ceiling.
