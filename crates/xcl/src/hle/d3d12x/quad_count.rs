//! Diagnostic (not for upstream): how much a title uses the console's quad sample mask, command
//! list #142 of the newer layout (`SetQuadSampleMaskX(u64)`, the name as in xcl-dev/xcl PR #1274;
//! a stub otherwise). Counts only – nothing is applied, no shader and no io word changes:
//! the calls, those with a mask other than all ones, and the title's graphics work recorded
//! while such a mask is current on its list (direct draws, DispatchMesh, and graphics
//! ExecuteIndirect calls by kind, each counted once at the title's call). One tally line every
//! ten seconds of graphics work, so a window with no use reads as zeros, not as no line.

use std::collections::{BTreeMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

/// what the title recorded (`note`)
#[derive(Clone, Copy)]
pub(super) enum Work {
    Draw = 0,
    DispatchMesh = 1,
    IndirectDraw = 2,
    IndirectMesh = 3,
}

const KINDS: usize = 4;
/// calls of #142, of them with a mask other than all ones
static CALLS: AtomicU64 = AtomicU64::new(0);
static PARTIAL_CALLS: AtomicU64 = AtomicU64::new(0);
/// the title's graphics work by kind: all of it, and with a partial mask current on its list
static ALL: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
static MASKED: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
/// list resets / ClearStates that found a partial mask current (it is dropped there: the count
/// says how much rests on that)
static DROPPED: AtomicU64 = AtomicU64::new(0);
/// lists with a partial mask current (`PARTIAL_LISTS`' size: the lock is taken only above 0)
static PARTIAL_NOW: AtomicUsize = AtomicUsize::new(0);
static PARTIAL_LISTS: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
/// the masks seen and their calls (the first 16 values)
static MASKS: Mutex<BTreeMap<u64, u64>> = Mutex::new(BTreeMap::new());
/// work noted, and the time (ms since the first note) of the last tally line
static NOTES: AtomicU64 = AtomicU64::new(0);
static LAST_LINE_MS: AtomicU64 = AtomicU64::new(0);

/// Whether `mask` takes any sample away.
fn partial(mask: u64) -> bool {
    mask != u64::MAX
}

fn set_current(list: usize, on: bool) -> bool {
    if !on && PARTIAL_NOW.load(Ordering::Relaxed) == 0 {
        return false;
    }
    let mut g = PARTIAL_LISTS.lock().unwrap();
    let set = g.get_or_insert_with(HashSet::new);
    let was = if on { !set.insert(list) } else { set.remove(&list) };
    PARTIAL_NOW.store(set.len(), Ordering::Relaxed);
    was
}

fn current(list: usize) -> bool {
    PARTIAL_NOW.load(Ordering::Relaxed) != 0 && PARTIAL_LISTS.lock().unwrap().as_ref().is_some_and(|s| s.contains(&list))
}

/// Command list #142 on the newer layout (on the older one #142 is another method: answered 0,
/// as the stub answers).
pub(super) unsafe extern "system" fn set_quad_sample_mask_x(this: *mut c_void, mask: u64) -> u64 {
    if !super::real::QUEUE_SHIFTED.load(Ordering::Acquire) {
        return 0;
    }
    let n = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if partial(mask) {
        PARTIAL_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    set_current(this as usize, partial(mask));
    {
        let mut g = MASKS.lock().unwrap();
        if g.len() < 16 || g.contains_key(&mask) {
            *g.entry(mask).or_default() += 1;
        }
    }
    if n <= 4 {
        crate::xlog_info!("D3D12.X: quad sample mask count: list #142 call {n}, mask {mask:#018x}");
    }
    0
}

/// The list's recording starts over or its state is cleared.
pub(super) fn reset(list: *mut c_void) {
    if set_current(list as usize, false) {
        DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}

/// A mid-list split: `to` goes on with `from`'s mask.
pub(super) fn inherit(from: *mut c_void, to: *mut c_void) {
    if PARTIAL_NOW.load(Ordering::Relaxed) != 0 || current(to as usize) {
        let on = current(from as usize);
        set_current(to as usize, on);
    }
}

/// The title recorded graphics work on `list`.
pub(super) fn note(list: *mut c_void, work: Work) {
    ALL[work as usize].fetch_add(1, Ordering::Relaxed);
    if current(list as usize) {
        MASKED[work as usize].fetch_add(1, Ordering::Relaxed);
    }
    if NOTES.fetch_add(1, Ordering::Relaxed) % 2048 == 0 {
        tally();
    }
}

fn tally() {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let now = START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64 + 1;
    let last = LAST_LINE_MS.load(Ordering::Relaxed);
    if last != 0 && now < last + 10_000 {
        return;
    }
    if LAST_LINE_MS.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    crate::xlog_info!("D3D12.X: quad sample mask count: {}", text());
}

fn text() -> String {
    let v = |a: &[AtomicU64; KINDS]| -> [u64; KINDS] { std::array::from_fn(|k| a[k].load(Ordering::Relaxed)) };
    let (all, masked) = (v(&ALL), v(&MASKED));
    let masks = MASKS.lock().unwrap().iter().map(|(m, c)| format!("{m:#018x} {c}")).collect::<Vec<_>>().join(", ");
    format!(
        "#142 calls {}, with a partial mask {}; with a partial mask current / all: draws {}/{}, DispatchMesh {}/{}, ExecuteIndirect draws {}/{}, ExecuteIndirect DispatchMesh {}/{}; resets that dropped a partial mask {}; masks and calls: [{}]",
        CALLS.load(Ordering::Relaxed),
        PARTIAL_CALLS.load(Ordering::Relaxed),
        masked[0],
        all[0],
        masked[1],
        all[1],
        masked[2],
        all[2],
        masked[3],
        all[3],
        DROPPED.load(Ordering::Relaxed),
        masks
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A partial mask counts the list's work until an all-ones mask or a reset; another list's
    /// work does not count; a split carries it.
    #[test]
    fn work_counts_under_a_partial_mask_of_its_own_list() {
        let (a, b, c) = (0x7142_0000usize as *mut c_void, 0x7142_1000usize as *mut c_void, 0x7142_2000usize as *mut c_void);
        let masked = |k: Work| MASKED[k as usize].load(Ordering::Relaxed);
        assert!(partial(0x0000_FFFF_FFFF_0000) && !partial(u64::MAX));
        let before = (masked(Work::Draw), masked(Work::IndirectMesh));
        set_current(a as usize, true);
        note(a, Work::Draw);
        note(b, Work::Draw);
        inherit(a, c);
        note(c, Work::IndirectMesh);
        assert_eq!((masked(Work::Draw), masked(Work::IndirectMesh)), (before.0 + 1, before.1 + 1));
        set_current(a as usize, false);
        note(a, Work::Draw);
        assert_eq!(masked(Work::Draw), before.0 + 1);
        let dropped = DROPPED.load(Ordering::Relaxed);
        reset(c);
        reset(b);
        assert_eq!(DROPPED.load(Ordering::Relaxed), dropped + 1);
        note(c, Work::IndirectMesh);
        assert_eq!(masked(Work::IndirectMesh), before.1 + 1);
        assert!(text().contains("#142 calls"));
    }
}
