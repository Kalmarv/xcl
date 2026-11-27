//! `XCL_DRED=1`: which translated pipeline a command list had bound where the GPU faulted.
//!
//! DRED's breadcrumbs give the operations a list got through, not which pipeline was bound, so
//! a page fault in a translated shader names no console shader. With `XCL_DRED=1` every
//! `SetPipelineState` XCL records for the title is followed by a `SetMarker`, and XCL keeps the
//! pipeline's label (its console shaders' hashes) in the list's order. The runtime kept no
//! marker text here (breadcrumb contexts came back empty: Windows 26300, NVIDIA 621.40), but it
//! records each marker as an operation, so the markers before the operation DRED stopped at
//! give the label bound there. DRED's count can lag behind the GPU (NVIDIA: a fault in the
//! second dispatch left the first one not done), so the labels bound after it are given too.
//! The title's own markers are counted, unlabelled, to keep the count in step; DRED gave the
//! list's name and not its pointer, so XCL names each list `xcl list 0x…` it marks.
//!
//! A hung GPU faults nowhere: what matters then is how much work the operation in flight was.
//! So the dispatches XCL records for the title are kept too, in the list's order between its
//! markers (`Work`): a Dispatch's groups and its loops' share, and for an ExecuteIndirect the
//! slot its counts are checked in (`indirect_guard`), whose record says whether the GPU got to
//! read them and what they were. The label of a compute pipeline names its translated code too.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::{ID3D12GraphicsCommandList, D3D12_AUTO_BREADCRUMB_OP_SETMARKER};

/// translated pipeline → its label
static LABELS: Mutex<Option<HashMap<usize, String>>> = Mutex::new(None);
/// PC command list → its last `RECORDINGS` recordings, oldest first
static MARKS: Mutex<Option<HashMap<usize, Vec<Recording>>>> = Mutex::new(None);

/// One recording of a list.
#[derive(Default)]
struct Recording {
    /// the label of each marker (`None`: the title's own)
    marks: Vec<Option<String>>,
    /// the title's dispatches, each with the number of markers recorded before it
    work: Vec<(usize, Work)>,
}

/// A dispatch of the title's as XCL recorded it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Work {
    /// `Dispatch`: the groups asked for, the groups recorded (folded or clamped ones differ) and
    /// what it puts into `IO_LOOP_BUDGET` (`recomp::dispatch_loop_budget`: the passes the lanes
    /// of one thread group may run together in one entry of a loop, which a program that reads
    /// it divides by its threads; 0: none given)
    Dispatch { groups: [u32; 3], recorded: [u32; 3], loop_budget: u32 },
    /// an ExecuteIndirect whose counts XCL's program reads first: the nth it checked
    /// (`indirect_guard::counts_of`)
    Checked(u32),
}

/// Work kept per recording: of a list with more, the first ones (a hung list's report says so).
const WORK_KEPT: usize = 16384;

/// Recordings kept per list: a list can be Reset and recorded again while an earlier
/// submission of it still runs, and that one can be the one that faults.
const RECORDINGS: usize = 4;

thread_local! {
    /// XCL's own marker is being recorded: the list's SetMarker hook must not count it again
    static OWN_MARKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_DRED").is_some())
}

/// Name a translated pipeline by its console shaders (`("ms", bytes)`, …).
pub fn label(pso: usize, shaders: &[(&str, Option<&[u8]>)]) {
    if !on() || pso == 0 {
        return;
    }
    let text = shaders.iter().filter_map(|(k, b)| b.map(|b| format!("{k} {:016x}", crate::gpu::shader::fnv64(b)))).collect::<Vec<_>>().join(" ");
    LABELS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, text);
}

/// The translated code of the compute pipeline `label` named: its size and hash, which tell the
/// forms one console program has apart (the two direct lowerings, HLSL; a cached one's producer
/// is not known here). `chain`: built as a chain of pieces from it.
pub fn label_code(pso: usize, dxil: &[u8], chain: bool) {
    if !on() || pso == 0 {
        return;
    }
    if let Some(l) = LABELS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&pso)) {
        *l += &code_text(dxil, chain);
    }
}

fn code_text(dxil: &[u8], chain: bool) -> String {
    format!(" (DXIL {} B {:016x}{})", dxil.len(), crate::gpu::shader::fnv64(dxil), if chain { ", a chain" } else { "" })
}

pub(super) fn label_of(pso: usize) -> Option<String> {
    LABELS.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// After `SetPipelineState(pso)` on `list`: a marker, and the label of what is bound now.
/// `pso` is the title's object, `bound` what reached the PC list (a stand-in while it builds).
pub unsafe fn mark_pipeline(list: *mut c_void, pso: usize, bound: usize) {
    if !on() || list.is_null() {
        return;
    }
    // nothing labelled: a stand-in still building (named by what it is being built from), or a
    // pipeline of PC shaders
    let text = label_of(bound)
        .or_else(|| super::pso_async::describe(pso))
        .unwrap_or_else(|| format!("pipeline {pso:#x} (a stand-in, or not translated)"));
    let first = !MARKS.lock().unwrap().as_ref().and_then(|m| m.get(&(list as usize))).and_then(|r| r.last()).is_some_and(|v| v.marks.iter().any(Option::is_some));
    push(list as usize, Some(text));
    if let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) {
        // DRED's node may give the list's name and not its pointer (NVIDIA, Windows 26300)
        if first {
            let _ = l.SetName(&windows::core::HSTRING::from(format!("xcl list {:#x}", list as usize)));
        }
        let w: Vec<u16> = "xcl pipeline".encode_utf16().chain([0]).collect();
        OWN_MARKER.with(|f| f.set(true));
        l.SetMarker(0, Some(w.as_ptr() as *const c_void), (w.len() * 2) as u32);
        OWN_MARKER.with(|f| f.set(false));
    }
}

/// The title's own `SetMarker` on `list`: counted, so the labels after it stay in step.
pub fn title_marker(list: *mut c_void) {
    if on() && !OWN_MARKER.with(|f| f.get()) {
        push(list as usize, None);
    }
}

fn push(list: usize, mark: Option<String>) {
    with_recording(list, |r| r.marks.push(mark));
}

/// A dispatch of the title's was recorded on `list`.
pub fn note_work(list: *mut c_void, work: impl FnOnce() -> Work) {
    if on() && !list.is_null() {
        push_work(list as usize, work());
    }
}

fn push_work(list: usize, work: Work) {
    with_recording(list, |r| {
        if r.work.len() < WORK_KEPT {
            r.work.push((r.marks.len(), work));
        }
    });
}

fn with_recording(list: usize, f: impl FnOnce(&mut Recording)) {
    let mut g = MARKS.lock().unwrap();
    let recordings = g.get_or_insert_with(HashMap::new).entry(list).or_default();
    if recordings.is_empty() {
        recordings.push(Recording::default());
    }
    f(recordings.last_mut().unwrap());
}

/// A list was Reset: a new recording, as DRED's history of it starts again too.
pub fn reset(list: *mut c_void) {
    if on() {
        let mut g = MARKS.lock().unwrap();
        let recordings = g.get_or_insert_with(HashMap::new).entry(list as usize).or_default();
        recordings.push(Recording::default());
        if recordings.len() > RECORDINGS {
            recordings.remove(0);
        }
    }
}

/// A list was released: its address may be another list's next.
pub fn forget_list(list: *mut c_void) {
    if on() {
        if let Some(m) = MARKS.lock().unwrap().as_mut() {
            m.remove(&(list as usize));
        }
    }
}

/// A pipeline was released: its address may be another pipeline's next.
pub fn forget_pso(pso: usize) {
    if on() {
        if let Some(m) = LABELS.lock().unwrap().as_mut() {
            m.remove(&pso);
        }
    }
}

/// The recording a DRED history is of: the newest one with as many markers as the history
/// (else the newest).
fn recording_of<'a>(recordings: &'a [Recording], history: &[i32]) -> Option<&'a Recording> {
    let total = markers_before(history, history.len());
    recordings.iter().rev().find(|r| r.marks.len() == total).or(recordings.last())
}

/// How many markers come before operation `done` of a list's history.
fn markers_before(history: &[i32], done: usize) -> usize {
    history.iter().take(done).filter(|&&op| op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER.0).count()
}

/// The list a DRED node is about: its pointer, or the one in the name XCL gave it.
pub fn node_list(ptr: usize, name: Option<&str>) -> usize {
    if ptr != 0 {
        return ptr;
    }
    name.and_then(|n| n.strip_prefix("xcl list 0x")).and_then(|h| usize::from_str_radix(h, 16).ok()).unwrap_or(0)
}

/// The labels of the pipelines bound after operation `done` (DRED's count can lag behind the
/// GPU: the faulting operation may be one of the later ones).
pub fn bound_after(list: usize, history: &[i32], done: usize) -> Vec<String> {
    let (n, total) = (markers_before(history, done), markers_before(history, history.len()));
    let g = MARKS.lock().unwrap();
    let Some(marks) = g.as_ref().and_then(|m| m.get(&list)).and_then(|r| recording_of(r, history)).map(|r| &r.marks) else { return Vec::new() };
    marks.iter().take(total).skip(n).flatten().cloned().collect()
}

/// Operations `from..to` of a list's DRED history by name; a marker of XCL's shows as the label
/// of the pipeline bound from there. From the first operation not done to the list's end it is
/// what the list was about to run: DRED's count can lag, so the one that faulted is among them.
pub fn ops(list: usize, history: &[i32], from: usize, to: usize) -> String {
    let g = MARKS.lock().unwrap();
    let marks = g.as_ref().and_then(|m| m.get(&list)).and_then(|r| recording_of(r, history)).map(|r| &r.marks);
    let mut n = markers_before(history, from);
    let mut out = Vec::new();
    for &op in history.iter().take(to).skip(from) {
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER.0 {
            let label = marks.and_then(|m| m.get(n)).cloned().flatten();
            n += 1;
            out.push(label.map_or_else(|| "marker".to_string(), |l| format!("[{l}]")));
        } else {
            out.push(op_name(op));
        }
    }
    out.join(" ")
}

/// A breadcrumb operation by name (D3D12_AUTO_BREADCRUMB_OP), the ones a title's lists hold.
fn op_name(op: i32) -> String {
    let name = match op {
        3 => "Draw",
        4 => "DrawIndexed",
        5 => "ExecuteIndirect",
        6 => "Dispatch",
        7 => "CopyBufferRegion",
        8 => "CopyTextureRegion",
        9 => "CopyResource",
        10 => "CopyTiles",
        11 => "ResolveSubresource",
        12 => "ClearRTV",
        13 => "ClearUAV",
        14 => "ClearDSV",
        15 => "Barrier",
        18 => "ResolveQueryData",
        23 | 24 => "AtomicCopy",
        25 => "ResolveSubresourceRegion",
        26 => "WriteBufferImmediate",
        42 => "DispatchMesh",
        45 => "Barrier2",
        46 => "Begin",
        n => return format!("op{n}"),
    };
    name.to_string()
}

/// The label of the pipeline `list` had bound at operation `done` of its DRED history.
pub fn bound_at(list: usize, history: &[i32], done: usize) -> Option<String> {
    let n = markers_before(history, done);
    let g = MARKS.lock().unwrap();
    let marks = &recording_of(g.as_ref()?.get(&list)?, history)?.marks;
    marks.get(..n.min(marks.len()))?.iter().rev().find_map(|m| m.clone())
}

/// Entries of `work`'s text at most: a list's report stays a line.
const WORK_SHOWN: usize = 48;

/// The title's dispatches `list` recorded from the pipeline bound at operation `from` of its
/// DRED history to the one bound at operation `to`, in the list's order under each pipeline's
/// label: how much work the operations around the one in flight were. `counts`: what the GPU
/// read for the nth checked ExecuteIndirect (`indirect_guard::counts_of`: a dispatch's groups,
/// a draw's vertices or indices, instances and first one; None: it has not run that check, or
/// the check's slot has been used again since).
pub fn work(list: usize, history: &[i32], from: usize, to: usize, counts: impl Fn(u32) -> Option<[u32; 3]>) -> String {
    let (first, last) = (markers_before(history, from), markers_before(history, to));
    let g = MARKS.lock().unwrap();
    let Some(r) = g.as_ref().and_then(|m| m.get(&list)).and_then(|r| recording_of(r, history)) else { return String::new() };
    let dims = |d: [u32; 3]| format!("{} x {} x {}", d[0], d[1], d[2]);
    let mut out: Vec<String> = Vec::new();
    let (mut under, mut shown, mut more) = (usize::MAX, 0, 0);
    for &(at, w) in r.work.iter().filter(|(at, _)| (first..=last).contains(at)) {
        if shown == WORK_SHOWN {
            more += 1;
            continue;
        }
        shown += 1;
        let text = match w {
            Work::Dispatch { groups, recorded, loop_budget } => format!(
                "Dispatch of {} groups{}, {}",
                dims(groups),
                if recorded != groups { format!(" (recorded as {})", dims(recorded)) } else { String::new() },
                if loop_budget == 0 { "no IO_LOOP_BUDGET".to_string() } else { format!("IO_LOOP_BUDGET {loop_budget} a group") }
            ),
            Work::Checked(nth) => match counts(nth) {
                Some(c) => format!("ExecuteIndirect, the GPU read {:#x}, {:#x}, {:#x}", c[0], c[1], c[2]),
                None => "ExecuteIndirect, no counts read back".to_string(),
            },
        };
        if at != under {
            under = at;
            let label = r.marks.get(..at.min(r.marks.len())).and_then(|m| m.iter().rev().find_map(|l| l.clone()));
            out.push(format!("[{}] {text}", label.as_deref().unwrap_or("no pipeline marked")));
        } else {
            out.push(text);
        }
    }
    if more != 0 {
        out.push(format!("… {more} more"));
    }
    if r.work.len() == WORK_KEPT {
        out.push(format!("(only the list's first {WORK_KEPT} were kept)"));
    }
    out.join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recording(marks: Vec<Option<String>>) -> Recording {
        Recording { marks, work: Vec::new() }
    }

    #[test]
    fn a_fault_in_an_earlier_recording_finds_that_recording() {
        // the list was Reset and recorded again while its first submission (2 markers) ran
        let recordings = vec![recording(vec![Some("cs old".to_string()), Some("ms old".to_string())]), recording(vec![Some("cs new".to_string())])];
        let old_history = [46, 0, 6, 0, 6];
        assert_eq!(recording_of(&recordings, &old_history).unwrap().marks[1].as_deref(), Some("ms old"));
        let new_history = [46, 0, 6];
        assert_eq!(recording_of(&recordings, &new_history).unwrap().marks[0].as_deref(), Some("cs new"));
        // no recording with that many markers: the newest
        assert_eq!(recording_of(&recordings, &[46, 0, 0, 0, 6]).unwrap().marks[0].as_deref(), Some("cs new"));
    }

    #[test]
    fn node_list_takes_the_pointer_or_the_name_xcl_gave() {
        assert_eq!(node_list(0x1000, Some("xcl list 0x2000")), 0x1000);
        assert_eq!(node_list(0, Some("xcl list 0x2c234208c30")), 0x2c2_3420_8c30);
        assert_eq!(node_list(0, Some("the title's list")), 0);
        assert_eq!(node_list(0, None), 0);
    }

    #[test]
    fn markers_before_counts_only_markers_ahead_of_the_failing_op() {
        // BEGIN_COMMAND_LIST, SETPIPELINESTATE-like ops, markers, a dispatch
        let h = [46, 0, 6, 1, 0, 2, 6, 0, 6];
        assert_eq!(markers_before(&h, 0), 0);
        assert_eq!(markers_before(&h, 2), 1);
        assert_eq!(markers_before(&h, 6), 2);
        assert_eq!(markers_before(&h, 8), 3);
        assert_eq!(markers_before(&h, 100), 3);
    }

    #[test]
    fn bound_at_skips_the_titles_markers() {
        let list = 0x1234_5670usize;
        {
            let mut g = MARKS.lock().unwrap();
            let m = g.get_or_insert_with(HashMap::new);
            m.insert(list, vec![recording(vec![Some("cs a".into()), None, Some("ms b ps c".into())])]);
        }
        // ops: begin, our marker (cs a), dispatch, the title's marker, dispatch, our marker (ms b), dispatch
        let h = [46, 0, 6, 0, 6, 0, 6];
        assert_eq!(bound_at(list, &h, 2).as_deref(), Some("cs a"));
        // after the title's marker the last label is still the compute one
        assert_eq!(bound_at(list, &h, 4).as_deref(), Some("cs a"));
        assert_eq!(bound_at(list, &h, 6).as_deref(), Some("ms b ps c"));
        assert_eq!(bound_at(list, &h, 1), None);
        assert_eq!(bound_at(0xdead, &h, 6), None);
        assert_eq!(bound_after(list, &h, 2), vec!["ms b ps c".to_string()]);
        assert_eq!(bound_after(list, &h, 6), Vec::<String>::new());
        // from the first dispatch on: names, XCL's markers as labels, the title's as "marker"
        assert_eq!(ops(list, &h, 2, 16), "Dispatch marker Dispatch [ms b ps c] Dispatch");
        assert_eq!(ops(list, &h, 0, 3), "Begin [cs a] Dispatch");
        assert_eq!(ops(0xdead, &[46, 0, 42, 99], 0, 16), "Begin marker DispatchMesh op99");
        MARKS.lock().unwrap().as_mut().unwrap().remove(&list);
    }

    #[test]
    fn work_is_given_in_order_under_the_pipeline_it_ran_with() {
        let list = 0x1234_5680usize;
        // (as the hooks note it: each with the markers recorded before it)
        push_work(list, Work::Checked(6));
        push(list, Some("cs a".into()));
        push_work(list, Work::Checked(7));
        push_work(list, Work::Dispatch { groups: [4, 2, 1], recorded: [4, 2, 1], loop_budget: 1 << 25 });
        push(list, None);
        push_work(list, Work::Checked(8));
        // (the same pipeline bound a second time)
        push(list, Some("cs a".into()));
        push_work(list, Work::Dispatch { groups: [70_000, 1, 1], recorded: [65_535, 2, 1], loop_budget: 0 });
        push(list, Some("cs b".into()));
        push_work(list, Work::Checked(9));
        // begin, a check + ExecuteIndirect, cs a, a check + ExecuteIndirect, a dispatch, the
        // title's marker, a check + ExecuteIndirect, cs a, a dispatch, cs b, a check + ExecuteIndirect
        let h = [46, 6, 5, 0, 6, 5, 6, 0, 6, 5, 0, 6, 0, 6, 5];
        // (the checks of slots 6 and 7 ran, those of 8 and 9 did not)
        let counts = |slot| [(6, [1, 1, 1]), (7, [3, 1, 0xffff_0001])].into_iter().find(|&(s, _)| s == slot).map(|(_, c)| c);
        assert_eq!(
            work(list, &h, 0, h.len(), counts),
            "[no pipeline marked] ExecuteIndirect, the GPU read 0x1, 0x1, 0x1; \
             [cs a] ExecuteIndirect, the GPU read 0x3, 0x1, 0xffff0001; Dispatch of 4 x 2 x 1 groups, IO_LOOP_BUDGET 33554432 a group; \
             [cs a] ExecuteIndirect, no counts read back; \
             [cs a] Dispatch of 70000 x 1 x 1 groups (recorded as 65535 x 2 x 1), no IO_LOOP_BUDGET; \
             [cs b] ExecuteIndirect, no counts read back"
        );
        // from the second cs a on (operation 11 is its dispatch): what came before it is left out
        assert_eq!(work(list, &h, 11, 12, counts), "[cs a] Dispatch of 70000 x 1 x 1 groups (recorded as 65535 x 2 x 1), no IO_LOOP_BUDGET");
        assert_eq!(work(0xdead, &h, 0, h.len(), counts), "");
        MARKS.lock().unwrap().as_mut().unwrap().remove(&list);
    }

    #[test]
    fn a_label_names_the_translated_code() {
        assert_eq!(code_text(&[1, 2, 3], false), format!(" (DXIL 3 B {:016x})", crate::gpu::shader::fnv64(&[1, 2, 3])));
        assert!(code_text(&[1, 2, 3], true).ends_with(", a chain)"));
    }
}
