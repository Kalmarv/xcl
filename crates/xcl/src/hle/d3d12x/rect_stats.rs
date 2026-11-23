//! Rect-list diagnostics for the report (`D3d12Stats::rect_lists`): how a title's rect lists
//! (IASetPrimitiveTopology 17, pipeline topology type 19) reach XCL. The pipelines it makes with
//! topology type 19, by what XCL made of them; the draws while topology 17 is set, by the kind
//! of pipeline bound; those `bind::draw_rects` expanded on the CPU; and every console topology
//! the PC does not have, by value. Counts only: nothing here changes how anything is drawn.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RectListStats {
    /// pipelines made with topology type 19, by `pipeline_kind` (`failed`: refused)
    #[serde(default)]
    pub pipelines: BTreeMap<String, u64>,
    /// draws while topology 17 is set, by the bound pipeline's `pipeline_kind` (`:indexed`)
    #[serde(default)]
    pub draws: BTreeMap<String, u64>,
    /// draws of topology 17 drawn as two triangles a rect, made on the CPU (`bind::draw_rects`)
    #[serde(default)]
    pub cpu_expanded: u64,
    /// IASetPrimitiveTopology calls with a console topology the PC does not have, by value
    #[serde(default)]
    pub console_topologies: BTreeMap<u32, u64>,
}

static PIPELINES: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());
static DRAWS: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());
static CPU_EXPANDED: AtomicU64 = AtomicU64::new(0);
static TOPOLOGIES: Mutex<BTreeMap<u32, u64>> = Mutex::new(BTreeMap::new());

/// The console's pipeline topology type of rect lists (draws of topology 17, `list_x::RECT_LIST`).
const RECT_LIST_TYPE: u32 = 19;

/// Whether a D3D12_GRAPHICS_PIPELINE_STATE_DESC has the rect lists' topology type.
pub(super) unsafe fn desc_is_rect_list(desc: *const std::ffi::c_void) -> bool {
    use windows::Win32::Graphics::Direct3D12::D3D12_GRAPHICS_PIPELINE_STATE_DESC;
    crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>())
        && (*(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC)).PrimitiveTopologyType.0 as u32 == RECT_LIST_TYPE
}

/// Whether a pipeline stream (D3D12_PIPELINE_STATE_STREAM_DESC) has a PRIMITIVE_TOPOLOGY
/// subobject (14) of the rect lists' type.
pub(super) unsafe fn stream_is_rect_list(desc: *const std::ffi::c_void) -> bool {
    if !crate::gpu::shader::readable(desc as usize, 16) {
        return false;
    }
    let (size, base) = (*(desc as *const usize), *((desc as usize + 8) as *const usize));
    if !crate::gpu::shader::readable(base, size) {
        return false;
    }
    let buf = std::slice::from_raw_parts(base as *const u8, size);
    let mut off = 0usize;
    while off + 8 <= size {
        let ty = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        let Some((psize, palign)) = super::null::subobject_payload(ty) else { return false };
        let at = off + if palign == 8 { 8 } else { 4 };
        if ty == 14 {
            return buf.get(at..at + 4).is_some_and(|p| p == RECT_LIST_TYPE.to_le_bytes());
        }
        off = (at + psize + 7) & !7;
    }
    false
}

pub fn snapshot() -> RectListStats {
    RectListStats {
        pipelines: PIPELINES.lock().map(|m| m.clone()).unwrap_or_default(),
        draws: DRAWS.lock().map(|m| m.clone()).unwrap_or_default(),
        cpu_expanded: CPU_EXPANDED.load(Ordering::Relaxed),
        console_topologies: TOPOLOGIES.lock().map(|m| m.clone()).unwrap_or_default(),
    }
}

/// What a pipeline is to XCL: `translated` (console VS + PS), `console_gs` / `mesh` (a console
/// GS or MS in two passes), `stand_in` (still being built in the background), `untranslated`
/// (the PC's own shaders, or a stand-in for a pipeline XCL could not translate), `none`.
pub fn pipeline_kind(pso: usize) -> &'static str {
    if pso == 0 {
        return "none";
    }
    if super::pso_async::is_pending(pso) {
        return "stand_in";
    }
    if let Some(t) = super::real::two_pass_of(pso) {
        return match t.kind {
            "GS" => "console_gs",
            "MS" => "mesh",
            _ => "two_pass",
        };
    }
    if super::real::is_translated_graphics(super::pso_async::resolve(pso)) {
        "translated"
    } else {
        "untranslated"
    }
}

fn add<K: Ord>(m: &Mutex<BTreeMap<K, u64>>, k: K) {
    if let Ok(mut m) = m.lock() {
        *m.entry(k).or_default() += 1;
    }
}

/// A pipeline the title made with topology type 19: `hr`, and the pipeline at `ppv`.
pub(super) unsafe fn note_pipeline(hr: i32, ppv: *mut *mut std::ffi::c_void) {
    let kind = if hr < 0 || ppv.is_null() || (*ppv).is_null() { "failed" } else { pipeline_kind(*ppv as usize) };
    add(&PIPELINES, kind.to_string());
}

/// A draw while topology 17 is set, with the bound pipeline `pso`.
pub(super) fn note_draw(pso: usize, indexed: bool) {
    let kind = pipeline_kind(pso);
    add(&DRAWS, if indexed { format!("{kind}:indexed") } else { kind.to_string() });
}

/// A draw of topology 17 expanded on the CPU.
pub(super) fn note_cpu_expanded() {
    CPU_EXPANDED.fetch_add(1, Ordering::Relaxed);
}

/// IASetPrimitiveTopology with the console topology `t`, one the PC does not have.
pub(super) fn note_console_topology(t: u32) {
    add(&TOPOLOGIES, t);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts land in the snapshot under their keys (deltas: other tests may count too).
    #[test]
    fn counts_reach_the_snapshot() {
        let before = snapshot();
        note_console_topology(0xEE01);
        note_console_topology(0xEE01);
        note_cpu_expanded();
        let pso = 0x7E57_6000usize; // not a pipeline XCL knows
        note_draw(pso, true);
        note_draw(0, false);
        unsafe { note_pipeline(-1, std::ptr::null_mut()) };
        let after = snapshot();
        let delta = |a: &BTreeMap<String, u64>, b: &BTreeMap<String, u64>, k: &str| b.get(k).copied().unwrap_or(0) - a.get(k).copied().unwrap_or(0);
        assert_eq!(after.console_topologies.get(&0xEE01), Some(&2));
        assert!(after.cpu_expanded > before.cpu_expanded);
        assert!(delta(&before.draws, &after.draws, "untranslated:indexed") >= 1);
        assert!(delta(&before.draws, &after.draws, "none") >= 1);
        assert!(delta(&before.pipelines, &after.pipelines, "failed") >= 1);
    }

    /// `pipeline_kind` of a translated VS + PS pipeline, an unknown one and none.
    #[test]
    fn pipeline_kinds() {
        let pso = 0x7E57_7000usize;
        assert_eq!(pipeline_kind(0), "none");
        assert_eq!(pipeline_kind(pso), "untranslated");
        super::super::real::register_translated_graphics_for_test(pso);
        assert_eq!(pipeline_kind(pso), "translated");
        super::super::real::forget_pso_for_test(pso);
        assert_eq!(pipeline_kind(pso), "untranslated");
    }

    /// A desc or a stream of the rect lists' topology type (19) is recognised; others are not.
    #[test]
    fn descs_and_streams_of_the_rect_list_type() {
        use windows::Win32::Graphics::Direct3D12::{D3D12_GRAPHICS_PIPELINE_STATE_DESC, D3D12_PRIMITIVE_TOPOLOGY_TYPE};
        let at = |t: i32| D3D12_GRAPHICS_PIPELINE_STATE_DESC { PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE(t), ..Default::default() };
        let (rect, tri) = (at(19), at(3));
        // SAMPLE_MASK (9), then PRIMITIVE_TOPOLOGY (14) if given: 4-byte type, 4-byte payload each
        let stream = |t: Option<u32>| {
            let mut v: Vec<u8> = [9u32, u32::MAX].iter().flat_map(|w| w.to_le_bytes()).collect();
            if let Some(t) = t {
                v.extend([14u32, t].iter().flat_map(|w| w.to_le_bytes()));
            }
            v
        };
        let (s19, s3, none) = (stream(Some(19)), stream(Some(3)), stream(None));
        let d = |s: &Vec<u8>| [s.len(), s.as_ptr() as usize];
        unsafe {
            assert!(desc_is_rect_list(&rect as *const _ as *const std::ffi::c_void));
            assert!(!desc_is_rect_list(&tri as *const _ as *const std::ffi::c_void));
            assert!(stream_is_rect_list(d(&s19).as_ptr() as *const std::ffi::c_void));
            assert!(!stream_is_rect_list(d(&s3).as_ptr() as *const std::ffi::c_void));
            assert!(!stream_is_rect_list(d(&none).as_ptr() as *const std::ffi::c_void));
        }
    }

    /// A report without `rect_lists` (an older XCL's) still reads; the field round-trips.
    #[test]
    fn reports_with_and_without_rect_lists_deserialize() {
        let mut v = serde_json::to_value(super::super::real::D3d12Stats::default()).unwrap();
        v.as_object_mut().unwrap().remove("rect_lists");
        let d: super::super::real::D3d12Stats = serde_json::from_value(v).unwrap();
        assert_eq!(d.rect_lists, RectListStats::default());
        let s = RectListStats { console_topologies: BTreeMap::from([(17, 3)]), cpu_expanded: 2, ..Default::default() };
        let back: RectListStats = serde_json::from_value(serde_json::to_value(&s).unwrap()).unwrap();
        assert_eq!(back, s);
    }
}
