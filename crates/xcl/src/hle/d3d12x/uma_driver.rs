//! (scratch, `XCL_UMA_DRIVER=custom|upload|mixed:<start_mb>:<size_mb>[,...]`) A coherent UMA
//! window in the driver's memory instead of the title's own pages opened with
//! `OpenExistingHeapFromAddress`: a whole-frame replay ran window 3's programs 4-270x faster from
//! a driver CUSTOM L0 heap than from the imported memory (RT1 vs RH-imp). The title keeps its
//! addresses: the window's pages are copied out, the window is released, every other large free
//! stretch of address space is reserved for a moment, so the driver's CPU mapping of the new heap
//! can only land in the hole the window left, and the pages go back.
//!
//! `mixed:<start_mb>:<size_mb>` backs `[start, start+size)` (<= 2048 MiB, where NVIDIA's
//! `GPU_UPLOAD` heap stays in real local VRAM instead of demoting to write-combined host memory)
//! with `GPU_UPLOAD` and the remaining sub-ranges with `CUSTOM` write-back `L0` heaps, placing
//! each segment's CPU mapping back-to-back at its exact virtual address and unifying them on the
//! GPU through a single `CreateReservedResource` mapped via `UpdateTileMappings`.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_FREE, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS,
    PAGE_READWRITE,
};

const MEM_WRITE_WATCH: u32 = 0x20_0000;
const TILE: usize = 0x1_0000;

#[link(name = "kernel32")]
extern "system" {
    fn GetWriteWatch(flags: u32, base: *const c_void, size: usize, addresses: *mut *mut c_void, count: *mut usize, granularity: *mut u32) -> u32;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    Single { upload: bool },
    /// Contiguous `(start_bytes, len_bytes)` ranges inside the window to back with `GPU_UPLOAD`
    /// (real ReBAR VRAM when `len_bytes <= 2 GiB`); all other ranges use `CUSTOM` write-back `L0`.
    Mixed(Vec<(usize, usize)>),
}

/// Parse `XCL_UMA_DRIVER`:
/// - `custom` (CUSTOM heap, write-back CPU pages, L0: host memory)
/// - `upload` (single GPU_UPLOAD heap)
/// - `mixed:<start_mb>:<size_mb>[,<start_mb>:<size_mb>...]`
pub(super) fn mode() -> Option<Mode> {
    let raw = std::env::var("XCL_UMA_DRIVER").ok()?;
    parse_mode(&raw)
}

fn parse_mode(raw: &str) -> Option<Mode> {
    let s = raw.trim();
    match s {
        "custom" => Some(Mode::Single { upload: false }),
        "upload" => Some(Mode::Single { upload: true }),
        _ if s.starts_with("mixed:") => {
            let mut ranges = Vec::new();
            for part in s["mixed:".len()..].split(',') {
                let (a, b) = part.trim().split_once(':')?;
                let start_mb: usize = a.trim().parse().ok()?;
                let size_mb: usize = b.trim().parse().ok()?;
                if size_mb == 0 {
                    continue;
                }
                ranges.push((start_mb << 20, size_mb << 20));
            }
            if ranges.is_empty() {
                None
            } else {
                ranges.sort_by_key(|r| r.0);
                Some(Mode::Mixed(ranges))
            }
        }
        _ => None,
    }
}

/// The written pages of `[base, base + size)` that hold anything but zeros, with their bytes; an
/// error when the write watch cannot be read (the window must then stay as it is).
unsafe fn written_pages(base: usize, size: usize) -> Result<Vec<(usize, Vec<u8>)>, String> {
    let mut out = Vec::new();
    let mut buf: Vec<*mut c_void> = vec![std::ptr::null_mut(); 1 << 16];
    let end = base + size;
    let mut at = base;
    while at < end {
        let (mut n, mut page) = (buf.len(), 0u32);
        let status = GetWriteWatch(0, at as *const c_void, end - at, buf.as_mut_ptr(), &mut n, &mut page);
        if status != 0 {
            return Err(format!("GetWriteWatch at {at:#x} failed ({status})"));
        }
        for &p in &buf[..n] {
            let s = std::slice::from_raw_parts(p as *const u8, page as usize);
            if s.iter().any(|&b| b != 0) {
                out.push((p as usize, s.to_vec()));
            }
        }
        if n < buf.len() {
            break;
        }
        at = buf[n - 1] as usize + page as usize;
    }
    Ok(out)
}

/// Reserves every free stretch of `min_region` or more (leaving `headroom` at the start of each
/// stretch for the driver's own small mappings) outside `[hole, hole + len)`.
unsafe fn groom_with(hole: usize, len: usize, min_region: usize, headroom: usize) -> Vec<usize> {
    let (top, mut a, mut got) = (0x7FFF_FFFE_0000usize, 0x1_0000usize, Vec::new());
    while a < top {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(a as *const c_void, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) == 0 {
            break;
        }
        let (base, end) = (m.BaseAddress as usize, (m.BaseAddress as usize + m.RegionSize).min(top));
        a = end;
        if m.State != MEM_FREE || m.RegionSize < min_region {
            continue;
        }
        let s = ((base + 0xFFFF) & !0xFFFF).saturating_add(headroom);
        for (s, e) in [(s, end.min(hole)), (s.max(hole + len), end)] {
            let mut s = s;
            while s < e {
                let n = (e - s).min(1 << 40);
                let p = VirtualAlloc(s as *const c_void, n, MEM_RESERVE, PAGE_NOACCESS);
                if !p.is_null() {
                    got.push(p as usize);
                }
                s += n;
            }
        }
    }
    got
}

unsafe fn ungroom(got: Vec<usize>) {
    for p in got {
        VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
    }
}

fn heap_props(upload: bool) -> D3D12_HEAP_PROPERTIES {
    if upload {
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE(5), ..Default::default() }
    } else {
        D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        }
    }
}

fn buffer_desc(size: usize) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    }
}

unsafe fn create_mapped_segment(
    device: &ID3D12Device,
    size: usize,
    upload: bool,
    want_watch: bool,
) -> Result<(ID3D12Heap, ID3D12Resource, usize, bool), String> {
    let properties = heap_props(upload);
    let desc = |flags| D3D12_HEAP_DESC { SizeInBytes: size as u64, Properties: properties, Alignment: 0x1_0000, Flags: flags };
    let mut heap: Option<ID3D12Heap> = None;
    let mut watched =
        want_watch && device.CreateHeap(&desc(D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH), &mut heap).is_ok() && heap.is_some();
    if !watched {
        heap = None;
        device.CreateHeap(&desc(D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS), &mut heap).map_err(|e| format!("CreateHeap: {e}"))?;
        watched = false;
    }
    let heap = heap.ok_or("CreateHeap")?;
    let buf_desc = buffer_desc(size);
    let mut buf: Option<ID3D12Resource> = None;
    device.CreatePlacedResource(&heap, 0, &buf_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf).map_err(|e| format!("CreatePlacedResource: {e}"))?;
    let buf = buf.ok_or("CreatePlacedResource")?;
    let mut p: *mut c_void = std::ptr::null_mut();
    buf.Map(0, Some(&D3D12_RANGE::default()), Some(&mut p)).map_err(|e| format!("Map: {e}"))?;
    Ok((heap, buf, p as usize, watched))
}

/// Window remake result: the GPU buffer shaders bind, per-segment `(offset, len, heap_raw)` for
/// placed resources inside the window, and the write-watched CPU sub-ranges `(va, len)`.
pub(super) struct Remade {
    pub buf_raw: usize,
    pub gpu_va: u64,
    pub heaps: Vec<(usize, usize, usize)>,
    pub watched_regions: Vec<(usize, usize)>,
    pub all_watched: bool,
}

/// Partition `[0, size)` into contiguous `(offset, len, upload)` segments.
fn build_segments(size: usize, vram_ranges: &[(usize, usize)]) -> Result<Vec<(usize, usize, bool)>, String> {
    let mut segs = Vec::new();
    let mut cur = 0usize;
    for &(start, len) in vram_ranges {
        let s = start.min(size) & !(TILE - 1);
        let e = start.saturating_add(len).min(size) & !(TILE - 1);
        if e <= s {
            continue;
        }
        if s < cur {
            return Err(format!("overlapping mixed ranges at {s:#x} < {cur:#x}"));
        }
        if s > cur {
            segs.push((cur, s - cur, false));
        }
        segs.push((s, e - s, true));
        cur = e;
    }
    if cur < size {
        segs.push((cur, size - cur, false));
    }
    if segs.is_empty() {
        return Err("empty segment list".into());
    }
    Ok(segs)
}

/// Window `[base, base + size)` (a whole `VirtualAlloc` allocation of its own) remade as driver
/// heap(s) mapped at `base`, its pages as they were. On failure the window is the title's memory
/// again, its pages as they were.
pub(super) unsafe fn remake(device: &ID3D12Device, base: usize, size: usize, mode: &Mode) -> Result<Remade, String> {
    match mode {
        Mode::Single { upload } => remake_single(device, base, size, *upload),
        Mode::Mixed(ranges) => remake_mixed(device, base, size, ranges),
    }
}

unsafe fn remake_single(device: &ID3D12Device, base: usize, size: usize, upload: bool) -> Result<Remade, String> {
    let t0 = std::time::Instant::now();
    let pages = written_pages(base, size)?;
    let kept: usize = pages.iter().map(|p| p.1.len()).sum();
    if VirtualFree(base as *mut c_void, 0, MEM_RELEASE) == 0 {
        return Err(format!("VirtualFree of the window at {base:#x} failed"));
    }
    let restore = |why: String| -> String {
        let p = VirtualAlloc(base as *const c_void, size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
        if p as usize != base {
            return format!("{why}; and the window could not be made again at {base:#x} (got {p:?})");
        }
        for (a, bytes) in &pages {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), *a as *mut u8, bytes.len());
        }
        why
    };
    let groomed = groom_with(base, size, 1 << 30, 256 << 20);
    let reservations = groomed.len();
    let want_watch = std::env::var("XCL_UMA_DRIVER_WATCH").map_or(true, |v| v.trim() != "0");
    let made = create_mapped_segment(device, size, upload, want_watch);
    ungroom(groomed);
    match made {
        Ok((heap, buf, p, watched)) if p == base => {
            for (a, bytes) in &pages {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), *a as *mut u8, bytes.len());
            }
            crate::xlog_info!(
                "D3D12.X: XCL_UMA_DRIVER: window at {base:#x} ({} MB) is a {} heap mapped at its address, {} ({reservations} reservations, {} pages kept, {} KB, {} ms)",
                size >> 20,
                if upload { "GPU_UPLOAD" } else { "CUSTOM write-back L0" },
                if watched { "write-watched" } else { "not write-watched" },
                pages.len(),
                kept >> 10,
                t0.elapsed().as_millis()
            );
            let gpu_va = buf.GetGPUVirtualAddress();
            let heap_raw = heap.into_raw() as usize;
            let buf_raw = buf.into_raw() as usize;
            Ok(Remade {
                buf_raw,
                gpu_va,
                heaps: vec![(0, size, heap_raw)],
                watched_regions: if watched { vec![(base, size)] } else { Vec::new() },
                all_watched: watched,
            })
        }
        Ok((heap, buf, p, _)) => {
            buf.Unmap(0, None);
            drop((buf, heap));
            Err(restore(format!("Map gave {p:#x}, not the window's {base:#x} ({reservations} reservations)")))
        }
        Err(e) => Err(restore(e)),
    }
}

unsafe fn remake_mixed(device: &ID3D12Device, base: usize, size: usize, vram_ranges: &[(usize, usize)]) -> Result<Remade, String> {
    let t0 = std::time::Instant::now();
    let spec = build_segments(size, vram_ranges)?;
    let min_seg = spec.iter().map(|s| s.1).min().unwrap_or(64 << 20);
    let pages = written_pages(base, size)?;
    let kept: usize = pages.iter().map(|p| p.1.len()).sum();
    if VirtualFree(base as *mut c_void, 0, MEM_RELEASE) == 0 {
        return Err(format!("VirtualFree of the window at {base:#x} failed"));
    }
    let restore = |why: String, placed: Vec<(usize, usize, bool, bool, ID3D12Heap, ID3D12Resource)>, placeholders: Vec<usize>| -> String {
        for (_, _, _, _, h, b) in placed {
            b.Unmap(0, None);
            drop((b, h));
        }
        for p in placeholders {
            if p != 0 {
                VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
            }
        }
        let p = VirtualAlloc(base as *const c_void, size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
        if p as usize != base {
            return format!("{why}; and the window could not be made again at {base:#x} (got {p:?})");
        }
        for (a, bytes) in &pages {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), *a as *mut u8, bytes.len());
        }
        why
    };

    // Reserve each segment slot inside [base, base + size) independently so segment i's Map()
    // cannot land in segment j's slot.
    let mut placeholders: Vec<usize> = Vec::with_capacity(spec.len());
    for &(off, len, _) in &spec {
        let want = base + off;
        let p = VirtualAlloc(want as *const c_void, len, MEM_RESERVE, PAGE_NOACCESS);
        if p as usize != want {
            if !p.is_null() {
                VirtualFree(p, 0, MEM_RELEASE);
            }
            return Err(restore(format!("placeholder reserve at {want:#x} ({} MB) got {p:?}", len >> 20), Vec::new(), placeholders));
        }
        placeholders.push(want);
    }

    // Groom every free stretch outside [base, base + size) >= min_seg, leaving 16 MiB headroom
    // for small driver internal allocations so no stray hole >= min_seg exists elsewhere.
    let headroom = (16 << 20).min(min_seg / 4);
    let groomed = groom_with(base, size, min_seg, headroom);
    let reservations = groomed.len();
    let want_watch = std::env::var("XCL_UMA_DRIVER_WATCH").map_or(true, |v| v.trim() != "0");

    let mut placed: Vec<(usize, usize, bool, bool, ID3D12Heap, ID3D12Resource)> = Vec::with_capacity(spec.len());
    let mut fail: Option<String> = None;
    for (idx, &(off, len, upload)) in spec.iter().enumerate() {
        let want = base + off;
        let ph = std::mem::replace(&mut placeholders[idx], 0);
        if VirtualFree(ph as *mut c_void, 0, MEM_RELEASE) == 0 {
            fail = Some(format!("VirtualFree placeholder {want:#x} failed"));
            break;
        }
        // Do not request ALLOW_WRITE_WATCH on GPU_UPLOAD segments: PAGE_WRITECOMBINE BAR mappings
        // refuse VirtualProtect(PAGE_READONLY) in watch_trap (causing tens of thousands of retries
        // and ~8 ms/look GetWriteWatch scans); unwatched textures in GPU_UPLOAD are still uploaded
        // on creation via read_vram_bytes DMA and rechecked at spaced milestones.
        let seg_watch = want_watch && !upload;
        match create_mapped_segment(device, len, upload, seg_watch) {
            Ok((heap, buf, p, watched)) if p == want => {
                placed.push((off, len, upload, watched, heap, buf));
            }
            Ok((heap, buf, p, _)) => {
                buf.Unmap(0, None);
                drop((buf, heap));
                fail = Some(format!("segment {idx} ({} MB {}) Map gave {p:#x}, wanted {want:#x}", len >> 20, if upload { "GPU_UPLOAD" } else { "CUSTOM_L0" }));
                break;
            }
            Err(e) => {
                fail = Some(format!("segment {idx} ({} MB {}): {e}", len >> 20, if upload { "GPU_UPLOAD" } else { "CUSTOM_L0" }));
                break;
            }
        }
    }
    ungroom(groomed);
    if let Some(why) = fail {
        return Err(restore(why, placed, placeholders));
    }

    // Create ONE reserved buffer over the whole window and map each segment's 64 KiB tiles onto
    // its heap via a temporary DIRECT queue + fence wait.
    let reserved_res = (|| -> Result<ID3D12Resource, String> {
        let mut res: Option<ID3D12Resource> = None;
        device
            .CreateReservedResource(&buffer_desc(size), D3D12_RESOURCE_STATE_COMMON, None, &mut res)
            .map_err(|e| format!("CreateReservedResource: {e}"))?;
        let res = res.ok_or("CreateReservedResource")?;
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })
            .map_err(|e| format!("CreateCommandQueue: {e}"))?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|e| format!("CreateFence: {e}"))?;
        for (off, len, _, _, heap, _) in &placed {
            let first = (*off / TILE) as u32;
            let count = (*len / TILE) as u32;
            let coord = D3D12_TILED_RESOURCE_COORDINATE { X: first, ..Default::default() };
            let region = D3D12_TILE_REGION_SIZE { NumTiles: count, UseBox: false.into(), ..Default::default() };
            let flag = D3D12_TILE_RANGE_FLAG_NONE;
            let heap_start = 0u32;
            queue.UpdateTileMappings(
                &res,
                1,
                Some(&coord),
                Some(&region),
                heap,
                1,
                Some(&flag),
                Some(&heap_start),
                Some(&count),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        queue.Signal(&fence, 1).map_err(|e| format!("Signal: {e}"))?;
        crate::gpu::fence::wait(&fence, 1, device, None, 2000).map_err(|e| format!("tile map wait: {e}"))?;
        Ok(res)
    })();
    let res = match reserved_res {
        Ok(r) => r,
        Err(why) => return Err(restore(why, placed, placeholders)),
    };

    for (a, bytes) in &pages {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), *a as *mut u8, bytes.len());
    }

    let mut desc_parts = Vec::with_capacity(placed.len());
    let mut heaps = Vec::with_capacity(placed.len());
    let mut watched_regions = Vec::new();
    let mut all_watched = true;
    for (off, len, upload, watched, heap, placed_buf) in placed {
        desc_parts.push(format!(
            "{:#x}+{}MB:{}{}",
            off,
            len >> 20,
            if upload { "GPU_UPLOAD" } else { "CUSTOM_L0" },
            if watched { "(ww)" } else { "" }
        ));
        if watched {
            watched_regions.push((base + off, len));
        } else {
            all_watched = false;
        }
        heaps.push((off, len, heap.into_raw() as usize));
        // Keep the mapped placed buffer alive for the lifetime of the process so the CPU mapping stays valid.
        std::mem::forget(placed_buf);
    }
    crate::xlog_info!(
        "D3D12.X: XCL_UMA_DRIVER: mixed reserved window at {base:#x} ({} MB) [{}] ({reservations} reservations, {} pages kept, {} KB, {} ms)",
        size >> 20,
        desc_parts.join(", "),
        pages.len(),
        kept >> 10,
        t0.elapsed().as_millis()
    );
    let vram_cpu_ranges: Vec<(usize, usize)> = spec.iter().filter(|s| s.2).map(|&(off, len, _)| (base + off, base + off + len)).collect();
    if !vram_cpu_ranges.is_empty() {
        init_vram_reader(device, base, size, &res, vram_cpu_ranges);
    }
    let gpu_va = res.GetGPUVirtualAddress();
    Ok(Remade { buf_raw: res.into_raw() as usize, gpu_va, heaps, watched_regions, all_watched })
}

static VRAM_READER_ACTIVE: AtomicBool = AtomicBool::new(false);
static VRAM_READER: Mutex<Option<VramReader>> = Mutex::new(None);

struct VramReader {
    window_base: usize,
    window_end: usize,
    vram_ranges: Vec<(usize, usize)>,
    device: ID3D12Device,
    src_buf: ID3D12Resource,
    queue: ID3D12CommandQueue,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    fence_val: u64,
    readback: ID3D12Resource,
    readback_ptr: *mut u8,
    readback_cap: usize,
}

unsafe impl Send for VramReader {}

unsafe fn make_readback(device: &ID3D12Device, cap: usize) -> Result<(ID3D12Resource, *mut u8), String> {
    let mut rb: Option<ID3D12Resource> = None;
    device
        .CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: cap as u64,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
        .map_err(|e| format!("readback CreateCommittedResource({cap}): {e}"))?;
    let rb = rb.ok_or("readback")?;
    let mut p: *mut c_void = std::ptr::null_mut();
    rb.Map(0, None, Some(&mut p)).map_err(|e| format!("readback Map: {e}"))?;
    Ok((rb, p as *mut u8))
}

unsafe fn init_vram_reader(device: &ID3D12Device, window_base: usize, window_size: usize, src_buf: &ID3D12Resource, vram_ranges: Vec<(usize, usize)>) {
    let res = (|| -> Result<VramReader, String> {
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COPY, ..Default::default() })
            .map_err(|e| format!("copy queue: {e}"))?;
        let alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY).map_err(|e| format!("copy alloc: {e}"))?;
        let list: ID3D12GraphicsCommandList =
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &alloc, None).map_err(|e| format!("copy list: {e}"))?;
        list.Close().map_err(|e| format!("copy list close: {e}"))?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|e| format!("copy fence: {e}"))?;
        let cap = 32 << 20;
        let (readback, readback_ptr) = make_readback(device, cap)?;
        Ok(VramReader {
            window_base,
            window_end: window_base + window_size,
            vram_ranges,
            device: device.clone(),
            src_buf: src_buf.clone(),
            queue,
            alloc,
            list,
            fence,
            fence_val: 0,
            readback,
            readback_ptr,
            readback_cap: cap,
        })
    })();
    match res {
        Ok(r) => {
            *VRAM_READER.lock().unwrap() = Some(r);
            VRAM_READER_ACTIVE.store(true, Ordering::Release);
        }
        Err(e) => crate::xlog_warn!("D3D12.X: XCL_UMA_DRIVER: VRAM DMA reader disabled: {e}"),
    }
}

/// Read `[at, at + len)` via GPU DMA (`CopyBufferRegion` to a `READBACK` buffer) when it overlaps
/// a `GPU_UPLOAD` VRAM segment of a mixed driver window. Avoids ~12-84 MiB/s CPU BAR reads during
/// `title_textures` hashing and detiling. Returns `None` if `[at, at + len)` does not overlap VRAM.
pub(super) unsafe fn read_vram_bytes(at: usize, len: usize) -> Option<Vec<u8>> {
    if len == 0 || !VRAM_READER_ACTIVE.load(Ordering::Acquire) {
        return None;
    }
    let mut g = VRAM_READER.lock().unwrap();
    let r = g.as_mut()?;
    let end = at.checked_add(len)?;
    if at < r.window_base || end > r.window_end {
        return None;
    }
    if !r.vram_ranges.iter().any(|&(s, e)| at < e && end > s) {
        return None;
    }
    if len > r.readback_cap {
        let new_cap = len.next_power_of_two().max(32 << 20);
        let (rb, ptr) = make_readback(&r.device, new_cap).ok()?;
        r.readback.Unmap(0, None);
        r.readback = rb;
        r.readback_ptr = ptr;
        r.readback_cap = new_cap;
    }
    r.alloc.Reset().ok()?;
    r.list.Reset(&r.alloc, None).ok()?;
    r.list.CopyBufferRegion(&r.readback, 0, &r.src_buf, (at - r.window_base) as u64, len as u64);
    r.list.Close().ok()?;
    let cl: ID3D12CommandList = r.list.cast().ok()?;
    r.queue.ExecuteCommandLists(&[Some(cl)]);
    r.fence_val += 1;
    let fv = r.fence_val;
    r.queue.Signal(&r.fence, fv).ok()?;
    crate::gpu::fence::wait(&r.fence, fv, &r.device, None, 5000).ok()?;
    let mut out = vec![0u8; len];
    std::ptr::copy_nonoverlapping(r.readback_ptr, out.as_mut_ptr(), len);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mode_variants() {
        assert_eq!(parse_mode("custom"), Some(Mode::Single { upload: false }));
        assert_eq!(parse_mode("upload"), Some(Mode::Single { upload: true }));
        assert_eq!(parse_mode("mixed:1536:1792"), Some(Mode::Mixed(vec![(1536 << 20, 1792 << 20)])));
        assert_eq!(parse_mode("mixed:2048:512,512:256"), Some(Mode::Mixed(vec![(512 << 20, 256 << 20), (2048 << 20, 512 << 20)])));
        assert_eq!(parse_mode("other"), None);
    }

    #[test]
    fn build_segments_splits_cleanly() {
        let segs = build_segments(4095 << 20, &[(1536 << 20, 1792 << 20)]).unwrap();
        assert_eq!(segs, vec![(0, 1536 << 20, false), (1536 << 20, 1792 << 20, true), (3328 << 20, 767 << 20, false)]);
    }

    #[test]
    fn mixed_reserved_window_places_and_is_coherent() {
        let dev: ID3D12Device = unsafe {
            let mut d: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0, &mut d).is_err() {
                return;
            }
            let Some(d) = d else { return };
            d
        };
        unsafe {
            let size = 256 << 20;
            let base = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            // Seed a non-zero page in each of the 3 segments before remake
            std::ptr::write_volatile((base + 0x1000) as *mut u32, 0x1111_2222);
            std::ptr::write_volatile((base + (64 << 20) + 0x1000) as *mut u32, 0x3333_4444);
            std::ptr::write_volatile((base + (192 << 20) + 0x1000) as *mut u32, 0x5555_6666);

            let mode = Mode::Mixed(vec![(64 << 20, 128 << 20)]);
            let remade = remake(&dev, base, size, &mode).expect("remake_mixed failed");
            assert_eq!(remade.heaps.len(), 3);
            assert_eq!(std::ptr::read_volatile((base + 0x1000) as *const u32), 0x1111_2222);
            assert_eq!(std::ptr::read_volatile((base + (64 << 20) + 0x1000) as *const u32), 0x3333_4444);
            assert_eq!(std::ptr::read_volatile((base + (192 << 20) + 0x1000) as *const u32), 0x5555_6666);

            // Verify GPU sees new CPU writes through the single reserved buffer across all 3 segments
            std::ptr::write_volatile((base + 0x2000) as *mut u32, 0xCAFE_0001);
            std::ptr::write_volatile((base + (64 << 20) + 0x2000) as *mut u32, 0xCAFE_0002);
            std::ptr::write_volatile((base + (192 << 20) + 0x2000) as *mut u32, 0xCAFE_0003);

            let queue: ID3D12CommandQueue =
                dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut rb: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: 256,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut rb,
            )
            .unwrap();
            let rb = rb.unwrap();
            let raw_ptr = remade.buf_raw as *mut c_void;
            let src = ID3D12Resource::from_raw_borrowed(&raw_ptr).unwrap();
            list.CopyBufferRegion(&rb, 0, src, 0x2000, 4);
            list.CopyBufferRegion(&rb, 4, src, (64 << 20) + 0x2000, 4);
            list.CopyBufferRegion(&rb, 8, src, (192 << 20) + 0x2000, 4);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            queue.Signal(&fence, 1).unwrap();
            crate::gpu::fence::wait(&fence, 1, &dev, None, 2000).unwrap();
            let mut m = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut m)).unwrap();
            let got = std::slice::from_raw_parts(m as *const u32, 3).to_vec();
            rb.Unmap(0, None);
            assert_eq!(got, vec![0xCAFE_0001, 0xCAFE_0002, 0xCAFE_0003]);

            // Verify DMA read_vram_bytes reads the GPU_UPLOAD segment cleanly and returns None for CUSTOM_L0
            assert!(read_vram_bytes(base + 0x2000, 4).is_none());
            let dma = read_vram_bytes(base + (64 << 20) + 0x2000, 4).expect("DMA read of GPU_UPLOAD failed");
            assert_eq!(u32::from_le_bytes(dma.try_into().unwrap()), 0xCAFE_0002);
        }
    }
}
