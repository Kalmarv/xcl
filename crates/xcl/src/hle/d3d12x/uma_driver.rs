//! (scratch, `XCL_UMA_DRIVER=custom|upload`) A coherent UMA window in the driver's memory instead
//! of the title's own pages opened with `OpenExistingHeapFromAddress`: a whole-frame replay ran
//! window 3's programs 4-270x faster from a driver CUSTOM L0 heap than from the imported memory
//! (RT1 vs RH-imp). The title keeps its addresses: the window's pages are copied out, the window
//! is released, every other large free stretch of address space is reserved for a moment, so the
//! driver's CPU mapping of the new heap can only land in the hole the window left, and the pages
//! go back. The heap asks for D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH, so CPU writes to the window are
//! still found by `GetWriteWatch` (A10: CUSTOM write-back L0 heaps of 64 KiB and 4 GiB report
//! written pages and reset on the RTX 5070 Ti); a heap made without it (`XCL_UMA_DRIVER_WATCH=0`,
//! or a driver refusing the flag) is not write-watched (`uma::set_driver_window`).

use std::ffi::c_void;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_FREE, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS,
    PAGE_READWRITE,
};

const MEM_WRITE_WATCH: u32 = 0x20_0000;

#[link(name = "kernel32")]
extern "system" {
    fn GetWriteWatch(flags: u32, base: *const c_void, size: usize, addresses: *mut *mut c_void, count: *mut usize, granularity: *mut u32) -> u32;
}

/// `custom` (CUSTOM heap, write-back CPU pages, L0: host memory) or `upload` (GPU_UPLOAD: video
/// memory the CPU writes through the BAR); None: the import as before.
pub(super) fn mode() -> Option<bool> {
    match std::env::var("XCL_UMA_DRIVER").ok()?.trim() {
        "custom" => Some(false),
        "upload" => Some(true),
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

/// Reserves every free stretch of 1 GiB or more (but its first 256 MiB, room for the driver's own
/// small mappings) outside `[hole, hole + len)`.
unsafe fn groom(hole: usize, len: usize) -> Vec<usize> {
    let (top, mut a, mut got) = (0x7FFF_FFFE_0000usize, 0x1_0000usize, Vec::new());
    while a < top {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(a as *const c_void, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) == 0 {
            break;
        }
        let (base, end) = (m.BaseAddress as usize, (m.BaseAddress as usize + m.RegionSize).min(top));
        a = end;
        if m.State != MEM_FREE || m.RegionSize < 1 << 30 {
            continue;
        }
        let s = ((base + 0xFFFF) & !0xFFFF) + (256 << 20);
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

/// Window `[base, base + size)` (a whole `VirtualAlloc` allocation of its own) remade as a driver
/// heap mapped at `base`, its pages as they were: the heap, a buffer over all of it, mapped for
/// good, and whether the heap is write-watched. On failure the window is the title's memory again,
/// its pages as they were.
pub(super) unsafe fn remake(device: &ID3D12Device, base: usize, size: usize, upload: bool) -> Result<(ID3D12Heap, ID3D12Resource, bool), String> {
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
    let groomed = groom(base, size);
    let reservations = groomed.len();
    let want_watch = std::env::var("XCL_UMA_DRIVER_WATCH").map_or(true, |v| v.trim() != "0");
    let made = (|| -> Result<(ID3D12Heap, ID3D12Resource, usize, bool), String> {
        let properties = if upload {
            D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE(5), ..Default::default() }
        } else {
            D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            }
        };
        let desc = |flags| D3D12_HEAP_DESC { SizeInBytes: size as u64, Properties: properties, Alignment: 0x1_0000, Flags: flags };
        let mut heap: Option<ID3D12Heap> = None;
        let mut watched = want_watch && device.CreateHeap(&desc(D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH), &mut heap).is_ok() && heap.is_some();
        if !watched {
            heap = None;
            device.CreateHeap(&desc(D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS), &mut heap).map_err(|e| format!("CreateHeap: {e}"))?;
            watched = false;
        }
        let heap = heap.ok_or("CreateHeap")?;
        let buf_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buf: Option<ID3D12Resource> = None;
        device.CreatePlacedResource(&heap, 0, &buf_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf).map_err(|e| format!("CreatePlacedResource: {e}"))?;
        let buf = buf.ok_or("CreatePlacedResource")?;
        let mut p: *mut c_void = std::ptr::null_mut();
        buf.Map(0, Some(&D3D12_RANGE::default()), Some(&mut p)).map_err(|e| format!("Map: {e}"))?;
        Ok((heap, buf, p as usize, watched))
    })();
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
            Ok((heap, buf, watched))
        }
        Ok((heap, buf, p, _)) => {
            buf.Unmap(0, None);
            drop((buf, heap));
            Err(restore(format!("Map gave {p:#x}, not the window's {base:#x} ({reservations} reservations)")))
        }
        Err(e) => Err(restore(e)),
    }
}

/// `heap` and `buf` as the raw pointers `open_uma` keeps.
pub(super) fn raw(heap: ID3D12Heap, buf: ID3D12Resource) -> (usize, u64, usize) {
    let va = unsafe { buf.GetGPUVirtualAddress() };
    (buf.into_raw() as usize, va, heap.into_raw() as usize)
}
