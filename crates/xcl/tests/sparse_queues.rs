//! CPU writes into a sparse window go to the GPU on XCL's upload queue (`sparse::before_execute`).
//! A window is one buffer, so while the title's queue still has a write into it in flight, the
//! upload is a second writer on another queue – the debug layer's "still referenced by write GPU
//! operations in-flight on another Command Queue" (OBJECT_ACCESSED_WHILE_STILL_IN_USE, 1047).
//! Its own process: the debug layer has to be on before the first device exists.

use windows::core::Interface;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

use xcl::hle::d3d12x::sparse;

/// The debug layer's messages with id `id` since the last call (then all cleared).
unsafe fn messages(q: &ID3D12InfoQueue, id: D3D12_MESSAGE_ID) -> Vec<String> {
    let mut out = Vec::new();
    for i in 0..q.GetNumStoredMessages() {
        let mut len = 0usize;
        if q.GetMessage(i, None, &mut len).is_err() || len == 0 {
            continue;
        }
        let mut buf = vec![0u64; len.div_ceil(8)];
        let m = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
        if q.GetMessage(i, Some(m), &mut len).is_ok() && (*m).ID == id {
            let d = std::slice::from_raw_parts((*m).pDescription as *const u8, (*m).DescriptionByteLength.saturating_sub(1));
            out.push(String::from_utf8_lossy(d).into_owned());
        }
    }
    q.ClearStoredMessages();
    out
}

/// The title's queue is still writing the first half of a sparse allocation (copies that take a
/// while) when the CPU's write to the second half is uploaded: the upload must wait for it.
/// Other bytes, the same buffer – D3D12 allows one writer at a time.
#[test]
fn uploads_wait_for_the_title_queues_writes_into_the_window() {
    unsafe {
        match xcl::gpu::agility::configuration_interface::<ID3D12Debug>(&CLSID_D3D12Debug) {
            Ok(d) => d.EnableDebugLayer(),
            Err(e) => {
                println!("SKIPPED: no D3D12 debug layer ({e})");
                return;
            }
        }
        std::env::set_var("XCL_UMA_MB", "64");
        std::env::set_var("XCL_COHERENT_MB", "64");
        let Ok(dev) = xcl::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
            println!("SKIPPED: no GPU");
            return;
        };
        let Ok(info) = dev.cast::<ID3D12InfoQueue>() else {
            println!("SKIPPED: the device has no debug layer");
            return;
        };
        let windows = sparse::init(&dev);
        let Some(&(k, raw, _)) = windows.first() else {
            println!("SKIPPED: no sparse windows");
            return;
        };
        let raw = raw as *mut std::ffi::c_void;
        let window = ID3D12Resource::from_raw_borrowed(&raw).unwrap();
        let base = xcl::gpu::uma::get().unwrap().window(k).0;
        const HALF: usize = 32 << 20;
        let (a, _) = sparse::alloc_in(k, 2 * HALF, 0x1_0000).expect("sparse allocation");
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let drain = || {
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&done, 1).unwrap();
            done.SetEventOnCompletion(1, HANDLE::default()).unwrap();
        };
        // as before each of the title's submits (the tile mappings)
        sparse::before_execute(&queue, &[], 0x1000);
        drain();
        let setup = messages(&info, D3D12_MESSAGE_ID_OBJECT_ACCESSED_WHILE_STILL_IN_USE);

        // the title's write into the first half: 200 copies of a buffer of its own
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: HALF as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut src: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut src).unwrap();
        let src = src.unwrap();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        for _ in 0..200 {
            list.CopyBufferRegion(window, (a - base) as u64, &src, 0, HALF as u64);
        }
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);

        // meanwhile the CPU writes the second half, and the title submits again
        std::ptr::write_bytes((a + HALF) as *mut u8, 0x5A, 0x1000);
        sparse::before_execute(&queue, &[a + HALF], 0x1000);
        let upload = messages(&info, D3D12_MESSAGE_ID_OBJECT_ACCESSED_WHILE_STILL_IN_USE);
        drain();
        println!("setup: {} messages 1047; upload: {} messages 1047, first: {:?}", setup.len(), upload.len(), upload.first());
        assert!(setup.is_empty(), "setup: {setup:?}");
        assert!(upload.is_empty(), "the upload started while the title's write was in flight: {upload:?}");
    }
}
