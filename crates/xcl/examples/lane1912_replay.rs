//! GPU dispatch and complete resident-resource readback for scripts/lane1912.
//! Each invocation owns a device; a failed fence or device removal ends the invocation.
use std::mem::{transmute_copy, ManuallyDrop};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};
use windows::core::Interface;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};
use xcl::gpu::{self, recomp};
const TILE: u64 = 65536;
// probe.py uses the same descriptor and diagnostic-buffer layout. REQ/BOUND/PAGES are dword offsets.
const LIMIT: usize = 16384;
const REQ: usize = 256;
const BOUND: usize = REQ + LIMIT;
const PAGES: usize = BOUND + LIMIT;
const RES_WORDS: usize = PAGES + 8192;
fn err(x: windows::core::Error) -> String {
    format!("{x}")
}

fn buffer(dev: &ID3D12Device, size: u64, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES, uav: bool) -> Result<ID3D12Resource, String> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
        ..Default::default()
    };
    let mut r: Option<ID3D12Resource> = None;
    unsafe {
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r)
            .map_err(err)?
    };
    r.ok_or_else(|| "buffer".into())
}

fn transition(r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
}

struct Gpu {
    dev: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
}

impl Gpu {
    unsafe fn submit(&mut self, list: &ID3D12GraphicsCommandList) -> Result<(), String> {
        list.Close().map_err(err)?;
        self.queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
        self.value += 1;
        self.queue.Signal(&self.fence, self.value).map_err(err)?;
        let start = std::time::Instant::now();
        while self.fence.GetCompletedValue() < self.value {
            if start.elapsed().as_secs() > 25 {
                return Err("GPU fence timeout".into());
            }
            self.dev.GetDeviceRemovedReason().map_err(err)?;
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        self.dev.GetDeviceRemovedReason().map_err(|e| format!("device removed: {e}"))
    }
}

fn subresources(d: &D3D12_RESOURCE_DESC) -> u32 {
    d.MipLevels as u32 * if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { d.DepthOrArraySize as u32 }
}

/// The copies between `tex` (all subresources) and its footprints in `buf` at `at` (from `buf` when `upload`).
unsafe fn texture_copies(dev: &ID3D12Device, list: &ID3D12GraphicsCommandList, tex: &ID3D12Resource, buf: &ID3D12Resource, at: u64, upload: bool) {
    let d = tex.GetDesc();
    let n = subresources(&d);
    let mut fp = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
    dev.GetCopyableFootprints(&d, 0, n, at, Some(fp.as_mut_ptr()), None, None, None);
    for (k, f) in fp.iter().enumerate() {
        let a = D3D12_TEXTURE_COPY_LOCATION {
            pResource: transmute_copy(buf),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *f },
        };
        let t = D3D12_TEXTURE_COPY_LOCATION {
            pResource: transmute_copy(tex),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k as u32 },
        };
        if upload {
            list.CopyTextureRegion(&t, 0, 0, 0, &a, None);
        } else {
            list.CopyTextureRegion(&a, 0, 0, 0, &t, None);
        }
    }
}

fn texture_bytes(dev: &ID3D12Device, d: &D3D12_RESOURCE_DESC) -> u64 {
    let mut total = 0u64;
    unsafe { dev.GetCopyableFootprints(d, 0, subresources(d), 0, None, None, None, Some(&mut total)) };
    total
}

struct Fixture {
    io: Vec<u32>,
    res: Vec<u32>,
    ranges: Vec<(usize, u64, u64)>,
    before: Vec<u8>,
    layout: BTreeMap<u32, u32>,
    seed: u64,
    groups: [u32; 3],
}
unsafe fn dispatch(gpu: &mut Gpu, pso: &ID3D12PipelineState, rs: &ID3D12RootSignature, c: &Fixture) -> Result<(Vec<u32>, Vec<u32>), String> {
    let dev = gpu.dev.clone();
    let groups = c.groups;
    let io_bytes = (c.io.len() * 4) as u64;
    let mem_bytes = c.before.len() as u64;
    // Reserve the entire 32-bit offset space in each window. Back and read back the discovered tiles.
    let mut tiles: Vec<Vec<u32>> = vec![Vec::new(); 4];
    for &(k, off, n) in &c.ranges {
        tiles[k].extend((off / TILE) as u32..(off + n).div_ceil(TILE) as u32);
    }
    let mut ntiles = 0u32;
    for t in tiles.iter_mut() {
        t.sort_unstable();
        t.dedup();
        ntiles += t.len() as u32;
    }
    let pool: ID3D12Heap = {
        let mut h: Option<ID3D12Heap> = None;
        dev.CreateHeap(
            &D3D12_HEAP_DESC {
                SizeInBytes: ntiles.max(1) as u64 * TILE,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
            },
            &mut h,
        )
        .map_err(err)?;
        h.ok_or("tile pool")?
    };
    let mut wins: Vec<ID3D12Resource> = Vec::new();
    let mut next = 0u32;
    for k in 0..4 {
        let bytes = 1u64 << 32;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut r: Option<ID3D12Resource> = None;
        dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r)
            .map_err(|e| format!("window {k}: CreateReservedResource ({bytes:#x} B): {e}"))?;
        let r = r.ok_or("reserved")?;
        let t = &tiles[k];
        let mut i = 0;
        let (mut coords, mut regions, mut offsets, mut counts) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        while i < t.len() {
            let mut run = 1u32;
            while i + (run as usize) < t.len() && t[i + run as usize] == t[i] + run {
                run += 1;
            }
            coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: t[i], ..Default::default() });
            regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
            offsets.push(next);
            counts.push(run);
            next += run;
            i += run as usize;
        }
        if !coords.is_empty() {
            let flags = vec![D3D12_TILE_RANGE_FLAG_NONE; coords.len()];
            gpu.queue.UpdateTileMappings(
                &r,
                coords.len() as u32,
                Some(coords.as_ptr()),
                Some(regions.as_ptr()),
                &pool,
                flags.len() as u32,
                Some(flags.as_ptr()),
                Some(offsets.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        wins.push(r);
    }

    let mut texs = Vec::new();
    let mut tex_total = 0;
    for (&idx, &kind) in &c.layout {
        let volume = kind >= 8;
        let writable = kind % 8 >= 4;
        let format = match kind % 4 {
            0 => DXGI_FORMAT_R32G32B32A32_FLOAT,
            1 => DXGI_FORMAT_R32G32B32A32_SINT,
            2 => DXGI_FORMAT_R32G32B32A32_UINT,
            _ => DXGI_FORMAT_R32_UINT,
        };
        let width = if kind % 4 == 3 { 64 } else { 16 };
        let rd = D3D12_RESOURCE_DESC {
            Dimension: if volume { D3D12_RESOURCE_DIMENSION_TEXTURE3D } else { D3D12_RESOURCE_DIMENSION_TEXTURE2D },
            Width: width,
            Height: 16,
            DepthOrArraySize: if volume { 8 } else { 1 },
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: if writable { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
            ..Default::default()
        };
        let mut tx: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &rd,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut tx,
        )
        .map_err(err)?;
        let at = tex_total;
        tex_total += texture_bytes(&dev, &rd);
        texs.push((idx, tx.ok_or("texture")?, at, kind));
    }
    let tex_state = |t: &ID3D12Resource| {
        if t.GetDesc().Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0 {
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        }
    };
    let tex_upload = buffer(&dev, tex_total.max(256), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false)?;
    let tex_in: Vec<u32> = {
        let mut r = Rng(c.seed | 1);
        let words: Vec<u32> = (0..tex_total / 4).map(|_| (r.next() & 0x007F_FFFF) | 0x3F00_0000).collect();
        let mut p = std::ptr::null_mut();
        tex_upload.Map(0, None, Some(&mut p)).map_err(err)?;
        std::ptr::copy_nonoverlapping(words.as_ptr(), p as *mut u32, words.len());
        tex_upload.Unmap(0, None);
        words
    };
    let tex_readback = buffer(&dev, tex_total.max(256), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false)?;
    let res_bytes = (c.res.len() * 4) as u64;
    let upload = buffer(&dev, io_bytes + mem_bytes + res_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false)?;
    let mut p = std::ptr::null_mut();
    upload.Map(0, None, Some(&mut p)).map_err(err)?;
    std::ptr::copy_nonoverlapping(c.io.as_ptr(), p as *mut u32, c.io.len());
    std::ptr::copy_nonoverlapping(c.before.as_ptr(), (p as *mut u8).add(io_bytes as usize), c.before.len());
    std::ptr::copy_nonoverlapping(c.res.as_ptr(), (p as *mut u8).add((io_bytes + mem_bytes) as usize) as *mut u32, c.res.len());
    upload.Unmap(0, None);
    let io_buf = buffer(&dev, io_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
    let res_buf = buffer(&dev, res_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
    let out_bytes = io_bytes + res_bytes + mem_bytes;
    let readback = buffer(&dev, out_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false)?;

    let heap: ID3D12DescriptorHeap = dev
        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: (LIMIT + 16) as u32,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            ..Default::default()
        })
        .map_err(err)?;
    let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
    for (idx, t, _, kind) in &texs {
        let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: heap.GetCPUDescriptorHandleForHeapStart().ptr + *idx as usize * inc };
        if kind % 8 >= 4 {
            dev.CreateUnorderedAccessView(t, None::<&ID3D12Resource>, None, at)
        } else {
            dev.CreateShaderResourceView(t, None, at)
        }
    }
    let sheap: ID3D12DescriptorHeap = dev
        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            NumDescriptors: 2048,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
        .map_err(err)?;
    let sinc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
    let sd = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    for k in 0..2048usize {
        dev.CreateSampler(&sd, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: sheap.GetCPUDescriptorHandleForHeapStart().ptr + k * sinc });
    }
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(err)?;

    let bind = |l: &ID3D12GraphicsCommandList| {
        l.SetDescriptorHeaps(&[Some(heap.clone()), Some(sheap.clone())]);
        l.SetComputeRootSignature(rs);
        l.SetComputeRootUnorderedAccessView(0, io_buf.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(1, res_buf.GetGPUVirtualAddress());
        for (i, w) in wins.iter().enumerate() {
            l.SetComputeRootUnorderedAccessView(2 + i as u32, w.GetGPUVirtualAddress());
        }
    };
    let all = |before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES| -> Vec<D3D12_RESOURCE_BARRIER> {
        std::iter::once(&io_buf).chain(wins.iter()).map(|r| transition(r, before, after)).collect()
    };
    // 1: one dispatch on fresh inputs, outputs read back
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, pso).map_err(err)?;
    let mut b = all(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    b.push(transition(&res_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST));
    list.ResourceBarrier(&b);
    for (_, t, at, _) in &texs {
        texture_copies(&dev, &list, t, &tex_upload, *at, true);
    }
    let tb: Vec<_> = texs.iter().map(|(_, t, ..)| transition(t, D3D12_RESOURCE_STATE_COPY_DEST, tex_state(t))).collect();
    if !tb.is_empty() {
        list.ResourceBarrier(&tb);
    }
    list.CopyBufferRegion(&res_buf, 0, &upload, io_bytes + mem_bytes, res_bytes);
    list.ResourceBarrier(&[transition(&res_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
    list.CopyBufferRegion(&io_buf, 0, &upload, 0, io_bytes);
    let mut src = io_bytes;
    for &(window, offset, bytes) in &c.ranges {
        list.CopyBufferRegion(&wins[window], offset, &upload, src, bytes);
        src += bytes;
    }
    list.ResourceBarrier(&all(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    bind(&list);
    list.Dispatch(groups[0], groups[1], groups[2]);
    let mut b = all(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    b.push(transition(&res_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));
    list.ResourceBarrier(&b);
    list.CopyBufferRegion(&readback, 0, &io_buf, 0, io_bytes);
    list.CopyBufferRegion(&readback, io_bytes, &res_buf, 0, res_bytes);
    let mut dst = io_bytes + res_bytes;
    for &(k, off, n) in &c.ranges {
        list.CopyBufferRegion(&readback, dst, &wins[k], off, n);
        dst += n;
    }
    let tb: Vec<_> = texs.iter().map(|(_, t, ..)| transition(t, tex_state(t), D3D12_RESOURCE_STATE_COPY_SOURCE)).collect();
    if !tb.is_empty() {
        list.ResourceBarrier(&tb);
    }
    for (_, t, at, _) in &texs {
        texture_copies(&dev, &list, t, &tex_readback, *at, false);
    }
    let tb: Vec<_> = texs.iter().map(|(_, t, ..)| transition(t, D3D12_RESOURCE_STATE_COPY_SOURCE, tex_state(t))).collect();
    if !tb.is_empty() {
        list.ResourceBarrier(&tb);
    }
    let mut b = all(D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    b.push(transition(&res_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    list.ResourceBarrier(&b);
    gpu.submit(&list)?;
    let mut q = std::ptr::null_mut();
    readback.Map(0, None, Some(&mut q)).map_err(err)?;
    let mut out = std::slice::from_raw_parts(q as *const u32, (out_bytes / 4) as usize).to_vec();
    readback.Unmap(0, None);
    tex_readback.Map(0, None, Some(&mut q)).map_err(err)?;
    out.extend_from_slice(std::slice::from_raw_parts(q as *const u32, (tex_total / 4) as usize));
    tex_readback.Unmap(0, None);
    let words = |b: &[u8]| b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<u32>>();
    let initial: Vec<u32> = c.io.iter().copied().chain(c.res.iter().copied()).chain(words(&c.before)).chain(tex_in).collect();
    Ok((out, initial))
}
fn input_word(r: &mut Rng, i: usize, mode: u64) -> u32 {
    let x = r.next();
    match (i % 4, mode % 3) {
        (0, 0) => (1 + x % 120) * 16,
        (0, 1) => (1 + x % 15) * 4,
        (0, _) => 4 + 2 * (x % 32),
        (1, _) => x % 4,
        (2, _) => 1 + x % 64,
        (_, 0) => x % 16,
        (_, 1) => x % 8,
        _ => x % 32,
    }
}
fn fixture(seed: u64, mode: u64, tiles: &BTreeSet<(usize, u32)>, layout: &BTreeMap<u32, u32>, probe: bool) -> Fixture {
    let mut rng = Rng(seed ^ 0x857b675901);
    let mut io: Vec<u32> = (0..128).map(|i| if i % 2 == 0 { input_word(&mut rng, 0, mode) } else { rng.next() % 4 }).collect();
    io[112] = 0;
    let mut res: Vec<u32> = (0..64).map(|_| rng.next()).collect();
    if probe {
        res.resize(RES_WORDS, 0);
        for (&idx, &kind) in layout {
            if idx < LIMIT as u32 {
                res[BOUND + idx as usize] = kind + 1;
            }
        }
    }
    let mut ranges = Vec::new();
    let mut before = Vec::new();
    for &(w, p) in tiles {
        ranges.push((w, u64::from(p) * TILE, TILE));
        let mut rg = Rng((seed ^ 0x6270_13b9_0cda_7e93 ^ ((w as u64 + 1) << 33) ^ (u64::from(p) << 17)) | 1);
        for i in 0..TILE as usize / 4 {
            before.extend_from_slice(&input_word(&mut rg, i, mode).to_le_bytes());
        }
    }
    Fixture { io, res, ranges, before, layout: layout.clone(), seed, groups: [1, 1, 1] }
}
fn pipeline(dev: &ID3D12Device, root: &Path, form: &str, h: &str) -> Result<(ID3D12PipelineState, ID3D12RootSignature), String> {
    let dx = fs::read(root.join(form).join(format!("{h}.dxil"))).map_err(|e| format!("{form}: {e}"))?;
    let sd = if form == "probe" { "ref" } else { form };
    let st = fs::read_to_string(root.join(sd).join(format!("{h}.samplers"))).map_err(|e| e.to_string())?;
    let mut samplers = Vec::new();
    for line in st.lines() {
        let w: Vec<_> = line.split_whitespace().collect();
        if w.len() != 5 {
            return Err("invalid sampler sidecar".into());
        }
        let a: Vec<u32> = w[..4].iter().map(|s| u32::from_str_radix(s, 16)).collect::<Result<_, _>>().map_err(|e| e.to_string())?;
        samplers.push((a.try_into().unwrap(), w[4] == "1"));
    }
    let rs = recomp::pso::abi_root_signature_with(dev, true, &samplers)?;
    let p = unsafe {
        dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: transmute_copy(&rs),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dx.as_ptr() as _, BytecodeLength: dx.len() },
            ..Default::default()
        })
        .map_err(|e| format!("{form} PSO: {e}"))?
    };
    Ok((p, rs))
}
/// Discover resource types and pages on fresh inputs, then compare both unmodified shaders.
fn trial(g: &mut Gpu, ps: &[(ID3D12PipelineState, ID3D12RootSignature)], h: &str, seed: u64, mode: u64) -> Result<(), String> {
    let mut layout: BTreeMap<u32, u32> = (0..16).map(|k| (LIMIT as u32 + k, k)).collect();
    let mut tiles: BTreeSet<_> = (0..4).map(|w| (w, 0)).collect();
    for _ in 0..8 {
        let f = fixture(seed, mode, &tiles, &layout, true);
        let (o, _) = unsafe { dispatch(g, &ps[0].0, &ps[0].1, &f)? };
        let meta = &o[128..128 + RES_WORDS];
        if meta[64] != 0 {
            return Err(format!("unrepresentable descriptor or sampler index ({})", meta[64]));
        }
        let mut steady = true;
        for (i, &bits) in meta[REQ..REQ + LIMIT].iter().enumerate() {
            if bits.count_ones() > 1 {
                return Err(format!("conflicting resource types at slot {i} ({bits:x})"));
            }
            if bits != 0 {
                let kind = bits.trailing_zeros();
                if kind >= 16 {
                    return Err("invalid resource type in preflight".into());
                }
                if layout.insert(i as u32, kind) != Some(kind) {
                    steady = false
                }
            }
        }
        for (i, &bits) in meta[PAGES..].iter().enumerate() {
            for b in 0..32 {
                if bits >> b & 1 != 0 {
                    steady &= !tiles.insert((i / 2048, ((i % 2048) * 32 + b) as u32));
                }
            }
        }
        if tiles.len() > 128 || layout.len() > 512 {
            return Err("synthetic fixture exceeds per-job resource cap".into());
        }
        if !steady {
            continue;
        }
        // Keep identical texture data and allocation order in the probe and both executable forms.
        let f = fixture(seed, mode, &tiles, &layout, false);
        let run = |g: &mut Gpu, idx: usize| unsafe { dispatch(g, &ps[idx].0, &ps[idx].1, &f) };
        let (reference, initial) = run(g, 1)?;
        let (lane, initial2) = run(g, 2)?;
        let mut checked = o[..128 + 64].to_vec();
        checked.extend_from_slice(&o[128 + RES_WORDS..]);
        let probe_diff = checked.iter().zip(&reference).filter(|(a, b)| a != b).count();
        if checked.len() != reference.len() {
            return Err("probe output size mismatch".into());
        }
        if initial != initial2 || reference.len() != lane.len() || reference.len() != initial.len() {
            return Err("fixture layout mismatch".into());
        }
        let writes = |a: &[u32]| a.iter().zip(&initial).filter(|(l, r)| l != r).count();
        let diff = reference.iter().zip(&lane).filter(|(l, r)| l != r).count();
        let first = reference.iter().zip(&lane).position(|(l, r)| l != r);
        let mut stable = "untested";
        if diff > 0 {
            let mut signed_zero = 0;
            let mut both_nan = 0;
            for (&a, &b) in reference.iter().zip(&lane) {
                if a == b {
                    continue;
                }
                if a | b == 0x80000000 && a ^ b == 0x80000000 {
                    signed_zero += 1
                } else if f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan() {
                    both_nan += 1
                }
            }
            println!("DETAIL,{h},{seed},{signed_zero},{both_nan},{}", diff - signed_zero - both_nan);
            let (a, _) = run(g, 1)?;
            let (b, _) = run(g, 2)?;
            stable = if a == reference && b == lane { "repeatable" } else { "unstable" };
        }
        let (at, rv, lv, resource, word) = if let Some(i) = first {
            let (resource, word) = if i < 128 {
                ("io".into(), i)
            } else if i < 192 {
                ("res".into(), i - 128)
            } else if i < 192 + tiles.len() * 16384 {
                let q = i - 192;
                let &(w, t) = tiles.iter().nth(q / 16384).unwrap();
                (format!("window{w}"), t as usize * 16384 + q % 16384)
            } else {
                let mut q = i - 192 - tiles.len() * 16384;
                let mut found = None;
                for (slot, k) in &layout {
                    let len = if *k >= 8 { 8192 } else { 1024 };
                    if q < len {
                        found = Some((format!("texture{slot}"), q));
                        break;
                    }
                    q -= len;
                }
                found.ok_or("texture output layout")?
            };
            (i.to_string(), format!("{:08x}", reference[i]), format!("{:08x}", lane[i]), resource, word.to_string())
        } else {
            (String::new(), String::new(), String::new(), String::new(), String::new())
        };
        println!(
            "ROW,{h},{seed},{mode},{},{},{},{diff},{at},{resource},{word},{rv},{lv},{stable},{},{},{probe_diff}",
            reference.len(),
            writes(&reference),
            writes(&lane),
            tiles.len(),
            layout.len() - 16
        );
        return Ok(());
    }
    Err("descriptor/page discovery did not stabilize".into())
}
fn main() -> Result<(), String> {
    use windows::Win32::Graphics::{Direct3D::D3D_FEATURE_LEVEL_12_0, Dxgi::*};
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: lane1912_replay DIRECTORY HASH SEED [SEED ...]".into());
    }
    let root = Path::new(&args[1]);
    let h = &args[2];
    let mut gpu = unsafe {
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(err)?;
        let sel = gpu::adapter::Selector::from_env_or("auto");
        let (adapter, info) = gpu::adapter::candidates(&factory, &sel).into_iter().next().ok_or("no adapter")?;
        let dev = gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_0).map_err(err)?;
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS).cast(),
            std::mem::size_of_val(&options) as u32,
        )
        .map_err(err)?;
        if options.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
            return Err("input discovery requires tiled resources tier 2 for zero reads from unmapped tiles".into());
        }
        let name_len = info.Description.iter().position(|&c| c == 0).unwrap_or(info.Description.len());
        eprintln!("Adapter: {}", String::from_utf16_lossy(&info.Description[..name_len]));
        let queue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(err)?;
        let fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
        Gpu { dev, queue, fence, value: 0 }
    };
    let ps = ["probe", "ref", "lane"].iter().map(|f| pipeline(&gpu.dev, root, f, h)).collect::<Result<Vec<_>, _>>()?;
    for s in &args[3..] {
        let seed = s.parse::<u64>().map_err(|e| e.to_string())?;
        let mut why = String::new();
        for attempt in 0..3 {
            let mode = (seed % 3 + attempt) % 3;
            match trial(&mut gpu, &ps, h, seed, mode) {
                Ok(()) => {
                    why.clear();
                    break;
                }
                Err(e) => {
                    if e.contains("removed") || e.contains("timeout") || unsafe { gpu.dev.GetDeviceRemovedReason() }.is_err() {
                        return Err(e);
                    }
                    why = e;
                }
            }
        }
        if !why.is_empty() {
            println!("SKIP,{h},{seed},{why}")
        }
    }
    Ok(())
}
