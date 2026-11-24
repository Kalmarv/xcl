//! The builder against DXC. Each fixture is a small compute program twice: in HLSL, and built
//! through the builder the way DXC lays that HLSL out. For the built program:
//! - `dxil.dll`'s validator accepts the container [`Module::finish`] writes;
//! - DXC's listing of that container is its listing of what it compiles the HLSL to (with
//!   `-validator-version 1.6`, the version the writer writes, and `-Qstrip_reflect`: the container
//!   has no reflection data), but for [`DIFFER`];
//! - the text [`Module::to_text`] prints, through DXC's assembler, has the same listing;
//! - on the GPU the container gives what DXC's gives: the same u1, memory and written texture.
//!
//! Checked with DXC 1.7.2308 (Windows SDK 10.0.26100) and 1.9.2609 (`XCL_DXC_DIR`).
//! `XCL_DXIL_TEST_DUMP=<folder>` keeps both containers and the text of each fixture.

use std::ffi::c_void;

use windows::core::{Interface, GUID, HRESULT};
use windows::Win32::Graphics::Direct3D::Dxc::{IDxcAssembler, IDxcUtils, CLSID_DxcAssembler, CLSID_DxcUtils, DXC_CP_UTF8};
use windows::Win32::Graphics::Direct3D12::{D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32_UINT};

use super::check::{self, Differ};
use super::dxop::{Component, HeapRes, Shape};
use super::module::{flags, Binary, Cast, Fcmp, FnAttr, Func, Icmp, Module, Rmw, Ty, Val};
use super::ops::opcode;
use crate::gpu::recomp::dxc;
use crate::gpu::recomp::pso::StaticSampler;
use crate::gpu::recomp::runner::{self, TestSampler, TestTexture, TexDim};

/// What the listing of a built program may differ from DXC's in: the container's digest, the
/// writer's name, DXC's TBAA tags on loads and stores, and the order of the type and `declare`
/// lines with the numbers of the attribute groups (DXC declares a function when one of its
/// passes first needs it, the builder when the program first calls it).
const DIFFER: [Differ; 4] = [check::SHADER_HASH, Differ::Ident, Differ::Tbaa, Differ::DeclarationOrder];

struct Fixture {
    hlsl: &'static str,
    /// DXC arguments the HLSL needs
    arguments: &'static [&'static str],
    build: fn() -> Module,
    /// what the listings may differ in besides [`DIFFER`]
    differ: &'static [Differ<'static>],
    /// the first dwords of u0 for the run on the GPU (as the kernel runner runs a kernel)
    io: &'static [u32],
    /// the first dword of u1 after it, when the program writes one whatever the adapter
    res: Option<u32>,
}

/// The container DXC's assembler makes of a module's text (not signed: nothing here reads the
/// digest). For these tests only: the program's own way to the assembler is not the builder's.
fn assemble(text: &str) -> Result<Vec<u8>, String> {
    type CreateInstance = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut c_void) -> HRESULT;
    unsafe fn instance<T: Interface>(create: CreateInstance, class: &GUID) -> Result<T, String> {
        let mut p = std::ptr::null_mut();
        let hr = create(class, &T::IID, &mut p);
        if hr.is_err() || p.is_null() {
            return Err(format!("DxcCreateInstance: {hr:?}"));
        }
        Ok(T::from_raw(p))
    }
    unsafe {
        let create = std::mem::transmute::<usize, CreateInstance>(dxc::create_instance().ok_or("no DXC")?);
        let (utils, assembler): (IDxcUtils, IDxcAssembler) = (instance(create, &CLSID_DxcUtils)?, instance(create, &CLSID_DxcAssembler)?);
        let blob = utils.CreateBlob(text.as_ptr() as *const c_void, text.len() as u32, DXC_CP_UTF8).map_err(|e| format!("CreateBlob: {e}"))?;
        let result = assembler.AssembleToContainer(&blob).map_err(|e| format!("AssembleToContainer: {e}"))?;
        let bytes = |pointer: *mut c_void, size: usize| std::slice::from_raw_parts(pointer as *const u8, size).to_vec();
        if result.GetStatus().map_err(|e| format!("{e}"))?.is_err() {
            let said = result.GetErrorBuffer().map_err(|e| format!("{e}"))?;
            return Err(String::from_utf8_lossy(&bytes(said.GetBufferPointer(), said.GetBufferSize())).to_string());
        }
        let container = result.GetResult().map_err(|e| format!("{e}"))?;
        Ok(bytes(container.GetBufferPointer(), container.GetBufferSize()))
    }
}

/// The checks of the module comment.
fn check(fixture: &Fixture) {
    if !check::available() {
        return;
    }
    let dump = |kind: &str, data: &[u8]| {
        if let Ok(dir) = std::env::var("XCL_DXIL_TEST_DUMP") {
            let name = std::thread::current().name().unwrap_or("fixture").rsplit("::").next().unwrap().to_string();
            std::fs::write(format!("{dir}/{name}.{kind}"), data).unwrap();
        }
    };
    let arguments = [&["-validator-version", "1.6", "-Qstrip_reflect"], fixture.arguments].concat();
    let (reference, theirs) = check::reference(fixture.hlsl, "main", "cs_6_6", &arguments).unwrap();
    let mut built = (fixture.build)();
    // DXC's layout has every definition before its uses
    built.check_dominance = true;
    let ours = built.finish().unwrap();
    dump("xcl.dxil", &ours);
    dump("dxc.dxil", &reference);
    let listing = check::disassemble(&ours).unwrap_or_default();
    if let Err(messages) = check::validate(&ours) {
        panic!("dxil.dll refuses the program:\n{messages}\n{listing}");
    }
    if let Err(e) = check::compare(&listing, &theirs, &[&DIFFER[..], fixture.differ].concat()) {
        panic!("{e}\n--- the writer's\n{listing}\n--- DXC's\n{theirs}");
    }

    let text = built.to_text().unwrap();
    dump("xcl.ll", text.as_bytes());
    let assembled = assemble(&text).unwrap_or_else(|e| panic!("DXC's assembler refuses the text:\n{e}\n{text}"));
    if let Err(e) = check::same_listing(&ours, &assembled) {
        panic!("the text assembles to another module than the bitcode: {e}\n--- the text\n{text}");
    }

    if !check::gpu_available() {
        return;
    }
    // for the programs that take them from the heaps: a texture of floats at 0, one to write
    // at 3, and a sampler at 0
    let floats = (0..32).map(|k| k as f32 + 0.5).flat_map(f32::to_le_bytes).collect();
    let textures = [
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 2, depth: 1, texel_bytes: 16, data: floats, uav: false },
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 2, height: 1, depth: 1, texel_bytes: 4, data: vec![0; 8], uav: true },
    ];
    let samplers = [TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_POINT, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None }];
    let state = runner::KernelState { sgpr: fixture.io.to_vec(), ..Default::default() };
    // `xcl_ss0` is in the root signature of a program that has it
    let statics: &[StaticSampler] = if fixture.hlsl.contains("xcl_ss0") { &[([0; 4], false)] } else { &[] };
    let run = |container: &[u8]| runner::run_with_statics(container, &state, &[0; 16], &textures, &samplers, statics).map(|r| (r.raw, r.mem, r.textures));
    let expected = match run(&reference) {
        Ok(result) => result,
        // waves of 32 lanes or native 16-bit types, on an adapter without them
        Err(e) => {
            eprintln!("DXC's container does not run: {e}");
            return runner::skip_test("no GPU for this program");
        }
    };
    if let Some(res) = fixture.res {
        assert_eq!(expected.0[0], res, "what DXC's program gives");
    }
    match run(&ours) {
        Ok(result) => assert!(result == expected, "the built program gives u1 {:x?}.., DXC's {:x?}..", &result.0[..8], &expected.0[..8]),
        Err(e) => panic!("the built program does not run: {e}"),
    }
}

/// The entry function of a new module in its first block.
fn entry(m: &mut Module) -> (Func<'_>, super::module::Block) {
    let mut f = m.entry();
    let b = f.block();
    f.at(b);
    (f, b)
}

#[test]
fn an_empty_program() {
    check(&Fixture {
        hlsl: "[numthreads(4, 2, 1)] void main() {}",
        arguments: &[],
        differ: &[],
        io: &[],
        res: None,
        build: || {
            let mut m = Module::compute([4, 2, 1]);
            entry(&mut m).0.ret();
            m
        },
    });
}

#[test]
fn a_loop_with_phis_over_root_buffers() {
    const HLSL: &str = "
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
[numthreads(32, 1, 1)]
void main(uint gi : SV_GroupIndex)
{
    uint n = io.Load(4u);
    uint s = 0u;
    for (uint i = 0u; i != n; i++) s += countbits(i);
    res.Store(gi * 4u, s);
}";
    fn build() -> Module {
        let mut m = Module::compute([32, 1, 1]);
        let (io, res) = (m.root_uav(0, false), m.root_uav(1, false));
        let (mut f, b0) = entry(&mut m);
        // DXC's layout: the loop's preheader, the block the loop leaves through, the exit, the loop
        let (preheader, leave, exit, body) = (f.block(), f.block(), f.block(), f.block());
        let res_bound = f.bind(res);
        let io_bound = f.bind(io);
        let gi = f.op(opcode::FlattenedThreadIdInGroup, Ty::I32, &[]);
        let io = f.annotate_bound(io_bound, io);
        let n = f.raw_load(io, Val::i32(4));
        let none = f.icmp(Icmp::Eq, n, Val::i32(0));
        f.br_if(none, exit, preheader);
        f.at(preheader);
        f.br(body);
        f.at(leave);
        f.br(exit);

        f.at(exit);
        let sum = f.phi(exit, Ty::I32);
        let offset = f.shl(gi, Val::i32(2));
        let res = f.annotate_bound(res_bound, res);
        f.raw_store(res, offset, sum);
        f.ret();

        f.at(body);
        let (s, i) = (f.phi(body, Ty::I32), f.phi(body, Ty::I32));
        let bits = f.op(opcode::Countbits, Ty::I32, &[i]);
        let s2 = f.add(bits, s);
        let i2 = f.add(i, Val::i32(1));
        let done = f.icmp(Icmp::Eq, i2, n);
        f.br_if(done, leave, body);
        f.incoming(sum, Val::i32(0), b0);
        f.incoming(sum, s2, leave);
        f.incoming(s, s2, body);
        f.incoming(s, Val::i32(0), preheader);
        f.incoming(i, i2, body);
        f.incoming(i, Val::i32(0), preheader);
        m
    }
    check(&Fixture { hlsl: HLSL, arguments: &[], build, differ: &[], io: &[0, 37], res: Some(90) });
}

#[test]
fn wave_operations_with_a_wave_size() {
    const HLSL: &str = "
RWByteAddressBuffer res : register(u1);
[WaveSize(32)]
[numthreads(32, 1, 1)]
void main()
{
    uint lane = WaveGetLaneIndex();
    uint4 ballot = WaveActiveBallot((lane & 1u) != 0u);
    res.Store(lane * 4u, WaveActiveSum(lane) + WaveReadLaneAt(lane, 3u) + ballot.x);
}";
    check(&Fixture { hlsl: HLSL, arguments: &[], build: wave_program, differ: &[Differ::Line("declare i32 @dx.op.waveGetLaneIndex(i32)")], io: &[], res: None });
}

/// The program of `wave_operations_with_a_wave_size`.
fn wave_program() -> Module {
    {
        let mut m = Module::compute([32, 1, 1]);
        m.wave_size(32);
        let res = m.root_uav(1, false);
        let (mut f, _) = entry(&mut m);
        let bound = f.bind(res);
        let lane = f.op(opcode::WaveGetLaneIndex, Ty::VOID, &[]);
        let odd = f.and(lane, Val::i32(1));
        let odd = f.icmp(Icmp::Ne, odd, Val::i32(0));
        let ballot = f.op(opcode::WaveActiveBallot, Ty::VOID, &[odd]);
        let ballot = f.extract(ballot, 0);
        // the sum (operation 0) of unsigned values (1)
        let sum = f.op(opcode::WaveActiveOp, Ty::I32, &[lane, Val::i8(0), Val::i8(1)]);
        let third = f.op(opcode::WaveReadLaneAt, Ty::I32, &[lane, Val::i32(3)]);
        let value = f.add(ballot, sum);
        let value = f.add(value, third);
        let offset = f.shl(lane, Val::i32(2));
        let res = f.annotate_bound(bound, res);
        f.raw_store(res, offset, value);
        f.ret();
        m
    }
}

/// `dxil.dll` (1.7.2308 and 1.9.2609) takes `dx.op.waveGetLaneIndex` declared `readonly` (DXC
/// since 1.8.2405, and the builder: it is what the table of the operations says) and `readnone`
/// (DXC before).
#[test]
fn the_validator_takes_the_lane_index_readonly_or_readnone() {
    if !check::available() {
        return;
    }
    for attr in [FnAttr::ReadOnly, FnAttr::ReadNone] {
        let mut m = wave_program();
        m.functions.iter_mut().find(|f| f.name == "dx.op.waveGetLaneIndex").unwrap().attr = Some(attr);
        assert_eq!(check::validate(&m.finish().unwrap()), Ok(()), "{attr:?}");
    }
}

#[test]
fn textures_and_samplers_of_the_descriptor_heaps() {
    const HLSL: &str = "
RWByteAddressBuffer io : register(u0);
SamplerState xcl_ss0 : register(s0, space1);
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint t = io.Load(0u);
    Texture2D<float4> tex = ResourceDescriptorHeap[NonUniformResourceIndex(t)];
    SamplerState ss = SamplerDescriptorHeap[NonUniformResourceIndex(t & 0xFFFu)];
    RWTexture2D<uint> out_tex = ResourceDescriptorHeap[NonUniformResourceIndex(t + 3u)];
    float4 a = tex.SampleLevel(ss, float2(0.25, 0.75), 0.0);
    float4 b = tex.SampleLevel(xcl_ss0, float2(0.5, 0.5), 1.0);
    out_tex[tid.xy] = asuint(a.x + b.y);
}";
    fn build() -> Module {
        let mut m = Module::compute([8, 8, 1]);
        let (io, sampler) = (m.root_uav(0, false), m.static_sampler(0, false));
        let (mut f, _) = entry(&mut m);
        let (io_bound, sampler_bound) = (f.bind(io), f.bind(sampler));
        let x = f.op(opcode::ThreadId, Ty::I32, &[Val::i32(0)]);
        let y = f.op(opcode::ThreadId, Ty::I32, &[Val::i32(1)]);
        let io = f.annotate_bound(io_bound, io);
        let t = f.raw_load(io, Val::i32(0));
        let texture = f.heap_handle(t, HeapRes::Texture { shape: Shape::Texture2D, component: Component::F32, count: 4, uav: false }, true);
        let index = f.and(t, Val::i32(4095));
        let heap_sampler = f.heap_handle(index, HeapRes::Sampler { comparison: false }, true);
        let index = f.add(t, Val::i32(3));
        let target = f.heap_handle(index, HeapRes::Texture { shape: Shape::Texture2D, component: Component::U32, count: 1, uav: true }, true);
        // SampleLevel(srv, sampler, coord0..3, offset0..2, LOD)
        let (no_coord, no_offset) = (Val::undef(Ty::F32), Val::undef(Ty::I32));
        let sample = |f: &mut Func, sampler: Val, u: f32, v: f32, lod: f32| {
            f.op(opcode::SampleLevel, Ty::F32, &[texture, sampler, Val::f32(u), Val::f32(v), no_coord, no_coord, Val::i32(0), Val::i32(0), no_offset, Val::f32(lod)])
        };
        let a = sample(&mut f, heap_sampler, 0.25, 0.75, 0.0);
        let a = f.extract(a, 0);
        let sampler = f.annotate_bound(sampler_bound, sampler);
        let b = sample(&mut f, sampler, 0.5, 0.5, 1.0);
        let b = f.extract(b, 1);
        let sum = f.binary_with(Binary::FAdd, flags::FAST, b, a);
        let bits = f.cast(Cast::BitCast, sum, Ty::I32);
        // TextureStore(srv, coord0..2, value0..3, mask)
        f.op(opcode::TextureStore, Ty::I32, &[target, x, y, no_offset, bits, bits, bits, bits, Val::i8(15)]);
        f.ret();
        m
    }
    check(&Fixture { hlsl: HLSL, arguments: &[], build, differ: &[], io: &[0], res: None });
}

#[test]
fn group_shared_memory_scratch_and_atomics() {
    const HLSL: &str = "
RWByteAddressBuffer io : register(u0);
globallycoherent RWByteAddressBuffer mem : register(u2);
groupshared uint xcl_lds0[8192];
[numthreads(64, 1, 1)]
void main(uint gi : SV_GroupIndex)
{
    uint scratch[16];
    uint a = io.Load(gi * 4u);
    scratch[a & 15u] = a;
    uint old;
    InterlockedAdd(xcl_lds0[a & 8191u], 1u, old);
    uint old2;
    InterlockedCompareExchange(xcl_lds0[gi], old, a, old2);
    uint old3;
    mem.InterlockedMax(8u, a, old3);
    GroupMemoryBarrierWithGroupSync();
    xcl_lds0[gi] = old2 + old3 + scratch[old & 15u];
}";
    fn build() -> Module {
        let mut m = Module::compute([64, 1, 1]);
        let lds = m.groupshared(8192);
        let (io, mem) = (m.root_uav(0, false), m.root_uav(2, true));
        let (mut f, _) = entry(&mut m);
        let (mem_bound, io_bound) = (f.bind(mem), f.bind(io));
        let gi = f.op(opcode::FlattenedThreadIdInGroup, Ty::I32, &[]);
        let scratch = f.alloca(16);
        let offset = f.shl(gi, Val::i32(2));
        let io = f.annotate_bound(io_bound, io);
        let a = f.raw_load(io, offset);
        let index = f.and(a, Val::i32(15));
        let slot = f.gep(scratch, index);
        f.store(slot, a);
        let index = f.and(a, Val::i32(8191));
        let counter = f.gep(lds, index);
        let old = f.atomic_rmw(Rmw::Add, counter, Val::i32(1));
        let mine = f.gep(lds, gi);
        let old2 = f.cmpxchg(mine, old, a);
        let mem = f.annotate_bound(mem_bound, mem);
        // AtomicBinOp(handle, operation (7: unsigned max), offset0..2, value)
        let none = Val::undef(Ty::I32);
        let old3 = f.op(opcode::AtomicBinOp, Ty::I32, &[mem, Val::i32(7), Val::i32(8), none, none, a]);
        // a group sync (1) and a fence of the group-shared memory (8)
        f.op(opcode::Barrier, Ty::VOID, &[Val::i32(9)]);
        let index = f.and(old, Val::i32(15));
        let slot = f.gep(scratch, index);
        let kept = f.load(slot);
        let sum = f.add(kept, old3);
        let sum = f.add(sum, old2);
        f.store(mine, sum);
        f.ret();
        m
    }
    check(&Fixture { hlsl: HLSL, arguments: &["-disable-lifetime-markers"], build, differ: &[], io: &[5, 6, 7], res: None });
}

#[test]
fn sixty_four_bit_integers_doubles_and_halves() {
    const HLSL: &str = "
RWByteAddressBuffer io : register(u0);
[numthreads(1, 1, 1)]
void main()
{
    uint a = io.Load(0u);
    uint b = io.Load(4u);
    uint64_t wide = ((uint64_t)a << 32u) | (uint64_t)b;
    wide = wide * 3ull + (wide >> (b & 63u));
    double d = asdouble(a, b);
    d = d / 3.0 + (double)(int)b;
    precise float f = asfloat(a) * 2.0f - asfloat(b);
    float16_t h = (float16_t)f + (float16_t)1.5;
    bool lt = f < asfloat(b);
    uint lo, hi;
    asuint(d, lo, hi);
    io.Store(8u, (uint)(wide >> 32u) + (uint)wide + (lt ? lo : hi) + (uint)(int)h + (uint)(int)d + firstbithigh(a) + f32tof16(f));
}";
    fn build() -> Module {
        let mut m = Module::compute([1, 1, 1]);
        let io = m.root_uav(0, false);
        let (mut f, _) = entry(&mut m);
        let io = f.handle(io);
        let a = f.raw_load(io, Val::i32(0));
        let b = f.raw_load(io, Val::i32(4));
        let a64 = f.cast(Cast::ZExt, a, Ty::I64);
        let high = f.binary_with(Binary::Shl, flags::NUW, a64, Val::i64(32));
        let b64 = f.cast(Cast::ZExt, b, Ty::I64);
        let wide = f.or(b64, high);
        let tripled = f.mul(wide, Val::i64(3));
        let amount = f.and(b64, Val::i64(63));
        let shifted = f.lshr(wide, amount);
        let wide = f.add(tripled, shifted);
        let d = f.op(opcode::MakeDouble, Ty::F64, &[a, b]);
        let third = f.binary_with(Binary::FMul, flags::FAST, d, Val::f64(f64::from_bits(0x3FD5_5555_5555_5555)));
        let signed = f.cast(Cast::SiToFp, b, Ty::F64);
        let d = f.binary_with(Binary::FAdd, flags::FAST, third, signed);
        // `precise`: no fast-math flags
        let fa = f.cast(Cast::BitCast, a, Ty::F32);
        let doubled = f.fmul(fa, Val::f32(2.0));
        let fb = f.cast(Cast::BitCast, b, Ty::F32);
        let fl = f.fsub(doubled, fb);
        let h = f.cast(Cast::FpTrunc, fl, Ty::F16);
        let h = f.binary_with(Binary::FAdd, flags::FAST, h, Val::f16(0x3E00));
        let less = f.fcmp_with(Fcmp::Olt, flags::FAST, fl, fb);
        let halves = f.op(opcode::SplitDouble, Ty::F64, &[d]);
        let (lo, hi) = (f.extract(halves, 0), f.extract(halves, 1));
        let top = f.lshr(wide, Val::i64(32));
        let top = f.cast(Cast::Trunc, top, Ty::I32);
        let bottom = f.cast(Cast::Trunc, wide, Ty::I32);
        let chosen = f.select(less, lo, hi);
        let from_half = f.cast(Cast::FpToSi, h, Ty::I32);
        let from_double = f.cast(Cast::FpToSi, d, Ty::I32);
        // firstbithigh: 31 minus the bit DXIL counts from the top, or -1
        let first = f.op(opcode::FirstbitHi, Ty::I32, &[a]);
        let from_low = f.sub(Val::i32(31), first);
        let none = f.icmp(Icmp::Eq, first, Val::i32(u32::MAX));
        let first = f.select(none, Val::i32(u32::MAX), from_low);
        let packed = f.op(opcode::LegacyF32ToF16, Ty::VOID, &[fl]);
        let mut sum = f.add(packed, from_double);
        for term in [chosen, first, bottom, from_half, top] {
            sum = f.add(sum, term);
        }
        f.raw_store(io, Val::i32(8), sum);
        f.ret();
        m
    }
    check(&Fixture { hlsl: HLSL, arguments: &["-enable-16bit-types"], build, differ: &[], io: &[0x4009_21FB, 0x5444_2D18], res: None });
}

/// What the builder does not look at unless `XCL_RECOMP_DXIL_VERIFY=1` asks: `dxil.dll` refuses
/// a program with a use its definition does not dominate.
#[test]
fn the_validator_refuses_a_use_its_definition_does_not_dominate() {
    if !check::available() {
        return;
    }
    let build = || {
        let mut m = Module::compute([1, 1, 1]);
        let res = m.root_uav(1, false);
        let (mut f, _) = entry(&mut m);
        let (then, join) = (f.block(), f.block());
        let res = f.handle(res);
        let x = f.raw_load(res, Val::i32(0));
        let zero = f.icmp(Icmp::Eq, x, Val::i32(0));
        f.br_if(zero, then, join);
        f.at(then);
        // of one arm only
        let y = f.add(x, Val::i32(1));
        f.br(join);
        f.at(join);
        f.raw_store(res, Val::i32(4), y);
        f.ret();
        m
    };
    let mut unchecked = build();
    unchecked.check_dominance = false;
    let said = check::validate(&unchecked.finish().unwrap()).expect_err("dxil.dll accepts the program");
    assert!(said.contains("Instruction does not dominate all uses!"), "{said}");
    let mut checked = build();
    checked.check_dominance = true;
    assert_eq!(checked.finish().map_err(|r| r.kind), Err("not-dominated"));
    assert_eq!(checked.to_text().map_err(|r| r.kind), Err("not-dominated"));
}
