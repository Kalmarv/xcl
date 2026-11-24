//! Reference programs for the tests of the DXIL writer: small compute programs in HLSL with
//! their input and the result a GPU gives. DXC's container of one ([`Fixture::reference`]) holds
//! the forms the writer has to produce for the same program; a container XCL wrote for it is
//! checked with [`check::validate`], [`check::compare`] against the reference's listing, and
//! [`Fixture::run`] against [`Fixture::expected`].
//!
//! Every program reads `io` (u0) and writes `res` (u1), as a kernel does, and is written the way
//! the writer names things: the group-shared array is `xcl_lds0`, a memory window (u2) is
//! `globallycoherent`.

use windows::Win32::Graphics::Direct3D12::{D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32_UINT};

use super::super::runner::{self, TestSampler, TestTexture, TexDim};
use super::check;

const BUFFERS: &str = "\
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
";

/// What DXC compiles the references with besides `-T cs_6_6 -E main`: the validator version the
/// writer writes, no reflection part, and no lifetime intrinsics around a local array (the
/// writer has none).
pub const ARGUMENTS: &[&str] = &["-validator-version", "1.6", "-Qstrip_reflect", "-disable-lifetime-markers"];

/// A reference program.
pub struct Fixture {
    pub name: &'static str,
    /// Its source.
    pub hlsl: String,
    /// The first dwords of u0.
    pub io: Vec<u32>,
    /// The textures on the resource heap: `textures[i]` at index `2 * i`, or at `2 * i + 1` when
    /// it is one the program writes.
    pub textures: Vec<TestTexture>,
    /// The samplers on the sampler heap.
    pub samplers: Vec<TestSampler>,
    /// The first dwords of u1 after the run.
    pub expected: Vec<u32>,
}

impl Fixture {
    fn new(name: &'static str, hlsl: &str, io: Vec<u32>, expected: Vec<u32>) -> Fixture {
        Fixture { name, hlsl: format!("{BUFFERS}{hlsl}"), io, textures: Vec::new(), samplers: Vec::new(), expected }
    }

    /// DXC's container of the program and its listing.
    pub fn reference(&self) -> Result<(Vec<u8>, String), String> {
        check::reference(&self.hlsl, "main", "cs_6_6", ARGUMENTS)
    }

    /// Run a container of the program on the GPU: the dwords of u1 that [`expected`](Self::expected) covers.
    pub fn run(&self, container: &[u8]) -> Result<Vec<u32>, String> {
        let mut res = check::run(container, &self.io, &self.textures, &self.samplers)?;
        res.truncate(self.expected.len());
        Ok(res)
    }
}

/// A value per thread of a group.
fn per_thread() -> Vec<u32> {
    (0..64u32).map(|gi| gi * gi + 0x1_0001 * (gi + 1)).collect()
}

fn sum(values: &[u32]) -> u32 {
    values.iter().fold(0, |a, &v| a.wrapping_add(v))
}

/// A 4 x 2 texture of distinct floats, and its texel.
fn float_texture(k: u32) -> TestTexture {
    let data = (0..2).flat_map(|y| (0..4).flat_map(move |x| float_texel(k, x, y))).flat_map(f32::to_le_bytes).collect();
    TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 2, depth: 1, texel_bytes: 16, data, uav: false }
}

fn float_texel(k: u32, x: u32, y: u32) -> [f32; 4] {
    std::array::from_fn(|c| (k * 1000 + (y * 4 + x) * 4 + c as u32) as f32 + 0.5)
}

/// One block: loads, arithmetic and a store.
pub fn store() -> Fixture {
    let hlsl = "[numthreads(1, 1, 1)]
void main() {
    res.Store(4, io.Load(0) * 3 + 1);
}
";
    Fixture::new("store", hlsl, vec![14], vec![0, 43])
}

/// A loop whose trip count is read from `io`: blocks, a conditional branch and phis.
pub fn loop_with_phis() -> Fixture {
    let hlsl = "[numthreads(1, 1, 1)]
void main() {
    uint n = io.Load(0);
    uint x = io.Load(4);
    uint sum = 0;
    for (uint i = 0; i < n; i++) {
        sum += x;
        x = x * 1664525 + 1013904223;
    }
    res.Store2(0, uint2(sum, x));
}
";
    let (n, start) = (9u32, 0x1234_5678u32);
    let (mut x, mut sum) = (start, 0u32);
    for _ in 0..n {
        sum = sum.wrapping_add(x);
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
    }
    Fixture::new("loop", hlsl, vec![n, start], vec![sum, x])
}

/// Values of other types than i32: 64-bit integers, floats and doubles, casts between them,
/// comparisons and selects. The numbers are chosen so that no operation rounds.
pub fn types() -> Fixture {
    let hlsl = "[numthreads(1, 1, 1)]
void main() {
    uint a = io.Load(0);
    uint b = io.Load(4);
    uint64_t wide = (uint64_t(a) << 32 | b) * 0x9E3779B97F4A7C15ull;
    float f = asfloat(io.Load(8));
    float g = float(a) * f + 0.5;
    uint pick = f < g ? uint(g) : b;
    uint apart = a > b ? a - b : b - a;
    double d = asdouble(io.Load(12), io.Load(16)) * 0.5;
    uint lo, hi;
    asuint(d, lo, hi);
    res.Store4(0, uint4(uint(wide), uint(wide >> 32), asuint(g), pick));
    res.Store3(16, uint3(apart, lo, hi));
}
";
    let (a, b, f, double) = (3u32, 0x8765_4321u32, 2.5f32, 0x4028_0000_1234_5678u64);
    let wide = (u64::from(a) << 32 | u64::from(b)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let g = a as f32 * f + 0.5;
    let pick = if f < g { g as u32 } else { b };
    let half = (f64::from_bits(double) * 0.5).to_bits();
    let io = vec![a, b, f.to_bits(), double as u32, (double >> 32) as u32];
    Fixture::new("types", hlsl, io, vec![wide as u32, (wide >> 32) as u32, g.to_bits(), pick, a.abs_diff(b), half as u32, (half >> 32) as u32])
}

/// A texel of a texture taken from the resource heap by an index read from `io`.
pub fn heap_texture() -> Fixture {
    let hlsl = "[numthreads(1, 1, 1)]
void main() {
    Texture2D<float4> t = ResourceDescriptorHeap[io.Load(0)];
    res.Store4(0, asuint(t.Load(int3(io.Load(4), io.Load(8), 0))));
}
";
    // texel (2, 1) of the second texture
    let mut fixture = Fixture::new("heap_texture", hlsl, vec![2, 2, 1], float_texel(1, 2, 1).map(f32::to_bits).to_vec());
    fixture.textures = vec![float_texture(0), float_texture(1)];
    fixture
}

/// A texture sampled with a sampler from the sampler heap, and a texture from the resource
/// heap written and read.
pub fn heap_sampler_and_store() -> Fixture {
    let hlsl = "[numthreads(1, 1, 1)]
void main() {
    Texture2D<float4> t = ResourceDescriptorHeap[io.Load(0)];
    SamplerState s = SamplerDescriptorHeap[io.Load(4)];
    RWTexture2D<uint> o = ResourceDescriptorHeap[io.Load(8)];
    float4 c = t.SampleLevel(s, float2(asfloat(io.Load(12)), asfloat(io.Load(16))), 0);
    o[uint2(1, 0)] = asuint(c.y);
    res.Store2(0, uint2(asuint(c.x), o[uint2(1, 0)]));
}
";
    // the middle of texel (3, 1) of the first texture, with a point sampler; the second texture
    // is the one written (its view is at 2 * 1 + 1)
    let [x, y, ..] = float_texel(0, 3, 1);
    let mut fixture = Fixture::new("heap_sampler_and_store", hlsl, vec![0, 0, 3, 0.875f32.to_bits(), 0.75f32.to_bits()], vec![x.to_bits(), y.to_bits()]);
    let written = TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 2, height: 1, depth: 1, texel_bytes: 4, data: vec![0; 8], uav: true };
    fixture.textures = vec![float_texture(0), written];
    fixture.samplers = vec![TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_POINT, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None }];
    fixture
}

/// Group-shared memory: stores and loads of other threads' slots, an atomic add, a compare and
/// exchange, and barriers.
pub fn lds() -> Fixture {
    let hlsl = "groupshared uint xcl_lds0[65];
[numthreads(64, 1, 1)]
void main(uint gi : SV_GroupIndex) {
    uint v = io.Load(gi * 4);
    if (gi == 0) {
        xcl_lds0[64] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    xcl_lds0[gi] = v;
    InterlockedAdd(xcl_lds0[64], v);
    uint was;
    InterlockedCompareExchange(xcl_lds0[gi], v, ~v, was);
    GroupMemoryBarrierWithGroupSync();
    res.Store3(gi * 12, uint3(xcl_lds0[63 - gi], xcl_lds0[64], was));
}
";
    let io = per_thread();
    let expected = (0..64).flat_map(|gi| [!io[63 - gi], sum(&io), io[gi]]).collect();
    Fixture::new("lds", hlsl, io, expected)
}

/// A local array written and read by an index read from `io`: an alloca.
pub fn scratch() -> Fixture {
    let hlsl = "[numthreads(1, 1, 1)]
void main() {
    uint scratch[16];
    scratch[io.Load(0) & 15] = io.Load(8);
    scratch[io.Load(4) & 15] = io.Load(12);
    res.Store2(0, uint2(scratch[io.Load(4) & 15], scratch[io.Load(0) & 15]));
}
";
    Fixture::new("scratch", hlsl, vec![3, 9, 0xAAAA_0001, 0xBBBB_0002], vec![0xBBBB_0002, 0xAAAA_0001])
}

/// An atomic add in a memory window (u2, which the runner gives zeroed) by every thread of a
/// group, read by all after a barrier.
pub fn memory_atomic() -> Fixture {
    let hlsl = "globallycoherent RWByteAddressBuffer mem : register(u2);
[numthreads(64, 1, 1)]
void main(uint gi : SV_GroupIndex) {
    uint was;
    mem.InterlockedAdd(8, io.Load(gi * 4), was);
    AllMemoryBarrierWithGroupSync();
    res.Store(gi * 4, mem.Load(8));
}
";
    let io = per_thread();
    let expected = vec![sum(&io); 64];
    Fixture::new("memory_atomic", hlsl, io, expected)
}

/// Wave operations in waves of `lanes` (32 or 64): the wave's size, the lane, a sum over the
/// wave, another lane's value and a ballot. Thread `gi` of the group is lane `gi % lanes` of
/// wave `gi / lanes`, which the console programs rely on as well.
pub fn wave(lanes: u32) -> Fixture {
    let hlsl = format!(
        "[WaveSize({lanes})]
[numthreads(64, 1, 1)]
void main(uint gi : SV_GroupIndex) {{
    uint v = io.Load(gi * 4);
    uint4 odd = WaveActiveBallot((v & 1) != 0);
    res.Store4(gi * 24, uint4(WaveGetLaneCount(), WaveGetLaneIndex(), WaveActiveSum(v), WaveReadLaneAt(v, 0)));
    res.Store2(gi * 24 + 16, odd.xy);
}}
"
    );
    let io = per_thread();
    let expected = (0..64u32)
        .flat_map(|gi| {
            let wave = &io[(gi / lanes * lanes) as usize..][..lanes as usize];
            let odd = wave.iter().enumerate().fold(0u64, |a, (lane, v)| a | u64::from(v & 1) << lane);
            [lanes, gi % lanes, sum(wave), wave[0], odd as u32, (odd >> 32) as u32]
        })
        .collect();
    Fixture::new("wave", &hlsl, io, expected)
}

/// The wave size [`wave`] can run with on this adapter: 64 where the GPU has such waves, else
/// 32. `None`: neither.
pub fn wave_lanes() -> Option<u32> {
    match (runner::wave64_supported(), runner::wave64_programs_supported()) {
        (Ok(true), _) => Some(64),
        (_, Ok(true)) => Some(32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DXC's listing of each program has the forms the fixture is there for, and the validator
    /// accepts its container.
    #[test]
    fn dxc_compiles_the_references_to_the_planned_forms() {
        if !check::available() {
            return;
        }
        let forms: [(Fixture, &[&str]); 10] = [
            (store(), &["@dx.op.createHandleFromBinding(i32 217", "@dx.op.annotateHandle(i32 216", "@dx.op.rawBufferLoad.i32(i32 139", "@dx.op.rawBufferStore.i32(i32 140", "mul i32"]),
            (loop_with_phis(), &["phi i32", "br i1", "br label", "icmp eq i32"]),
            (types(), &["zext i32", "shl nuw i64", "mul i64", "lshr i64", "trunc i64", "uitofp i32", "fptoui float", "fcmp fast olt float", "icmp ugt i32", "select i1", "@dx.op.makeDouble.f64(i32 101", "@dx.op.splitDouble.f64(i32 102", "fmul fast double", "bitcast i32"]),
            (heap_texture(), &["@dx.op.createHandleFromHeap(i32 218", "i1 false, i1 false)", "@dx.op.textureLoad.f32(i32 66", "bitcast float"]),
            (heap_sampler_and_store(), &["i1 true, i1 false)", "@dx.op.sampleLevel.f32(i32 62", "@dx.op.textureStore.i32(i32 67", "@dx.op.textureLoad.i32(i32 66"]),
            (lds(), &["@\"\\01?xcl_lds0@@3PAIA\" = external addrspace(3) global [65 x i32]", "atomicrmw add i32 addrspace(3)*", "cmpxchg i32 addrspace(3)*", "@dx.op.barrier(i32 80, i32 9)", "getelementptr [65 x i32]", ", !tbaa !", "@dx.op.flattenedThreadIdInGroup.i32(i32 96)"]),
            (scratch(), &["alloca [16 x i32]", "getelementptr inbounds [16 x i32]", "store i32", "load i32"]),
            (memory_atomic(), &["@dx.op.atomicBinOp.i32(i32 78", "@dx.op.barrier(i32 80, i32 11)", "i32 20491"]),
            (wave(32), &["@dx.op.waveGetLaneCount(i32 112)", "@dx.op.waveGetLaneIndex(i32 111)", "@dx.op.waveActiveOp.i32(i32 119", "@dx.op.waveReadLaneAt.i32(i32 117", "@dx.op.waveActiveBallot(i32 116", "!{i32 32}"]),
            (wave(64), &["@dx.op.waveActiveOp.i32(i32 119", "!{i32 64}"]),
        ];
        for (fixture, forms) in forms {
            let (container, listing) = fixture.reference().unwrap_or_else(|e| panic!("{}: {e}", fixture.name));
            assert_eq!(check::validate(&container), Ok(()), "{}", fixture.name);
            assert!(listing.contains("!dx.valver = !{!1}\n") && listing.contains("!1 = !{i32 1, i32 6}\n") && !listing.contains("lifetime"), "{}:\n{listing}", fixture.name);
            for form in forms {
                assert!(listing.contains(form), "{}: no `{form}` in\n{listing}", fixture.name);
            }
        }
    }

    #[test]
    fn dxc_references_give_the_expected_results_on_the_gpu() {
        if !check::available() || !check::gpu_available() {
            return;
        }
        for fixture in [store(), loop_with_phis(), types(), heap_texture(), heap_sampler_and_store(), lds(), scratch(), memory_atomic()] {
            let (container, _) = fixture.reference().unwrap();
            assert_eq!(fixture.run(&container), Ok(fixture.expected.clone()), "{}", fixture.name);
        }
    }

    #[test]
    fn the_dxc_wave_reference_gives_the_expected_result_on_the_gpu() {
        if !check::available() || !check::gpu_available() {
            return;
        }
        let Some(lanes) = wave_lanes() else {
            runner::skip_test("no wave64");
            return;
        };
        let fixture = wave(lanes);
        let (container, _) = fixture.reference().unwrap();
        assert_eq!(fixture.run(&container), Ok(fixture.expected.clone()), "waves of {lanes}");
    }
}
