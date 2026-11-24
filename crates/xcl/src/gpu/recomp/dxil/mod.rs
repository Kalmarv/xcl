//! A DXIL module without HLSL: assembled and signed from LLVM IR text (`assemble`).
//!
//! Direct DXIL output (opt-in `XCL_RECOMP_DXIL=1`): a builder of compute programs for the lowering
//! ([`module::Module`]), which writes one as the LLVM 3.7 bitcode DXIL is stored as, in the DXBC
//! container with its digest, or prints it as LLVM assembly; and the table of the DXIL operations.

pub mod assemble;
pub mod bitstream;
#[cfg(test)]
mod builder_tests;
#[cfg(test)]
pub(crate) mod check;
pub mod container;
pub mod dxop;
#[cfg(test)]
pub(crate) mod fixtures;
mod meta;
pub mod module;
mod number;
pub mod ops;
mod text;
mod write;

#[cfg(test)]
mod tests {
    use super::ops::{self, opcode, slot, stage, Attr, Ty};
    use crate::gpu::recomp::{dxc, runner};

    /// The tables are indexed by opcode, `op` finds each operation, and the named opcodes are
    /// their operations'.
    #[test]
    fn operations_are_found_by_opcode() {
        for (table, id) in [(&ops::CORE_OPS[..], 0), (&ops::EXPERIMENTAL_OPS[..], 0x8000)] {
            for (index, o) in table.iter().enumerate() {
                assert_eq!(o.opcode, id << 16 | index as u32, "{}", o.name);
                assert!(std::ptr::eq(ops::op(o.opcode).unwrap(), o), "{}", o.name);
                assert!(o.overloads.len() <= 4 && o.overloads.iter().all(|&(types, elements)| types != 0 && (elements == 0 || types & slot::VECTOR != 0)), "{}", o.name);
            }
        }
        assert!(ops::op(ops::CORE_OPS.len() as u32).is_none() && ops::op(0x10000).is_none() && ops::op(0x80000000 + ops::EXPERIMENTAL_OPS.len() as u32).is_none());
        assert_eq!(ops::op(opcode::ThreadId).unwrap().name, "ThreadId");
        assert_eq!(ops::op(opcode::ExperimentalNop).unwrap().name, "ExperimentalNop");
        assert_eq!(*ops::CORE_OPS_IN_DXIL_1.last().unwrap() as usize, ops::CORE_OPS.len());
    }

    /// Known answers for operations XCL's programs use: their numbers and declarations.
    #[test]
    fn operations_xcl_uses_are_as_known() {
        let handle = ops::op(opcode::CreateHandleFromHeap).unwrap();
        assert_eq!((handle.opcode, handle.function, handle.attr, handle.ret), (218, "createHandleFromHeap", Attr::ReadNone, Ty::Handle));
        assert_eq!(handle.params, [("index", Ty::I32), ("samplerHeap", Ty::I1), ("nonUniformIndex", Ty::I1)]);
        assert!(handle.overloads.is_empty() && handle.opcode < ops::CORE_OPS_IN_DXIL_1[6] && handle.opcode >= ops::CORE_OPS_IN_DXIL_1[5]);
        let annotate = ops::op(opcode::AnnotateHandle).unwrap();
        assert_eq!((annotate.opcode, annotate.ret, annotate.params), (216, Ty::Handle, &[("res", Ty::Handle), ("props", Ty::ResourceProperties)][..]));
        let load = ops::op(opcode::RawBufferLoad).unwrap();
        assert_eq!((load.opcode, load.class, load.function, load.attr, load.ret), (139, "RawBufferLoad", "rawBufferLoad", Attr::ReadOnly, Ty::ResRet));
        assert_eq!(load.params.iter().map(|p| p.1).collect::<Vec<_>>(), [Ty::Handle, Ty::I32, Ty::I32, Ty::I8, Ty::I32]);
        assert_eq!(load.overloads, [(slot::F16 | slot::F32 | slot::F64 | slot::I16 | slot::I32 | slot::I64, 0)]);
        let store = ops::op(opcode::RawBufferStore).unwrap();
        assert_eq!((store.opcode, store.attr, store.ret, store.params.len()), (140, Attr::None, Ty::Void, 9));
        // two operations of one class: one function for each overload
        let (umax, umin) = (ops::op(opcode::UMax).unwrap(), ops::op(opcode::UMin).unwrap());
        assert_eq!((umax.opcode, umin.opcode, umax.function, umin.function), (39, 40, "binary", "binary"));
        assert_eq!((umax.ret, umax.params), (Ty::Overload, &[("a", Ty::Overload), ("b", Ty::Overload)][..]));
        // a class with an operation that has no overload (`dx.op.wavePrefixOp`) next to one that has
        let bits = ops::op(opcode::WavePrefixBitCount).unwrap();
        assert_eq!((bits.opcode, bits.class, bits.overloads.len(), bits.ret, bits.params), (136, "WavePrefixOp", 0, Ty::I32, &[("value", Ty::I1)][..]));
        assert_eq!(ops::op(opcode::WavePrefixOp).unwrap().overloads.len(), 1);
        assert_eq!(ops::op(opcode::WaveActiveBallot).unwrap().ret, Ty::FourI32);
        assert_eq!(ops::op(226).map(|o| (o.name, o.class, o.params.len())), Some(("Reserved0", "Reserved", 0)));
        // the shader model an operation came with, and the stages that have it
        assert_eq!((handle.min_sm, annotate.min_sm, load.min_sm, umax.min_sm), ((6, 6), (6, 6), (6, 2), (6, 0)));
        assert_eq!((handle.stages, ops::op(opcode::Discard).unwrap().stages), (0xFFFF, stage::PIXEL));
        assert_eq!(ops::op(opcode::ThreadId).unwrap().stages, stage::COMPUTE | stage::MESH | stage::AMPLIFICATION | stage::NODE);
    }

    /// `ty` as DXC's listing writes it, in a function with the overload `suffix`.
    fn listed(ty: Ty, suffix: &str) -> Option<String> {
        let overload = match suffix {
            "f16" => "half",
            "f32" => "float",
            "f64" => "double",
            "i1" | "i8" | "i16" | "i32" | "i64" => suffix,
            _ => "",
        };
        Some(match ty {
            Ty::Void => "void".into(),
            Ty::I1 | Ty::I1LikeOverload => "i1".into(),
            Ty::I8 => "i8".into(),
            Ty::I16 => "i16".into(),
            Ty::I32 | Ty::I32LikeOverload => "i32".into(),
            Ty::I64 => "i64".into(),
            Ty::F16 => "half".into(),
            Ty::F32 => "float".into(),
            Ty::F64 => "double".into(),
            Ty::Overload if !overload.is_empty() => overload.into(),
            Ty::ResRet if !overload.is_empty() => format!("%dx.types.ResRet.{suffix}"),
            Ty::CBufRet if !overload.is_empty() => format!("%dx.types.CBufRet.{suffix}"),
            Ty::Handle => "%dx.types.Handle".into(),
            Ty::Dimensions => "%dx.types.Dimensions".into(),
            Ty::I32C => "%dx.types.i32c".into(),
            Ty::TwoI32 => "%dx.types.twoi32".into(),
            Ty::FourI32 => "%dx.types.fouri32".into(),
            Ty::SplitDouble => "%dx.types.splitdouble".into(),
            Ty::ResourceProperties => "%dx.types.ResourceProperties".into(),
            Ty::ResBind => "%dx.types.ResBind".into(),
            _ => return None,
        })
    }

    /// DXC's listing of a container.
    pub(super) fn disassemble(container: &[u8]) -> String {
        use windows::core::{Interface, GUID, HRESULT};
        use windows::Win32::Graphics::Direct3D::Dxc::{DxcBuffer, IDxcBlob, IDxcCompiler3, IDxcResult, CLSID_DxcCompiler};
        type CreateInstance = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut std::ffi::c_void) -> HRESULT;
        unsafe {
            let create = std::mem::transmute::<usize, CreateInstance>(dxc::create_instance().unwrap());
            let mut p = std::ptr::null_mut();
            create(&CLSID_DxcCompiler, &IDxcCompiler3::IID, &mut p).ok().unwrap();
            let compiler = IDxcCompiler3::from_raw(p);
            let buffer = DxcBuffer { Ptr: container.as_ptr() as _, Size: container.len(), Encoding: 0 };
            let result: IDxcResult = compiler.Disassemble(&buffer).unwrap();
            let blob: IDxcBlob = result.GetResult().unwrap();
            String::from_utf8_lossy(std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())).trim_end_matches('\0').to_string()
        }
    }

    /// The `dx.op` functions DXC declares in a program are the table's: the function of the
    /// class with the overload's suffix, its return and parameter types and its attribute.
    #[test]
    fn dxc_declares_the_operations_as_the_table_has_them() {
        if dxc::location().is_err() {
            runner::skip_test("no DXC");
            return;
        }
        let src = "RWByteAddressBuffer mem : register(u2);
cbuffer constants : register(b0) { float4 scale; };
[numthreads(4, 1, 1)] void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint flat : SV_GroupIndex) {
    ByteAddressBuffer bytes = ResourceDescriptorHeap[tid.x];
    Texture2D<float4> tex = ResourceDescriptorHeap[tid.x + 1];
    RWTexture2D<float4> target = ResourceDescriptorHeap[gid.x + 2];
    Buffer<uint> typed = ResourceDescriptorHeap[gtid.x + 3];
    SamplerState samp = SamplerDescriptorHeap[flat];
    uint a = bytes.Load(0) + typed[1];
    float f = asfloat(mem.Load(4)) * scale.x;
    uint2 size;
    tex.GetDimensions(size.x, size.y);
    float4 colour = tex.SampleLevel(samp, float2(f, sqrt(f)), 0) + tex.Load(int3(tid.xy, 0)) + f16tof32(a) + max(f, 2.0) + mad(f, f, 1.0);
    uint4 ballot = WaveActiveBallot(a > 1);
    uint waves = WaveActiveCountBits(a > 1) + WavePrefixCountBits(a > 2) + WavePrefixSum(a) + WaveGetLaneIndex() + WaveGetLaneCount() + WaveReadLaneAt(a, 1) + WaveActiveSum(a) + WaveActiveBitOr(a) + (WaveActiveAnyTrue(a > 3) ? 1 : 0);
    uint bits = countbits(a) + firstbithigh(a) + firstbitlow(a) + reversebits(a) + max(a, 3u) + min(a, size.x) + mad(a, 3u, size.y) + f32tof16(f);
    uint64_t wide = (uint64_t)a * 0x100000001ull;
    target[tid.xy] = colour;
    mem.Store(0, waves + bits + ballot.x + ballot.w);
    mem.Store(8, wide);
    uint was;
    mem.InterlockedAdd(16, a, was);
    mem.Store(20, was);
}
";
        let listing = disassemble(&dxc::compile(src, "main", "cs_6_6", &[]).unwrap());
        let (mut seen, mut attrs) = (Vec::new(), "");
        for line in listing.lines() {
            if let Some(a) = line.strip_prefix("; Function Attrs:") {
                attrs = a;
            }
            let Some((ret, rest)) = line.strip_prefix("declare ").and_then(|l| l.split_once(" @dx.op.")) else { continue };
            let (name, rest) = rest.split_once('(').unwrap();
            let (params, _) = rest.split_once(')').unwrap();
            let (function, suffix) = name.split_once('.').unwrap_or((name, ""));
            let o = ops::CORE_OPS.iter().find(|o| o.function == function && o.overloads.is_empty() == suffix.is_empty()).unwrap_or_else(|| panic!("no operation for {line}"));
            let want_params: Vec<String> = std::iter::once(Ty::I32).chain(o.params.iter().map(|p| p.1)).map(|t| listed(t, suffix).unwrap_or_else(|| panic!("{line}: {t:?}"))).collect();
            assert_eq!((ret, params), (listed(o.ret, suffix).unwrap().as_str(), want_params.join(", ").as_str()), "{}: {line}", o.name);
            let mut got: Vec<&str> = attrs.split_whitespace().collect();
            got.sort();
            let mut want = vec!["nounwind"];
            want.extend(match o.attr {
                Attr::None => None,
                Attr::ReadNone => Some("readnone"),
                Attr::ReadOnly => Some("readonly"),
                Attr::ArgMemOnly => Some("argmemonly"),
                Attr::NoDuplicate => Some("noduplicate"),
                Attr::NoReturn => Some("noreturn"),
            });
            want.sort();
            // the table is DXC 1.9.2609's: before 1.8.2405 WaveGetLaneIndex was readnone
            let release: Vec<u32> = dxc::toolchain().dxcompiler.file_version.split('.').filter_map(|n| n.parse().ok()).collect();
            if o.name == "WaveGetLaneIndex" && release.as_slice() < [1, 8, 2405].as_slice() {
                want[1] = "readnone";
            }
            assert_eq!(got, want, "{}: {line}", o.name);
            if !suffix.is_empty() {
                let bit = ["f16", "f32", "f64", "i1", "i8", "i16", "i32", "i64"].iter().position(|s| *s == suffix).unwrap();
                assert!(ops::CORE_OPS.iter().any(|x| x.class == o.class && x.overloads.first().is_some_and(|m| m.0 >> bit & 1 == 1)), "{line}: no {} takes {suffix}", o.class);
            }
            seen.push(name.to_string());
        }
        for function in ["createHandleFromHeap", "annotateHandle", "rawBufferLoad.i32", "rawBufferStore.i32", "threadId.i32", "wavePrefixOp", "wavePrefixOp.i32", "waveActiveBallot", "sampleLevel.f32", "atomicBinOp.i32"] {
            assert!(seen.iter().any(|s| s == function), "DXC declared no dx.op.{function}: {seen:?}");
        }
        println!("{} dx.op functions compared: {seen:?}", seen.len());
    }
}
