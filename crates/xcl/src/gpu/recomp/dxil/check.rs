//! Test-time checks of DXIL containers against Microsoft's own tools, for the tests of this
//! module: `dxil.dll`'s validator says whether a container is valid DXIL, and DXC's listing of
//! a container is compared with its listing of what it compiles from HLSL for the same program.
//! The libraries are the ones [`dxc`] loads. D3D12 does not need `dxil.dll` to accept a
//! container, so nothing here is part of the program. A container also runs on the GPU, through
//! the kernel [`runner`].
//!
//! A test starts with `if !check::available() { return; }`, one that runs on the GPU with
//! `if !check::gpu_available() { return; }` as well.

use std::collections::BTreeSet;
use std::ffi::c_void;
use std::sync::OnceLock;

use windows::core::{Interface, GUID, HRESULT};
use windows::Win32::Graphics::Direct3D::Dxc::{
    DxcBuffer, DxcValidatorFlags_Default, IDxcBlob, IDxcCompiler3, IDxcOperationResult, IDxcResult, IDxcUtils, IDxcValidator, CLSID_DxcCompiler, CLSID_DxcUtils,
    CLSID_DxcValidator, DXC_CP_ACP,
};

use super::super::{dxc, runner};

type CreateInstance = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut c_void) -> HRESULT;

/// `DxcCreateInstance` of `dxcompiler.dll`.
fn dxcompiler() -> Result<CreateInstance, String> {
    dxc::location()?;
    let f = dxc::create_instance().ok_or("DXC is not loaded")?;
    Ok(unsafe { std::mem::transmute::<usize, CreateInstance>(f) })
}

/// `DxcCreateInstance` of the `dxil.dll` beside it.
fn dxil() -> Result<CreateInstance, String> {
    static DXIL: OnceLock<Result<usize, String>> = OnceLock::new();
    let f = DXIL
        .get_or_init(|| {
            use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
            let file = dxc::location()?.join("dxil.dll");
            if !file.is_file() {
                return Err(format!("no {}", file.display()));
            }
            // dxc.rs loaded this file before dxcompiler.dll: this is the same module
            unsafe {
                let h = LoadLibraryW(crate::util::wide(&file.to_string_lossy()).as_ptr());
                let f = if h.is_null() { None } else { GetProcAddress(h, b"DxcCreateInstance\0".as_ptr()) };
                f.map(|f| f as usize).ok_or_else(|| format!("{} has no DxcCreateInstance", file.display()))
            }
        })
        .clone()?;
    Ok(unsafe { std::mem::transmute::<usize, CreateInstance>(f) })
}

fn create<T: Interface>(create: CreateInstance, class: &GUID) -> Result<T, String> {
    let mut p = std::ptr::null_mut();
    let hr = unsafe { create(class, &T::IID, &mut p) };
    if hr.is_err() || p.is_null() {
        return Err(format!("DxcCreateInstance: {hr:?}"));
    }
    Ok(unsafe { T::from_raw(p) })
}

unsafe fn text(blob: &IDxcBlob) -> String {
    let s = std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize());
    String::from_utf8_lossy(s).trim_end_matches('\0').to_string()
}

/// What a failed operation said: its error text, else its status.
unsafe fn failure(what: &str, r: &IDxcOperationResult, status: HRESULT) -> String {
    let said = r.GetErrorBuffer().ok().map(|b| text(&b)).unwrap_or_default();
    if said.trim().is_empty() {
        format!("{what}: {status:?}")
    } else {
        said
    }
}

/// Whether DXC and its `dxil.dll` are here. When they are not, the calling test is reported as
/// `SKIPPED <test>: no DXC` and returns.
pub fn available() -> bool {
    if let Err(e) = dxcompiler().and(dxil()) {
        eprintln!("{e}");
        runner::skip_test("no DXC");
        return false;
    }
    true
}

/// Validate a DXIL container with `dxil.dll`'s `IDxcValidator`; the error is the validator's
/// text. The container is not changed (the validator does not look at its digest).
pub fn validate(container: &[u8]) -> Result<(), String> {
    let utils: IDxcUtils = create(dxcompiler()?, &CLSID_DxcUtils)?;
    let validator: IDxcValidator = create(dxil()?, &CLSID_DxcValidator)?;
    unsafe {
        let blob = utils.CreateBlob(container.as_ptr() as *const c_void, container.len() as u32, DXC_CP_ACP).map_err(|e| format!("CreateBlob: {e}"))?;
        let r = validator.Validate(&blob, DxcValidatorFlags_Default).map_err(|e| format!("Validate: {e}"))?;
        let status = r.GetStatus().map_err(|e| format!("{e}"))?;
        if status.is_err() {
            return Err(failure("Validate", &r, status));
        }
    }
    Ok(())
}

/// DXC's listing of a container (`IDxcCompiler3::Disassemble`, what `dxc -dumpbin` prints).
pub fn disassemble(container: &[u8]) -> Result<String, String> {
    let compiler: IDxcCompiler3 = create(dxcompiler()?, &CLSID_DxcCompiler)?;
    let buf = DxcBuffer { Ptr: container.as_ptr() as *const c_void, Size: container.len(), Encoding: DXC_CP_ACP.0 };
    unsafe {
        let r: IDxcResult = compiler.Disassemble(&buf).map_err(|e| format!("Disassemble: {e}"))?;
        let status = r.GetStatus().map_err(|e| format!("{e}"))?;
        if status.is_err() {
            return Err(failure("Disassemble", &r, status));
        }
        Ok(text(&r.GetResult().map_err(|e| format!("{e}"))?))
    }
}

/// The container DXC compiles from `hlsl` and its listing, to [`compare`] with the listing of
/// the container XCL writes for the same program. `profile` e.g. "cs_6_6"; `extra` are dxc
/// arguments.
pub fn reference(hlsl: &str, entry: &str, profile: &str, extra: &[&str]) -> Result<(Vec<u8>, String), String> {
    let container = dxc::compile(hlsl, entry, profile, extra)?;
    let listing = disassemble(&container)?;
    Ok((container, listing))
}

/// Whether there is a GPU to [`run`] on. When there is none, the calling test is reported as
/// `SKIPPED <test>: no GPU` and returns.
pub fn gpu_available() -> bool {
    if let Err(e) = runner::adapter() {
        eprintln!("{e}");
        runner::skip_test("no GPU");
        return false;
    }
    true
}

/// Run a compute container on the GPU as the kernel runner runs a kernel ([`runner::run_with`]):
/// XCL's ABI root signature, one thread group, `io` as the first dwords of u0, a zeroed u1,
/// `textures[i]` at `ResourceDescriptorHeap[2 * i]` (its UAV at `2 * i + 1`) and `samplers[j]`
/// at `SamplerDescriptorHeap[j]`. Returns u1 (`KERNEL_RES_DWORDS` dwords).
pub fn run(container: &[u8], io: &[u32], textures: &[runner::TestTexture], samplers: &[runner::TestSampler]) -> Result<Vec<u32>, String> {
    use super::super::{KERNEL_LANES, KERNEL_SCC_SLOT, KERNEL_SGPRS, KERNEL_VGPRS};
    // the runner's u0 is the SGPRs, SCC in one of their slots, then each lane's VGPRs
    let mut dwords = io.to_vec();
    if dwords.len() > KERNEL_SGPRS + KERNEL_LANES * KERNEL_VGPRS {
        return Err(format!("{} dwords do not fit u0", dwords.len()));
    }
    dwords.resize(KERNEL_SGPRS + KERNEL_LANES * KERNEL_VGPRS, 0);
    let state = runner::KernelState {
        sgpr: dwords[..KERNEL_SGPRS].to_vec(),
        vgpr: dwords[KERNEL_SGPRS..].chunks(KERNEL_VGPRS).map(<[u32]>::to_vec).collect(),
        scc: dwords[KERNEL_SCC_SLOT],
    };
    Ok(runner::run_with(container, &state, &[], textures, samplers)?.raw)
}

/// What the listings of two containers of one program may differ in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Differ<'a> {
    /// Lines that contain this text.
    Line(&'a str),
    /// The node `!llvm.ident` names: the compiler's name and version.
    Ident,
    /// TBAA: the `!tbaa` attachments of loads and stores, and the nodes they lead to.
    Tbaa,
}

/// The digest of the container, which the listing's header prints.
pub const SHADER_HASH: Differ = Differ::Line("; shader hash: ");

/// The numbers of the metadata nodes `text` names (`!7`).
fn node_refs(text: &str) -> Vec<u32> {
    let number = |at: usize| text[at + 1..].split(|c: char| !c.is_ascii_digit()).next().and_then(|n| n.parse().ok());
    text.match_indices('!').filter_map(|(at, _)| number(at)).collect()
}

/// Drop the lines that define these nodes and the nodes they lead to.
fn drop_nodes(lines: &mut Vec<String>, mut nodes: Vec<u32>) {
    let mut dropped = BTreeSet::new();
    while let Some(n) = nodes.pop() {
        let head = format!("!{n} = ");
        if let Some(line) = lines.iter().find(|l| l.starts_with(&head)) {
            if dropped.insert(head.clone()) {
                nodes.extend(node_refs(&line[head.len()..]));
            }
        }
    }
    lines.retain(|l| !dropped.iter().any(|head| l.starts_with(head)));
}

/// The lines of a listing without what `differ` names.
fn kept(listing: &str, differ: &[Differ]) -> Vec<String> {
    let mut lines: Vec<String> = listing.lines().map(str::to_string).collect();
    for d in differ {
        match *d {
            Differ::Line(text) => lines.retain(|l| !l.contains(text)),
            Differ::Ident => {
                let named = lines.iter().filter_map(|l| l.strip_prefix("!llvm.ident = ")).flat_map(node_refs).collect();
                drop_nodes(&mut lines, named);
            }
            Differ::Tbaa => {
                const TAG: &str = ", !tbaa !";
                let mut tags = Vec::new();
                for l in &mut lines {
                    while let Some(at) = l.find(TAG) {
                        let digits = l[at + TAG.len()..].chars().take_while(char::is_ascii_digit).count();
                        let end = at + TAG.len() + digits;
                        tags.extend(l[at + TAG.len()..end].parse::<u32>());
                        l.replace_range(at..end, "");
                    }
                }
                drop_nodes(&mut lines, tags);
            }
        }
    }
    lines
}

/// Compare two listings line by line, `ours` of a container XCL wrote and DXC's `reference`
/// for the same program, both without what `differ` names. The error shows the first line
/// that differs. Nothing is renumbered: a metadata node that only one listing has (other than
/// the TBAA nodes, which come last) shifts every later `!N` and shows as a difference.
pub fn compare(ours: &str, reference: &str, differ: &[Differ]) -> Result<(), String> {
    let (a, b) = (kept(ours, differ), kept(reference, differ));
    let Some(at) = (0..a.len().max(b.len())).find(|&i| a.get(i) != b.get(i)) else {
        return Ok(());
    };
    let line = |l: &[String]| l.get(at).map_or("(the listing ends)", String::as_str).to_string();
    let before = a[at.saturating_sub(3)..at].join("\n");
    Err(format!("the listings differ at line {} (of {} and {} compared):\n{before}\n- ours:      {}\n- reference: {}", at + 1, a.len(), b.len(), line(&a), line(&b)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const HLSL: &str = "
RWByteAddressBuffer res : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint a = res.Load(id.x * 4);
    res.Store(4, a * 3 + 1);
}
";

    /// A program with `lds`: its loads and stores carry TBAA.
    const HLSL_LDS: &str = "
RWByteAddressBuffer res : register(u0);
groupshared uint lds[64];
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    lds[gi] = res.Load(id.x * 4);
    GroupMemoryBarrierWithGroupSync();
    res.Store(gi * 4, lds[63 - gi]);
}
";

    /// Some lines of DXC 1.9.2609's listing of `HLSL_LDS`.
    const LISTING: &str = "\
; shader hash: 868e2ccc077259ade84ccc43a91d88a9
define void @main() {
  store i32 %7, i32 addrspace(3)* %8, align 4, !tbaa !10
  %11 = load i32, i32 addrspace(3)* %10, align 4, !tbaa !10
  ret void
}
!llvm.ident = !{!0}
!dx.version = !{!1}
!0 = !{!\"dxcoob 1.9.2609.5 (01b62ad47)\"}
!1 = !{i32 1, i32 6}
!10 = !{!11, !11, i64 0}
!11 = !{!\"int\", !12, i64 0}
!12 = !{!\"omnipotent char\", !13, i64 0}
!13 = !{!\"Simple C/C++ TBAA\"}
";

    /// What another writer gives for it: its own digest and name, no TBAA.
    const OURS: &str = "\
; shader hash: 00000000000000000000000000000000
define void @main() {
  store i32 %7, i32 addrspace(3)* %8, align 4
  %11 = load i32, i32 addrspace(3)* %10, align 4
  ret void
}
!llvm.ident = !{!0}
!dx.version = !{!1}
!0 = !{!\"xcl\"}
!1 = !{i32 1, i32 6}
";

    #[test]
    fn listings_compare_without_what_is_named() {
        let all = [SHADER_HASH, Differ::Ident, Differ::Tbaa];
        assert_eq!(compare(OURS, LISTING, &all), Ok(()));
        assert_eq!(compare(LISTING, LISTING, &[]), Ok(()));
        // Each of the three is needed, and the error shows the first line that differs.
        let said = compare(OURS, LISTING, &all[1..]).unwrap_err();
        assert!(said.contains("at line 1 ") && said.contains("- reference: ; shader hash: 868e"), "{said}");
        let said = compare(OURS, LISTING, &[SHADER_HASH, Differ::Tbaa]).unwrap_err();
        assert!(said.contains("- ours:      !0 = !{!\"xcl\"}"), "{said}");
        let said = compare(OURS, LISTING, &all[..2]).unwrap_err();
        assert!(said.contains("at line 2 ") && said.contains("- reference:   store i32 %7, i32 addrspace(3)* %8, align 4, !tbaa !10"), "{said}");
        // A line that is not named still counts, and so does a listing that ends early.
        let said = compare(&OURS.replace("i32 6}", "i32 5}"), LISTING, &all).unwrap_err();
        assert!(said.contains("- ours:      !1 = !{i32 1, i32 5}") && said.contains("- reference: !1 = !{i32 1, i32 6}"), "{said}");
        let said = compare(&OURS.replace("  ret void\n}\n", ""), LISTING, &all).unwrap_err();
        assert!(said.contains("- ours:      !llvm.ident") && said.contains("- reference:   ret void"), "{said}");
        let said = compare(OURS.trim_end_matches("!1 = !{i32 1, i32 6}\n"), LISTING, &all).unwrap_err();
        assert!(said.contains("- ours:      (the listing ends)"), "{said}");
    }

    #[test]
    fn a_dxc_container_matches_its_listing() {
        if !available() {
            return;
        }
        let (container, listing) = reference(HLSL, "main", "cs_6_6", &[]).unwrap();
        assert!(listing.contains("define void @main()") && listing.contains("; shader hash: "), "{listing}");
        assert_eq!(compare(&disassemble(&container).unwrap(), &listing, &[]), Ok(()));
        // DXC compiles the same program to the same listing, and another program to another.
        let (_, again) = reference(HLSL, "main", "cs_6_6", &[]).unwrap();
        assert_eq!(compare(&again, &listing, &[]), Ok(()));
        let (_, other) = reference(&HLSL.replace("a * 3", "a * 5"), "main", "cs_6_6", &[]).unwrap();
        let said = compare(&other, &listing, &[SHADER_HASH]).unwrap_err();
        assert!(said.contains("- ours:      ") && said.contains("mul i32") && said.contains(", 5"), "{said}");
    }

    /// u0 reaches the program as it was given, past the runner's register slots too, and u1
    /// comes back.
    #[test]
    fn a_dxc_container_runs_on_the_gpu() {
        if !available() || !gpu_available() {
            return;
        }
        let hlsl = "
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
[numthreads(1, 1, 1)]
void main() {
    res.Store4(8, uint4(io.Load(0) * 3 + 1, io.Load(125 * 4), io.Load(128 * 4), io.Load(700 * 4)));
}
";
        let container = dxc::compile(hlsl, "main", "cs_6_6", &[]).unwrap();
        let mut io = vec![0u32; 701];
        (io[0], io[125], io[128], io[700]) = (14, 0xDEAD_BEEF, 77, 0x1234_5678);
        let res = run(&container, &io, &[], &[]).unwrap();
        assert_eq!(res.len(), crate::gpu::recomp::KERNEL_RES_DWORDS);
        assert_eq!(res[..7], [0, 0, 43, 0xDEAD_BEEF, 77, 0x1234_5678, 0]);
        // What D3D12 refuses comes back as the error.
        assert!(run(b"DXBC", &[], &[], &[]).unwrap_err().contains("CreateComputePipelineState"));
    }

    /// What `Differ::Tbaa` and `Differ::Ident` look for is what the DXC in use prints.
    #[test]
    fn tbaa_and_the_compilers_name_are_found_in_a_dxc_listing() {
        if !available() {
            return;
        }
        let (_, listing) = reference(HLSL_LDS, "main", "cs_6_6", &[]).unwrap();
        assert!(listing.contains(", !tbaa !") && listing.contains("omnipotent char") && listing.contains("!{!\"dxc"), "{listing}");
        let rest = kept(&listing, &[Differ::Ident, Differ::Tbaa]).join("\n");
        assert!(!rest.contains("tbaa") && !rest.contains("omnipotent char") && !rest.contains("TBAA") && !rest.contains("!{!\"dxc"), "{rest}");
        assert!(rest.contains("store i32") && rest.contains("!llvm.ident = !{!"), "{rest}");
        assert_eq!(listing.lines().count() - rest.lines().count(), 5, "the name's node and four of TBAA");
    }

    /// Where the data of a container's part starts.
    fn part(container: &[u8], kind: &[u8; 4]) -> usize {
        let word = |at: usize| u32::from_le_bytes(container[at..at + 4].try_into().unwrap()) as usize;
        (0..word(28)).map(|i| word(32 + 4 * i)).find(|&at| &container[at..at + 4] == kind).expect("part") + 8
    }

    #[test]
    fn a_dxc_container_validates() {
        if !available() {
            return;
        }
        let container = dxc::compile(HLSL, "main", "cs_6_6", &[]).unwrap();
        assert_eq!(validate(&container), Ok(()));
    }

    #[test]
    fn the_validator_says_what_is_wrong() {
        if !available() {
            return;
        }
        let mut container = dxc::compile(HLSL, "main", "cs_6_6", &[]).unwrap();
        // The PSV0 part says a smallest wave of 4 lanes (after its size, at 16 in the runtime
        // info), the module none. DXC 1.9 names `'PSV0' part`, 1.7 `'Pipeline State Validation'`.
        let wave = part(&container, b"PSV0") + 4 + 16;
        assert_eq!(container[wave], 0);
        container[wave] = 4;
        let said = validate(&container).unwrap_err();
        assert!(said.contains("PSV0") || said.contains("Pipeline State Validation"), "{said}");
        // Not a container at all.
        assert!(validate(b"DXBC").is_err());
    }

    /// The shader model of the program header must be the module's. dxil.dll 1.9.2609 refuses
    /// a container where it is not; 1.7.2308 accepts it (1.8 was not tried). So a container
    /// XCL wrote is shown right in this only by a run with dxil.dll 1.9 or newer.
    #[test]
    fn the_validator_checks_the_shader_model_of_the_program_header() {
        if !available() {
            return;
        }
        let (major, minor) = dxc::toolchain().dxil.version().unwrap_or_default();
        if (major, minor) < (1, 9) {
            runner::skip_test(&format!("dxil.dll {major}.{minor} is older than 1.9, which checks the program header's shader model (1.7 does not)"));
            return;
        }
        let mut container = dxc::compile(HLSL, "main", "cs_6_6", &[]).unwrap();
        let program = part(&container, b"DXIL");
        assert_eq!(container[program], 0x66);
        container[program] = 0x65;
        let said = validate(&container).unwrap_err();
        assert!(said.contains("6.5") && said.contains("6.6"), "{said}");
    }
}
