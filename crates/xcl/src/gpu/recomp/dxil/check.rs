//! Test-time checks of DXIL containers against Microsoft's own tools, for the tests of this
//! module: `dxil.dll`'s validator says whether a container is valid DXIL. The libraries are the
//! ones [`dxc`] loads. D3D12 does not need `dxil.dll` to accept a container, so nothing here is
//! part of the program.
//!
//! A test starts with `if !check::available() { return; }`.

use std::ffi::c_void;
use std::sync::OnceLock;

use windows::core::{Interface, GUID, HRESULT};
use windows::Win32::Graphics::Direct3D::Dxc::{
    DxcValidatorFlags_Default, IDxcBlob, IDxcOperationResult, IDxcUtils, IDxcValidator, CLSID_DxcUtils, CLSID_DxcValidator, DXC_CP_ACP,
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
