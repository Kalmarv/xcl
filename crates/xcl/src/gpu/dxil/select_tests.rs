//! `Module::optimize`'s select folding (`fold_guarded_selects`) on the GPU: the same program
//! translated with and without it gives the same words in every register of every lane,
//! inactive lanes included.

use super::kernel::{translate, KernelOptions};
use super::module::NO_SELECT_FOLD;
use crate::gpu::rdna2::encode;
use crate::gpu::recomp::runner::{self, KernelState};

const V: u64 = 256;

fn prog(list: &[(&str, &str, &[(&str, u64)])]) -> Vec<u8> {
    let mut v = Vec::new();
    for (n, e, f) in list {
        v.extend(encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}")));
    }
    v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    v
}

fn translate_fold(code: &[u8], o: KernelOptions, fold: bool) -> Vec<u8> {
    NO_SELECT_FOLD.with(|c| c.set(!fold));
    let r = translate(code, o);
    NO_SELECT_FOLD.with(|c| c.set(false));
    r.unwrap_or_else(|e| panic!("translate (fold {fold}): {e}"))
}

/// Chains of VALU writes under a narrowed EXEC, narrowed again and restored, with the
/// registers read by `v_readlane` and DPP while lanes are off; NaN, signed zeros, infinities
/// and denormals as operands; sentinels in the lanes that never run.
#[test]
fn gpu_select_fold_keeps_inactive_lanes() {
    match (runner::adapter(), runner::wave64_programs_supported()) {
        (Ok(_), Ok(true)) => {}
        (a, _) => return runner::skip_test(if a.is_err() { "no GPU" } else { "no wave64" }),
    }
    let code = prog(&[
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 0)]),
        ("V_MUL_LEGACY_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V), ("VSRC1", 1)]),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 4), ("VSRC1", 2)]),
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 5), ("SRC0", V + 4), ("SRC1", V + 3), ("SRC2", V + 4)]),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 5), ("VSRC1", 4)]),
        ("V_CMP_GT_F32", "ENC_VOPC", &[("SRC0", V + 4), ("VSRC1", 5)]),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 4), ("VSRC1", 5)]),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 4), ("SRC1", 131)]),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 21), ("SRC0", V + 6), ("SRC1", 137)]),
        ("S_AND_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 2)]),
        ("V_SUB_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 4), ("VSRC1", 6)]),
        ("V_MAX_F32", "ENC_VOP2", &[("VDST", 7), ("SRC0", V + 4), ("VSRC1", 0)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", V + 4)]),
        ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 9), ("SRC0", 0xFA), ("VSRC0", 4), ("DPP_CTRL", 0x111), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)]),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 4), ("SRC1", 133)]),
        ("V_MUL_LEGACY_F32", "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 6), ("VSRC1", 6)]),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 10)]),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V + 4), ("VSRC1", 6)]),
        ("V_MUL_LEGACY_F32", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 7), ("VSRC1", 8)]),
    ]);
    // NaNs, signed zeros, infinities, denormals, ordinary values
    const EDGE: [u32; 12] = [0x7FC0_1234, 0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x0000_0001, 0x807F_FFFF, 0x3FC0_0000, 0xC000_0000, 0x3F80_0000, 0x4120_0000, 0xBE80_0000];
    let mut st = KernelState::default();
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        for w in r.iter_mut().take(4) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *w = if x % 3 == 0 { x as u32 } else { EDGE[(x >> 8) as usize % EDGE.len()] };
        }
        for (k, w) in r.iter_mut().enumerate().skip(4).take(8) {
            *w = 0xA000_0000 | (l as u32) << 8 | k as u32;
        }
    }
    (st.sgpr[0], st.sgpr[1]) = (0xF0F0_5A3C, 0x0FF0_C3A5);
    (st.sgpr[2], st.sgpr[3]) = (0x3333_FF0F, 0xFFFF_3333);
    let mut ran = 0;
    for (wave64, split) in [(false, false), (true, true), (true, false)] {
        if wave64 && !split && runner::wave64_supported() != Ok(true) {
            continue;
        }
        let o = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
        let (old, new) = (translate_fold(&code, o, false), translate_fold(&code, o, true));
        assert!(new.len() < old.len(), "wave64 {wave64} split {split}: nothing folded ({} -> {} B)", old.len(), new.len());
        let a = runner::run(&old, &st, &[]).expect("GPU run without the fold");
        let b = runner::run(&new, &st, &[]).expect("GPU run with the fold");
        let lanes = if wave64 { 64 } else { 32 };
        let mut bad = Vec::new();
        for l in 0..lanes {
            for k in 0..12 {
                if a.vgpr[l][k] != b.vgpr[l][k] {
                    bad.push(format!("lane {l} v{k}: {:#010x} without, {:#010x} with", a.vgpr[l][k], b.vgpr[l][k]));
                }
            }
            // lanes off from the start keep their sentinels
            let on = (st.sgpr[l / 32] >> (l % 32)) & 1 == 1;
            if !on {
                for k in [4, 5, 7, 8] {
                    assert_eq!(b.vgpr[l][k], st.vgpr[l][k], "wave64 {wave64} split {split}: lane {l} v{k} was written while off");
                }
            }
        }
        for k in 0..128 {
            if a.sgpr[k] != b.sgpr[k] {
                bad.push(format!("s{k}: {:#010x} without, {:#010x} with", a.sgpr[k], b.sgpr[k]));
            }
        }
        if a.scc != b.scc {
            bad.push(format!("scc: {} without, {} with", a.scc, b.scc));
        }
        assert!(bad.is_empty(), "wave64 {wave64} split {split}:\n{}", bad.join("\n"));
        ran += 1;
    }
    assert!(ran >= 2);
}
