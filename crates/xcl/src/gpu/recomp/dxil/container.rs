//! The DXBC container a DXIL program is handed to D3D12 in, written with its digest.
//!
//! The layout: a header (`DXBC`, a 16-byte digest, version 1, the container's size, the number
//! of parts), the directory of the parts' offsets right after it, then the parts one after
//! another in the directory's order, each its four-character kind, its length and its bytes.
//! The digest is MD5's block function over the bytes from 20 on with the compiler's own padding
//! ([`checksum`]); D3D12 refuses a container whose digest is not the one it computes.
//!
//! Ported from Kalmarv/dx (https://github.com/Kalmarv/dx, `dxcontainer/src/lib.rs` at commit
//! 02d837a0d4788eee8eb64ac7682d5cc2160012aa: `serialize`, `checksum`, `repair_digest`) and from
//! primase-foundations 0.3.1 (https://github.com/Kalmarv/primase-foundations, `src/hash.rs`: the
//! MD5 block function of RFC 1321). Both are published under GPL-3.0-or-later and contributed
//! to XCL under GPL-2.0-only by their author. Changed for XCL (2026-11-24): one module, parts as
//! (kind, bytes) pairs, `String` errors, no reader.

/// MD5's per-round shift amounts and sine constants (RFC 1321 §3.4).
const MD5_SHIFT: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23,
    4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];
const MD5_SINE: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];
/// MD5's initial state (RFC 1321 §3.3).
const MD5_INIT: [u32; 4] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476];

/// MD5's block function (RFC 1321 §3.4): `state` advanced by one 64-byte block.
fn md5_block(state: &mut [u32; 4], block: &[u8; 64]) {
    let mut m = [0u32; 16];
    for (w, bytes) in m.iter_mut().zip(block.chunks_exact(4)) {
        *w = u32::from_le_bytes(bytes.try_into().unwrap());
    }
    let [mut a, mut b, mut c, mut d] = *state;
    for (i, (&shift, &sine)) in MD5_SHIFT.iter().zip(&MD5_SINE).enumerate() {
        let (f, g) = match i / 16 {
            0 => ((b & c) | (!b & d), i),
            1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
            2 => (b ^ c ^ d, (3 * i + 5) % 16),
            _ => (c ^ (b | !d), (7 * i) % 16),
        };
        let rotated = a.wrapping_add(f).wrapping_add(sine).wrapping_add(m[g]).rotate_left(shift);
        (a, b, c, d) = (d, b.wrapping_add(rotated), b, c);
    }
    for (s, x) in state.iter_mut().zip([a, b, c, d]) {
        *s = s.wrapping_add(x);
    }
}

/// The digest a container stores at bytes 4 to 20: MD5's block function over its bytes from 20
/// on, the last block padded as the compiler does (the bit count first, then the rest, `0x80`,
/// zeros, and at 60 the bit count over 4, or 1; a rest of 56 bytes or more padded in a block of
/// its own first). `None` for fewer than 20 bytes or more than a 32-bit bit count holds.
pub fn checksum(container: &[u8]) -> Option<[u8; 16]> {
    let data = container.get(20..)?;
    let bits = u32::try_from(data.len()).ok()?.wrapping_mul(8);
    let mut blocks = data.chunks_exact(64);
    let mut state = MD5_INIT;
    for block in &mut blocks {
        md5_block(&mut state, block.try_into().unwrap());
    }
    let rest = blocks.remainder();
    let mut last = [0u8; 64];
    if rest.len() < 56 {
        last[4..4 + rest.len()].copy_from_slice(rest);
        last[4 + rest.len()] = 0x80;
    } else {
        let mut first = [0u8; 64];
        first[..rest.len()].copy_from_slice(rest);
        first[rest.len()] = 0x80;
        md5_block(&mut state, &first);
    }
    last[..4].copy_from_slice(&bits.to_le_bytes());
    last[60..].copy_from_slice(&((bits >> 2) | 1).to_le_bytes());
    md5_block(&mut state, &last);
    let mut out = [0u8; 16];
    for (o, s) in out.chunks_exact_mut(4).zip(state) {
        o.copy_from_slice(&s.to_le_bytes());
    }
    Some(out)
}

/// Writes `parts` (kind, bytes) as a container: the header, the directory of the parts'
/// offsets, then the parts in the order given, each its kind, its length and its bytes with
/// nothing between them, and the digest [`checksum`] computes over the result. A part's bytes
/// are written as given (a caller that wants 4-byte aligned offsets pads its parts). Refuses a
/// kind twice and a size that does not fit the header's 32 bits.
pub fn serialize(parts: &[([u8; 4], &[u8])]) -> Result<Vec<u8>, String> {
    let too_big = || "DXBC container: larger than the header's 32-bit size can state".to_string();
    let mut offset = 32 + 4 * parts.len();
    let mut directory = Vec::with_capacity(parts.len());
    for (i, (kind, data)) in parts.iter().enumerate() {
        if let Some(first) = parts[..i].iter().position(|p| p.0 == *kind) {
            return Err(format!("DXBC container: parts {first} and {i} are both {}", String::from_utf8_lossy(kind)));
        }
        directory.push(u32::try_from(offset).map_err(|_| too_big())?);
        offset += 8 + data.len();
    }
    let size = u32::try_from(offset).map_err(|_| too_big())?;
    let mut out = Vec::with_capacity(offset);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0; 16]);
    for word in [1, size, parts.len() as u32] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    for at in directory {
        out.extend_from_slice(&at.to_le_bytes());
    }
    for (kind, data) in parts {
        out.extend_from_slice(kind);
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
    }
    repair_digest(&mut out)?;
    Ok(out)
}

/// Rewrites the digest at bytes 4 to 20 of `container` to the one [`checksum`] computes over
/// the rest of it: for a container whose bytes were edited in place.
pub fn repair_digest(container: &mut [u8]) -> Result<(), String> {
    if container.get(..4) != Some(b"DXBC") || container.len() < 20 {
        return Err("DXBC container: no DXBC magic and 16-byte digest".into());
    }
    let digest = checksum(container).ok_or("DXBC container: too long for the digest's 32-bit bit count")?;
    container[4..20].copy_from_slice(&digest);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::recomp::{dxc, runner};

    /// The MD5 block function with RFC 1321's own padding gives the RFC's test suite (§A.5).
    #[test]
    fn md5_block_function_matches_rfc_1321() {
        fn md5(bytes: &[u8]) -> [u8; 16] {
            let mut tail = bytes.to_vec();
            tail.push(0x80);
            while tail.len() % 64 != 56 {
                tail.push(0);
            }
            tail.extend_from_slice(&(bytes.len() as u64 * 8).to_le_bytes());
            let mut state = MD5_INIT;
            for block in tail.chunks_exact(64) {
                md5_block(&mut state, block.try_into().unwrap());
            }
            let mut out = [0u8; 16];
            for (o, s) in out.chunks_exact_mut(4).zip(state) {
                o.copy_from_slice(&s.to_le_bytes());
            }
            out
        }
        let hex = |d: [u8; 16]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        for (input, want) in [
            ("", "d41d8cd98f00b204e9800998ecf8427e"),
            ("a", "0cc175b9c0f1b6a831c399e269772661"),
            ("abc", "900150983cd24fb0d6963f7d28e17f72"),
            ("message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
            ("abcdefghijklmnopqrstuvwxyz", "c3fcd3d76192e4007dfb496cca67e13b"),
            ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", "d174ab98d277d9f5a5611c2c9f419d9f"),
            ("12345678901234567890123456789012345678901234567890123456789012345678901234567890", "57edf4a22be3c955ac49da2e2107b67a"),
        ] {
            assert_eq!(hex(md5(input.as_bytes())), want, "{input:?}");
        }
    }

    /// The layout `serialize` writes, read back with the crate's own part reader, and a digest
    /// that follows each edit; both padding branches (a rest under and over 56 bytes) are hit.
    #[test]
    fn serialize_writes_the_layout_and_the_digest() {
        let bytes = serialize(&[(*b"ISGN", &[]), (*b"SFI0", &[1, 2, 3, 4, 5, 6, 7, 8])]).unwrap();
        assert_eq!(bytes.len(), 32 + 8 + 8 + 16);
        assert_eq!(&bytes[20..32], &[1, 0, 0, 0, 64, 0, 0, 0, 2, 0, 0, 0]);
        let parts = crate::gpu::shader::container_parts(&bytes).unwrap();
        assert_eq!((parts[0].fourcc.as_str(), parts[0].offset, parts[0].size), ("ISGN", 48, 0));
        assert_eq!((parts[1].fourcc.as_str(), parts[1].offset, parts[1].size), ("SFI0", 56, 8));
        assert_eq!(checksum(&bytes).unwrap()[..], bytes[4..20]);
        for len in 0..140 {
            let data = vec![0x5a; len];
            let mut bytes = serialize(&[(*b"DXIL", &data)]).unwrap();
            assert_eq!(checksum(&bytes).unwrap()[..], bytes[4..20], "{len}");
            let before = checksum(&bytes).unwrap();
            *bytes.last_mut().unwrap() ^= 1;
            assert_ne!(checksum(&bytes).unwrap(), before, "{len}: the last byte is in the digest");
            repair_digest(&mut bytes).unwrap();
            assert_eq!(checksum(&bytes).unwrap()[..], bytes[4..20], "{len}");
        }
        assert!(serialize(&[(*b"DXIL", &[]), (*b"DXIL", &[])]).is_err());
        assert!(repair_digest(&mut [0u8; 19]).is_err());
    }

    /// A container DXC wrote (and dxil.dll signed): its digest zeroed and computed again is
    /// the one DXC stored, and `serialize` of its parts is the container byte for byte.
    #[test]
    fn digest_of_a_dxc_container() {
        if dxc::location().is_err() {
            runner::skip_test("no DXC");
            return;
        }
        let src = "RWByteAddressBuffer mem : register(u2);\n[numthreads(1, 1, 1)] void main() { mem.Store(0, 0x12345678u); }\n";
        let signed = dxc::compile(src, "main", "cs_6_6", &[]).unwrap();
        let stored: [u8; 16] = signed[4..20].try_into().unwrap();
        assert_ne!(stored, [0; 16], "DXC stored no digest (no dxil.dll next to it?)");
        let mut zeroed = signed.clone();
        zeroed[4..20].fill(0);
        assert_eq!(checksum(&zeroed), Some(stored));
        repair_digest(&mut zeroed).unwrap();
        assert_eq!(zeroed, signed);
        let parts = crate::gpu::shader::container_parts(&signed).unwrap();
        let parts: Vec<([u8; 4], &[u8])> = parts.iter().map(|p| (p.fourcc.as_bytes().try_into().unwrap(), &signed[p.offset..p.offset + p.size])).collect();
        assert_eq!(serialize(&parts).unwrap(), signed);
    }

    /// `value` as a bitstream VBR of `width`-bit chunks (the low chunk first, a chunk's top
    /// bit says another follows), one bool per bit in stream order.
    fn vbr_bits(mut value: u64, width: u32) -> Vec<bool> {
        let (mut bits, data) = (Vec::new(), width - 1);
        loop {
            let (chunk, more) = (value & ((1 << data) - 1), value >> data != 0);
            bits.extend((0..data).map(|i| chunk >> i & 1 == 1));
            bits.push(more);
            value >>= data;
            if !more {
                return bits;
            }
        }
    }

    /// `bytes` with the one place the bit pattern `from` is at replaced by `to` (a bitstream
    /// is read from each byte's low bit up). `None` unless `from` is there exactly once.
    fn replace_bits(bytes: &[u8], from: &[bool], to: &[bool]) -> Option<Vec<u8>> {
        assert_eq!(from.len(), to.len());
        let bit = |at: usize| bytes[at / 8] >> (at % 8) & 1 == 1;
        let found: Vec<usize> = (0..(bytes.len() * 8).saturating_sub(from.len())).filter(|&at| from.iter().enumerate().all(|(i, &b)| bit(at + i) == b)).collect();
        let [at] = found[..] else { return None };
        let mut out = bytes.to_vec();
        for (i, &b) in to.iter().enumerate() {
            out[(at + i) / 8] = out[(at + i) / 8] & !(1 << ((at + i) % 8)) | (b as u8) << ((at + i) % 8);
        }
        Some(out)
    }

    /// What D3D12 wants of the digest, which decides whether XCL needs dxil.dll for DXIL it
    /// writes itself: a compute pipeline is created, and runs, from a container whose DXIL part
    /// was changed after DXC and dxil.dll were done with it (the stored constant, in the
    /// bitcode) and whose digest XCL computed. The validator never saw that program, and the
    /// `HASH` part still holds the old program's hash. A container with no digest is refused.
    #[test]
    fn d3d12_takes_a_program_with_a_digest_of_ours() {
        if let Err(why) = runner::adapter() {
            println!("no GPU: {why}");
            runner::skip_test("no GPU");
            return;
        }
        if dxc::location().is_err() {
            runner::skip_test("no DXC");
            return;
        }
        let (old, new) = (0x12345678u32, 0x1234a8a7u32);
        let src = format!("RWByteAddressBuffer mem : register(u2);\n[numthreads(1, 1, 1)] void main() {{ mem.Store(0, {old:#x}u); }}\n");
        let signed = dxc::compile(&src, "main", "cs_6_6", &[]).unwrap();
        println!("{}; D3D12 Agility SDK: {:?}; dxcompiler {}", runner::adapter().unwrap(), crate::gpu::agility::loaded(), dxc::toolchain().dxcompiler);
        let stored = |container: &[u8]| runner::dispatch(container, [1, 1, 1], &[], &[0; 64]).map(|mem| mem[0]);
        assert_eq!(stored(&signed), Ok(old), "DXC's own container");

        // a constant is a signed VBR (the value doubled) of 8-bit chunks in DXC's constants
        // block, of 6-bit chunks in a record with no abbreviation
        let parts = crate::gpu::shader::container_parts(&signed).unwrap();
        let dxil = parts.iter().find(|p| p.fourcc == "DXIL").unwrap();
        let (start, end) = (dxil.offset, dxil.offset + dxil.size);
        let edited = [8, 6].iter().find_map(|&width| replace_bits(&signed[start..end], &vbr_bits(u64::from(old) * 2, width), &vbr_bits(u64::from(new) * 2, width)));
        let mut changed = signed.clone();
        changed[start..end].copy_from_slice(&edited.expect("the constant, once in the DXIL part"));
        let stale = stored(&changed);
        assert!(stale.is_err(), "a changed program under DXC's digest");
        repair_digest(&mut changed).unwrap();
        assert_ne!(changed[4..20], signed[4..20]);
        assert_eq!(stored(&changed), Ok(new), "a changed program under our digest");

        let mut unsigned = signed.clone();
        unsigned[4..20].fill(0);
        let none = stored(&unsigned);
        assert!(none.is_err(), "no digest");
        println!("a stale digest: {stale:?}; no digest: {none:?}");
        // the `BYPASS` digest (hlsl-specs INF-0004): runtimes from Agility SDK 1.615 on take it,
        // older ones do not, so only reported
        let mut bypass = changed.clone();
        bypass[4..20].fill(1);
        println!("the all-01 bypass digest: {:?}", stored(&bypass).map(|v| v == new));
    }
}
