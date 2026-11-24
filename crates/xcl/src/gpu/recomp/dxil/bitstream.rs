//! LLVM 3.7 bitstream writer, the container DXIL's bitcode is stored in: fixed and VBR fields,
//! blocks with a backpatched length, DEFINE_ABBREV and abbreviated records, UNABBREV_RECORD and
//! the BLOCKINFO block. It knows nothing of IR.
//!
//! Ported from LLVM 3.7 `include/llvm/Bitcode/BitstreamWriter.h` and `BitCodes.h`
//! (https://github.com/llvm/llvm-project, branch release/3.7.x, University of Illinois/NCSA
//! licence), 2026-11-24. The test reader follows `dxil-reader/src/bitstream.rs` of
//! https://github.com/Kalmarv/dx (commit 02d837a, contributed under GPL-2.0-only).
//!
//! The output is a sequence of little-endian 32-bit words; bit 0 of a field is the lowest bit of
//! the first word it starts in. Misuse (a value wider than its field, an abbreviation that does
//! not fit the record, unbalanced blocks) panics, as LLVM's asserts do.

/// Abbreviation ids every block has; the first defined abbreviation is `FIRST_ABBREV`.
pub const END_BLOCK: u32 = 0;
pub const ENTER_SUBBLOCK: u32 = 1;
pub const DEFINE_ABBREV: u32 = 2;
pub const UNABBREV_RECORD: u32 = 3;
pub const FIRST_ABBREV: u32 = 4;
/// Block id of BLOCKINFO and its SETBID record code.
pub const BLOCKINFO_BLOCK_ID: u32 = 0;
pub const BLOCKINFO_CODE_SETBID: u32 = 1;

const BLOCK_ID_WIDTH: u32 = 8;
const CODE_LEN_WIDTH: u32 = 4;

/// One operand of an abbreviation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbbrevOp {
    /// The value is always this; nothing is stored.
    Literal(u64),
    /// A fixed-width field of 0..=32 bits (0 stores nothing and means 0).
    Fixed(u32),
    /// A VBR field with chunks of 2..=32 bits.
    Vbr(u32),
    /// The rest of the record, a VBR6 count and then each element in the next operand's encoding.
    Array,
    /// A 6-bit character of `[a-zA-Z0-9._]`; the record holds the character.
    Char6,
    /// A VBR6 length, 32-bit alignment, the bytes and 32-bit alignment again.
    Blob,
}

/// An abbreviation: how each of a record's values (its code first) is stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Abbrev(pub Vec<AbbrevOp>);

impl Abbrev {
    fn check(&self) {
        let ops = &self.0;
        assert!(!ops.is_empty(), "empty abbreviation");
        for (i, op) in ops.iter().enumerate() {
            match *op {
                AbbrevOp::Fixed(w) => assert!(w <= 32, "fixed width {w} above 32"),
                AbbrevOp::Vbr(w) => assert!((2..=32).contains(&w), "VBR width {w}"),
                AbbrevOp::Array => {
                    assert!(i + 2 == ops.len(), "array not second to last");
                    assert!(
                        !matches!(ops[i + 1], AbbrevOp::Array | AbbrevOp::Blob),
                        "array element is an array or blob"
                    );
                }
                AbbrevOp::Blob => assert!(i + 1 == ops.len(), "blob not last"),
                AbbrevOp::Literal(_) | AbbrevOp::Char6 => {}
            }
        }
    }
}

/// Whether `c` has a char6 code.
pub fn is_char6(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'.' || c == b'_'
}

fn encode_char6(c: u8) -> u32 {
    match c {
        b'a'..=b'z' => u32::from(c - b'a'),
        b'A'..=b'Z' => u32::from(c - b'A') + 26,
        b'0'..=b'9' => u32::from(c - b'0') + 52,
        b'.' => 62,
        b'_' => 63,
        _ => panic!("{c:#x} is not a char6 character"),
    }
}

struct Scope {
    prev_width: u32,
    prev_abbrevs: Vec<Abbrev>,
    /// Index of the block's length word in `out`.
    size_word: usize,
    id: u32,
}

/// Writes a bitstream into a byte vector.
pub struct BitWriter {
    out: Vec<u8>,
    /// Bits not yet in `out`: `cur_bits` (< 32) of them, from bit 0.
    cur: u64,
    cur_bits: u32,
    /// Abbreviation id width of the current block (2 at the top level).
    width: u32,
    abbrevs: Vec<Abbrev>,
    scopes: Vec<Scope>,
    /// Abbreviations BLOCKINFO gave each block id.
    block_info: Vec<(u32, Vec<Abbrev>)>,
    /// In BLOCKINFO, the block id of the last SETBID.
    info_target: Option<u32>,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            cur: 0,
            cur_bits: 0,
            width: 2,
            abbrevs: Vec::new(),
            scopes: Vec::new(),
            block_info: Vec::new(),
            info_target: None,
        }
    }

    /// Position in the stream, in bits.
    pub fn bit_pos(&self) -> u64 {
        self.out.len() as u64 * 8 + u64::from(self.cur_bits)
    }

    /// The stream; every block must be closed. Pads to a 32-bit boundary.
    pub fn finish(mut self) -> Vec<u8> {
        assert!(self.scopes.is_empty(), "unclosed block");
        self.align32();
        self.out
    }

    /// `value` in `width` (0..=64) bits.
    pub fn emit(&mut self, value: u64, width: u32) {
        assert!(width <= 64, "field width {width}");
        assert!(width == 64 || value >> width == 0, "{value:#x} does not fit {width} bits");
        let (mut value, mut width) = (value, width);
        while width > 0 {
            let n = width.min(32);
            self.cur |= (value & ((1u64 << n) - 1)) << self.cur_bits;
            self.cur_bits += n;
            if self.cur_bits >= 32 {
                self.out.extend_from_slice(&(self.cur as u32).to_le_bytes());
                self.cur >>= 32;
                self.cur_bits -= 32;
            }
            value >>= n;
            width -= n;
        }
    }

    /// `value` as VBR with `width`-bit chunks (2..=32).
    pub fn emit_vbr(&mut self, mut value: u64, width: u32) {
        assert!((2..=32).contains(&width), "VBR width {width}");
        let high = 1u64 << (width - 1);
        while value >= high {
            self.emit((value & (high - 1)) | high, width);
            value >>= width - 1;
        }
        self.emit(value, width);
    }

    /// Zero bits up to the next 32-bit boundary.
    pub fn align32(&mut self) {
        if self.cur_bits > 0 {
            self.out.extend_from_slice(&(self.cur as u32).to_le_bytes());
            self.cur = 0;
            self.cur_bits = 0;
        }
    }

    fn emit_code(&mut self, code: u32) {
        self.emit(u64::from(code), self.width);
    }

    /// ENTER_SUBBLOCK: `id`, abbreviation ids of `width` bits inside, the length filled in by
    /// [`exit_block`](Self::exit_block). The block starts with the abbreviations BLOCKINFO gave
    /// its id so far.
    pub fn enter_block(&mut self, id: u32, width: u32) {
        assert!((2..=32).contains(&width), "abbreviation width {width}");
        self.emit_code(ENTER_SUBBLOCK);
        self.emit_vbr(u64::from(id), BLOCK_ID_WIDTH);
        self.emit_vbr(u64::from(width), CODE_LEN_WIDTH);
        self.align32();
        let size_word = self.out.len() / 4;
        self.emit(0, 32);
        let inherited = self
            .block_info
            .iter()
            .find(|(b, _)| *b == id)
            .map(|(_, a)| a.clone())
            .unwrap_or_default();
        self.scopes.push(Scope {
            prev_width: std::mem::replace(&mut self.width, width),
            prev_abbrevs: std::mem::replace(&mut self.abbrevs, inherited),
            size_word,
            id,
        });
        if id == BLOCKINFO_BLOCK_ID {
            self.info_target = None;
        }
    }

    /// END_BLOCK, 32-bit alignment, and the block's length (in words after its length word).
    pub fn exit_block(&mut self) {
        let scope = self.scopes.pop().expect("exit_block without a block");
        self.emit_code(END_BLOCK);
        self.align32();
        let words = u32::try_from(self.out.len() / 4 - scope.size_word - 1).expect("block over 2^32 words");
        let at = scope.size_word * 4;
        self.out[at..at + 4].copy_from_slice(&words.to_le_bytes());
        self.width = scope.prev_width;
        self.abbrevs = scope.prev_abbrevs;
    }

    fn encode_abbrev(&mut self, abbrev: &Abbrev) {
        abbrev.check();
        self.emit_code(DEFINE_ABBREV);
        self.emit_vbr(abbrev.0.len() as u64, 5);
        for op in &abbrev.0 {
            let (encoding, data) = match *op {
                AbbrevOp::Literal(v) => {
                    self.emit(1, 1);
                    self.emit_vbr(v, 8);
                    continue;
                }
                AbbrevOp::Fixed(w) => (1, Some(w)),
                AbbrevOp::Vbr(w) => (2, Some(w)),
                AbbrevOp::Array => (3, None),
                AbbrevOp::Char6 => (4, None),
                AbbrevOp::Blob => (5, None),
            };
            self.emit(0, 1);
            self.emit(encoding, 3);
            if let Some(w) = data {
                self.emit_vbr(u64::from(w), 5);
            }
        }
    }

    /// DEFINE_ABBREV in the current block; returns its abbreviation id.
    pub fn define_abbrev(&mut self, abbrev: Abbrev) -> u32 {
        assert!(self.current_block() != Some(BLOCKINFO_BLOCK_ID), "use blockinfo_abbrev in BLOCKINFO");
        self.encode_abbrev(&abbrev);
        self.abbrevs.push(abbrev);
        FIRST_ABBREV + (self.abbrevs.len() - 1) as u32
    }

    /// Open the BLOCKINFO block.
    pub fn enter_blockinfo(&mut self, width: u32) {
        self.enter_block(BLOCKINFO_BLOCK_ID, width);
    }

    /// In BLOCKINFO: DEFINE_ABBREV for every later block `block_id` (after a SETBID when the
    /// previous one named another block). Returns its abbreviation id in those blocks.
    pub fn blockinfo_abbrev(&mut self, block_id: u32, abbrev: Abbrev) -> u32 {
        assert!(self.current_block() == Some(BLOCKINFO_BLOCK_ID), "not in BLOCKINFO");
        if self.info_target != Some(block_id) {
            self.unabbrev_record(BLOCKINFO_CODE_SETBID, &[u64::from(block_id)]);
            self.info_target = Some(block_id);
        }
        self.encode_abbrev(&abbrev);
        let list = match self.block_info.iter().position(|(b, _)| *b == block_id) {
            Some(i) => &mut self.block_info[i].1,
            None => {
                self.block_info.push((block_id, Vec::new()));
                &mut self.block_info.last_mut().unwrap().1
            }
        };
        list.push(abbrev);
        FIRST_ABBREV + (list.len() - 1) as u32
    }

    fn current_block(&self) -> Option<u32> {
        self.scopes.last().map(|s| s.id)
    }

    /// UNABBREV_RECORD: the code and each operand as VBR6.
    pub fn unabbrev_record(&mut self, code: u32, ops: &[u64]) {
        self.emit_code(UNABBREV_RECORD);
        self.emit_vbr(u64::from(code), 6);
        self.emit_vbr(ops.len() as u64, 6);
        for &v in ops {
            self.emit_vbr(v, 6);
        }
    }

    /// A record: unabbreviated when `abbrev` is `None`, else through that abbreviation (an array
    /// takes the operands that remain).
    pub fn record(&mut self, code: u32, ops: &[u64], abbrev: Option<u32>) {
        match abbrev {
            None => self.unabbrev_record(code, ops),
            Some(id) => self.abbreviated(id, code, ops, None),
        }
    }

    /// A record through an abbreviation that ends in a blob, which holds `blob`.
    pub fn record_with_blob(&mut self, abbrev: u32, code: u32, ops: &[u64], blob: &[u8]) {
        self.abbreviated(abbrev, code, ops, Some(blob));
    }

    fn scalar(&mut self, op: AbbrevOp, v: u64) {
        match op {
            AbbrevOp::Literal(l) => assert!(v == l, "{v} for literal {l}"),
            AbbrevOp::Fixed(0) => assert!(v == 0, "{v} for a 0-bit field"),
            AbbrevOp::Fixed(w) => self.emit(v, w),
            AbbrevOp::Vbr(w) => self.emit_vbr(v, w),
            AbbrevOp::Char6 => {
                let c = u8::try_from(v).unwrap_or(0xff);
                self.emit(u64::from(encode_char6(c)), 6);
            }
            AbbrevOp::Array | AbbrevOp::Blob => unreachable!(),
        }
    }

    fn abbreviated(&mut self, id: u32, code: u32, ops: &[u64], mut blob: Option<&[u8]>) {
        let abbrev = id
            .checked_sub(FIRST_ABBREV)
            .and_then(|i| self.abbrevs.get(i as usize))
            .cloned()
            .unwrap_or_else(|| panic!("undefined abbreviation {id}"));
        self.emit_code(id);
        let mut values = std::iter::once(u64::from(code)).chain(ops.iter().copied()).peekable();
        let mut i = 0;
        while let Some(&op) = abbrev.0.get(i) {
            match op {
                AbbrevOp::Array => {
                    let element = abbrev.0[i + 1];
                    let rest: Vec<u64> = values.by_ref().collect();
                    self.emit_vbr(rest.len() as u64, 6);
                    for v in rest {
                        self.scalar(element, v);
                    }
                    i += 1;
                }
                AbbrevOp::Blob => {
                    let bytes = blob.take().expect("blob abbreviation without a blob");
                    self.emit_vbr(bytes.len() as u64, 6);
                    self.align32();
                    self.out.extend_from_slice(bytes);
                    while self.out.len() % 4 != 0 {
                        self.out.push(0);
                    }
                }
                op => {
                    let v = values.next().expect("record shorter than its abbreviation");
                    self.scalar(op, v);
                }
            }
            i += 1;
        }
        assert!(values.peek().is_none(), "record longer than its abbreviation");
        assert!(blob.is_none(), "blob for an abbreviation without one");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A record as the test reader returns it: the abbreviation id it was stored with
    /// (`UNABBREV_RECORD` for none), its code, operands and blob.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Rec {
        abbrev: u32,
        code: u64,
        ops: Vec<u64>,
        blob: Option<Vec<u8>>,
    }
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum El {
        Block(u32, Vec<El>),
        Rec(Rec),
    }

    /// A small bitstream reader for the tests only.
    struct Reader<'a> {
        data: &'a [u8],
        pos: usize,
        info: Vec<(u32, Vec<Abbrev>)>,
    }
    impl Reader<'_> {
        fn read(&mut self, width: u32) -> u64 {
            let mut v = 0u64;
            for i in 0..width as usize {
                let bit = self.pos + i;
                v |= u64::from((self.data[bit / 8] >> (bit % 8)) & 1) << i;
            }
            self.pos += width as usize;
            v
        }
        fn vbr(&mut self, width: u32) -> u64 {
            let high = 1u64 << (width - 1);
            let (mut v, mut shift) = (0u64, 0);
            loop {
                let chunk = self.read(width);
                v |= (chunk & (high - 1)) << shift;
                if chunk & high == 0 {
                    return v;
                }
                shift += width - 1;
            }
        }
        fn align32(&mut self) {
            self.pos = self.pos.div_ceil(32) * 32;
        }
        fn define_abbrev(&mut self) -> Abbrev {
            let n = self.vbr(5) as usize;
            let mut ops = Vec::new();
            while ops.len() < n {
                if self.read(1) == 1 {
                    ops.push(AbbrevOp::Literal(self.vbr(8)));
                    continue;
                }
                ops.push(match self.read(3) {
                    1 => AbbrevOp::Fixed(self.vbr(5) as u32),
                    2 => AbbrevOp::Vbr(self.vbr(5) as u32),
                    3 => AbbrevOp::Array,
                    4 => AbbrevOp::Char6,
                    5 => AbbrevOp::Blob,
                    e => panic!("encoding {e}"),
                });
            }
            Abbrev(ops)
        }
        fn scalar(&mut self, op: AbbrevOp) -> u64 {
            match op {
                AbbrevOp::Literal(v) => v,
                AbbrevOp::Fixed(w) => self.read(w),
                AbbrevOp::Vbr(w) => self.vbr(w),
                AbbrevOp::Char6 => {
                    let chars = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._";
                    u64::from(chars[self.read(6) as usize])
                }
                AbbrevOp::Array | AbbrevOp::Blob => panic!("array or blob as an element"),
            }
        }
        /// A block's body after its header, up to and including END_BLOCK.
        fn block(&mut self, id: u32, width: u32, end: usize) -> Vec<El> {
            let mut abbrevs: Vec<Abbrev> =
                self.info.iter().find(|(b, _)| *b == id).map(|(_, a)| a.clone()).unwrap_or_default();
            let mut info_target = None;
            let mut elements = Vec::new();
            loop {
                match self.read(width) as u32 {
                    END_BLOCK => {
                        self.align32();
                        assert_eq!(self.pos, end, "block length");
                        return elements;
                    }
                    ENTER_SUBBLOCK => {
                        let (child, child_width) = self.header();
                        let words = self.read(32) as usize;
                        let child_end = self.pos + words * 32;
                        let inner = self.block(child, child_width, child_end);
                        elements.push(El::Block(child, inner));
                    }
                    DEFINE_ABBREV => {
                        let abbrev = self.define_abbrev();
                        if id == BLOCKINFO_BLOCK_ID {
                            let target = info_target.expect("abbreviation before SETBID");
                            match self.info.iter_mut().find(|(b, _)| *b == target) {
                                Some((_, list)) => list.push(abbrev),
                                None => self.info.push((target, vec![abbrev])),
                            }
                        } else {
                            abbrevs.push(abbrev);
                        }
                    }
                    UNABBREV_RECORD => {
                        let code = self.vbr(6);
                        let n = self.vbr(6) as usize;
                        let ops: Vec<u64> = (0..n).map(|_| self.vbr(6)).collect();
                        if id == BLOCKINFO_BLOCK_ID && code == u64::from(BLOCKINFO_CODE_SETBID) {
                            info_target = Some(ops[0] as u32);
                        }
                        elements.push(El::Rec(Rec { abbrev: UNABBREV_RECORD, code, ops, blob: None }));
                    }
                    n => {
                        let abbrev = abbrevs[(n - FIRST_ABBREV) as usize].clone();
                        let (mut values, mut blob) = (Vec::new(), None);
                        let mut i = 0;
                        while let Some(&op) = abbrev.0.get(i) {
                            match op {
                                AbbrevOp::Array => {
                                    for _ in 0..self.vbr(6) {
                                        values.push(self.scalar(abbrev.0[i + 1]));
                                    }
                                    i += 1;
                                }
                                AbbrevOp::Blob => {
                                    let len = self.vbr(6) as usize;
                                    self.align32();
                                    let start = self.pos / 8;
                                    blob = Some(self.data[start..start + len].to_vec());
                                    self.pos += len * 8;
                                    self.align32();
                                }
                                op => values.push(self.scalar(op)),
                            }
                            i += 1;
                        }
                        let code = values.remove(0);
                        elements.push(El::Rec(Rec { abbrev: n, code, ops: values, blob }));
                    }
                }
            }
        }
        /// After ENTER_SUBBLOCK: the block id and abbreviation width, up to the length word.
        fn header(&mut self) -> (u32, u32) {
            let header = (self.vbr(8) as u32, self.vbr(4) as u32);
            self.align32();
            header
        }
    }

    /// The top-level blocks of a stream (abbreviation width 2) that starts `skip` bytes in.
    fn read(data: &[u8], skip: usize) -> Vec<El> {
        assert_eq!(data.len() % 4, 0);
        let mut r = Reader { data, pos: skip * 8, info: Vec::new() };
        let mut blocks = Vec::new();
        while r.pos < data.len() * 8 {
            assert_eq!(r.read(2), u64::from(ENTER_SUBBLOCK), "top level holds only blocks");
            let (id, width) = r.header();
            let words = r.read(32) as usize;
            let end = r.pos + words * 32;
            let inner = r.block(id, width, end);
            blocks.push(El::Block(id, inner));
        }
        blocks
    }

    fn rec(abbrev: u32, code: u64, ops: &[u64], blob: Option<&[u8]>) -> El {
        El::Rec(Rec { abbrev, code, ops: ops.to_vec(), blob: blob.map(<[u8]>::to_vec) })
    }

    #[test]
    fn fixed_and_vbr_fields_pack_from_bit_zero() {
        let mut w = BitWriter::new();
        w.emit(0b101, 3);
        w.emit(0x3F, 6);
        // VBR6 of 100: 0b1_00100 (the low five bits, continued), then 0b0_00011.
        w.emit_vbr(100, 6);
        w.emit(0xDEAD_BEEF, 32);
        // VBR8 of 1 << 40: five chunks of 0x80, then 0x20.
        w.emit_vbr(1 << 40, 8);
        assert_eq!(w.bit_pos(), 3 + 6 + 12 + 32 + 48);
        assert_eq!(
            w.finish(),
            [0xFD, 0xC9, 0xE1, 0xDD, 0xB7, 0xD5, 0x1B, 0x10, 0x10, 0x10, 0x10, 0x10, 0x04, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn block_with_an_unabbreviated_record() {
        let mut w = BitWriter::new();
        w.enter_block(8, 3);
        w.unabbrev_record(1, &[2]);
        w.exit_block();
        // Word 0: ENTER_SUBBLOCK (2 bits), id 8 (VBR8), width 3 (VBR4) = 1 | 8 << 2 | 3 << 10.
        // Word 1: the length, 1 word.
        // Word 2: UNABBREV_RECORD (3 bits), code 1, 1 operand, 2 (VBR6 each), END_BLOCK
        //         = 3 | 1 << 3 | 1 << 9 | 2 << 15.
        assert_eq!(w.finish(), [0x21, 0x0C, 0, 0, 0x01, 0, 0, 0, 0x0B, 0x02, 0x01, 0]);
    }

    #[test]
    fn abbreviated_records_with_array_char6_and_blob() {
        let mut w = BitWriter::new();
        w.enter_block(8, 3);
        let a = w.define_abbrev(Abbrev(vec![
            AbbrevOp::Literal(5),
            AbbrevOp::Fixed(4),
            AbbrevOp::Vbr(6),
            AbbrevOp::Array,
            AbbrevOp::Char6,
        ]));
        assert_eq!(a, 4);
        let ops = [9, 300, u64::from(b'a'), u64::from(b'Z'), u64::from(b'_')];
        w.record(5, &ops, Some(a));
        let b = w.define_abbrev(Abbrev(vec![AbbrevOp::Literal(6), AbbrevOp::Blob]));
        assert_eq!(b, 5);
        w.record_with_blob(b, 6, &[], b"xyz");
        w.exit_block();
        let bytes = w.finish();
        // The header and the length (6 words), then from bit 64:
        // DEFINE_ABBREV (3 bits), 5 operands (VBR5); literal 5 (1, then VBR8); fixed 4 and VBR 6
        // (0, the encoding in 3 bits, the width as VBR5); array and char6 (0, the encoding);
        // abbreviation 4: 9 (4 bits), 300 (VBR6: 0b101100, 0b001001), 3 elements (VBR6),
        // 'a' = 0, 'Z' = 51, '_' = 63 (6 bits each);
        // DEFINE_ABBREV, 2 operands, literal 6, blob;
        // abbreviation 5: length 3 (VBR6), alignment, "xyz" and a zero byte;
        // END_BLOCK and alignment.
        assert_eq!(
            bytes,
            [
                0x21, 0x0C, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x2A, 0x0B, 0x84, 0x90, 0x31, 0x64, 0xB2, 0xC9,
                0x00, 0xCC, 0xBF, 0x44, 0x03, 0xED, 0x00, 0x00, 0x78, 0x79, 0x7A, 0x00, 0x00, 0x00, 0x00, 0x00
            ]
        );
        assert_eq!(read(&bytes, 0), [El::Block(8, vec![rec(4, 5, &ops, None), rec(5, 6, &[], Some(b"xyz"))])]);
    }

    #[test]
    fn round_trip_with_blockinfo_and_nested_blocks() {
        let mut w = BitWriter::new();
        w.emit(u64::from(b'B'), 8);
        w.emit(u64::from(b'C'), 8);
        w.emit(0xDEC0, 16);
        w.enter_block(8, 3);
        w.enter_blockinfo(2);
        let wide = Abbrev(vec![AbbrevOp::Fixed(3), AbbrevOp::Vbr(8), AbbrevOp::Fixed(32), AbbrevOp::Fixed(0)]);
        let array = Abbrev(vec![AbbrevOp::Literal(1), AbbrevOp::Array, AbbrevOp::Vbr(6)]);
        assert_eq!(w.blockinfo_abbrev(12, wide), 4);
        assert_eq!(w.blockinfo_abbrev(12, array), 5);
        let blob = Abbrev(vec![AbbrevOp::Literal(2), AbbrevOp::Blob]);
        assert_eq!(w.blockinfo_abbrev(14, blob), 4);
        w.exit_block();
        w.unabbrev_record(1, &[u64::MAX, 0, 63, 64]);
        let mut expected = vec![
            El::Block(0, vec![rec(UNABBREV_RECORD, 1, &[12], None), rec(UNABBREV_RECORD, 1, &[14], None)]),
            rec(UNABBREV_RECORD, 1, &[u64::MAX, 0, 63, 64], None),
        ];
        for round in 0..2u64 {
            w.enter_block(12, 4);
            w.record(7, &[1 << 33, 0xFFFF_FFFF, 0], Some(4));
            w.record(1, &[], Some(5));
            w.record(1, &[round, u64::MAX], Some(5));
            // A block's own abbreviations are numbered after those BLOCKINFO gave it.
            let local = w.define_abbrev(Abbrev(vec![AbbrevOp::Fixed(0), AbbrevOp::Char6]));
            assert_eq!(local, 6);
            w.record(0, &[u64::from(b'.')], Some(local));
            w.enter_block(14, 3);
            for len in 0..6 {
                w.record_with_blob(4, 2, &[], &[0xA5; 6][..len]);
            }
            w.exit_block();
            w.exit_block();
            let blobs = (0..6).map(|len| rec(4, 2, &[], Some(&[0xA5; 6][..len]))).collect();
            expected.push(El::Block(12, vec![
                rec(4, 7, &[1 << 33, 0xFFFF_FFFF, 0], None),
                rec(5, 1, &[], None),
                rec(5, 1, &[round, u64::MAX], None),
                rec(6, 0, &[u64::from(b'.')], None),
                El::Block(14, blobs),
            ]));
        }
        w.exit_block();
        w.enter_block(9, 32);
        w.exit_block();
        let bytes = w.finish();
        assert_eq!(&bytes[..4], b"BC\xC0\xDE");
        assert_eq!(read(&bytes, 4), [El::Block(8, expected), El::Block(9, vec![])]);
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn a_value_wider_than_its_field_panics() {
        let mut w = BitWriter::new();
        w.enter_block(8, 3);
        let a = w.define_abbrev(Abbrev(vec![AbbrevOp::Literal(1), AbbrevOp::Fixed(4)]));
        w.record(1, &[16], Some(a));
    }

    #[test]
    #[should_panic(expected = "array not second to last")]
    fn an_array_must_be_second_to_last() {
        let mut w = BitWriter::new();
        w.enter_block(8, 3);
        w.define_abbrev(Abbrev(vec![AbbrevOp::Array, AbbrevOp::Fixed(4), AbbrevOp::Fixed(4)]));
    }
}
