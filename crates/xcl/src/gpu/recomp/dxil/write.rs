//! Writes a built module as LLVM 3.7 bitcode records, in the order and the forms DXC writes
//! them: the attribute groups and lists of the declared functions, the type table, the target,
//! the global variables and functions, the module's constants and metadata, the names, and the
//! entry function's body ([`super::number`]).
//!
//! After LLVM 3.7 `lib/Bitcode/Writer/BitcodeWriter.cpp` and `include/llvm/Bitcode/LLVMBitCodes.h`
//! (https://github.com/llvm/llvm-project, branch release/3.7.x, University of Illinois/NCSA
//! licence), checked against the records of DXC 1.9.2609's own modules.

use std::collections::HashMap;

use super::module::{FnAttr, Module, Refused, Ty, TypeDef, Val};
use super::number::{Numbered, Sink};

const MODULE_BLOCK: u32 = 8;
const PARAMATTR_BLOCK: u32 = 9;
const PARAMATTR_GROUP_BLOCK: u32 = 10;
const VALUE_SYMTAB_BLOCK: u32 = 14;
const METADATA_BLOCK: u32 = 15;
const TYPE_BLOCK: u32 = 17;

const TRIPLE: &str = "dxil-ms-dx";
/// DXC's data layouts: with minimum-precision types (the default), and with native 16-bit ones.
const LAYOUT_MIN_PRECISION: &str = "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64";
const LAYOUT_NATIVE_16: &str = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64";

/// The metadata kinds DXC's context has; it writes them all whether or not one is attached.
const KINDS: [&str; 17] = [
    "dbg",
    "tbaa",
    "prof",
    "fpmath",
    "range",
    "tbaa.struct",
    "invariant.load",
    "alias.scope",
    "noalias",
    "nontemporal",
    "llvm.mem.parallel_loop_access",
    "nonnull",
    "dereferenceable",
    "dereferenceable_or_null",
    "dx.temp",
    "dx.dbg.varlayout",
    "dx.break.br",
];

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(super) enum Md {
    String(String),
    /// a constant or a function as metadata
    Value(Val),
    /// operands by index; None is `null`
    Node(Vec<Option<u32>>),
}

/// The module's metadata: nodes are unique, as LLVM's are (`!dx.version` and `!dx.valver` 1.6
/// name the same node), and an operand is created before the node that has it.
#[derive(Default, Debug)]
pub(super) struct Metadata {
    pub nodes: Vec<Md>,
    index: HashMap<Md, u32>,
    pub named: Vec<(&'static str, Vec<u32>)>,
}

impl Metadata {
    fn add(&mut self, md: Md) -> Option<u32> {
        let next = self.nodes.len() as u32;
        let id = *self.index.entry(md.clone()).or_insert(next);
        if id == next {
            self.nodes.push(md);
        }
        Some(id)
    }
    pub fn string(&mut self, s: &str) -> Option<u32> {
        self.add(Md::String(s.into()))
    }
    pub fn value(&mut self, v: Val) -> Option<u32> {
        self.add(Md::Value(v))
    }
    pub fn i32(&mut self, v: u32) -> Option<u32> {
        self.value(Val::i32(v))
    }
    pub fn node(&mut self, operands: Vec<Option<u32>>) -> Option<u32> {
        self.add(Md::Node(operands))
    }
    /// `!name = !{nodes}`
    pub fn name(&mut self, name: &'static str, nodes: &[Option<u32>]) {
        self.named.push((name, nodes.iter().flatten().copied().collect()));
    }
    /// The constants the metadata names: they are the module's.
    fn constants(&self) -> Vec<Val> {
        let mut seen = HashMap::new();
        let constant = |md: &Md| match md {
            Md::Value(v) if v.is_constant() && seen.insert(*v, ()).is_none() => Some(*v),
            _ => None,
        };
        self.nodes.iter().filter_map(constant).collect()
    }
}

fn bytes(s: &str) -> Vec<u64> {
    s.bytes().map(u64::from).collect()
}

/// The MODULE block of `m` with `metadata`.
pub(super) fn write_module(m: &mut Module, metadata: &Metadata, native_16: bool, sink: &mut impl Sink) -> Result<(), Refused> {
    let n = Numbered::new(m, metadata.constants())?;
    let m = &*m;
    sink.enter(MODULE_BLOCK, 3);
    // relative value ids
    sink.record(1, &[1]);

    // one attribute group and one list of it for each attribute set of the declarations, all on
    // the function itself (0xFFFFFFFF); a function names its list from 1
    let mut sets: Vec<FnAttr> = Vec::new();
    for f in &m.functions {
        if let Some(a) = f.attr.filter(|a| !sets.contains(a)) {
            sets.push(a);
        }
    }
    let list = |a: Option<FnAttr>| a.map_or(0, |a| sets.iter().position(|&s| s == a).unwrap() as u64 + 1);
    if !sets.is_empty() {
        sink.enter(PARAMATTR_GROUP_BLOCK, 3);
        for &a in &sets {
            // an attribute is [0, kind]: noduplicate 12, nounwind 18, readnone 20, readonly 21
            let kinds: &[u64] = match a {
                FnAttr::None => &[0, 18],
                FnAttr::ReadNone => &[0, 18, 0, 20],
                FnAttr::ReadOnly => &[0, 18, 0, 21],
                FnAttr::NoDuplicate => &[0, 12, 0, 18],
            };
            sink.record(3, &[&[list(Some(a)), 0xFFFF_FFFF][..], kinds].concat());
        }
        sink.exit();
        sink.enter(PARAMATTR_BLOCK, 3);
        sets.iter().for_each(|&a| sink.record(2, &[list(Some(a))]));
        sink.exit();
    }

    sink.enter(TYPE_BLOCK, 4);
    sink.record(1, &[n.type_ids.iter().filter(|&&t| t != u32::MAX).count() as u64]);
    for (_, def) in m.types.iter().enumerate().filter(|&(t, _)| n.type_ids[t] != u32::MAX) {
        let ids = |tys: &[Ty]| tys.iter().map(|&t| n.type_id(t)).collect::<Vec<u64>>();
        match def {
            TypeDef::Void => sink.record(2, &[]),
            TypeDef::Float => sink.record(3, &[]),
            TypeDef::Double => sink.record(4, &[]),
            TypeDef::Int(bits) => sink.record(7, &[u64::from(*bits)]),
            TypeDef::Pointer(to, space) => sink.record(8, &[n.type_id(*to), u64::from(*space)]),
            TypeDef::Half => sink.record(10, &[]),
            TypeDef::Array(len, of) => sink.record(11, &[*len, n.type_id(*of)]),
            // not packed
            TypeDef::Struct(None, elements) => sink.record(18, &[&[0][..], &ids(elements)[..]].concat()),
            TypeDef::Struct(Some(name), elements) => {
                sink.record(19, &bytes(name));
                sink.record(20, &[&[0][..], &ids(elements)[..]].concat());
            }
            // not vararg
            TypeDef::Function(ret, params) => sink.record(21, &[&[0, n.type_id(*ret)][..], &ids(params)[..]].concat()),
        }
    }
    sink.exit();

    sink.record(2, &bytes(TRIPLE));
    sink.record(3, &bytes(if native_16 { LAYOUT_NATIVE_16 } else { LAYOUT_MIN_PRECISION }));
    for g in &m.globals {
        // the value type with its address space (bit 1: the type is explicit), no initializer,
        // external, align 4 (log2 + 1), no section
        sink.record(7, &[n.type_id(g.ty), u64::from(g.address_space) << 2 | 2, 0, 0, 3, 0]);
    }
    for f in &m.functions {
        // [type, calling convention, declaration, linkage, attribute list, alignment, section, visibility, gc,
        // unnamed_addr, prologue data, DLL storage, comdat, prefix data, personality]
        let mut record = [0; 15];
        (record[0], record[2], record[4]) = (n.type_id(f.ty), u64::from(f.attr.is_some()), list(f.attr));
        sink.record(8, &record);
    }
    n.write_constants(m, &n.module_constants, sink);

    sink.enter(METADATA_BLOCK, 3);
    for md in &metadata.nodes {
        match md {
            Md::String(s) => sink.record(1, &bytes(s)),
            Md::Value(v) => sink.record(2, &[n.type_id(v.ty), u64::from(n.id(m, *v))]),
            // an operand is its index + 1, 0 for null
            Md::Node(operands) => sink.record(3, &operands.iter().map(|o| o.map_or(0, |o| u64::from(o) + 1)).collect::<Vec<_>>()),
        }
    }
    for (name, nodes) in &metadata.named {
        sink.record(4, &bytes(name));
        sink.record(10, &nodes.iter().map(|&o| u64::from(o)).collect::<Vec<_>>());
    }
    sink.exit();
    sink.enter(METADATA_BLOCK, 3);
    for (k, kind) in KINDS.iter().enumerate() {
        sink.record(6, &[&[k as u64][..], &bytes(kind)[..]].concat());
    }
    sink.exit();

    sink.enter(VALUE_SYMTAB_BLOCK, 4);
    let names = m.globals.iter().map(|g| &g.name).chain(m.functions.iter().map(|f| &f.name));
    for (id, name) in names.enumerate() {
        sink.record(1, &[&[id as u64][..], &bytes(name)[..]].concat());
    }
    sink.exit();

    n.write_function(m, sink);
    sink.exit();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::module::Kind;
    use super::super::number::tests::Listing;
    use super::*;

    #[test]
    fn metadata_nodes_are_unique() {
        let mut md = Metadata::default();
        let (one, six) = (md.i32(1), md.i32(6));
        let version = md.node(vec![one, six]);
        assert_eq!((one, six), (md.i32(1), md.i32(6)));
        assert_eq!(version, md.node(vec![one, six]));
        assert_ne!(version, md.node(vec![six, one]));
        assert_eq!(md.string("cs"), md.string("cs"));
        md.name("dx.version", &[version]);
        md.name("dx.valver", &[version]);
        assert_eq!((md.nodes.len(), md.constants()), (5, vec![Val::i32(1), Val::i32(6)]));
        assert_eq!(md.named, [("dx.version", vec![2]), ("dx.valver", vec![2])]);
    }

    /// A whole module, record by record: `void main() { put(i32 31, i32 undef) }` with a
    /// group-shared array, `!dx.version = !{!{i32 1, i32 6}}` and a node that names `main`.
    #[test]
    fn a_module_is_written_in_dxc_order() {
        let mut m = Module::compute([1, 1, 1]);
        m.groupshared(4);
        let put = m.declare("put", Ty::VOID, &[Ty::I32, Ty::I32], FnAttr::ReadOnly);
        let pure = m.declare("pure", Ty::F16, &[], FnAttr::ReadNone);
        let mut f = m.entry();
        let b = f.block();
        f.at(b);
        f.call(put, &[Val::i32(31), Val::undef(Ty::I32)]);
        f.call(pure, &[]);
        f.ret();
        let main = Val { ty: m.pointer(m.functions[0].ty, 0), kind: Kind::Function(0) };
        let mut md = Metadata::default();
        let (one, six) = (md.i32(1), md.i32(6));
        let version = md.node(vec![one, six]);
        md.name("dx.version", &[version]);
        let (function, name) = (md.value(main), md.string("main"));
        let entry = md.node(vec![function, name, None]);
        md.name("dx.entryPoints", &[entry]);
        let mut out = Listing::default();
        write_module(&mut m, &md, false, &mut out).unwrap();
        let text = |s: &str| format!("{:?}", bytes(s));
        // types: void 0, i32 1, half 2, void () 3, [4 x i32] 4, its pointer 5, void (i32, i32) 6, half () 7, void ()* 8
        // values: 0 the array, 1 main, 2 put, 3 pure, 4 = i32 1, 5 = i32 6; in the function 6 = 31, 7 = undef
        let expected = [
            "<8>".to_string(),
            "  1 [1]".into(),
            "  <10>".into(),
            "    3 [1, 4294967295, 0, 18, 0, 21]".into(),
            "    3 [2, 4294967295, 0, 18, 0, 20]".into(),
            "  <9>".into(),
            "    2 [1]".into(),
            "    2 [2]".into(),
            "  <17>".into(),
            "    1 [9]".into(),
            "    2 []".into(),
            "    7 [32]".into(),
            "    10 []".into(),
            "    21 [0, 0]".into(),
            "    11 [4, 1]".into(),
            "    8 [4, 3]".into(),
            "    21 [0, 0, 1, 1]".into(),
            "    21 [0, 2]".into(),
            "    8 [3, 0]".into(),
            format!("  2 {}", text("dxil-ms-dx")),
            format!("  3 {}", text(LAYOUT_MIN_PRECISION)),
            "  7 [4, 14, 0, 0, 3, 0]".into(),
            "  8 [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]".into(),
            "  8 [6, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]".into(),
            "  8 [7, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]".into(),
            "  <11>".into(),
            "    1 [1]".into(),
            "    4 [2]".into(),
            "    4 [12]".into(),
            "  <15>".into(),
            "    2 [1, 4]".into(),
            "    2 [1, 5]".into(),
            "    3 [1, 2]".into(),
            "    2 [8, 1]".into(),
            format!("    1 {}", text("main")),
            "    3 [4, 5, 0]".into(),
            format!("    4 {}", text("dx.version")),
            "    10 [2]".into(),
            format!("    4 {}", text("dx.entryPoints")),
            "    10 [5]".into(),
        ];
        assert_eq!(out.0[..expected.len()], expected);
        let rest = &out.0[expected.len()..];
        assert_eq!(rest[0], "  <15>");
        assert_eq!((rest[1].as_str(), rest[17].as_str()), (&*format!("    6 {:?}", [&[0][..], &bytes("dbg")[..]].concat()), &*format!("    6 {:?}", [&[16][..], &bytes("dx.break.br")[..]].concat())));
        assert_eq!(
            rest[18..],
            [
                "  <14>".to_string(),
                format!("    1 {:?}", [&[0][..], &bytes("\u{1}?xcl_lds0@@3PAIA")[..]].concat()),
                format!("    1 {:?}", [&[1][..], &bytes("main")[..]].concat()),
                format!("    1 {:?}", [&[2][..], &bytes("put")[..]].concat()),
                format!("    1 {:?}", [&[3][..], &bytes("pure")[..]].concat()),
                "  <12>".into(),
                "    1 [1]".into(),
                "    <11>".into(),
                "      1 [1]".into(),
                "      4 [62]".into(),
                "      3 []".into(),
                "    34 [0, 32768, 6, 6, 2, 1]".into(),
                "    34 [0, 32768, 7, 5]".into(),
                "    10 []".into(),
            ]
        );
    }

    #[test]
    fn a_refused_function_writes_nothing() {
        let mut m = Module::compute([1, 1, 1]);
        let mut f = m.entry();
        let b = f.block();
        f.at(b);
        f.add(Val::i32(1), Val::f32(1.0));
        let mut out = Listing::default();
        assert_eq!(write_module(&mut m, &Metadata::default(), false, &mut out).unwrap_err().kind, "operand-type");
        assert!(out.0.is_empty());
    }
}
