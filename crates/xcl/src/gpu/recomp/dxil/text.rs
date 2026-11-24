//! Prints a built module as LLVM 3.7 assembly, laid out as DXC's listings are: the form
//! `IDxcAssembler::AssembleToContainer` parses. It is the module [`super::write`] writes as
//! bitcode, from the same checked and ordered function ([`super::number`]): the functions, the
//! instructions, their flags and the metadata are the same, so both disassemble to one listing.
//!
//! An unnamed value is `%n`, numbered through the function with its blocks (the entry block is
//! `%0`). A floating-point constant is always in LLVM's hexadecimal form, which is exact.
//!
//! After LLVM 3.7 `lib/IR/AsmWriter.cpp` (https://github.com/llvm/llvm-project, branch
//! release/3.7.x, University of Illinois/NCSA licence).

use std::fmt::Write;

use super::module::{Binary, Cast, FnAttr, Inst, Kind, Module, Refused, Rmw, Ty, TypeDef, Val};
use super::number::Numbered;
use super::write::{Md, Metadata, LAYOUT_MIN_PRECISION, LAYOUT_NATIVE_16, TRIPLE};

/// A string between quotes: LLVM's `PrintEscapedString`.
fn escaped(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_graphic() && b != b'"' && b != b'\\' || b == b' ' {
            out.push(b as char);
        } else {
            write!(out, "\\{b:02X}").unwrap();
        }
    }
    out
}

/// A global's name after its `@`: quoted unless it is an identifier.
fn name(s: &str) -> String {
    let plain = |c: char| c.is_ascii_alphanumeric() || "-$._".contains(c);
    if !s.is_empty() && s.chars().all(plain) && !s.starts_with(|c: char| c.is_ascii_digit()) {
        s.into()
    } else {
        format!("\"{}\"", escaped(s))
    }
}

/// The bits of the `double` a `float` constant is printed as.
fn float_as_double(bits: u32) -> u64 {
    let f = f32::from_bits(bits);
    if f.is_nan() {
        // the payload is kept (a conversion would quieten a signalling NaN)
        u64::from(bits >> 31) << 63 | 0x7FF << 52 | u64::from(bits & 0x7F_FFFF) << 29
    } else {
        f64::from(f).to_bits()
    }
}

/// ` nuw nsw`, ` exact` or the fast-math flags of an instruction.
fn flag_names(float: bool, exact: bool, flags: u8) -> String {
    let names: &[(u8, &str)] = match (float, exact) {
        // `fast` stands for all five
        (true, _) if flags & 1 != 0 => &[(1, " fast")],
        (true, _) => &[(2, " nnan"), (4, " ninf"), (8, " nsz"), (16, " arcp")],
        (false, true) => &[(1, " exact")],
        (false, false) => &[(1, " nuw"), (2, " nsw")],
    };
    names.iter().filter(|&&(bit, _)| flags & bit != 0).map(|&(_, name)| name).collect()
}

struct Printer<'a> {
    m: &'a Module,
    /// each instruction's `%n` and each block's
    values: Vec<u32>,
    blocks: Vec<u32>,
}

impl Printer<'_> {
    /// A type as LLVM writes it (the names of the module's struct types are identifiers).
    fn ty(&self, ty: Ty) -> String {
        self.m.type_name(ty)
    }

    /// A value without its type.
    fn value(&self, v: Val) -> String {
        match v.kind {
            Kind::Inst(i) => format!("%{}", self.values[i as usize]),
            Kind::Int(x) if v.ty == Ty::I1 => (if x != 0 { "true" } else { "false" }).into(),
            Kind::Int(x) => x.to_string(),
            Kind::Float(bits) => match v.ty {
                Ty::F16 => format!("0xH{bits:04X}"),
                Ty::F32 => format!("0x{:016X}", float_as_double(bits as u32)),
                _ => format!("0x{bits:016X}"),
            },
            Kind::Undef => "undef".into(),
            Kind::Aggregate(a) => format!("{{ {} }}", self.typed_list(&self.m.aggregates[a as usize].1)),
            Kind::Global(g) => format!("@{}", name(&self.m.globals[g as usize].name)),
            Kind::Function(f) => format!("@{}", name(&self.m.functions[f as usize].name)),
            // a refused module is not printed
            Kind::Bad => "<refused>".into(),
        }
    }

    fn typed(&self, v: Val) -> String {
        format!("{} {}", self.ty(v.ty), self.value(v))
    }

    fn typed_list(&self, values: &[Val]) -> String {
        values.iter().map(|&v| self.typed(v)).collect::<Vec<_>>().join(", ")
    }

    fn label(&self, b: super::module::Block) -> String {
        format!("label %{}", self.blocks[b.0 as usize])
    }

    /// Instruction `i` without its `%n = `.
    fn instruction(&self, i: u32) -> String {
        let (m, ty) = (self.m, self.m.body.tys[i as usize]);
        match m.body.insts[i as usize] {
            Inst::Binary { op, flags, a, b } => {
                let (mnemonic, float, exact) = match op {
                    Binary::Add => ("add", false, false),
                    Binary::Sub => ("sub", false, false),
                    Binary::Mul => ("mul", false, false),
                    Binary::UDiv => ("udiv", false, true),
                    Binary::SDiv => ("sdiv", false, true),
                    Binary::URem => ("urem", false, false),
                    Binary::SRem => ("srem", false, false),
                    Binary::Shl => ("shl", false, false),
                    Binary::LShr => ("lshr", false, true),
                    Binary::AShr => ("ashr", false, true),
                    Binary::And => ("and", false, false),
                    Binary::Or => ("or", false, false),
                    Binary::Xor => ("xor", false, false),
                    Binary::FAdd => ("fadd", true, false),
                    Binary::FSub => ("fsub", true, false),
                    Binary::FMul => ("fmul", true, false),
                    Binary::FDiv => ("fdiv", true, false),
                    Binary::FRem => ("frem", true, false),
                };
                format!("{mnemonic}{} {}, {}", flag_names(float, exact, flags), self.typed(a), self.value(b))
            }
            Inst::Cast { op, value } => {
                let mnemonic = match op {
                    Cast::Trunc => "trunc",
                    Cast::ZExt => "zext",
                    Cast::SExt => "sext",
                    Cast::FpToUi => "fptoui",
                    Cast::FpToSi => "fptosi",
                    Cast::UiToFp => "uitofp",
                    Cast::SiToFp => "sitofp",
                    Cast::FpTrunc => "fptrunc",
                    Cast::FpExt => "fpext",
                    Cast::BitCast => "bitcast",
                };
                format!("{mnemonic} {} to {}", self.typed(value), self.ty(ty))
            }
            Inst::Cmp { pred, flags, a, b } => {
                // LLVM's predicate numbers: the FP ones from 1 (oeq), the integer ones from 32 (eq)
                const INT: [&str; 10] = ["eq", "ne", "ugt", "uge", "ult", "ule", "sgt", "sge", "slt", "sle"];
                const FLOAT: [&str; 14] = ["oeq", "ogt", "oge", "olt", "ole", "one", "ord", "uno", "ueq", "ugt", "uge", "ult", "ule", "une"];
                let (mnemonic, pred) = if pred >= 32 { ("icmp", INT[pred as usize - 32]) } else { ("fcmp", FLOAT[pred as usize - 1]) };
                format!("{mnemonic}{} {pred} {}, {}", flag_names(true, false, flags), self.typed(a), self.value(b))
            }
            Inst::Select { cond, a, b } => format!("select {}, {}, {}", self.typed(cond), self.typed(a), self.typed(b)),
            Inst::ExtractValue { aggregate, index } => format!("extractvalue {}, {index}", self.typed(aggregate)),
            Inst::Phi(list) => {
                let incoming: Vec<String> = m.body.incoming[list as usize].iter().map(|&(v, from)| format!("[ {}, %{} ]", self.value(v), self.blocks[from.0 as usize])).collect();
                format!("phi {} {}", self.ty(ty), incoming.join(", "))
            }
            Inst::Call { function, args: (start, n) } => {
                let arguments = self.typed_list(&m.body.operands[start as usize..(start + n) as usize]);
                format!("call {} @{}({arguments})", self.ty(ty), name(&m.functions[function as usize].name))
            }
            Inst::Br(to) => format!("br {}", self.label(to)),
            Inst::BrIf { cond, then, otherwise } => format!("br {}, {}, {}", self.typed(cond), self.label(then), self.label(otherwise)),
            Inst::Switch { value, default, cases: (start, n) } => {
                let cases: String = m.body.cases[start as usize..(start + n) as usize].iter().map(|&(case, to)| format!("    {}, {}\n", self.typed(Val::i32(case)), self.label(to))).collect();
                format!("switch {}, {} [\n{cases}  ]", self.typed(value), self.label(default))
            }
            Inst::Ret => "ret void".into(),
            Inst::Alloca { words } => format!("alloca [{words} x i32], align 4"),
            Inst::Gep { base, index, inbounds } => {
                let TypeDef::Pointer(array, _) = m.def(base.ty) else { unreachable!("the base of a getelementptr") };
                format!("getelementptr{} {}, {}, i32 0, {}", if inbounds { " inbounds" } else { "" }, self.ty(*array), self.typed(base), self.typed(index))
            }
            Inst::Load { pointer } => format!("load {}, {}, align 4", self.ty(ty), self.typed(pointer)),
            Inst::Store { pointer, value } => format!("store {}, {}, align 4", self.typed(value), self.typed(pointer)),
            Inst::AtomicRmw { op, pointer, value } => {
                let mnemonic = match op {
                    Rmw::Xchg => "xchg",
                    Rmw::Add => "add",
                    Rmw::And => "and",
                    Rmw::Or => "or",
                    Rmw::Xor => "xor",
                    Rmw::Max => "max",
                    Rmw::Min => "min",
                    Rmw::UMax => "umax",
                    Rmw::UMin => "umin",
                };
                format!("atomicrmw {mnemonic} {}, {} seq_cst", self.typed(pointer), self.typed(value))
            }
            Inst::CmpXchg { pointer, expected, new } => format!("cmpxchg {}, {}, {} seq_cst seq_cst", self.typed(pointer), self.typed(expected), self.typed(new)),
            Inst::Removed => unreachable!("a replaced instruction is in no block"),
        }
    }
}

/// The module `m` with `metadata` as text.
pub(super) fn module_text(m: &mut Module, metadata: &Metadata, native_16: bool) -> Result<String, Refused> {
    let n = Numbered::new(m, metadata.constants())?;
    let m = &*m;
    let mut p = Printer { m, values: vec![u32::MAX; m.body.insts.len()], blocks: Vec::new() };
    let mut next = 0;
    for block in &n.order {
        p.blocks.push(next);
        next += 1;
        for &i in block.iter().filter(|&&i| m.body.tys[i as usize] != Ty::VOID) {
            p.values[i as usize] = next;
            next += 1;
        }
    }

    let mut out = format!("target datalayout = \"{}\"\ntarget triple = \"{TRIPLE}\"\n\n", if native_16 { LAYOUT_NATIVE_16 } else { LAYOUT_MIN_PRECISION });
    let types = out.len();
    for (_, def) in m.types.iter().enumerate().filter(|&(t, _)| n.type_ids[t] != u32::MAX) {
        if let TypeDef::Struct(Some(named), elements) = def {
            writeln!(out, "%{named} = type {{ {} }}", elements.iter().map(|&t| p.ty(t)).collect::<Vec<_>>().join(", ")).unwrap();
        }
    }
    if out.len() != types {
        out.push('\n');
    }
    for g in &m.globals {
        let space = if g.address_space == 0 { String::new() } else { format!("addrspace({}) ", g.address_space) };
        writeln!(out, "@{} = external {space}global {}, align 4", name(&g.name), p.ty(g.ty)).unwrap();
    }
    if !m.globals.is_empty() {
        out.push('\n');
    }

    // the entry function, then the declarations with their attribute groups (as `write` has them)
    let mut sets: Vec<FnAttr> = Vec::new();
    for f in &m.functions {
        let TypeDef::Function(ret, params) = m.def(f.ty) else { unreachable!("a function's type") };
        let params = params.iter().map(|&t| p.ty(t)).collect::<Vec<_>>().join(", ");
        let Some(attr) = f.attr else {
            writeln!(out, "define {} @{}({params}) {{", p.ty(*ret), name(&f.name)).unwrap();
            for (b, block) in n.order.iter().enumerate() {
                if b != 0 {
                    writeln!(out, "\n; <label>:{}", p.blocks[b]).unwrap();
                }
                for &i in block {
                    let result = if m.body.tys[i as usize] == Ty::VOID { String::new() } else { format!("%{} = ", p.values[i as usize]) };
                    writeln!(out, "  {result}{}", p.instruction(i)).unwrap();
                }
            }
            out.push_str("}\n\n");
            continue;
        };
        let group = sets.iter().position(|&s| s == attr).unwrap_or_else(|| {
            sets.push(attr);
            sets.len() - 1
        });
        writeln!(out, "declare {} @{}({params}) #{group}\n", p.ty(*ret), name(&f.name)).unwrap();
    }
    for (group, attr) in sets.iter().enumerate() {
        let attributes = match attr {
            FnAttr::None => "nounwind",
            FnAttr::ReadNone => "nounwind readnone",
            FnAttr::ReadOnly => "nounwind readonly",
            FnAttr::NoDuplicate => "noduplicate nounwind",
        };
        writeln!(out, "attributes #{group} = {{ {attributes} }}").unwrap();
    }
    if !sets.is_empty() {
        out.push('\n');
    }

    // a node is `!n`, numbered when it is first reached from the named ones, before its operands
    let mut numbers = vec![u32::MAX; metadata.nodes.len()];
    let mut numbered = Vec::new();
    let mut reach: Vec<u32> = metadata.named.iter().rev().flat_map(|(_, nodes)| nodes.iter().rev().copied()).collect();
    while let Some(id) = reach.pop() {
        let Md::Node(operands) = &metadata.nodes[id as usize] else { continue };
        if numbers[id as usize] == u32::MAX {
            numbers[id as usize] = numbered.len() as u32;
            numbered.push(id);
            reach.extend(operands.iter().rev().flatten());
        }
    }
    let operand = |o: &Option<u32>| match o.map(|id| (id, &metadata.nodes[id as usize])) {
        None => "null".into(),
        Some((_, Md::String(s))) => format!("!\"{}\"", escaped(s)),
        Some((_, Md::Value(v))) => p.typed(*v),
        Some((id, Md::Node(_))) => format!("!{}", numbers[id as usize]),
    };
    let list = |operands: &[Option<u32>]| format!("!{{{}}}", operands.iter().map(operand).collect::<Vec<_>>().join(", "));
    for (named, nodes) in &metadata.named {
        writeln!(out, "!{named} = {}", list(&nodes.iter().map(|&id| Some(id)).collect::<Vec<_>>())).unwrap();
    }
    out.push('\n');
    for (number, &id) in numbered.iter().enumerate() {
        let Md::Node(operands) = &metadata.nodes[id as usize] else { unreachable!("only nodes are numbered") };
        writeln!(out, "!{number} = {}", list(operands)).unwrap();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::module::{flags, Binary, Cast, Fcmp, Module, Ty, Val};
    use super::{escaped, float_as_double, name};

    #[test]
    fn names_and_constants_are_written_as_llvm_reads_them() {
        assert_eq!((name("dx.op.barrier"), name("\u{1}?xcl_lds0@@3PAIA")), ("dx.op.barrier".into(), "\"\\01?xcl_lds0@@3PAIA\"".into()));
        assert_eq!(escaped("a \"b\"\\\n"), "a \\22b\\22\\5C\\0A");
        // 0.1f, -0.0f, the largest float, an infinity and a signalling NaN with its payload
        assert_eq!(float_as_double(0x3DCC_CCCD), 0x3FB9_9999_A000_0000);
        assert_eq!(float_as_double(0x8000_0000), 0x8000_0000_0000_0000);
        assert_eq!(float_as_double(0x7F7F_FFFF), 0x47EF_FFFF_E000_0000);
        assert_eq!(float_as_double(0xFF80_0000), 0xFFF0_0000_0000_0000);
        assert_eq!(float_as_double(0x7FA0_0001), 0x7FF4_0000_2000_0000);
    }

    #[test]
    fn a_program_as_text() {
        let mut m = Module::compute([1, 1, 1]);
        let lds = m.groupshared(4);
        let mut f = m.entry();
        let (b0, one, two, end) = (f.block(), f.block(), f.block(), f.block());
        f.at(b0);
        let p = f.gep(lds, Val::i32(1));
        let x = f.load(p);
        let y = f.binary_with(Binary::Add, flags::NUW | flags::NSW, x, Val::i32(0xFFFF_FFFF));
        let z = f.binary_with(Binary::LShr, flags::EXACT, y, Val::i32(2));
        f.switch(z, end, &[(0, one), (9, two)]);
        f.at(one);
        let h = f.cast(Cast::UiToFp, z, Ty::F32);
        let h = f.binary_with(Binary::FMul, flags::FAST, h, Val::f32(0.1));
        let less = f.fcmp(Fcmp::Ult, h, Val::f32(1.0));
        let s = f.select(less, x, y);
        f.br(end);
        f.at(two);
        f.br(end);
        f.at(end);
        let r = f.phi(end, Ty::I32);
        f.incoming(r, z, b0);
        f.incoming(r, s, one);
        f.incoming(r, Val::i32(7), two);
        f.store(p, r);
        f.ret();
        let expected = r#"target datalayout = "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-ms-dx"

@"\01?xcl_lds0@@3PAIA" = external addrspace(3) global [4 x i32], align 4

define void @main() {
  %1 = getelementptr [4 x i32], [4 x i32] addrspace(3)* @"\01?xcl_lds0@@3PAIA", i32 0, i32 1
  %2 = load i32, i32 addrspace(3)* %1, align 4
  %3 = add nuw nsw i32 %2, -1
  %4 = lshr exact i32 %3, 2
  switch i32 %4, label %11 [
    i32 0, label %5
    i32 9, label %10
  ]

; <label>:5
  %6 = uitofp i32 %4 to float
  %7 = fmul fast float %6, 0x3FB99999A0000000
  %8 = fcmp ult float %7, 0x3FF0000000000000
  %9 = select i1 %8, i32 %2, i32 %3
  br label %11

; <label>:10
  br label %11

; <label>:11
  %12 = phi i32 [ %4, %0 ], [ %9, %5 ], [ 7, %10 ]
  store i32 %12, i32 addrspace(3)* %1, align 4
  ret void
}

!llvm.ident = !{!0}
!dx.version = !{!1}
!dx.valver = !{!1}
!dx.shaderModel = !{!2}
!dx.entryPoints = !{!3}

!0 = !{!"xcl"}
!1 = !{i32 1, i32 6}
!2 = !{!"cs", i32 6, i32 6}
!3 = !{void ()* @main, !"main", null, null, !4}
!4 = !{i32 4, !5}
!5 = !{i32 1, i32 1, i32 1}
"#;
        assert_eq!(m.to_text().unwrap(), expected);
        // the text is of the module as it is: asking again gives it again, and so after `finish`
        m.finish().unwrap();
        assert_eq!(m.to_text().unwrap(), expected);
    }
}
