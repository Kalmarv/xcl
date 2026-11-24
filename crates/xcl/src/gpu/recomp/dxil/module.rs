//! DXIL module builder: what a lowering calls to describe one compute program – types, SSA values
//! and constants, basic blocks, instructions and phis. Nothing is numbered here: [`Val`]s name
//! instructions by their creation order, and `write` numbers them as LLVM does when the module is
//! finished. So a phi can be added to a block that already has instructions, a value can be
//! used in a block created before the one that defines it, and [`Func::replace`] costs nothing.
//!
//! A call never panics on a wrong program and returns no `Result`: the first mistake is kept as
//! a [`Refused`] (with the block and the instruction's index in it) and `finish` returns it.
//!
//! Only the forms DXC itself writes for HLSL are offered (dxil-spirv reads no others): the
//! instruction list is DXIL's (no `unreachable`, no vectors), predicates and atomic operations
//! are those HLSL has.

use std::collections::HashMap;

use super::dxop::Resource;

/// A type, by its index in the module's type list; the scalar types are the same in every module.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Ty(pub(super) u32);

impl Ty {
    pub const VOID: Ty = Ty(0);
    pub const I1: Ty = Ty(1);
    /// only as a constant: the mask arguments of `dx.op` calls
    pub const I8: Ty = Ty(2);
    pub const I32: Ty = Ty(3);
    pub const I64: Ty = Ty(4);
    /// `half`; a module that uses it has native 16-bit types (as `-enable-16bit-types`)
    pub const F16: Ty = Ty(5);
    pub const F32: Ty = Ty(6);
    pub const F64: Ty = Ty(7);
    pub(super) const SCALARS: u32 = 8;

    /// i32 or i64: what arithmetic works on
    fn is_int(self) -> bool {
        self == Ty::I32 || self == Ty::I64
    }
    fn is_float(self) -> bool {
        self == Ty::F16 || self == Ty::F32 || self == Ty::F64
    }
    /// what a register, a phi or a select holds
    fn is_value(self) -> bool {
        self == Ty::I1 || self.is_int() || self.is_float()
    }
    fn bits(self) -> u32 {
        [0, 1, 8, 32, 64, 16, 32, 64].get(self.0 as usize).copied().unwrap_or(0)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(super) enum TypeDef {
    Void,
    Int(u32),
    Half,
    Float,
    Double,
    /// pointee, address space (3: group shared)
    Pointer(Ty, u32),
    Array(u64, Ty),
    /// `%name = type { ... }`, or a literal `{ ... }`
    Struct(Option<String>, Vec<Ty>),
    /// return type, parameters
    Function(Ty, Vec<Ty>),
}

/// An SSA value or a constant. Constants need no builder: `Val::i32(7)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Val {
    pub(super) ty: Ty,
    pub(super) kind: Kind,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum Kind {
    /// the instruction's index in [`Body::insts`]
    Inst(u32),
    /// sign-extended from the type's width, as LLVM stores it (`i1 true` is -1)
    Int(i64),
    /// the bit pattern, in the type's width
    Float(u64),
    Undef,
    /// index in [`Module::aggregates`]
    Aggregate(u32),
    /// index in [`Module::globals`]
    Global(u32),
    /// index in [`Module::functions`] (metadata names the entry function)
    Function(u32),
    /// what a refused call returns: using it reports nothing more
    Bad,
}

impl Val {
    pub fn bool(v: bool) -> Val {
        Val { ty: Ty::I1, kind: Kind::Int(-i64::from(v)) }
    }
    pub fn i8(v: u8) -> Val {
        Val { ty: Ty::I8, kind: Kind::Int(i64::from(v as i8)) }
    }
    pub fn i32(v: u32) -> Val {
        Val { ty: Ty::I32, kind: Kind::Int(i64::from(v as i32)) }
    }
    pub fn i64(v: u64) -> Val {
        Val { ty: Ty::I64, kind: Kind::Int(v as i64) }
    }
    /// A `half` from its bits.
    pub fn f16(bits: u16) -> Val {
        Val { ty: Ty::F16, kind: Kind::Float(u64::from(bits)) }
    }
    pub fn f32(v: f32) -> Val {
        Val { ty: Ty::F32, kind: Kind::Float(u64::from(v.to_bits())) }
    }
    pub fn f64(v: f64) -> Val {
        Val { ty: Ty::F64, kind: Kind::Float(v.to_bits()) }
    }
    pub fn undef(ty: Ty) -> Val {
        Val { ty, kind: Kind::Undef }
    }
    pub fn ty(self) -> Ty {
        self.ty
    }
    /// Whether the value is a constant (it needs no instruction).
    pub fn is_constant(self) -> bool {
        matches!(self.kind, Kind::Int(_) | Kind::Float(_) | Kind::Undef | Kind::Aggregate(_))
    }
    pub(super) fn bad(ty: Ty) -> Val {
        Val { ty, kind: Kind::Bad }
    }
}

/// A basic block of the function, in creation order; the first is the entry.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Block(pub(super) u32);

/// Binary operations, numbered as bitcode numbers them (an FP operation shares its integer one's).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Binary {
    Add,
    Sub,
    Mul,
    UDiv,
    SDiv,
    URem,
    SRem,
    Shl,
    LShr,
    AShr,
    And,
    Or,
    Xor,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
}

impl Binary {
    fn is_float(self) -> bool {
        matches!(self, Binary::FAdd | Binary::FSub | Binary::FMul | Binary::FDiv | Binary::FRem)
    }
    pub(super) fn code(self) -> u64 {
        match self {
            Binary::Add | Binary::FAdd => 0,
            Binary::Sub | Binary::FSub => 1,
            Binary::Mul | Binary::FMul => 2,
            Binary::UDiv => 3,
            Binary::SDiv | Binary::FDiv => 4,
            Binary::URem => 5,
            Binary::SRem | Binary::FRem => 6,
            Binary::Shl => 7,
            Binary::LShr => 8,
            Binary::AShr => 9,
            Binary::And => 10,
            Binary::Or => 11,
            Binary::Xor => 12,
        }
    }
    /// The flag bits the operation can carry.
    fn flags(self) -> u8 {
        match self {
            Binary::Add | Binary::Sub | Binary::Mul | Binary::Shl => flags::NUW | flags::NSW,
            Binary::UDiv | Binary::SDiv | Binary::LShr | Binary::AShr => flags::EXACT,
            op if op.is_float() => flags::FAST,
            _ => 0,
        }
    }
}

/// Instruction flags for [`Func::binary_with`]. None is set unless the caller sets it (DXC sets
/// `FAST` on FP arithmetic that is not `precise`).
pub mod flags {
    pub const NUW: u8 = 1;
    pub const NSW: u8 = 2;
    pub const EXACT: u8 = 1;
    /// all five fast-math flags
    pub const FAST: u8 = 31;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cast {
    Trunc = 0,
    ZExt = 1,
    SExt = 2,
    FpToUi = 3,
    FpToSi = 4,
    UiToFp = 5,
    SiToFp = 6,
    FpTrunc = 7,
    FpExt = 8,
    BitCast = 11,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Icmp {
    Eq = 32,
    Ne = 33,
    Ugt = 34,
    Uge = 35,
    Ult = 36,
    Ule = 37,
    Sgt = 38,
    Sge = 39,
    Slt = 40,
    Sle = 41,
}

/// Ordered (`O..`: false if an operand is a NaN) and unordered (`U..`: true then) comparisons.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fcmp {
    Oeq = 1,
    Ogt = 2,
    Oge = 3,
    Olt = 4,
    Ole = 5,
    One = 6,
    Ord = 7,
    Uno = 8,
    Ueq = 9,
    Ugt = 10,
    Uge = 11,
    Ult = 12,
    Ule = 13,
    Une = 14,
}

/// `atomicrmw` operations (those `Interlocked*` has).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rmw {
    Xchg = 0,
    Add = 1,
    And = 3,
    Or = 5,
    Xor = 6,
    Max = 7,
    Min = 8,
    UMax = 9,
    UMin = 10,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Inst {
    Binary { op: Binary, flags: u8, a: Val, b: Val },
    /// to the instruction's type
    Cast { op: Cast, value: Val },
    /// `pred` is LLVM's predicate number; fast-math flags on an FP comparison
    Cmp { pred: u8, flags: u8, a: Val, b: Val },
    Select { cond: Val, a: Val, b: Val },
    ExtractValue { aggregate: Val, index: u32 },
    /// index in [`Body::incoming`]
    Phi(u32),
    /// a declared function ([`Module::functions`]) and its arguments, a range of [`Body::operands`]
    Call { function: u32, args: (u32, u32) },
    Br(Block),
    BrIf { cond: Val, then: Block, otherwise: Block },
    /// the cases are a range of [`Body::cases`]
    Switch { value: Val, default: Block, cases: (u32, u32) },
    Ret,
    /// `[words x i32]`, in the entry block
    Alloca { words: u32 },
    /// `getelementptr [n x i32], base, i32 0, index`; `inbounds` on an alloca, as DXC writes it
    Gep { base: Val, index: Val, inbounds: bool },
    Load { pointer: Val },
    Store { pointer: Val, value: Val },
    AtomicRmw { op: Rmw, pointer: Val, value: Val },
    CmpXchg { pointer: Val, expected: Val, new: Val },
    /// replaced by another value ([`Func::replace`])
    Removed,
}

#[derive(Default, Debug)]
pub(super) struct BlockData {
    pub phis: Vec<u32>,
    pub insts: Vec<u32>,
    pub terminated: bool,
}

/// The body of the entry function.
#[derive(Default, Debug)]
pub(super) struct Body {
    pub insts: Vec<Inst>,
    /// each instruction's type (`VOID` for a store or a terminator)
    pub tys: Vec<Ty>,
    pub blocks: Vec<BlockData>,
    /// the allocas created outside the entry block: they go to its start
    pub allocas: Vec<u32>,
    pub operands: Vec<Val>,
    pub cases: Vec<(u32, Block)>,
    pub incoming: Vec<Vec<(Val, Block)>>,
    pub replaced: HashMap<u32, Val>,
    cur: Option<u32>,
}

/// Why a module was not written. `kind` is short and stable (for counting fallbacks per reason).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Refused {
    pub kind: &'static str,
    /// the block and the index of the instruction in it, when it is about one
    pub at: Option<(u32, u32)>,
    pub text: String,
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.at {
            Some((b, i)) => write!(f, "DXIL writer: {} (block {b}, instruction {i}): {}", self.kind, self.text),
            None => write!(f, "DXIL writer: {}: {}", self.kind, self.text),
        }
    }
}

/// A global variable: a group-shared array (the console's LDS).
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Global {
    pub name: String,
    /// `[words x i32]`
    pub ty: Ty,
    pub address_space: u32,
}

/// Attributes of a declared function, besides `nounwind`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum FnAttr {
    None,
    ReadNone,
    ReadOnly,
    NoDuplicate,
}

/// A function of the module: the entry (the first, the only one with a body) or a declaration.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Function {
    pub name: String,
    pub ty: Ty,
    /// None: the entry, which has no attributes
    pub attr: Option<FnAttr>,
}

/// One compute program.
#[derive(Debug)]
pub struct Module {
    pub(super) types: Vec<TypeDef>,
    type_ids: HashMap<TypeDef, Ty>,
    pub(super) aggregates: Vec<(Ty, Vec<Val>)>,
    aggregate_ids: HashMap<(Ty, Vec<Val>), u32>,
    pub(super) globals: Vec<Global>,
    pub(super) functions: Vec<Function>,
    function_ids: HashMap<String, u32>,
    /// the declared function of each (operation, overload) in use
    pub(super) op_functions: HashMap<(u32, Ty), u32>,
    pub(super) resources: Vec<Resource>,
    pub(super) body: Body,
    pub(super) numthreads: [u32; 3],
    pub(super) wave_size: Option<u32>,
    pub(super) refused: Option<Refused>,
}

impl Module {
    /// A compute program (`cs_6_6`) of `numthreads` threads per group.
    pub fn compute(numthreads: [u32; 3]) -> Module {
        let types = vec![TypeDef::Void, TypeDef::Int(1), TypeDef::Int(8), TypeDef::Int(32), TypeDef::Int(64), TypeDef::Half, TypeDef::Float, TypeDef::Double];
        let type_ids = types.iter().cloned().zip((0..).map(Ty)).collect();
        let mut m = Module {
            types,
            type_ids,
            aggregates: Vec::new(),
            aggregate_ids: HashMap::new(),
            globals: Vec::new(),
            functions: Vec::new(),
            function_ids: HashMap::new(),
            op_functions: HashMap::new(),
            resources: Vec::new(),
            body: Body::default(),
            numthreads,
            wave_size: None,
            refused: None,
        };
        let main = m.ty(TypeDef::Function(Ty::VOID, Vec::new()));
        m.functions.push(Function { name: "main".into(), ty: main, attr: None });
        m
    }

    pub(super) fn ty(&mut self, def: TypeDef) -> Ty {
        if let Some(&t) = self.type_ids.get(&def) {
            return t;
        }
        let t = Ty(self.types.len() as u32);
        self.types.push(def.clone());
        self.type_ids.insert(def, t);
        t
    }

    pub(super) fn def(&self, ty: Ty) -> &TypeDef {
        &self.types[ty.0 as usize]
    }

    pub(super) fn pointer(&mut self, to: Ty, address_space: u32) -> Ty {
        self.ty(TypeDef::Pointer(to, address_space))
    }

    /// A constant struct, e.g. the `%dx.types.ResBind` of a `dx.op` call.
    pub(super) fn aggregate(&mut self, ty: Ty, elements: Vec<Val>) -> Val {
        let key = (ty, elements);
        let next = self.aggregates.len() as u32;
        let id = *self.aggregate_ids.entry(key.clone()).or_insert(next);
        if id == next {
            self.aggregates.push(key);
        }
        Val { ty, kind: Kind::Aggregate(id) }
    }

    /// The function `name`, declared on first use.
    pub(super) fn declare(&mut self, name: &str, ret: Ty, params: &[Ty], attr: FnAttr) -> u32 {
        if let Some(&f) = self.function_ids.get(name) {
            return f;
        }
        let ty = self.ty(TypeDef::Function(ret, params.to_vec()));
        let f = self.functions.len() as u32;
        self.functions.push(Function { name: name.into(), ty, attr: Some(attr) });
        self.function_ids.insert(name.into(), f);
        f
    }

    /// A group-shared array of `words` i32 (the console's LDS), zero-based for `Func::gep`.
    /// D3D12 allows 32 KB of them per program.
    pub fn groupshared(&mut self, words: u32) -> Val {
        let array = self.ty(TypeDef::Array(u64::from(words), Ty::I32));
        let ty = self.pointer(array, 3);
        if words == 0 || words > 8192 {
            self.refuse("groupshared-size", None, format!("{words} words of group-shared memory (1 to 8192)"));
            return Val::bad(ty);
        }
        // the name DXC gives `groupshared uint xcl_lds<n>[]`
        let name = format!("\u{1}?xcl_lds{}@@3PAIA", self.globals.len());
        self.globals.push(Global { name, ty: array, address_space: 3 });
        Val { ty, kind: Kind::Global(self.globals.len() as u32 - 1) }
    }

    pub(super) fn refuse(&mut self, kind: &'static str, at: Option<(u32, u32)>, text: String) {
        if self.refused.is_none() {
            self.refused = Some(Refused { kind, at, text });
        }
    }

    /// The builder of the entry function, `void main()`; it can be taken more than once.
    pub fn entry(&mut self) -> Func<'_> {
        Func { m: self }
    }

    /// `v` after the replacements made so far.
    pub(super) fn resolve(&self, mut v: Val) -> Val {
        // a chain cannot be longer than the map; a cycle is refused by `replace`
        for _ in 0..=self.body.replaced.len() {
            match v.kind {
                Kind::Inst(i) => match self.body.replaced.get(&i) {
                    Some(&r) => v = r,
                    None => break,
                },
                _ => break,
            }
        }
        v
    }

    pub(super) fn type_name(&self, ty: Ty) -> String {
        match self.def(ty) {
            TypeDef::Void => "void".into(),
            TypeDef::Int(n) => format!("i{n}"),
            TypeDef::Half => "half".into(),
            TypeDef::Float => "float".into(),
            TypeDef::Double => "double".into(),
            TypeDef::Pointer(t, 0) => format!("{}*", self.type_name(*t)),
            TypeDef::Pointer(t, s) => format!("{} addrspace({s})*", self.type_name(*t)),
            TypeDef::Array(n, t) => format!("[{n} x {}]", self.type_name(*t)),
            TypeDef::Struct(Some(name), _) => format!("%{name}"),
            TypeDef::Struct(None, e) => format!("{{ {} }}", e.iter().map(|&t| self.type_name(t)).collect::<Vec<_>>().join(", ")),
            TypeDef::Function(r, p) => format!("{} ({})", self.type_name(*r), p.iter().map(|&t| self.type_name(t)).collect::<Vec<_>>().join(", ")),
        }
    }
}

/// Builds the entry function. Instructions go to the end of the block chosen with [`Func::at`].
pub struct Func<'m> {
    pub(super) m: &'m mut Module,
}

macro_rules! binary_ops {
    ($($name:ident $op:ident),*) => {
        $(pub fn $name(&mut self, a: Val, b: Val) -> Val { self.binary(Binary::$op, a, b) })*
    };
}

impl Func<'_> {
    /// A new block, after those created so far. The first one is the entry: nothing can branch
    /// to it, so a program whose first instruction is a loop head needs an empty block before.
    pub fn block(&mut self) -> Block {
        self.m.body.blocks.push(BlockData::default());
        Block(self.m.body.blocks.len() as u32 - 1)
    }

    /// Instructions are appended to `block` from now on.
    pub fn at(&mut self, block: Block) {
        if self.known(block) {
            self.m.body.cur = Some(block.0);
        }
    }

    fn known(&mut self, block: Block) -> bool {
        let ok = (block.0 as usize) < self.m.body.blocks.len();
        if !ok {
            self.refuse("unknown-block", format!("block {} of {}", block.0, self.m.body.blocks.len()));
        }
        ok
    }

    fn position(&self) -> Option<(u32, u32)> {
        let b = self.m.body.cur?;
        Some((b, self.m.body.blocks[b as usize].insts.len() as u32))
    }

    pub(super) fn refuse(&mut self, kind: &'static str, text: String) {
        let at = self.position();
        self.m.refuse(kind, at, text);
    }

    /// Checks an operand: it exists and has not been replaced away. False for a value a refused
    /// call returned (nothing more is reported then).
    fn operand(&mut self, v: Val) -> bool {
        match v.kind {
            Kind::Bad => false,
            Kind::Inst(i) if i as usize >= self.m.body.insts.len() => {
                self.refuse("unknown-value", format!("instruction {i} of {}", self.m.body.insts.len()));
                false
            }
            _ => true,
        }
    }

    fn push(&mut self, inst: Inst, ty: Ty) -> Val {
        let Some(b) = self.m.body.cur else {
            self.refuse("no-block", "an instruction before any Func::at".into());
            return Val::bad(ty);
        };
        if self.m.body.blocks[b as usize].terminated {
            self.refuse("after-terminator", "an instruction after the block's terminator".into());
            return Val::bad(ty);
        }
        let i = self.m.body.insts.len() as u32;
        self.m.body.insts.push(inst);
        self.m.body.tys.push(ty);
        let block = &mut self.m.body.blocks[b as usize];
        block.insts.push(i);
        block.terminated = matches!(inst, Inst::Br(_) | Inst::BrIf { .. } | Inst::Switch { .. } | Inst::Ret);
        Val { ty, kind: Kind::Inst(i) }
    }

    fn mismatch(&mut self, what: &str, tys: &[Ty]) {
        let names = tys.iter().map(|&t| self.m.type_name(t)).collect::<Vec<_>>().join(", ");
        self.refuse("operand-type", format!("{what} of {names}"));
    }

    pub fn binary(&mut self, op: Binary, a: Val, b: Val) -> Val {
        self.binary_with(op, 0, a, b)
    }

    /// A binary operation with [`flags`].
    pub fn binary_with(&mut self, op: Binary, flags: u8, a: Val, b: Val) -> Val {
        if !(self.operand(a) & self.operand(b)) {
            return Val::bad(a.ty);
        }
        let logic = matches!(op, Binary::And | Binary::Or | Binary::Xor);
        let fits = if op.is_float() { a.ty.is_float() } else { a.ty.is_int() || (logic && a.ty == Ty::I1) };
        if a.ty != b.ty || !fits {
            self.mismatch(&format!("{op:?}"), &[a.ty, b.ty]);
            return Val::bad(a.ty);
        }
        if flags & !op.flags() != 0 {
            self.refuse("flags", format!("flags {flags:#x} on {op:?}"));
            return Val::bad(a.ty);
        }
        self.push(Inst::Binary { op, flags, a, b }, a.ty)
    }

    binary_ops!(add Add, sub Sub, mul Mul, udiv UDiv, sdiv SDiv, urem URem, srem SRem, shl Shl, lshr LShr, ashr AShr, and And, or Or, xor Xor);
    binary_ops!(fadd FAdd, fsub FSub, fmul FMul, fdiv FDiv, frem FRem);

    pub fn cast(&mut self, op: Cast, value: Val, to: Ty) -> Val {
        if !self.operand(value) {
            return Val::bad(to);
        }
        let from = value.ty;
        let (int, wider) = (|t: Ty| t == Ty::I1 || t.is_int(), to.bits() > from.bits());
        let ok = match op {
            Cast::Trunc => int(from) && int(to) && to.bits() < from.bits(),
            Cast::ZExt | Cast::SExt => int(from) && int(to) && wider,
            Cast::FpToUi | Cast::FpToSi => from.is_float() && to.is_int(),
            Cast::UiToFp | Cast::SiToFp => int(from) && to.is_float(),
            Cast::FpTrunc => from.is_float() && to.is_float() && to.bits() < from.bits(),
            Cast::FpExt => from.is_float() && to.is_float() && wider,
            Cast::BitCast => from.is_int() != to.is_int() && from.is_float() != to.is_float() && from.bits() == to.bits(),
        };
        if !ok {
            self.mismatch(&format!("{op:?}"), &[from, to]);
            return Val::bad(to);
        }
        self.push(Inst::Cast { op, value }, to)
    }

    pub fn icmp(&mut self, pred: Icmp, a: Val, b: Val) -> Val {
        self.cmp(pred as u8, 0, a.ty == Ty::I1 || a.ty.is_int(), a, b)
    }

    pub fn fcmp(&mut self, pred: Fcmp, a: Val, b: Val) -> Val {
        self.fcmp_with(pred, 0, a, b)
    }

    /// An FP comparison with fast-math [`flags`].
    pub fn fcmp_with(&mut self, pred: Fcmp, flags: u8, a: Val, b: Val) -> Val {
        if flags & !flags::FAST != 0 {
            self.refuse("flags", format!("flags {flags:#x} on fcmp"));
            return Val::bad(Ty::I1);
        }
        self.cmp(pred as u8, flags, a.ty.is_float(), a, b)
    }

    fn cmp(&mut self, pred: u8, flags: u8, fits: bool, a: Val, b: Val) -> Val {
        if !(self.operand(a) & self.operand(b)) {
            return Val::bad(Ty::I1);
        }
        if a.ty != b.ty || !fits {
            self.mismatch(if pred >= 32 { "icmp" } else { "fcmp" }, &[a.ty, b.ty]);
            return Val::bad(Ty::I1);
        }
        self.push(Inst::Cmp { pred, flags, a, b }, Ty::I1)
    }

    /// `cond ? a : b`
    pub fn select(&mut self, cond: Val, a: Val, b: Val) -> Val {
        if !(self.operand(cond) & self.operand(a) & self.operand(b)) {
            return Val::bad(a.ty);
        }
        if cond.ty != Ty::I1 || a.ty != b.ty || !a.ty.is_value() {
            self.mismatch("select", &[cond.ty, a.ty, b.ty]);
            return Val::bad(a.ty);
        }
        self.push(Inst::Select { cond, a, b }, a.ty)
    }

    /// Element `index` of a struct value (the result of a `dx.op` call).
    pub fn extract(&mut self, aggregate: Val, index: u32) -> Val {
        if !self.operand(aggregate) {
            return Val::bad(Ty::I32);
        }
        let element = match self.m.def(aggregate.ty) {
            TypeDef::Struct(_, elements) => elements.get(index as usize).copied(),
            _ => None,
        };
        let Some(ty) = element else {
            let name = self.m.type_name(aggregate.ty);
            self.refuse("operand-type", format!("element {index} of {name}"));
            return Val::bad(Ty::I32);
        };
        self.push(Inst::ExtractValue { aggregate, index }, ty)
    }

    /// A phi at the head of `block`, which may already have instructions. Its incoming values
    /// are added with [`Func::incoming`], one per predecessor edge, any time before `finish`.
    pub fn phi(&mut self, block: Block, ty: Ty) -> Val {
        if !self.known(block) {
            return Val::bad(ty);
        }
        if !ty.is_value() {
            self.mismatch("phi", &[ty]);
            return Val::bad(ty);
        }
        let body = &mut self.m.body;
        let i = body.insts.len() as u32;
        body.insts.push(Inst::Phi(body.incoming.len() as u32));
        body.incoming.push(Vec::new());
        body.tys.push(ty);
        body.blocks[block.0 as usize].phis.push(i);
        Val { ty, kind: Kind::Inst(i) }
    }

    /// `phi` has `value` when control comes from `pred`.
    pub fn incoming(&mut self, phi: Val, value: Val, pred: Block) {
        if !(self.operand(phi) & self.operand(value)) || !self.known(pred) {
            return;
        }
        let list = match phi.kind {
            Kind::Inst(i) => match self.m.body.insts[i as usize] {
                Inst::Phi(list) => Some(list),
                _ => None,
            },
            _ => None,
        };
        let Some(list) = list else {
            self.refuse("not-a-phi", "incoming value for something that is not a phi (or was replaced)".into());
            return;
        };
        if value.ty != phi.ty {
            self.mismatch("phi incoming value", &[phi.ty, value.ty]);
            return;
        }
        self.m.body.incoming[list as usize].push((value, pred));
    }

    /// Every use of `old` (an instruction's value), made so far or later, becomes a use of `new`,
    /// and `old` is dropped. Constant time: for removing trivial phis while SSA is constructed.
    pub fn replace(&mut self, old: Val, new: Val) {
        if !(self.operand(old) & self.operand(new)) {
            return;
        }
        let Kind::Inst(i) = old.kind else {
            self.refuse("replace", "the replaced value is not an instruction".into());
            return;
        };
        if old.ty != new.ty || old.ty == Ty::VOID {
            self.mismatch("replace", &[old.ty, new.ty]);
            return;
        }
        if self.m.body.insts[i as usize] == Inst::Removed || self.m.resolve(new) == old {
            self.refuse("replace", format!("instruction {i} replaced twice or by itself"));
            return;
        }
        self.m.body.insts[i as usize] = Inst::Removed;
        self.m.body.replaced.insert(i, new);
    }

    pub fn br(&mut self, to: Block) {
        if self.known(to) {
            self.push(Inst::Br(to), Ty::VOID);
        }
    }

    /// To `then` if `cond`, else to `otherwise`.
    pub fn br_if(&mut self, cond: Val, then: Block, otherwise: Block) {
        if !self.operand(cond) || !(self.known(then) & self.known(otherwise)) {
            return;
        }
        if cond.ty != Ty::I1 {
            self.mismatch("br condition", &[cond.ty]);
            return;
        }
        self.push(Inst::BrIf { cond, then, otherwise }, Ty::VOID);
    }

    /// To the block of the case equal to `value` (an i32), else to `default`.
    pub fn switch(&mut self, value: Val, default: Block, cases: &[(u32, Block)]) {
        if !self.operand(value) || !self.known(default) || !cases.iter().all(|&(_, b)| self.known(b)) {
            return;
        }
        if value.ty != Ty::I32 {
            self.mismatch("switch", &[value.ty]);
            return;
        }
        let mut seen: Vec<u32> = cases.iter().map(|&(c, _)| c).collect();
        seen.sort_unstable();
        if seen.windows(2).any(|w| w[0] == w[1]) {
            self.refuse("switch-case", "a switch with the same case twice".into());
            return;
        }
        let start = self.m.body.cases.len() as u32;
        self.m.body.cases.extend_from_slice(cases);
        self.push(Inst::Switch { value, default, cases: (start, cases.len() as u32) }, Ty::VOID);
    }

    pub fn ret(&mut self) {
        self.push(Inst::Ret, Ty::VOID);
    }

    /// A per-thread array of `words` i32 (scratch memory), for [`Func::gep`]. It is always in the
    /// entry block: where it is created if that is there, else at the block's start.
    pub fn alloca(&mut self, words: u32) -> Val {
        let array = self.m.ty(TypeDef::Array(u64::from(words), Ty::I32));
        let ty = self.m.pointer(array, 0);
        if words == 0 {
            self.refuse("alloca-size", "an empty alloca".into());
            return Val::bad(ty);
        }
        if self.m.body.cur == Some(0) {
            return self.push(Inst::Alloca { words }, ty);
        }
        let body = &mut self.m.body;
        let i = body.insts.len() as u32;
        body.insts.push(Inst::Alloca { words });
        body.tys.push(ty);
        body.allocas.push(i);
        Val { ty, kind: Kind::Inst(i) }
    }

    /// A pointer to element `index` (an i32) of an array from [`Module::groupshared`] or
    /// [`Func::alloca`]. An index past the array's end is undefined.
    pub fn gep(&mut self, array: Val, index: Val) -> Val {
        let element = |f: &mut Self, space| f.m.pointer(Ty::I32, space);
        if !(self.operand(array) & self.operand(index)) {
            return Val::bad(element(self, 0));
        }
        let space = match self.m.def(array.ty) {
            TypeDef::Pointer(to, space) if matches!(self.m.def(*to), TypeDef::Array(..)) => Some(*space),
            _ => None,
        };
        let (Some(space), true) = (space, index.ty == Ty::I32) else {
            self.mismatch("getelementptr", &[array.ty, index.ty]);
            return Val::bad(element(self, 0));
        };
        let ty = element(self, space);
        self.push(Inst::Gep { base: array, index, inbounds: space == 0 }, ty)
    }

    /// The address space of an `i32*` from [`Func::gep`].
    fn element_pointer(&mut self, what: &str, pointer: Val) -> Option<u32> {
        match self.m.def(pointer.ty) {
            TypeDef::Pointer(to, space) if *to == Ty::I32 => Some(*space),
            _ => {
                self.mismatch(what, &[pointer.ty]);
                None
            }
        }
    }

    pub fn load(&mut self, pointer: Val) -> Val {
        if !self.operand(pointer) || self.element_pointer("load", pointer).is_none() {
            return Val::bad(Ty::I32);
        }
        self.push(Inst::Load { pointer }, Ty::I32)
    }

    pub fn store(&mut self, pointer: Val, value: Val) {
        if !(self.operand(pointer) & self.operand(value)) || self.element_pointer("store", pointer).is_none() {
            return;
        }
        if value.ty != Ty::I32 {
            self.mismatch("store", &[value.ty]);
            return;
        }
        self.push(Inst::Store { pointer, value }, Ty::VOID);
    }

    /// The group-shared pointer of an atomic operation and its i32 operands.
    fn atomic(&mut self, what: &str, pointer: Val, values: &[Val]) -> bool {
        if !values.iter().fold(self.operand(pointer), |ok, &v| ok & self.operand(v)) {
            return false;
        }
        match self.element_pointer(what, pointer) {
            Some(3) if values.iter().all(|v| v.ty == Ty::I32) => true,
            Some(3) => {
                self.mismatch(what, &values.iter().map(|v| v.ty).collect::<Vec<_>>());
                false
            }
            Some(_) => {
                self.refuse("atomic-not-groupshared", format!("{what} on memory that is not group shared"));
                false
            }
            None => false,
        }
    }

    /// `atomicrmw` on group-shared memory: the old value. (Atomics on a UAV are `dx.op` calls.)
    pub fn atomic_rmw(&mut self, op: Rmw, pointer: Val, value: Val) -> Val {
        if !self.atomic("atomicrmw", pointer, &[value]) {
            return Val::bad(Ty::I32);
        }
        self.push(Inst::AtomicRmw { op, pointer, value }, Ty::I32)
    }

    /// `cmpxchg` on group-shared memory: the old value (`new` is stored if it was `expected`).
    pub fn cmpxchg(&mut self, pointer: Val, expected: Val, new: Val) -> Val {
        if !self.atomic("cmpxchg", pointer, &[expected, new]) {
            return Val::bad(Ty::I32);
        }
        let pair = self.m.ty(TypeDef::Struct(None, vec![Ty::I32, Ty::I1]));
        let result = self.push(Inst::CmpXchg { pointer, expected, new }, pair);
        self.push(Inst::ExtractValue { aggregate: result, index: 0 }, Ty::I32)
    }

    /// A call of a declared function with arguments of its parameter types.
    #[cfg(test)]
    pub(super) fn call(&mut self, function: u32, args: &[Val]) -> Val {
        self.call_with(function, &[], args)
    }

    /// A call whose arguments are `head` and then `args` (a `dx.op` call's opcode comes first).
    pub(super) fn call_with(&mut self, function: u32, head: &[Val], args: &[Val]) -> Val {
        let all = || head.iter().chain(args);
        let (ret, fits) = match self.m.def(self.m.functions[function as usize].ty) {
            TypeDef::Function(ret, params) => (*ret, all().count() == params.len() && all().zip(params).all(|(a, &p)| a.ty == p)),
            _ => unreachable!("a function's type"),
        };
        if !all().fold(true, |ok, &a| ok & self.operand(a)) {
            return Val::bad(ret);
        }
        if !fits {
            let name = self.m.functions[function as usize].name.clone();
            let got = all().map(|a| self.m.type_name(a.ty)).collect::<Vec<_>>().join(", ");
            let ty = self.m.type_name(self.m.functions[function as usize].ty);
            self.refuse("call-arguments", format!("{name}: ({got}) for {ty}"));
            return Val::bad(ret);
        }
        let start = self.m.body.operands.len() as u32;
        self.m.body.operands.extend(all());
        self.push(Inst::Call { function, args: (start, (head.len() + args.len()) as u32) }, ret)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(m: &Module) -> (&'static str, Option<(u32, u32)>) {
        let r = m.refused.as_ref().expect("nothing was refused");
        (r.kind, r.at)
    }

    #[test]
    fn constants_are_stored_sign_extended_as_llvm_does() {
        assert_eq!(Val::bool(true).kind, Kind::Int(-1));
        assert_eq!(Val::i32(0xFFFF_FFFF).kind, Kind::Int(-1));
        assert_eq!(Val::i32(0x7FFF_FFFF).kind, Kind::Int(0x7FFF_FFFF));
        assert_eq!(Val::i8(0x80).kind, Kind::Int(-128));
        assert_eq!(Val::i64(u64::MAX).kind, Kind::Int(-1));
        assert_eq!(Val::f32(1.0).kind, Kind::Float(0x3F80_0000));
        assert_eq!(Val::f32(-0.0), Val::f32(-0.0));
        assert_ne!(Val::f32(0.0), Val::f32(-0.0));
        assert!(Val::undef(Ty::F64).is_constant() && Val::undef(Ty::F64).ty() == Ty::F64);
    }

    #[test]
    fn types_and_declarations_are_interned() {
        let mut m = Module::compute([1, 1, 1]);
        let a = m.ty(TypeDef::Struct(Some("dx.types.Handle".into()), vec![Ty::I32]));
        assert_eq!(a, m.ty(TypeDef::Struct(Some("dx.types.Handle".into()), vec![Ty::I32])));
        let pointer = m.pointer(Ty::I32, 3);
        assert_eq!(m.type_name(pointer), "i32 addrspace(3)*");
        let f = m.declare("dx.op.unaryBits.i32", Ty::I32, &[Ty::I32, Ty::I32], FnAttr::ReadNone);
        assert_eq!(f, m.declare("dx.op.unaryBits.i32", Ty::I32, &[Ty::I32, Ty::I32], FnAttr::ReadNone));
        assert_eq!((f, m.functions[0].name.as_str()), (1, "main"));
        let b = m.aggregate(a, vec![Val::i32(4107)]);
        assert_eq!(b, m.aggregate(a, vec![Val::i32(4107)]));
        assert_ne!(b, m.aggregate(a, vec![Val::i32(0)]));
    }

    #[test]
    fn a_loop_builds_with_phis_filled_in_later() {
        let mut m = Module::compute([32, 1, 1]);
        let mut f = m.entry();
        let (b0, head, body, exit) = (f.block(), f.block(), f.block(), f.block());
        f.at(b0);
        f.br(head);
        f.at(head);
        let c = f.icmp(Icmp::Ult, Val::i32(1), Val::i32(2));
        f.br_if(c, body, exit);
        f.at(body);
        // phis come after the block has its instructions, and are used before they are complete
        let i = f.phi(head, Ty::I32);
        let i2 = f.add(i, Val::i32(1));
        f.br(head);
        f.incoming(i, Val::i32(0), b0);
        f.incoming(i, i2, body);
        f.at(exit);
        f.ret();
        assert_eq!(m.refused, None);
        assert_eq!(m.body.blocks[1].phis.len(), 1);
        assert_eq!(m.body.incoming[0], vec![(Val::i32(0), b0), (i2, body)]);
        assert!(m.body.blocks.iter().all(|b| b.terminated));
    }

    #[test]
    fn replace_redirects_uses_through_chains() {
        let mut m = Module::compute([1, 1, 1]);
        let mut f = m.entry();
        let b = f.block();
        f.at(b);
        let (p, q) = (f.phi(b, Ty::I32), f.phi(b, Ty::I32));
        let x = f.add(Val::i32(1), Val::i32(2));
        f.replace(p, q);
        f.replace(q, x);
        assert_eq!((m.resolve(p), m.resolve(q), m.resolve(x)), (x, x, x));
        assert_eq!(m.refused, None);
        // a cycle is refused where it is made
        let mut f = m.entry();
        let r = f.phi(b, Ty::I32);
        f.replace(x, r);
        f.replace(r, p);
        assert_eq!(refused(&m).0, "replace");
    }

    #[test]
    fn memory_is_i32_arrays_and_atomics_are_group_shared_only() {
        let mut m = Module::compute([1, 1, 1]);
        let lds = m.groupshared(8192);
        let mut f = m.entry();
        let b = f.block();
        f.at(b);
        // in the entry block an alloca stays where it is created
        let scratch = f.alloca(1024);
        let (p, q) = (f.gep(lds, Val::i32(5)), f.gep(scratch, Val::i32(5)));
        let old = f.atomic_rmw(Rmw::Add, p, Val::i32(1));
        let old2 = f.cmpxchg(p, old, Val::i32(7));
        f.store(q, old2);
        let v = f.load(q);
        f.store(p, v);
        f.ret();
        assert_eq!(m.refused, None);
        assert_eq!(m.type_name(p.ty), "i32 addrspace(3)*");
        assert_eq!(m.type_name(scratch.ty), "[1024 x i32]*");
        assert_eq!(m.body.insts[1], Inst::Gep { base: lds, index: Val::i32(5), inbounds: false });
        assert_eq!(m.body.insts[2], Inst::Gep { base: scratch, index: Val::i32(5), inbounds: true });
        assert_eq!((m.body.allocas.len(), m.body.blocks[0].insts[0], m.globals[0].name.as_str()), (0, 0, "\u{1}?xcl_lds0@@3PAIA"));
        // an alloca is no place for an atomic
        f = m.entry();
        f.atomic_rmw(Rmw::Add, q, Val::i32(1));
        assert_eq!(refused(&m).0, "atomic-not-groupshared");
    }

    /// One wrong call in a fresh one-block function: the refusal's kind.
    fn refusal<R>(build: impl FnOnce(&mut Func, Block) -> R) -> &'static str {
        let mut m = Module::compute([1, 1, 1]);
        let mut f = m.entry();
        let b = f.block();
        f.at(b);
        build(&mut f, b);
        m.refused.map_or("", |r| r.kind)
    }

    #[test]
    fn wrong_programs_are_refused_with_their_place() {
        let mut m = Module::compute([1, 1, 1]);
        let mut f = m.entry();
        let b = f.block();
        f.at(b);
        let x = f.add(Val::i32(1), Val::i32(2));
        let y = f.add(x, Val::f32(1.0));
        // the value of a refused call can be used: only the first mistake is reported
        let z = f.mul(y, y);
        f.select(z, x, x);
        let r = m.refused.clone().unwrap();
        assert_eq!((r.kind, r.at), ("operand-type", Some((0, 1))));
        assert_eq!(r.to_string(), "DXIL writer: operand-type (block 0, instruction 1): Add of i32, float");

        assert_eq!(refusal(|f, _| f.fadd(Val::i32(1), Val::i32(2))), "operand-type");
        assert_eq!(refusal(|f, _| f.add(Val::bool(true), Val::bool(true))), "operand-type");
        assert_eq!(refusal(|f, _| f.and(Val::bool(true), Val::bool(true))), "");
        assert_eq!(refusal(|f, _| f.binary_with(Binary::FAdd, flags::FAST, Val::f32(1.0), Val::f32(2.0))), "");
        assert_eq!(refusal(|f, _| f.binary_with(Binary::And, flags::NUW, Val::i32(1), Val::i32(2))), "flags");
        assert_eq!(refusal(|f, _| f.cast(Cast::BitCast, Val::i32(1), Ty::F32)), "");
        assert_eq!(refusal(|f, _| f.cast(Cast::BitCast, Val::i32(1), Ty::F64)), "operand-type");
        assert_eq!(refusal(|f, _| f.cast(Cast::ZExt, Val::i64(1), Ty::I32)), "operand-type");
        assert_eq!(refusal(|f, _| f.cast(Cast::Trunc, Val::i64(1), Ty::I32)), "");
        assert_eq!(refusal(|f, _| f.cast(Cast::UiToFp, Val::bool(true), Ty::F32)), "");
        assert_eq!(refusal(|f, _| f.cast(Cast::ZExt, Val::i8(1), Ty::I32)), "operand-type");
        assert_eq!(refusal(|f, _| f.fcmp(Fcmp::Olt, Val::i32(1), Val::i32(2))), "operand-type");
        assert_eq!(refusal(|f, _| f.icmp(Icmp::Eq, Val::i32(1), Val::i64(2))), "operand-type");
        assert_eq!(refusal(|f, _| f.select(Val::i32(1), Val::i32(1), Val::i32(2))), "operand-type");
        assert_eq!(refusal(|f, _| f.extract(Val::i32(1), 0)), "operand-type");
        assert_eq!(refusal(|f, b| f.phi(b, Ty::I8)), "operand-type");
        assert_eq!(refusal(|f, b| f.incoming(Val::i32(1), Val::i32(1), b)), "not-a-phi");
        assert_eq!(refusal(|f, b| {
            let p = f.phi(b, Ty::I32);
            f.incoming(p, Val::i64(1), b)
        }), "operand-type");
        assert_eq!(refusal(|f, _| f.br(Block(7))), "unknown-block");
        assert_eq!(refusal(|f, b| f.br_if(Val::i32(1), b, b)), "operand-type");
        assert_eq!(refusal(|f, b| f.switch(Val::i32(1), b, &[(1, b), (1, b)])), "switch-case");
        assert_eq!(refusal(|f, _| {
            f.ret();
            f.ret()
        }), "after-terminator");
        assert_eq!(refusal(|f, _| {
            let a = f.alloca(4);
            let p = f.gep(a, Val::i32(0));
            f.atomic_rmw(Rmw::Add, p, Val::i32(1))
        }), "atomic-not-groupshared");
        assert_eq!(refusal(|f, _| {
            let a = f.alloca(4);
            f.load(a)
        }), "operand-type");
        assert_eq!(refusal(|f, _| f.gep(Val::i32(1), Val::i32(0))), "operand-type");
        assert_eq!(refusal(|f, _| f.add(Val { ty: Ty::I32, kind: Kind::Inst(99) }, Val::i32(1))), "unknown-value");

        let mut m = Module::compute([1, 1, 1]);
        m.entry().add(Val::i32(1), Val::i32(2));
        assert_eq!(refused(&m), ("no-block", None));
        let mut m = Module::compute([1, 1, 1]);
        m.groupshared(8193);
        assert_eq!(refused(&m).0, "groupshared-size");
    }

    #[test]
    fn calls_check_their_arguments() {
        let mut m = Module::compute([1, 1, 1]);
        let bits = m.declare("dx.op.unaryBits.i32", Ty::I32, &[Ty::I32, Ty::I32], FnAttr::ReadNone);
        let mut f = m.entry();
        let b = f.block();
        f.at(b);
        let n = f.call(bits, &[Val::i32(31), Val::i32(5)]);
        assert_eq!(n.ty(), Ty::I32);
        f.call(bits, &[Val::i32(31), Val::f32(5.0)]);
        let r = m.refused.unwrap();
        assert_eq!((r.kind, r.text.as_str()), ("call-arguments", "dx.op.unaryBits.i32: (i32, float) for i32 (i32, i32)"));
    }
}
