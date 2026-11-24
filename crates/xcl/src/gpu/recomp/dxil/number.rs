//! Checks a built function, numbers its values as LLVM 3.7 does and writes its FUNCTION block.
//!
//! Value ids: the module's globals, functions and constants (those its metadata names), then the
//! function's constants, then every instruction with a result, block by block in creation
//! order (an entry block's allocas first, a block's phis before its other instructions). An
//! operand is written relative (the instruction's own id minus the operand's): a later value
//! wraps in 32 bits and is followed by its type where the record form has one; phi operands are
//! signed. Basic blocks are numbered apart, from 0.
//!
//! The record forms are the ones DXC 1.7 / 1.9 writes (LLVM 3.7 `BitcodeWriter.cpp`,
//! University of Illinois/NCSA licence): explicit types in CALL, LOAD, GEP and ALLOCA.

use std::collections::HashMap;

use super::module::{Block, Inst, Kind, Module, Refused, Ty, TypeDef, Val};

/// Where records go: a bitstream, or a list in tests.
pub(super) trait Sink {
    fn enter(&mut self, block: u32, abbrev_width: u32);
    fn record(&mut self, code: u32, ops: &[u64]);
    fn exit(&mut self);
}

pub(super) const CONSTANTS_BLOCK: u32 = 11;
pub(super) const FUNCTION_BLOCK: u32 = 12;

/// The function ready to write: checked, operands resolved, everything numbered.
pub(super) struct Numbered {
    /// each block's instructions in their final order
    pub order: Vec<Vec<u32>>,
    /// each instruction's value id (unused for one with no result)
    ids: Vec<u32>,
    /// the module's constants and then the function's, and their value ids
    pub module_constants: Vec<Val>,
    pub function_constants: Vec<Val>,
    constant_ids: HashMap<Val, u32>,
    /// the first value id after the module's globals and functions
    first_constant: u32,
    /// each type's index in the type table (`u32::MAX`: not in it)
    pub type_ids: Vec<u32>,
}

/// LLVM's signed VBR form: the sign in the lowest bit.
pub(super) fn signed(v: i64) -> u64 {
    if v >= 0 {
        (v as u64) << 1
    } else {
        (v.unsigned_abs() << 1) | 1
    }
}

fn successors(m: &Module, inst: &Inst, mut f: impl FnMut(Block)) {
    match *inst {
        Inst::Br(to) => f(to),
        Inst::BrIf { then, otherwise, .. } => {
            f(then);
            f(otherwise);
        }
        Inst::Switch { default, cases: (start, n), .. } => {
            f(default);
            m.body.cases[start as usize..(start + n) as usize].iter().for_each(|&(_, b)| f(b));
        }
        _ => {}
    }
}

/// The operands of an instruction that are in the instruction itself (a call's and a phi's are in
/// the body's lists).
fn operands(inst: &mut Inst) -> Vec<&mut Val> {
    match inst {
        Inst::Binary { a, b, .. } | Inst::Cmp { a, b, .. } => vec![a, b],
        Inst::Cast { value, .. } | Inst::Switch { value, .. } => vec![value],
        Inst::Select { cond, a, b } => vec![cond, a, b],
        Inst::ExtractValue { aggregate, .. } => vec![aggregate],
        Inst::BrIf { cond, .. } => vec![cond],
        Inst::Gep { base, index, .. } => vec![base, index],
        Inst::Load { pointer } => vec![pointer],
        Inst::Store { pointer, value } | Inst::AtomicRmw { pointer, value, .. } => vec![pointer, value],
        Inst::CmpXchg { pointer, expected, new } => vec![pointer, expected, new],
        Inst::Phi(_) | Inst::Call { .. } | Inst::Br(_) | Inst::Ret | Inst::Alloca { .. } | Inst::Removed => Vec::new(),
    }
}

impl Numbered {
    /// Checks and numbers the function of `m`, whose module-level values are its globals, its
    /// functions and `module_constants` (in that order).
    pub fn new(m: &mut Module, module_constants: Vec<Val>) -> Result<Numbered, Refused> {
        if let Some(r) = &m.refused {
            return Err(r.clone());
        }
        let refuse = |kind, block: usize, inst: usize, text: String| Err(Refused { kind, at: Some((block as u32, inst as u32)), text });
        if m.body.blocks.is_empty() {
            return Err(Refused { kind: "no-block", at: None, text: "the function has no block".into() });
        }
        // every block ends in a terminator; its edges
        let mut preds: Vec<Vec<u32>> = vec![Vec::new(); m.body.blocks.len()];
        for (b, block) in m.body.blocks.iter().enumerate() {
            if !block.terminated {
                return refuse("no-terminator", b, block.insts.len(), "the block has no terminator".into());
            }
            successors(m, &m.body.insts[*block.insts.last().unwrap() as usize], |to| preds[to.0 as usize].push(b as u32));
        }
        if let Some(&from) = preds[0].first() {
            return refuse("branch-to-entry", from as usize, m.body.blocks[from as usize].insts.len() - 1, "a branch to the entry block".into());
        }
        // every block can be reached from the entry
        let (mut reached, mut todo) = (vec![false; preds.len()], vec![0]);
        reached[0] = true;
        while let Some(b) = todo.pop() {
            successors(m, &m.body.insts[*m.body.blocks[b].insts.last().unwrap() as usize], |to| {
                if !std::mem::replace(&mut reached[to.0 as usize], true) {
                    todo.push(to.0 as usize);
                }
            });
        }
        if let Some(b) = reached.iter().position(|&r| !r) {
            return refuse("unreachable-block", b, 0, "no path from the entry block reaches the block".into());
        }

        // uses of replaced values become uses of their replacements
        if !m.body.replaced.is_empty() {
            for i in 0..m.body.insts.len() {
                let mut inst = m.body.insts[i];
                operands(&mut inst).into_iter().for_each(|v| *v = m.resolve(*v));
                m.body.insts[i] = inst;
            }
            for i in 0..m.body.operands.len() {
                m.body.operands[i] = m.resolve(m.body.operands[i]);
            }
            for i in 0..m.body.incoming.len() {
                for k in 0..m.body.incoming[i].len() {
                    m.body.incoming[i][k].0 = m.resolve(m.body.incoming[i][k].0);
                }
            }
        }

        // the final order; a phi has one value per incoming edge
        let live = |i: &&u32| m.body.insts[**i as usize] != Inst::Removed;
        let mut order: Vec<Vec<u32>> = Vec::with_capacity(m.body.blocks.len());
        for (b, block) in m.body.blocks.iter().enumerate() {
            let allocas = if b == 0 { &m.body.allocas[..] } else { &[] };
            order.push(allocas.iter().chain(block.phis.iter().filter(live)).chain(block.insts.iter().filter(live)).copied().collect());
            preds[b].sort_unstable();
            for (k, &phi) in block.phis.iter().filter(live).enumerate() {
                let Inst::Phi(list) = m.body.insts[phi as usize] else { unreachable!("a block's phis") };
                let mut incoming = m.body.incoming[list as usize].clone();
                incoming.sort_by_key(|&(_, from)| from.0);
                if !incoming.iter().map(|&(_, from)| from.0).eq(preds[b].iter().copied()) {
                    let from: Vec<u32> = incoming.iter().map(|&(_, from)| from.0).collect();
                    return refuse("phi-incoming", b, k, format!("a phi with values from blocks {from:?} in a block entered from {:?}", preds[b]));
                }
                if incoming.windows(2).any(|w| w[0].1 == w[1].1 && w[0].0 != w[1].0) {
                    return refuse("phi-incoming", b, k, "a phi with two values from one block".into());
                }
            }
        }

        let mut n = Numbered {
            order,
            ids: vec![u32::MAX; m.body.insts.len()],
            module_constants,
            function_constants: Vec::new(),
            constant_ids: HashMap::new(),
            first_constant: (m.globals.len() + m.functions.len()) as u32,
            type_ids: Vec::new(),
        };
        for (k, &c) in n.module_constants.iter().enumerate() {
            n.constant_ids.insert(c, n.first_constant + k as u32);
        }
        n.collect_constants(m);
        let mut next = n.first_constant + (n.module_constants.len() + n.function_constants.len()) as u32;
        for &i in n.order.iter().flatten() {
            if m.body.tys[i as usize] != Ty::VOID {
                n.ids[i as usize] = next;
                next += 1;
            }
        }
        n.number_types(m);
        Ok(n)
    }

    /// The function's constants: those its instructions use that the module does not have, integers
    /// first and each type's together (DXC's order of the kinds; an aggregate's elements, which
    /// are integers, so come before it).
    fn collect_constants(&mut self, m: &Module) {
        let mut found: Vec<Val> = Vec::new();
        let mut seen: HashMap<Val, ()> = HashMap::new();
        let mut add = |v: Val, module: &HashMap<Val, u32>| {
            let elements = match v.kind {
                Kind::Aggregate(a) => &m.aggregates[a as usize].1[..],
                Kind::Int(_) | Kind::Float(_) | Kind::Undef => &[],
                _ => return,
            };
            for &c in elements.iter().chain([&v]) {
                if !module.contains_key(&c) && seen.insert(c, ()).is_none() {
                    found.push(c);
                }
            }
        };
        for &i in self.order.iter().flatten() {
            match m.body.insts[i as usize] {
                // the array size of an alloca and the first index of a getelementptr
                Inst::Alloca { .. } => add(Val::i32(1), &self.constant_ids),
                Inst::Gep { .. } => add(Val::i32(0), &self.constant_ids),
                Inst::Switch { cases: (start, n), .. } => m.body.cases[start as usize..(start + n) as usize].iter().for_each(|&(c, _)| add(Val::i32(c), &self.constant_ids)),
                _ => {}
            }
            m.each_operand(i, |v| add(v, &self.constant_ids));
        }
        let class = |v: &Val| match m.def(v.ty) {
            TypeDef::Int(_) => 0,
            TypeDef::Half | TypeDef::Float | TypeDef::Double => 1,
            _ => 2,
        };
        let mut rank: HashMap<Ty, usize> = HashMap::new();
        for v in &found {
            let next = rank.len();
            rank.entry(v.ty).or_insert(next);
        }
        found.sort_by_key(|v| (class(v), rank[&v.ty]));
        let first = self.first_constant + self.module_constants.len() as u32;
        for (k, &c) in found.iter().enumerate() {
            self.constant_ids.insert(c, first + k as u32);
        }
        self.function_constants = found;
    }

    /// The type table: the types of the module's values and what they are made of, in the order
    /// they were created (a type's parts come before it).
    fn number_types(&mut self, m: &Module) {
        let mut used = vec![false; m.types.len()];
        // the composite types exist because something uses them; of the scalars only those in use
        used[Ty::SCALARS as usize..].fill(true);
        let value_types = self.order.iter().flatten().map(|&i| m.body.tys[i as usize]);
        for t in value_types.chain(self.module_constants.iter().chain(&self.function_constants).map(|v| v.ty)) {
            used[t.0 as usize] = true;
        }
        for t in (0..m.types.len()).rev().filter(|&t| used[t]).collect::<Vec<_>>() {
            // parts have lower indices, so one pass from the end marks them all
            match &m.types[t] {
                TypeDef::Pointer(e, _) | TypeDef::Array(_, e) => used[e.0 as usize] = true,
                TypeDef::Struct(_, e) => e.iter().for_each(|e| used[e.0 as usize] = true),
                TypeDef::Function(r, p) => p.iter().chain([r]).for_each(|e| used[e.0 as usize] = true),
                _ => {}
            }
        }
        let mut next = 0;
        self.type_ids = used.iter().map(|&u| if u { next += 1; next - 1 } else { u32::MAX }).collect();
    }

    pub fn type_id(&self, ty: Ty) -> u64 {
        debug_assert_ne!(self.type_ids[ty.0 as usize], u32::MAX, "a type that is not in the table");
        u64::from(self.type_ids[ty.0 as usize])
    }

    /// The value id of anything but a refused value.
    pub fn id(&self, m: &Module, v: Val) -> u32 {
        match v.kind {
            Kind::Inst(i) => self.ids[i as usize],
            Kind::Global(g) => g,
            Kind::Function(f) => m.globals.len() as u32 + f,
            Kind::Int(_) | Kind::Float(_) | Kind::Undef | Kind::Aggregate(_) => self.constant_ids[&v],
            Kind::Bad => unreachable!("a refused value in a module that was not refused"),
        }
    }

    /// A CONSTANTS block: a SETTYPE before each run of one type.
    pub fn write_constants(&self, m: &Module, constants: &[Val], sink: &mut impl Sink) {
        if constants.is_empty() {
            return;
        }
        sink.enter(CONSTANTS_BLOCK, 4);
        let mut ty = None;
        for &c in constants {
            if ty != Some(c.ty) {
                sink.record(1, &[self.type_id(c.ty)]);
                ty = Some(c.ty);
            }
            match c.kind {
                // zero of any type is NULL
                Kind::Int(0) | Kind::Float(0) => sink.record(2, &[]),
                Kind::Undef => sink.record(3, &[]),
                Kind::Int(v) => sink.record(4, &[signed(v)]),
                Kind::Float(bits) => sink.record(6, &[bits]),
                Kind::Aggregate(a) => {
                    let ids: Vec<u64> = m.aggregates[a as usize].1.iter().map(|&e| u64::from(self.id(m, e))).collect();
                    sink.record(7, &ids);
                }
                _ => unreachable!("a constant"),
            }
        }
        sink.exit();
    }

    /// The FUNCTION block of the entry function.
    pub fn write_function(&self, m: &Module, sink: &mut impl Sink) {
        sink.enter(FUNCTION_BLOCK, 4);
        sink.record(1, &[self.order.len() as u64]);
        self.write_constants(m, &self.function_constants, sink);
        let mut ops: Vec<u64> = Vec::new();
        let mut next = self.first_constant + (self.module_constants.len() + self.function_constants.len()) as u32;
        for &i in self.order.iter().flatten() {
            ops.clear();
            // an operand: relative to this instruction; `typed` adds the type of a later value
            let rel = |ops: &mut Vec<u64>, v: Val| ops.push(u64::from(next.wrapping_sub(self.id(m, v))));
            let typed = |ops: &mut Vec<u64>, v: Val| {
                rel(ops, v);
                if self.id(m, v) >= next {
                    ops.push(self.type_id(v.ty));
                }
            };
            let ty = m.body.tys[i as usize];
            let code = match m.body.insts[i as usize] {
                Inst::Binary { op, flags, a, b } => {
                    typed(&mut ops, a);
                    rel(&mut ops, b);
                    ops.push(op.code());
                    if flags != 0 {
                        ops.push(u64::from(flags));
                    }
                    2
                }
                Inst::Cast { op, value } => {
                    typed(&mut ops, value);
                    ops.extend([self.type_id(ty), op as u64]);
                    3
                }
                Inst::Cmp { pred, flags, a, b } => {
                    typed(&mut ops, a);
                    rel(&mut ops, b);
                    ops.push(u64::from(pred));
                    if flags != 0 {
                        ops.push(u64::from(flags));
                    }
                    28
                }
                Inst::Select { cond, a, b } => {
                    typed(&mut ops, a);
                    rel(&mut ops, b);
                    typed(&mut ops, cond);
                    29
                }
                Inst::ExtractValue { aggregate, index } => {
                    typed(&mut ops, aggregate);
                    ops.push(u64::from(index));
                    26
                }
                Inst::Phi(list) => {
                    ops.push(self.type_id(ty));
                    for &(v, from) in &m.body.incoming[list as usize] {
                        ops.extend([signed(i64::from(next) - i64::from(self.id(m, v))), u64::from(from.0)]);
                    }
                    16
                }
                Inst::Call { function, args: (start, n) } => {
                    // no attributes at the call, the C calling convention with an explicit function type
                    ops.extend([0, 1 << 15, self.type_id(m.functions[function as usize].ty)]);
                    ops.push(u64::from(next - (m.globals.len() as u32 + function)));
                    // an argument has no type even when it is a later value
                    m.body.operands[start as usize..(start + n) as usize].iter().for_each(|&a| rel(&mut ops, a));
                    34
                }
                Inst::Br(to) => {
                    ops.push(u64::from(to.0));
                    11
                }
                Inst::BrIf { cond, then, otherwise } => {
                    ops.extend([u64::from(then.0), u64::from(otherwise.0)]);
                    rel(&mut ops, cond);
                    11
                }
                Inst::Switch { value, default, cases: (start, n) } => {
                    ops.push(self.type_id(Ty::I32));
                    rel(&mut ops, value);
                    ops.push(u64::from(default.0));
                    for &(c, to) in &m.body.cases[start as usize..(start + n) as usize] {
                        // a case is a constant's value id, not relative
                        ops.extend([u64::from(self.id(m, Val::i32(c))), u64::from(to.0)]);
                    }
                    12
                }
                Inst::Ret => 10,
                Inst::Alloca { .. } => {
                    let TypeDef::Pointer(array, _) = *m.def(ty) else { unreachable!("an alloca's type") };
                    // one array, align 4 (stored as log2 + 1), bit 6: the type is the allocated one
                    ops.extend([self.type_id(array), self.type_id(Ty::I32), u64::from(self.id(m, Val::i32(1))), 3 | 1 << 6]);
                    19
                }
                Inst::Gep { base, index, inbounds } => {
                    let TypeDef::Pointer(array, _) = *m.def(base.ty) else { unreachable!("an array pointer") };
                    ops.extend([u64::from(inbounds), self.type_id(array)]);
                    typed(&mut ops, base);
                    typed(&mut ops, Val::i32(0));
                    typed(&mut ops, index);
                    43
                }
                Inst::Load { pointer } => {
                    typed(&mut ops, pointer);
                    ops.extend([self.type_id(ty), 3, 0]);
                    20
                }
                Inst::Store { pointer, value } => {
                    typed(&mut ops, pointer);
                    typed(&mut ops, value);
                    ops.extend([3, 0]);
                    44
                }
                Inst::AtomicRmw { op, pointer, value } => {
                    typed(&mut ops, pointer);
                    rel(&mut ops, value);
                    // not volatile, seq_cst, across threads
                    ops.extend([op as u64, 0, 6, 1]);
                    38
                }
                Inst::CmpXchg { pointer, expected, new } => {
                    typed(&mut ops, pointer);
                    typed(&mut ops, expected);
                    rel(&mut ops, new);
                    // not volatile, seq_cst across threads on success, seq_cst on failure, not weak
                    ops.extend([0, 6, 1, 6, 0]);
                    46
                }
                Inst::Removed => unreachable!("a replaced instruction in the final order"),
            };
            sink.record(code, &ops);
            if ty != Ty::VOID {
                next += 1;
            }
        }
        sink.exit();
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::module::{FnAttr, Icmp, Rmw};
    use super::*;

    /// The records as text, one line each, blocks indented.
    #[derive(Default)]
    pub struct Listing(pub Vec<String>, usize);

    impl Sink for Listing {
        fn enter(&mut self, block: u32, _: u32) {
            self.0.push(format!("{}<{block}>", "  ".repeat(self.1)));
            self.1 += 1;
        }
        fn record(&mut self, code: u32, ops: &[u64]) {
            self.0.push(format!("{}{code} {ops:?}", "  ".repeat(self.1)));
        }
        fn exit(&mut self) {
            self.1 -= 1;
        }
    }

    fn function(m: &mut Module) -> Vec<String> {
        let n = Numbered::new(m, Vec::new()).unwrap();
        let mut out = Listing::default();
        n.write_function(m, &mut out);
        out.0
    }

    #[test]
    fn signed_operands_keep_the_sign_in_bit_zero() {
        assert_eq!([signed(0), signed(1), signed(-1), signed(5), signed(-5), signed(i64::MIN)], [0, 2, 3, 10, 11, 1]);
    }

    /// `for (i = 0; i != n; i++) s += i;` with the phis created last: values used before their
    /// definition in the layout, phi operands that are later values, and constants.
    #[test]
    fn a_loop_is_numbered_in_layout_order_with_relative_operands() {
        let mut m = Module::compute([1, 1, 1]);
        let get = m.declare("get", Ty::I32, &[Ty::I32], FnAttr::ReadNone);
        let put = m.declare("put", Ty::VOID, &[Ty::I32, Ty::I32], FnAttr::None);
        let mut f = m.entry();
        let (b0, head, exit) = (f.block(), f.block(), f.block());
        f.at(b0);
        let n = f.call(get, &[Val::i32(96)]);
        f.br(head);
        f.at(exit);
        let s = f.phi(head, Ty::I32);
        f.call(put, &[Val::i32(7), s]);
        f.ret();
        f.at(head);
        let i = f.phi(head, Ty::I32);
        let s2 = f.add(s, i);
        let i2 = f.add(i, Val::i32(1));
        let done = f.icmp(Icmp::Eq, i2, n);
        f.br_if(done, exit, head);
        f.incoming(s, Val::i32(0), b0);
        f.incoming(s, s2, head);
        f.incoming(i, i2, head);
        f.incoming(i, Val::i32(0), b0);
        // types: void 0, i1 1, i32 2, void () 3, i32 (i32) 4, void (i32, i32) 5; values: 0 main, 1 get, 2 put;
        // constants in the order of their first use, 3 = 96, 4 = 0, 5 = 1, 6 = 7; then 7 = n, 8 = s, 9 = i,
        // 10 = s2, 11 = i2, 12 = done
        assert_eq!(
            function(&mut m),
            [
                "<12>",
                "  1 [3]",
                "  <11>",
                "    1 [2]",
                "    4 [192]",
                "    2 []",
                "    4 [2]",
                "    4 [14]",
                "  34 [0, 32768, 4, 6, 4]",   // n = get(96): the callee 7 - 1, the argument 7 - 3
                "  11 [1]",
                "  16 [2, 8, 0, 5, 1]",       // s = phi [0, b0], [s2, head]: 8 - 4 = 4, 8 - 10 = -2
                "  16 [2, 5, 1, 10, 0]",      // i = phi [i2, head], [0, b0]: 9 - 11 = -2, 9 - 4 = 5
                "  2 [2, 1, 0]",              // s2 = add s, i
                "  2 [2, 6, 0]",              // i2 = add i, 1: 11 - 9, 11 - 5
                "  28 [1, 5, 32]",            // done = icmp eq i2, n
                "  11 [2, 1, 1]",             // br done, exit, head: the condition 13 - 12
                "  34 [0, 32768, 5, 11, 7, 5]", // put(7, s) in the last block: 13 - 2, 13 - 6, 13 - 8
                "  10 []",
            ]
        );
    }

    /// A value used in a block that is laid out before the one that defines it: the operand
    /// wraps in 32 bits and has its type after it, except as a call's argument.
    #[test]
    fn a_later_value_wraps_and_names_its_type() {
        let mut m = Module::compute([1, 1, 1]);
        let put = m.declare("put", Ty::VOID, &[Ty::I32], FnAttr::None);
        let mut f = m.entry();
        let (b0, user, def) = (f.block(), f.block(), f.block());
        f.at(b0);
        f.br(def);
        f.at(def);
        let x = f.add(Val::i32(5), Val::i32(5));
        let c = f.icmp(Icmp::Ne, x, Val::i32(5));
        f.br(user);
        f.at(user);
        let y = f.xor(x, x);
        f.select(c, y, x);
        f.call(put, &[x]);
        f.cast(super::super::module::Cast::ZExt, x, Ty::I64);
        f.ret();
        // types: void 0, i1 1, i32 2, i64 3, void () 4, void (i32) 5; values: 0 main, 1 put, 2 = 5,
        // then user's y 3, z 4, the zext 5, and def's x 6, c 7
        assert_eq!(
            function(&mut m),
            [
                "<12>",
                "  1 [3]",
                "  <11>",
                "    1 [2]",
                "    4 [10]",
                "  11 [2]",
                "  2 [4294967293, 2, 4294967293, 12]", // y = xor x, x: 3 - 6 wraps; the type once, after the first
                "  29 [1, 4294967294, 4294967293, 1]", // z = select c, y, x: y is 4 - 3, x 4 - 6, c 4 - 7 with i1
                "  34 [0, 32768, 5, 4, 4294967295]",   // put(x): 5 - 6, no type
                "  3 [4294967295, 2, 3, 1]",           // zext x to i64
                "  10 []",
                "  2 [4, 4, 0]",                       // x = add 5, 5
                "  28 [1, 5, 33]",                     // c = icmp ne x, 5
                "  11 [1]",
            ]
        );
    }

    #[test]
    fn memory_instructions_and_a_switch() {
        let mut m = Module::compute([1, 1, 1]);
        let lds = m.groupshared(64);
        let mut f = m.entry();
        let (b0, one, two, end) = (f.block(), f.block(), f.block(), f.block());
        f.at(b0);
        let p = f.gep(lds, Val::i32(3));
        let old = f.atomic_rmw(Rmw::UMax, p, Val::i32(9));
        let old2 = f.cmpxchg(p, old, Val::i32(9));
        f.switch(old2, end, &[(0, one), (9, two)]);
        f.at(one);
        // created outside the entry block: it goes to that block's start
        let scratch = f.alloca(16);
        let q = f.gep(scratch, old2);
        f.store(q, old2);
        f.load(q);
        f.br(end);
        f.at(two);
        f.br(end);
        f.at(end);
        f.ret();
        // types: void 0, i1 1, i32 2, void () 3, [64 x i32] 4, [64 x i32] addrspace(3)* 5, i32 addrspace(3)* 6,
        // { i32, i1 } 7, [16 x i32] 8, [16 x i32]* 9, i32* 10; values: 0 lds, 1 main, constants 2 = 1, 3 = 0,
        // 4 = 3, 5 = 9; 6 scratch, 7 p, 8 old, 9 the pair, 10 old2, 11 q, 12 v
        assert_eq!(
            function(&mut m),
            [
                "<12>",
                "  1 [4]",
                "  <11>",
                "    1 [2]",
                "    4 [2]",
                "    2 []",
                "    4 [6]",
                "    4 [18]",
                "  19 [8, 2, 2, 67]",          // alloca [16 x i32], i32 1, align 4
                "  43 [0, 4, 7, 4, 3]",        // p = getelementptr [64 x i32], lds, 0, 3
                "  38 [1, 3, 9, 0, 6, 1]",     // old = atomicrmw umax p, 9 seq_cst
                "  46 [2, 1, 4, 0, 6, 1, 6, 0]", // cmpxchg p, old, 9 seq_cst seq_cst
                "  26 [1, 0]",
                "  12 [2, 1, 3, 3, 1, 5, 2]",  // switch old2, end [0 -> one, 9 -> two]: cases by value id
                "  43 [1, 8, 5, 8, 1]",        // q = getelementptr inbounds [16 x i32], scratch, 0, old2
                "  44 [1, 2, 3, 0]",           // store old2, q, align 4
                "  20 [1, 2, 3, 0]",           // load i32, q, align 4
                "  11 [3]",
                "  11 [3]",
                "  10 []",
            ]
        );
    }

    #[test]
    fn constants_keep_integers_first_and_an_aggregate_after_its_elements() {
        let mut m = Module::compute([1, 1, 1]);
        let bind = m.ty(TypeDef::Struct(Some("dx.types.ResBind".into()), vec![Ty::I32, Ty::I32, Ty::I32, Ty::I8]));
        let take = m.declare("take", Ty::VOID, &[bind, Ty::F32, Ty::F64, Ty::I1, Ty::I32, Ty::F16], FnAttr::None);
        let b = m.aggregate(bind, vec![Val::i32(2), Val::i32(2), Val::i32(0), Val::i8(1)]);
        let mut f = m.entry();
        let b0 = f.block();
        f.at(b0);
        f.call(take, &[b, Val::f32(-0.0), Val::f64(0.0), Val::bool(true), Val::undef(Ty::I32), Val::f16(0x4000)]);
        f.call(take, &[b, Val::f32(0.5), Val::f64(0.0), Val::bool(false), Val::i32(0x8000_0000), Val::f16(0x4000)]);
        f.ret();
        let n = Numbered::new(&mut m, Vec::new()).unwrap();
        let mut out = Listing::default();
        n.write_constants(&m, &n.function_constants, &mut out);
        // types: void 0, i1 1, i8 2, i32 3, half 4, float 5, double 6, void () 7, ResBind 8, the function 9
        assert_eq!(
            out.0,
            [
                "<11>",
                "  1 [3]", // i32: 2, 0, undef, -2^31
                "  4 [4]",
                "  2 []",
                "  3 []",
                "  4 [4294967297]",
                "  1 [2]", // i8 1
                "  4 [2]",
                "  1 [1]", // i1 true (-1), false
                "  4 [3]",
                "  2 []",
                "  1 [5]", // float -0.0 (not NULL: only +0.0 is), 0.5
                "  6 [2147483648]",
                "  6 [1056964608]",
                "  1 [6]", // double 0.0
                "  2 []",
                "  1 [4]", // half 2.0
                "  6 [16384]",
                "  1 [8]", // the ResBind: value ids of 2, 2, 0 and the i8 (2 = main, take; then the constants)
                "  7 [2, 2, 3, 6]",
            ]
        );
    }

    #[test]
    fn a_module_constant_is_not_repeated_in_the_function() {
        let mut m = Module::compute([1, 1, 1]);
        let mut f = m.entry();
        let b0 = f.block();
        f.at(b0);
        let x = f.add(Val::i32(6), Val::i32(1));
        f.sub(x, Val::i32(6));
        f.ret();
        let n = Numbered::new(&mut m, vec![Val::i32(1), Val::i32(6)]).unwrap();
        assert_eq!((n.function_constants.len(), n.id(&m, Val::i32(1)), n.id(&m, Val::i32(6)), n.id(&m, x)), (0, 1, 2, 3));
    }

    #[test]
    fn replaced_values_are_gone_from_the_layout() {
        let mut m = Module::compute([1, 1, 1]);
        let mut f = m.entry();
        let (b0, b1) = (f.block(), f.block());
        f.at(b0);
        let x = f.add(Val::i32(1), Val::i32(1));
        f.br(b1);
        f.at(b1);
        // a trivial phi: used, then replaced by its only value
        let p = f.phi(b1, Ty::I32);
        f.incoming(p, x, b0);
        let y = f.mul(p, p);
        f.replace(p, x);
        let dead = f.mul(y, y);
        f.replace(dead, y);
        f.ret();
        assert_eq!(function(&mut m), ["<12>", "  1 [2]", "  <11>", "    1 [1]", "    4 [2]", "  2 [1, 1, 0]", "  11 [1]", "  2 [1, 1, 2]", "  10 []"]);
    }

    fn refused(build: impl FnOnce(&mut super::super::module::Func)) -> (&'static str, Option<(u32, u32)>) {
        let mut m = Module::compute([1, 1, 1]);
        build(&mut m.entry());
        let r = Numbered::new(&mut m, Vec::new()).err().expect("the function was not refused");
        (r.kind, r.at)
    }

    #[test]
    fn functions_that_are_not_well_formed_are_refused() {
        assert_eq!(refused(|_| {}), ("no-block", None));
        assert_eq!(
            refused(|f| {
                let (a, b) = (f.block(), f.block());
                f.at(a);
                f.br(b);
                f.at(b);
                f.add(Val::i32(1), Val::i32(1));
            }),
            ("no-terminator", Some((1, 1)))
        );
        assert_eq!(
            refused(|f| {
                let (a, b) = (f.block(), f.block());
                f.at(a);
                f.br(b);
                f.at(b);
                f.br(a);
            }),
            ("branch-to-entry", Some((1, 0)))
        );
        assert_eq!(
            refused(|f| {
                let (a, b) = (f.block(), f.block());
                f.at(a);
                f.ret();
                f.at(b);
                f.ret();
            }),
            ("unreachable-block", Some((1, 0)))
        );
        // a phi needs a value for each edge into its block, and no other
        let diamond = |f: &mut super::super::module::Func, values: &[(u32, u32)]| {
            let (a, l, r, end) = (f.block(), f.block(), f.block(), f.block());
            f.at(a);
            let c = f.icmp(Icmp::Eq, Val::i32(1), Val::i32(2));
            f.br_if(c, l, r);
            f.at(l);
            f.br(end);
            f.at(r);
            f.br(end);
            f.at(end);
            let p = f.phi(end, Ty::I32);
            values.iter().for_each(|&(v, from)| f.incoming(p, Val::i32(v), Block(from)));
            f.ret();
        };
        assert_eq!(refused(|f| diamond(f, &[(1, 1)])), ("phi-incoming", Some((3, 0))));
        assert_eq!(refused(|f| diamond(f, &[(1, 1), (2, 2), (3, 0)])), ("phi-incoming", Some((3, 0))));
        assert_eq!(refused(|f| diamond(f, &[(1, 1), (2, 1)])), ("phi-incoming", Some((3, 0))));
        let mut m = Module::compute([1, 1, 1]);
        diamond(&mut m.entry(), &[(1, 2), (2, 1)]);
        assert!(Numbered::new(&mut m, Vec::new()).is_ok());
        // both edges of a conditional branch to one block: a value for each, the same one
        let twice = |f: &mut super::super::module::Func, values: &[u32]| {
            let (a, end) = (f.block(), f.block());
            f.at(a);
            let c = f.icmp(Icmp::Eq, Val::i32(1), Val::i32(2));
            f.br_if(c, end, end);
            let p = f.phi(end, Ty::I32);
            values.iter().for_each(|&v| f.incoming(p, Val::i32(v), a));
            f.at(end);
            f.ret();
        };
        assert_eq!(refused(|f| twice(f, &[1, 2])), ("phi-incoming", Some((1, 0))));
        assert_eq!(refused(|f| twice(f, &[1])), ("phi-incoming", Some((1, 0))));
        let mut m = Module::compute([1, 1, 1]);
        twice(&mut m.entry(), &[1, 1]);
        assert!(Numbered::new(&mut m, Vec::new()).is_ok());
        // what a builder call refused comes back here
        assert_eq!(
            refused(|f| {
                let a = f.block();
                f.at(a);
                f.fadd(Val::i32(1), Val::i32(1));
                f.ret();
            }),
            ("operand-type", Some((0, 0)))
        );
    }
}
