use super::{BinOp, CastOp, HashMap, Module, Op, Ty, Value};
use super::super::Block;

impl Module {
    /// Share repeated integer expressions from scalar load lowering across sibling paths.
    /// The UAV reads, handle creation and wave operations keep their original positions.
    pub(super) fn share_scalar_load_arithmetic(&mut self, order: &[u32], idom: &[Option<usize>]) {
        let mut scalar = vec![false; self.insts.len()];
        for range in &self.scalar_load_ranges {
            scalar[range.clone()].fill(true);
        }
        let mut depth = vec![0; idom.len()];
        for &b in order.iter().skip(1) {
            depth[b as usize] = depth[idom[b as usize].unwrap()] + 1;
        }
        let common = |mut a: usize, mut b: usize| {
            while a != b {
                if depth[a] >= depth[b] {
                    a = idom[a].unwrap();
                } else {
                    b = idom[b].unwrap();
                }
            }
            a
        };

        // Number expressions in definition order. Equal subexpressions have the same
        // representative even before they have been given a common insertion block.
        struct Group {
            first: u32,
            block: usize,
            // First occurrence in the common dominator, when it already contains a copy.
            existing: Option<u32>,
        }
        let mut keys = HashMap::<(Op, Ty), usize>::new();
        let mut groups = Vec::<Group>::new();
        let mut group_of = vec![None; self.insts.len()];
        for &b in order {
            let block = b as usize;
            for &i in &self.blocks[block] {
                let def = &self.insts[i as usize];
                if !scalar[i as usize] || self.int_width(def.ty).is_none() {
                    continue;
                }
                match def.op {
                    Op::Bin(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::And | BinOp::Or | BinOp::Xor | BinOp::Shl | BinOp::LShr, _, _, 0)
                    | Op::Cast(CastOp::Trunc | CastOp::ZExt | CastOp::SExt, ..) | Op::Cmp(..) | Op::Select(..) => {}
                    _ => continue,
                }
                let mut key = def.op.clone();
                key.map_operands(|v| match v {
                    Value::Inst(j) => group_of[j as usize].map_or(v, |g: usize| Value::Inst(groups[g].first)),
                    _ => v,
                });
                let g = *keys.entry((key, def.ty)).or_insert_with(|| {
                    groups.push(Group { first: i, block, existing: Some(i) });
                    groups.len() - 1
                });
                let group = &mut groups[g];
                let target = common(group.block, block);
                if target != group.block {
                    group.existing = None;
                    group.block = target;
                }
                if target == block && group.existing.is_none() {
                    group.existing = Some(i);
                }
                group_of[i as usize] = Some(g);
            }
        }

        let mut insert = vec![Vec::new(); self.blocks.len()];
        for group in &groups {
            if group.existing.is_none() {
                insert[group.block].push(group.first);
                self.def_block[group.first as usize] = Block(group.block as u32);
            }
        }
        for &b in order {
            let block = &mut self.blocks[b as usize];
            block.retain(|&i| group_of[i as usize].is_none_or(|g| groups[g].existing == Some(i)));
            // The operands of equal expressions dominate all their occurrences. Their
            // own common dominators are ancestors of this one or the same block, in
            // which case their groups precede this group in definition order.
            if !insert[b as usize].is_empty() {
                block.splice(block.len() - 1..block.len() - 1, insert[b as usize].drain(..));
            }
            for &i in block.iter() {
                self.insts[i as usize].op.map_operands(|v| match v {
                    Value::Inst(j) => group_of[j as usize].map_or(v, |g| {
                        let group = &groups[g];
                        Value::Inst(group.existing.unwrap_or(group.first))
                    }),
                    _ => v,
                });
            }
        }
    }
}
