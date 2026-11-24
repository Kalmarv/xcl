//! An optional check of a built function, for debugging a lowering: every use of an instruction's
//! value is dominated by its definition. `finish` does not look at this by default (`dxil.dll`
//! reports it, and so does a driver, each in its own words); with `XCL_RECOMP_DXIL_VERIFY=1` it
//! does, and refuses with `not-dominated` and the place of the use.
//!
//! The dominators are found as in Cooper, Harvey and Kennedy, "A Simple, Fast Dominance
//! Algorithm" (2001).

use std::sync::OnceLock;

use super::module::{Inst, Kind, Module, Refused};

/// The switch that turns the check on for every module.
pub(super) const SWITCH: &str = "XCL_RECOMP_DXIL_VERIFY";

pub(super) fn wanted() -> bool {
    static WANTED: OnceLock<bool> = OnceLock::new();
    *WANTED.get_or_init(|| std::env::var(SWITCH).is_ok_and(|v| v.trim() == "1"))
}

/// Each block's immediate dominator (the entry's is itself). Every block can be reached.
fn dominators(preds: &[Vec<u32>]) -> Vec<u32> {
    let mut succs = vec![Vec::new(); preds.len()];
    for (b, from) in preds.iter().enumerate() {
        from.iter().for_each(|&p| succs[p as usize].push(b as u32));
    }
    // the blocks in postorder, and each block's place in it
    let (mut postorder, mut seen, mut stack) = (Vec::with_capacity(preds.len()), vec![false; preds.len()], vec![(0u32, 0usize)]);
    seen[0] = true;
    while let Some(&(b, next)) = stack.last() {
        match succs[b as usize].get(next) {
            Some(&to) => {
                stack.last_mut().unwrap().1 += 1;
                if !std::mem::replace(&mut seen[to as usize], true) {
                    stack.push((to, 0));
                }
            }
            None => {
                postorder.push(b);
                stack.pop();
            }
        }
    }
    let mut place = vec![0; preds.len()];
    postorder.iter().enumerate().for_each(|(k, &b)| place[b as usize] = k);

    const UNKNOWN: u32 = u32::MAX;
    let mut idom = vec![UNKNOWN; preds.len()];
    idom[0] = 0;
    let mut changed = true;
    while changed {
        changed = false;
        for &b in postorder.iter().rev().skip(1) {
            let mut new = UNKNOWN;
            for &p in preds[b as usize].iter().filter(|&&p| idom[p as usize] != UNKNOWN) {
                // the nearest block that dominates both
                let (mut x, mut y) = (p, if new == UNKNOWN { p } else { new });
                while x != y {
                    while place[x as usize] < place[y as usize] {
                        x = idom[x as usize];
                    }
                    while place[y as usize] < place[x as usize] {
                        y = idom[y as usize];
                    }
                }
                new = x;
            }
            changed |= std::mem::replace(&mut idom[b as usize], new) != new;
        }
    }
    idom
}

/// Checks the function of `m`, laid out as `order` (each block's instructions) with the edges
/// `preds` (each block's predecessors).
pub(super) fn check(m: &Module, order: &[Vec<u32>], preds: &[Vec<u32>]) -> Result<(), Refused> {
    let idom = dominators(preds);
    let dominates = |a: u32, mut b: u32| loop {
        if a == b {
            break true;
        }
        if b == 0 {
            break false;
        }
        b = idom[b as usize];
    };
    // where each instruction is: its block, and its index there
    let mut place = vec![(u32::MAX, 0); m.body.insts.len()];
    for (b, list) in order.iter().enumerate() {
        list.iter().enumerate().for_each(|(k, &i)| place[i as usize] = (b as u32, k as u32));
    }
    for (b, list) in order.iter().enumerate() {
        for (k, &i) in list.iter().enumerate() {
            let mut bad: Option<(u32, String)> = None;
            if let Inst::Phi(incoming) = m.body.insts[i as usize] {
                // a phi uses each value at the end of the block it comes from
                for &(v, from) in &m.body.incoming[incoming as usize] {
                    let Kind::Inst(d) = v.kind else { continue };
                    if bad.is_none() && !dominates(place[d as usize].0, from.0) {
                        bad = Some((d, format!(" for the edge from block {}", from.0)));
                    }
                }
            } else {
                m.each_operand(i, |v| {
                    let Kind::Inst(d) = v.kind else { return };
                    let (block, index) = place[d as usize];
                    let before = if block == b as u32 { index < k as u32 } else { dominates(block, b as u32) };
                    if bad.is_none() && !before {
                        bad = Some((d, String::new()));
                    }
                });
            }
            if let Some((d, edge)) = bad {
                let (block, index) = place[d as usize];
                let text = format!("the operand defined in block {block} (instruction {index}) does not dominate its use{edge}");
                return Err(Refused { kind: "not-dominated", at: Some((b as u32, k as u32)), text });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::module::{Block, Func, Icmp, Module, Ty, Val};
    use super::super::number::Numbered;

    /// Blocks `then` and `join` after the entry, which branches on a value: to `then`, which
    /// goes on to `join`, or to `join`.
    fn diamond(m: &mut Module) -> (Func<'_>, Val, Block, Block, Block) {
        let mut f = m.entry();
        let (b0, then, join) = (f.block(), f.block(), f.block());
        f.at(b0);
        let x = f.add(Val::i32(1), Val::i32(2));
        let c = f.icmp(Icmp::Eq, x, Val::i32(3));
        f.br_if(c, then, join);
        (f, x, b0, then, join)
    }

    fn checked(mut m: Module) -> Option<(&'static str, Option<(u32, u32)>)> {
        m.check_dominance = true;
        Numbered::new(&mut m, Vec::new()).err().map(|r| (r.kind, r.at))
    }

    #[test]
    fn a_use_its_definition_does_not_dominate_is_refused() {
        // a value of one arm used after the arms join
        fn one_arm() -> Module {
            let mut m = Module::compute([1, 1, 1]);
            let (mut f, x, _, then, join) = diamond(&mut m);
            f.at(then);
            let y = f.add(x, x);
            f.br(join);
            f.at(join);
            f.add(y, x);
            f.ret();
            m
        }
        assert_eq!(checked(one_arm()), Some(("not-dominated", Some((2, 0)))));
        // without the switch nothing looks
        let mut unchecked = one_arm();
        unchecked.check_dominance = false;
        assert!(Numbered::new(&mut unchecked, Vec::new()).is_ok());

        // a value used before it is made, in its own block
        let mut m = Module::compute([1, 1, 1]);
        let (mut f, x, _, then, join) = diamond(&mut m);
        f.at(then);
        let early = f.phi(then, Ty::I32);
        f.add(early, x);
        let late = f.add(x, x);
        f.br(join);
        f.at(join);
        f.ret();
        f.replace(early, late);
        assert_eq!(checked(m), Some(("not-dominated", Some((1, 0)))));

        // a phi's value that does not reach the end of the block it comes from
        let mut m = Module::compute([1, 1, 1]);
        let (mut f, x, b0, then, join) = diamond(&mut m);
        f.at(then);
        let y = f.add(x, x);
        f.br(join);
        f.at(join);
        let p = f.phi(join, Ty::I32);
        f.ret();
        f.incoming(p, y, b0);
        f.incoming(p, x, then);
        assert_eq!(checked(m), Some(("not-dominated", Some((2, 0)))));
    }

    #[test]
    fn values_that_dominate_their_uses_pass() {
        // through a phi, around a loop, and from the entry into every block
        let mut m = Module::compute([1, 1, 1]);
        let (mut f, x, b0, then, join) = diamond(&mut m);
        f.at(then);
        let i = f.phi(then, Ty::I32);
        let next = f.add(i, x);
        let again = f.icmp(Icmp::Ult, next, Val::i32(9));
        f.br_if(again, then, join);
        f.incoming(i, x, b0);
        f.incoming(i, next, then);
        f.at(join);
        let p = f.phi(join, Ty::I32);
        f.incoming(p, x, b0);
        f.incoming(p, next, then);
        f.add(p, x);
        f.ret();
        assert_eq!(checked(m), None);
    }
}
