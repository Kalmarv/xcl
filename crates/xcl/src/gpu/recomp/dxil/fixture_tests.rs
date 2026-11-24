//! The reference programs of [`fixtures`] built through the builder. For each one:
//! - `dxil.dll`'s validator accepts the container [`Module::finish`] writes (1.9 also holds the
//!   program header to the module's shader model);
//! - DXC assembles, validates and signs the text [`Module::to_text`] prints ([`assemble`]), and
//!   the listing of that container is the listing of the first;
//! - on the GPU the container gives the result the fixture expects, which is computed here and
//!   not taken from DXC's container.
//!
//! The programs follow DXC's layout of the HLSL, but their listings are not compared with DXC's.

use super::assemble::assemble;
use super::check;
use super::dxop::{Component, HeapRes, Shape};
use super::fixtures::{self, Fixture};
use super::module::{flags, Binary, Block, Cast, Fcmp, Func, Icmp, Module, Rmw, Ty, Val};
use super::ops::opcode;
use crate::gpu::recomp::runner;

/// The checks of the module comment, but the run on the GPU.
fn written(fixture: &Fixture, mut built: Module) -> Vec<u8> {
    // every program here has its definitions before their uses
    built.check_dominance = true;
    let ours = built.finish().unwrap_or_else(|refused| panic!("{}: {refused}", fixture.name));
    if let Err(said) = check::validate(&ours) {
        panic!("{}: dxil.dll refuses the program:\n{said}\n{}", fixture.name, check::disassemble(&ours).unwrap_or_default());
    }
    let text = built.to_text().unwrap_or_else(|refused| panic!("{}: {refused}", fixture.name));
    let assembled = assemble(&text).unwrap_or_else(|e| panic!("{}: DXC refuses the text:\n{e}\n{text}", fixture.name));
    if let Err(e) = check::same_listing(&ours, &assembled) {
        panic!("{}: the text assembles to another module than the bitcode: {e}\n--- the text\n{text}", fixture.name);
    }
    ours
}

/// The checks of the module comment.
fn check(fixture: &Fixture, built: Module) {
    if !check::available() {
        return;
    }
    let ours = written(fixture, built);
    if !check::gpu_available() {
        return;
    }
    assert_eq!(fixture.run(&ours), Ok(fixture.expected.clone()), "{}", fixture.name);
}

/// The entry function of a new module in its first block, and handles of `io` (u0) and `res`
/// (u1).
fn program(m: &mut Module) -> (Func<'_>, Val, Val) {
    let (io, res) = (m.root_uav(0, false), m.root_uav(1, false));
    let mut f = m.entry();
    let b = f.block();
    f.at(b);
    let (io, res) = (f.handle(io), f.handle(res));
    (f, io, res)
}

/// `buffer.Store`, `Store2`, `Store3` or `Store4` at byte `offset`.
fn store(f: &mut Func, buffer: Val, offset: Val, values: &[Val]) {
    let mut v = [Val::undef(Ty::I32); 4];
    v[..values.len()].copy_from_slice(values);
    // no element offset, a mask of the components, 4-byte alignment
    f.op(opcode::RawBufferStore, Ty::I32, &[buffer, offset, Val::undef(Ty::I32), v[0], v[1], v[2], v[3], Val::i8((1 << values.len()) - 1), Val::i32(4)]);
}

/// `texture.Load(int3(x, y, mip))` of a 2D texture: TextureLoad(srv, mip, coord0..2, offset0..2).
fn texel(f: &mut Func, overload: Ty, texture: Val, mip: Val, x: Val, y: Val) -> Val {
    let none = Val::undef(Ty::I32);
    f.op(opcode::TextureLoad, overload, &[texture, mip, x, y, none, none, none, none])
}

const FLOAT_TEXTURE: HeapRes = HeapRes::Texture { shape: Shape::Texture2D, component: Component::F32, count: 4, uav: false };

#[test]
fn a_store() {
    let mut m = Module::compute([1, 1, 1]);
    let (mut f, io, res) = program(&mut m);
    let a = f.raw_load(io, Val::i32(0));
    let tripled = f.mul(a, Val::i32(3));
    let value = f.add(tripled, Val::i32(1));
    f.raw_store(res, Val::i32(4), value);
    f.ret();
    check(&fixtures::store(), m);
}

#[test]
fn a_loop_with_phis() {
    let mut m = Module::compute([1, 1, 1]);
    let (mut f, io, res) = program(&mut m);
    // DXC's layout: the loop's preheader, the loop, the block it leaves through, the exit
    let (preheader, body, leave, exit) = (f.block(), f.block(), f.block(), f.block());
    let n = f.raw_load(io, Val::i32(0));
    let start = f.raw_load(io, Val::i32(4));
    let none = f.icmp(Icmp::Eq, n, Val::i32(0));
    f.br_if(none, exit, preheader);
    f.at(preheader);
    f.br(body);

    f.at(body);
    let (sum, x, i) = (f.phi(body, Ty::I32), f.phi(body, Ty::I32), f.phi(body, Ty::I32));
    let sum2 = f.add(x, sum);
    let scaled = f.mul(x, Val::i32(1664525));
    let x2 = f.add(scaled, Val::i32(1013904223));
    let i2 = f.binary_with(Binary::Add, flags::NUW, i, Val::i32(1));
    let done = f.icmp(Icmp::Eq, i2, n);
    f.br_if(done, leave, body);
    f.at(leave);
    f.br(exit);

    f.at(exit);
    let (sum_out, x_out) = (f.phi(exit, Ty::I32), f.phi(exit, Ty::I32));
    store(&mut f, res, Val::i32(0), &[sum_out, x_out]);
    f.ret();
    // the entry block is the first one
    let b0 = Block(0);
    for (phi, first, next, from) in [(sum, Val::i32(0), sum2, preheader), (x, start, x2, preheader), (i, Val::i32(0), i2, preheader)] {
        f.incoming(phi, next, body);
        f.incoming(phi, first, from);
    }
    for (phi, first, last) in [(sum_out, Val::i32(0), sum2), (x_out, start, x2)] {
        f.incoming(phi, first, b0);
        f.incoming(phi, last, leave);
    }
    check(&fixtures::loop_with_phis(), m);
}

#[test]
fn types_other_than_i32() {
    let mut m = Module::compute([1, 1, 1]);
    let (mut f, io, res) = program(&mut m);
    let a = f.raw_load(io, Val::i32(0));
    let b = f.raw_load(io, Val::i32(4));
    let a64 = f.cast(Cast::ZExt, a, Ty::I64);
    let high = f.binary_with(Binary::Shl, flags::NUW, a64, Val::i64(32));
    let b64 = f.cast(Cast::ZExt, b, Ty::I64);
    let wide = f.or(b64, high);
    let wide = f.mul(wide, Val::i64(0x9E37_79B9_7F4A_7C15));
    let bits = f.raw_load(io, Val::i32(8));
    let fl = f.cast(Cast::BitCast, bits, Ty::F32);
    let fa = f.cast(Cast::UiToFp, a, Ty::F32);
    let product = f.binary_with(Binary::FMul, flags::FAST, fl, fa);
    let g = f.binary_with(Binary::FAdd, flags::FAST, product, Val::f32(0.5));
    let less = f.fcmp_with(Fcmp::Olt, flags::FAST, fl, g);
    let whole = f.cast(Cast::FpToUi, g, Ty::I32);
    let pick = f.select(less, whole, b);
    let above = f.icmp(Icmp::Ugt, a, b);
    let (a_b, b_a) = (f.sub(a, b), f.sub(b, a));
    let apart = f.select(above, a_b, b_a);
    let hi = f.raw_load(io, Val::i32(16));
    let lo = f.raw_load(io, Val::i32(12));
    let d = f.op(opcode::MakeDouble, Ty::F64, &[lo, hi]);
    let half = f.binary_with(Binary::FMul, flags::FAST, d, Val::f64(0.5));
    let halves = f.op(opcode::SplitDouble, Ty::F64, &[half]);
    let (lo, hi) = (f.extract(halves, 0), f.extract(halves, 1));
    let bottom = f.cast(Cast::Trunc, wide, Ty::I32);
    let top = f.lshr(wide, Val::i64(32));
    let top = f.cast(Cast::Trunc, top, Ty::I32);
    let g = f.cast(Cast::BitCast, g, Ty::I32);
    store(&mut f, res, Val::i32(0), &[bottom, top, g, pick]);
    store(&mut f, res, Val::i32(16), &[apart, lo, hi]);
    f.ret();
    check(&fixtures::types(), m);
}

#[test]
fn a_texture_of_the_resource_heap() {
    let mut m = Module::compute([1, 1, 1]);
    let (mut f, io, res) = program(&mut m);
    let index = f.raw_load(io, Val::i32(0));
    let texture = f.heap_handle(index, FLOAT_TEXTURE, false);
    let x = f.raw_load(io, Val::i32(4));
    let y = f.raw_load(io, Val::i32(8));
    let loaded = texel(&mut f, Ty::F32, texture, Val::i32(0), x, y);
    let components: Vec<Val> = (0..4).map(|c| f.extract(loaded, c)).collect();
    let bits: Vec<Val> = components.into_iter().map(|c| f.cast(Cast::BitCast, c, Ty::I32)).collect();
    store(&mut f, res, Val::i32(0), &bits);
    f.ret();
    check(&fixtures::heap_texture(), m);
}

#[test]
fn a_sampler_of_the_sampler_heap_and_a_texture_written() {
    let mut m = Module::compute([1, 1, 1]);
    let (mut f, io, res) = program(&mut m);
    let index = f.raw_load(io, Val::i32(0));
    let texture = f.heap_handle(index, FLOAT_TEXTURE, false);
    let index = f.raw_load(io, Val::i32(4));
    let sampler = f.heap_handle(index, HeapRes::Sampler { comparison: false }, false);
    let index = f.raw_load(io, Val::i32(8));
    let target = f.heap_handle(index, HeapRes::Texture { shape: Shape::Texture2D, component: Component::U32, count: 1, uav: true }, false);
    let u = f.raw_load(io, Val::i32(12));
    let u = f.cast(Cast::BitCast, u, Ty::F32);
    let v = f.raw_load(io, Val::i32(16));
    let v = f.cast(Cast::BitCast, v, Ty::F32);
    // SampleLevel(srv, sampler, coord0..3, offset0..2, LOD)
    let (no_coord, none) = (Val::undef(Ty::F32), Val::undef(Ty::I32));
    let sample = f.op(opcode::SampleLevel, Ty::F32, &[texture, sampler, u, v, no_coord, no_coord, Val::i32(0), Val::i32(0), none, Val::f32(0.0)]);
    let (x, y) = (f.extract(sample, 0), f.extract(sample, 1));
    let y = f.cast(Cast::BitCast, y, Ty::I32);
    // TextureStore(srv, coord0..2, value0..3, mask)
    f.op(opcode::TextureStore, Ty::I32, &[target, Val::i32(1), Val::i32(0), none, y, y, y, y, Val::i8(15)]);
    let x = f.cast(Cast::BitCast, x, Ty::I32);
    let back = texel(&mut f, Ty::I32, target, none, Val::i32(1), Val::i32(0));
    let back = f.extract(back, 0);
    store(&mut f, res, Val::i32(0), &[x, back]);
    f.ret();
    check(&fixtures::heap_sampler_and_store(), m);
}

#[test]
fn group_shared_memory_with_atomics_and_barriers() {
    let mut m = Module::compute([64, 1, 1]);
    let lds = m.groupshared(65);
    let (mut f, io, res) = program(&mut m);
    let (init, rest) = (f.block(), f.block());
    let gi = f.op(opcode::FlattenedThreadIdInGroup, Ty::I32, &[]);
    let offset = f.shl(gi, Val::i32(2));
    let v = f.raw_load(io, offset);
    let first = f.icmp(Icmp::Eq, gi, Val::i32(0));
    f.br_if(first, init, rest);
    f.at(init);
    let total = f.gep(lds, Val::i32(64));
    f.store(total, Val::i32(0));
    f.br(rest);

    f.at(rest);
    // a group sync (1) and a fence of the group-shared memory (8)
    f.op(opcode::Barrier, Ty::VOID, &[Val::i32(9)]);
    let mine = f.gep(lds, gi);
    f.store(mine, v);
    let total = f.gep(lds, Val::i32(64));
    f.atomic_rmw(Rmw::Add, total, v);
    let not_v = f.xor(v, Val::i32(u32::MAX));
    let was = f.cmpxchg(mine, v, not_v);
    f.op(opcode::Barrier, Ty::VOID, &[Val::i32(9)]);
    let mirrored = f.sub(Val::i32(63), gi);
    let theirs = f.gep(lds, mirrored);
    let theirs = f.load(theirs);
    let sum = f.load(total);
    let offset = f.mul(gi, Val::i32(12));
    store(&mut f, res, offset, &[theirs, sum, was]);
    f.ret();
    check(&fixtures::lds(), m);
}

#[test]
fn a_local_array() {
    let mut m = Module::compute([1, 1, 1]);
    let (mut f, io, res) = program(&mut m);
    let scratch = f.alloca(16);
    let first = f.raw_load(io, Val::i32(8));
    let index = f.raw_load(io, Val::i32(0));
    let index = f.and(index, Val::i32(15));
    let first_slot = f.gep(scratch, index);
    f.store(first_slot, first);
    let second = f.raw_load(io, Val::i32(12));
    let index = f.raw_load(io, Val::i32(4));
    let index = f.and(index, Val::i32(15));
    let second_slot = f.gep(scratch, index);
    f.store(second_slot, second);
    let second = f.load(second_slot);
    let first = f.load(first_slot);
    store(&mut f, res, Val::i32(0), &[second, first]);
    f.ret();
    check(&fixtures::scratch(), m);
}

#[test]
fn an_atomic_add_in_a_memory_window() {
    let mut m = Module::compute([64, 1, 1]);
    let mem = m.root_uav(2, true);
    let (mut f, io, res) = program(&mut m);
    let mem = f.handle(mem);
    let gi = f.op(opcode::FlattenedThreadIdInGroup, Ty::I32, &[]);
    let offset = f.shl(gi, Val::i32(2));
    let v = f.raw_load(io, offset);
    // AtomicBinOp(handle, operation (0: add), offset0..2, value)
    let none = Val::undef(Ty::I32);
    f.op(opcode::AtomicBinOp, Ty::I32, &[mem, Val::i32(0), Val::i32(8), none, none, v]);
    // a group sync (1) and a fence of the UAVs, of the device (2) and of the group (8)
    f.op(opcode::Barrier, Ty::VOID, &[Val::i32(11)]);
    let sum = f.raw_load(mem, Val::i32(8));
    f.raw_store(res, offset, sum);
    f.ret();
    check(&fixtures::memory_atomic(), m);
}

/// The program of [`fixtures::wave`].
fn wave_program(lanes: u32) -> Module {
    let mut m = Module::compute([64, 1, 1]);
    m.wave_size(lanes);
    let (mut f, io, res) = program(&mut m);
    let gi = f.op(opcode::FlattenedThreadIdInGroup, Ty::I32, &[]);
    let offset = f.shl(gi, Val::i32(2));
    let v = f.raw_load(io, offset);
    let odd = f.and(v, Val::i32(1));
    let odd = f.icmp(Icmp::Ne, odd, Val::i32(0));
    let ballot = f.op(opcode::WaveActiveBallot, Ty::VOID, &[odd]);
    let ballot = [f.extract(ballot, 0), f.extract(ballot, 1)];
    let count = f.op(opcode::WaveGetLaneCount, Ty::VOID, &[]);
    let lane = f.op(opcode::WaveGetLaneIndex, Ty::VOID, &[]);
    // the sum (operation 0) of unsigned values (1)
    let sum = f.op(opcode::WaveActiveOp, Ty::I32, &[v, Val::i8(0), Val::i8(1)]);
    let first = f.op(opcode::WaveReadLaneAt, Ty::I32, &[v, Val::i32(0)]);
    let offset = f.mul(gi, Val::i32(24));
    store(&mut f, res, offset, &[count, lane, sum, first]);
    let offset = f.add(offset, Val::i32(16));
    store(&mut f, res, offset, &ballot);
    f.ret();
    m
}

#[test]
fn wave_operations_in_waves_of_32_and_of_64() {
    if !check::available() {
        return;
    }
    let ran = fixtures::wave_lanes();
    for lanes in [32, 64] {
        let fixture = fixtures::wave(lanes);
        let ours = written(&fixture, wave_program(lanes));
        if ran == Some(lanes) && check::gpu_available() {
            assert_eq!(fixture.run(&ours), Ok(fixture.expected.clone()), "waves of {lanes}");
        }
    }
    if ran.is_none() {
        runner::skip_test("no wave64");
    }
}
