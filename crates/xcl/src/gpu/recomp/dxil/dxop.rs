//! `dx.op` calls, typed from the generated operations table ([`super::ops`]), and the resources
//! of XCL's shader ABI (`pso::abi_root_signature`): root UAVs, static samplers and the directly
//! indexed descriptor heaps, as shader model 6.6 reaches them (`createHandleFromBinding` or
//! `createHandleFromHeap`, each followed by `annotateHandle`).

use super::module::{FnAttr, Func, Kind, Module, Ty, TypeDef, Val};
use super::ops::{self, opcode, slot, stage, Attr};

/// The module is DXIL 1.6, a compute program of shader model 6.6.
const DXIL_MINOR: usize = 6;
const SHADER_MODEL: (u8, u8) = (6, 6);

/// The operations whose texel offsets must be constants before shader model 6.7.
const CONSTANT_OFFSETS: [u32; 7] =
    [opcode::Sample, opcode::SampleBias, opcode::SampleLevel, opcode::SampleGrad, opcode::SampleCmp, opcode::SampleCmpLevelZero, opcode::TextureLoad];

/// An overload type's slot in the table and its suffix in the function's name.
fn overload_slot(ty: Ty) -> Option<(u16, &'static str)> {
    Some(match ty {
        Ty::F16 => (slot::F16, "f16"),
        Ty::F32 => (slot::F32, "f32"),
        Ty::F64 => (slot::F64, "f64"),
        Ty::I1 => (slot::I1, "i1"),
        Ty::I32 => (slot::I32, "i32"),
        Ty::I64 => (slot::I64, "i64"),
        _ => return None,
    })
}

/// A resource bound by the root signature.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Res(u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ResKind {
    /// `RWByteAddressBuffer` (`globallycoherent` or not)
    RawUav { coherent: bool },
    /// `SamplerState`, or `SamplerComparisonState`
    Sampler { comparison: bool },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Resource {
    pub kind: ResKind,
    pub register: u32,
    pub space: u32,
    /// a handle of it was created: DXC keeps no resource a program does not use
    pub used: bool,
}

impl Resource {
    /// DXIL's ResourceClass: 1 UAV, 3 sampler.
    pub fn class(&self) -> u8 {
        match self.kind {
            ResKind::RawUav { .. } => 1,
            ResKind::Sampler { .. } => 3,
        }
    }
    /// The two words of `%dx.types.ResourceProperties`: the resource kind (11 a raw buffer,
    /// 14 a sampler) with bit 12 for a UAV, bit 14 globally coherent and bit 15 a comparison
    /// sampler; the second word is for typed and structured resources.
    fn properties(&self) -> [u32; 2] {
        match self.kind {
            ResKind::RawUav { coherent } => [11 | 1 << 12 | u32::from(coherent) << 14, 0],
            ResKind::Sampler { comparison } => [14 | u32::from(comparison) << 15, 0],
        }
    }
}

/// A texture's dimensions (DXIL's ResourceKind).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    Texture1D = 1,
    Texture2D = 2,
    Texture3D = 4,
    TextureCube = 5,
    Texture1DArray = 6,
    Texture2DArray = 7,
    TextureCubeArray = 9,
}

/// What a texture's components are read as (DXIL's ComponentType).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Component {
    I32 = 4,
    U32 = 5,
    F32 = 9,
}

/// What a descriptor of a directly indexed heap is used as.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HeapRes {
    /// `Texture2D<float4>` and the like; `uav`: `RWTexture2D<..>`
    Texture { shape: Shape, component: Component, count: u8, uav: bool },
    /// from the sampler heap
    Sampler { comparison: bool },
}

impl Module {
    fn named(&mut self, name: &str, elements: Vec<Ty>) -> Ty {
        self.ty(TypeDef::Struct(Some(name.into()), elements))
    }

    /// `%dx.types.Handle`
    pub(super) fn handle_ty(&mut self) -> Ty {
        let pointer = self.pointer(Ty::I8, 0);
        self.named("dx.types.Handle", vec![pointer])
    }

    /// A type of the operations table, for the overload type of the call.
    fn op_ty(&mut self, ty: ops::Ty, overload: Ty) -> Option<Ty> {
        Some(match ty {
            ops::Ty::Void => Ty::VOID,
            ops::Ty::I1 | ops::Ty::I1LikeOverload => Ty::I1,
            ops::Ty::I8 => Ty::I8,
            ops::Ty::I32 | ops::Ty::I32LikeOverload => Ty::I32,
            ops::Ty::I64 => Ty::I64,
            ops::Ty::F16 => Ty::F16,
            ops::Ty::F32 => Ty::F32,
            ops::Ty::F64 => Ty::F64,
            ops::Ty::Overload => overload,
            ops::Ty::ResRet => {
                let name = format!("dx.types.ResRet.{}", overload_slot(overload)?.1);
                self.named(&name, vec![overload, overload, overload, overload, Ty::I32])
            }
            ops::Ty::Handle => self.handle_ty(),
            ops::Ty::Dimensions => self.named("dx.types.Dimensions", vec![Ty::I32; 4]),
            ops::Ty::SamplePos => self.named("dx.types.SamplePos", vec![Ty::F32; 2]),
            ops::Ty::I32C => self.named("dx.types.i32c", vec![Ty::I32, Ty::I1]),
            ops::Ty::TwoI32 => self.named("dx.types.twoi32", vec![Ty::I32; 2]),
            ops::Ty::FourI32 => self.named("dx.types.fouri32", vec![Ty::I32; 4]),
            ops::Ty::OverloadStruct4 if overload == Ty::I32 => self.named("dx.types.fouri32", vec![Ty::I32; 4]),
            ops::Ty::SplitDouble => self.named("dx.types.splitdouble", vec![Ty::I32; 2]),
            ops::Ty::ResourceProperties => self.named("dx.types.ResourceProperties", vec![Ty::I32; 2]),
            ops::Ty::ResBind => self.named("dx.types.ResBind", vec![Ty::I32, Ty::I32, Ty::I32, Ty::I8]),
            // 16-bit integers, vectors, extended overloads, node and ray tracing types
            _ => return None,
        })
    }

    /// The function of operation `opcode` for `overload`, declared on first use; or why not.
    fn op_function(&mut self, opcode: u32, overload: Ty) -> Result<u32, (&'static str, String)> {
        if let Some(&f) = self.op_functions.get(&(opcode, overload)) {
            return Ok(f);
        }
        let allowed = |op: &&ops::Op| opcode < ops::CORE_OPS_IN_DXIL_1[DXIL_MINOR] && op.class != "Reserved" && op.min_sm <= SHADER_MODEL && op.stages & stage::COMPUTE != 0;
        let Some(op) = ops::op(opcode).filter(allowed) else {
            let name = ops::op(opcode).map_or("no such operation", |op| op.name);
            return Err(("op-not-allowed", format!("operation {opcode:#x} ({name}) in a compute program of shader model 6.6")));
        };
        let suffix = match (op.overloads, overload_slot(overload)) {
            ([], _) if overload == Ty::VOID => String::new(),
            ([(scalars, _)], Some((slot, name))) if scalars & slot != 0 => format!(".{name}"),
            _ => return Err(("op-overload", format!("{} for {}", op.name, self.type_name(overload)))),
        };
        // 64-bit atomics on a resource need a flag of the resource in the container
        if overload == Ty::I64 && [opcode::AtomicBinOp, opcode::AtomicCompareExchange].contains(&opcode) {
            return Err(("op-overload", format!("{} for i64", op.name)));
        }
        let attr = match op.attr {
            Attr::None => FnAttr::None,
            Attr::ReadNone => FnAttr::ReadNone,
            Attr::ReadOnly => FnAttr::ReadOnly,
            Attr::NoDuplicate => FnAttr::NoDuplicate,
            Attr::ArgMemOnly | Attr::NoReturn => return Err(("op-type", format!("{}: an attribute the writer does not have", op.name))),
        };
        let mut params = vec![Ty::I32];
        for &(_, ty) in op.params.iter() {
            params.push(self.op_ty(ty, overload).ok_or(("op-type", format!("{}: a type the writer does not have", op.name)))?);
        }
        let ret = self.op_ty(op.ret, overload).ok_or(("op-type", format!("{}: a type the writer does not have", op.name)))?;
        let f = self.declare(&format!("dx.op.{}{suffix}", op.function), ret, &params, attr);
        self.op_functions.insert((opcode, overload), f);
        Ok(f)
    }

    fn resource(&mut self, kind: ResKind, register: u32, space: u32) -> Res {
        let new = Resource { kind, register, space, used: false };
        if self.resources.iter().any(|r| (r.class(), r.register, r.space) == (new.class(), register, space)) {
            self.refuse("resource-twice", None, format!("register {register} of space {space} bound twice"));
        }
        self.resources.push(new);
        Res(self.resources.len() as u32 - 1)
    }

    /// The `RWByteAddressBuffer` at `u<register>` (space 0): u0 `io`, u1 `res`, u2 to u5 the
    /// `mem` windows, which compute programs see `globallycoherent`.
    pub fn root_uav(&mut self, register: u32, coherent: bool) -> Res {
        self.resource(ResKind::RawUav { coherent }, register, 0)
    }

    /// The static sampler at `s<register>`, space 1.
    pub fn static_sampler(&mut self, register: u32, comparison: bool) -> Res {
        self.resource(ResKind::Sampler { comparison }, register, 1)
    }
}

impl Func<'_> {
    /// A call of the DXIL operation `opcode` ([`opcode`]) with `args` (after the opcode, which the
    /// writer passes itself). `overload` is the type the operation is used for where it has
    /// overloads (`Ty::I32` in `dx.op.unaryBits.i32`), else `Ty::VOID`. The result of some
    /// operations is a struct: see [`Func::extract`].
    pub fn op(&mut self, opcode: u32, overload: Ty, args: &[Val]) -> Val {
        let function = match self.m.op_function(opcode, overload) {
            Ok(f) => f,
            Err((kind, text)) => {
                self.refuse(kind, text);
                return Val::bad(Ty::I32);
            }
        };
        if CONSTANT_OFFSETS.contains(&opcode) {
            let params = ops::op(opcode).map_or(&[][..], |op| op.params);
            if params.iter().zip(args).any(|(&(name, _), a)| name.starts_with("offset") && !a.is_constant()) {
                self.refuse("op-not-allowed", format!("operation {opcode:#x} with a texel offset that is not constant (shader model 6.7)"));
                return Val::bad(Ty::I32);
            }
        }
        self.call_with(function, &[Val::i32(opcode)], args)
    }

    fn annotate(&mut self, handle: Val, [basic, extended]: [u32; 2]) -> Val {
        let ty = self.m.op_ty(ops::Ty::ResourceProperties, Ty::VOID).unwrap();
        let properties = self.m.aggregate(ty, vec![Val::i32(basic), Val::i32(extended)]);
        self.op(opcode::AnnotateHandle, Ty::VOID, &[handle, properties])
    }

    /// A handle of a bound resource: `createHandleFromBinding` and `annotateHandle`, here. It
    /// can be used wherever this place dominates.
    pub fn handle(&mut self, res: Res) -> Val {
        let bound = self.bind(res);
        self.annotate_bound(bound, res)
    }

    /// The `createHandleFromBinding` of [`Func::handle`] alone. (DXC puts it in the entry block
    /// and the `annotateHandle` where the resource is used.)
    pub(super) fn bind(&mut self, res: Res) -> Val {
        let r = &mut self.m.resources[res.0 as usize];
        r.used = true;
        let r = *r;
        let ty = self.m.op_ty(ops::Ty::ResBind, Ty::VOID).unwrap();
        // one register: the range's lower and upper bound, then the space and the class
        let bind = self.m.aggregate(ty, vec![Val::i32(r.register), Val::i32(r.register), Val::i32(r.space), Val::i8(r.class())]);
        self.op(opcode::CreateHandleFromBinding, Ty::VOID, &[bind, Val::i32(r.register), Val::bool(false)])
    }

    /// The `annotateHandle` of a [`Func::bind`].
    pub(super) fn annotate_bound(&mut self, bound: Val, res: Res) -> Val {
        let properties = self.m.resources[res.0 as usize].properties();
        self.annotate(bound, properties)
    }

    /// A handle of descriptor `index` (an i32) of the CBV/SRV/UAV heap, or of the sampler heap
    /// for a sampler: `ResourceDescriptorHeap[index]` used as `kind`. `non_uniform` as
    /// `NonUniformResourceIndex`: the index may differ between the lanes of a wave.
    pub fn heap_handle(&mut self, index: Val, kind: HeapRes, non_uniform: bool) -> Val {
        let (sampler, properties) = match kind {
            HeapRes::Texture { shape, component, count, uav } => {
                if !(1..=4).contains(&count) {
                    self.refuse("resource-type", format!("a texture of {count} components"));
                }
                (false, [shape as u32 | u32::from(uav) << 12, component as u32 | u32::from(count) << 8])
            }
            HeapRes::Sampler { comparison } => (true, [14 | u32::from(comparison) << 15, 0]),
        };
        let handle = self.op(opcode::CreateHandleFromHeap, Ty::VOID, &[index, Val::bool(sampler), Val::bool(non_uniform)]);
        self.annotate(handle, properties)
    }

    /// The dword at byte `offset` (an i32) of a raw buffer: `buffer.Load(offset)`.
    pub fn raw_load(&mut self, handle: Val, offset: Val) -> Val {
        // no element offset, the first component only, 4-byte alignment
        let loaded = self.op(opcode::RawBufferLoad, Ty::I32, &[handle, offset, Val::undef(Ty::I32), Val::i8(1), Val::i32(4)]);
        self.extract(loaded, 0)
    }

    /// `buffer.Store(offset, value)` of an i32.
    pub fn raw_store(&mut self, handle: Val, offset: Val, value: Val) {
        let none = Val::undef(Ty::I32);
        self.op(opcode::RawBufferStore, Ty::I32, &[handle, offset, none, value, none, none, none, Val::i8(1), Val::i32(4)]);
    }
}

impl Module {
    /// The opcode of a `dx.op` call.
    pub(super) fn called_op(&self, function: u32, first_operand: u32) -> Option<u32> {
        if !self.functions[function as usize].name.starts_with("dx.op.") {
            return None;
        }
        match self.body.operands[first_operand as usize].kind {
            Kind::Int(opcode) => Some(opcode as u32),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::module::Inst;
    use super::*;

    fn refusal<R>(build: impl FnOnce(&mut Func) -> R) -> &'static str {
        let mut m = Module::compute([1, 1, 1]);
        let mut f = m.entry();
        let b = f.block();
        f.at(b);
        build(&mut f);
        m.refused.map_or("", |r| r.kind)
    }

    #[test]
    fn functions_are_named_and_typed_from_the_table() {
        let mut m = Module::compute([1, 1, 1]);
        let io = m.root_uav(0, false);
        let mut f = m.entry();
        let b = f.block();
        f.at(b);
        let h = f.handle(io);
        let v = f.raw_load(h, Val::i32(4));
        let n = f.op(opcode::Countbits, Ty::I32, &[v]);
        let lane = f.op(opcode::WaveGetLaneIndex, Ty::VOID, &[]);
        let d = f.op(opcode::MakeDouble, Ty::F64, &[n, lane]);
        let halves = f.op(opcode::SplitDouble, Ty::F64, &[d]);
        let lo = f.extract(halves, 0);
        f.op(opcode::Barrier, Ty::VOID, &[Val::i32(9)]);
        f.raw_store(h, Val::i32(0), lo);
        // the same operation again declares nothing
        f.op(opcode::Countbits, Ty::I32, &[lo]);
        f.ret();
        assert_eq!(m.refused, None);
        let declared: Vec<(String, String, FnAttr)> = m.functions[1..].iter().map(|f| (f.name.clone(), m.type_name(f.ty), f.attr.unwrap())).collect();
        let expected = [
            ("dx.op.createHandleFromBinding", "%dx.types.Handle (i32, %dx.types.ResBind, i32, i1)", FnAttr::ReadNone),
            ("dx.op.annotateHandle", "%dx.types.Handle (i32, %dx.types.Handle, %dx.types.ResourceProperties)", FnAttr::ReadNone),
            ("dx.op.rawBufferLoad.i32", "%dx.types.ResRet.i32 (i32, %dx.types.Handle, i32, i32, i8, i32)", FnAttr::ReadOnly),
            ("dx.op.unaryBits.i32", "i32 (i32, i32)", FnAttr::ReadNone),
            ("dx.op.waveGetLaneIndex", "i32 (i32)", FnAttr::ReadOnly),
            ("dx.op.makeDouble.f64", "double (i32, i32, i32)", FnAttr::ReadNone),
            ("dx.op.splitDouble.f64", "%dx.types.splitdouble (i32, double)", FnAttr::ReadNone),
            ("dx.op.barrier", "void (i32, i32)", FnAttr::NoDuplicate),
            ("dx.op.rawBufferStore.i32", "void (i32, %dx.types.Handle, i32, i32, i32, i32, i32, i32, i8, i32)", FnAttr::None),
        ];
        assert_eq!(declared, expected.map(|(n, t, a)| (n.to_string(), t.to_string(), a)));
        // the opcode is the first argument; the resource's binding and properties are constants
        let Inst::Call { function, args: (start, n) } = m.body.insts[0] else { panic!() };
        let bind = m.body.operands[start as usize + 1];
        assert_eq!((m.called_op(function, start), n, bind.is_constant()), (Some(opcode::CreateHandleFromBinding), 4, true));
        let Kind::Aggregate(a) = bind.kind else { panic!() };
        assert_eq!(m.aggregates[a as usize].1, [Val::i32(0), Val::i32(0), Val::i32(0), Val::i8(1)]);
        let Inst::Call { args: (start, _), .. } = m.body.insts[1] else { panic!() };
        let Kind::Aggregate(a) = m.body.operands[start as usize + 2].kind else { panic!() };
        assert_eq!(m.aggregates[a as usize].1, [Val::i32(4107), Val::i32(0)]);
    }

    #[test]
    fn resources_have_the_properties_dxc_gives_them() {
        let properties = |build: &dyn Fn(&mut Module) -> Res| {
            let mut m = Module::compute([1, 1, 1]);
            let res = build(&mut m);
            let mut f = m.entry();
            let b = f.block();
            f.at(b);
            f.handle(res);
            let Kind::Aggregate(a) = m.body.operands.last().unwrap().kind else { panic!() };
            assert_eq!(m.refused, None);
            (m.aggregates[a as usize].1[0], m.aggregates[a as usize].1[1])
        };
        assert_eq!(properties(&|m| m.root_uav(2, true)), (Val::i32(20491), Val::i32(0)));
        assert_eq!(properties(&|m| m.static_sampler(0, false)), (Val::i32(14), Val::i32(0)));
        assert_eq!(properties(&|m| m.static_sampler(1, true)), (Val::i32(32782), Val::i32(0)));
        let heap = |kind: HeapRes| {
            let mut m = Module::compute([1, 1, 1]);
            let mut f = m.entry();
            let b = f.block();
            f.at(b);
            f.heap_handle(Val::i32(7), kind, true);
            let Kind::Aggregate(a) = m.body.operands.last().unwrap().kind else { panic!() };
            assert_eq!(m.refused, None);
            // createHandleFromHeap(i32 218, index, samplerHeap, nonUniformIndex)
            (m.body.operands[2] == Val::bool(true), m.aggregates[a as usize].1.clone())
        };
        let texture = |shape, component, count, uav| HeapRes::Texture { shape, component, count, uav };
        assert_eq!(heap(texture(Shape::Texture2D, Component::F32, 4, false)), (false, vec![Val::i32(2), Val::i32(1033)]));
        assert_eq!(heap(texture(Shape::Texture3D, Component::U32, 1, true)), (false, vec![Val::i32(4100), Val::i32(261)]));
        assert_eq!(heap(HeapRes::Sampler { comparison: false }), (true, vec![Val::i32(14), Val::i32(0)]));
    }

    #[test]
    fn operations_outside_the_program_are_refused() {
        assert_eq!(refusal(|f| f.op(opcode::Countbits, Ty::I32, &[Val::i32(1)])), "");
        // no such overload, an overload for an operation without one, and none for one with
        assert_eq!(refusal(|f| f.op(opcode::Countbits, Ty::F32, &[Val::f32(1.0)])), "op-overload");
        assert_eq!(refusal(|f| f.op(opcode::WaveGetLaneIndex, Ty::I32, &[])), "op-overload");
        assert_eq!(refusal(|f| f.op(opcode::Countbits, Ty::VOID, &[Val::i32(1)])), "op-overload");
        assert_eq!(refusal(|f| f.op(opcode::Countbits, Ty::I32, &[Val::i64(1)])), "call-arguments");
        assert_eq!(refusal(|f| f.op(opcode::Countbits, Ty::I32, &[])), "call-arguments");
        // a pixel shader's operation, one of DXIL 1.7, a reserved one and one that does not exist
        assert_eq!(refusal(|f| f.op(opcode::Discard, Ty::VOID, &[Val::bool(true)])), "op-not-allowed");
        assert_eq!(refusal(|f| f.op(ops::CORE_OPS_IN_DXIL_1[6], Ty::VOID, &[])), "op-not-allowed");
        assert_eq!(refusal(|f| f.op(0xe2, Ty::VOID, &[])), "op-not-allowed");
        assert_eq!(refusal(|f| f.op(0x8000_0000, Ty::VOID, &[])), "op-not-allowed");
        assert_eq!(refusal(|f| f.op(9999, Ty::VOID, &[])), "op-not-allowed");
        assert_eq!(refusal(|f| {
            let h = f.heap_handle(Val::i32(0), HeapRes::Texture { shape: Shape::Texture2D, component: Component::U32, count: 1, uav: true }, false);
            let none = Val::undef(Ty::I32);
            f.op(opcode::AtomicBinOp, Ty::I64, &[h, Val::i32(0), Val::i32(0), Val::i32(0), none, Val::i64(1)])
        }), "op-overload");
        let load = |f: &mut Func, offset: Val| {
            let h = f.heap_handle(Val::i32(0), HeapRes::Texture { shape: Shape::Texture2D, component: Component::F32, count: 4, uav: false }, false);
            let none = Val::undef(Ty::I32);
            f.op(opcode::TextureLoad, Ty::F32, &[h, Val::i32(0), Val::i32(1), Val::i32(2), none, offset, offset, none])
        };
        assert_eq!(refusal(|f| load(f, Val::i32(1))), "");
        assert_eq!(refusal(|f| {
            let x = f.add(Val::i32(1), Val::i32(1));
            load(f, x)
        }), "op-not-allowed");
        assert_eq!(refusal(|f| f.heap_handle(Val::i32(0), HeapRes::Texture { shape: Shape::Texture2D, component: Component::F32, count: 5, uav: false }, false)), "resource-type");
        let mut m = Module::compute([1, 1, 1]);
        m.root_uav(0, false);
        m.static_sampler(0, false);
        assert_eq!(m.refused, None);
        m.root_uav(0, true);
        assert_eq!(m.refused.unwrap().kind, "resource-twice");
    }
}
