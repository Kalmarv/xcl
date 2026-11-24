//! What `finish` adds to a built module: its shader flags, derived from what it uses, the
//! `dx.*` metadata (DXIL 1.6, `cs` 6.6, validator version 1.6), the bitcode, and the container.
//!
//! The flags follow DXC v1.9.2609 `ShaderFlags::CollectShaderFlags` (lib/DXIL/DxilShaderFlags.cpp)
//! and `DxilModule::CollectShaderFlagsForModule`, for a validator of version 1.6 (University of
//! Illinois/NCSA licence): a validator compares them with the ones it computes itself.

use super::bitstream::BitWriter;
use super::container::{compute_container, Binding, ComputeProgram};
use super::dxop::ResKind;
use super::module::{Binary, Cast, Inst, Kind, Module, Refused, Ty, TypeDef, Val};
use super::number::Sink;
use super::ops::opcode;
use super::write::{write_module, Metadata};

/// Bits of the shader flags (DXC's `ShaderFlags`), the i64 of the entry point's properties.
mod flag {
    pub const DOUBLES: u64 = 1 << 2;
    pub const RAW_AND_STRUCTURED_BUFFERS: u64 = 1 << 4;
    pub const LOW_PRECISION: u64 = 1 << 5;
    /// double division and conversions between doubles and integers
    pub const DOUBLE_EXTENSIONS: u64 = 1 << 6;
    pub const MSAD: u64 = 1 << 7;
    pub const TILED_RESOURCES: u64 = 1 << 12;
    /// a typed UAV of more than one component is read
    pub const UAV_LOAD_ADDITIONAL_FORMATS: u64 = 1 << 13;
    pub const UAVS_64: u64 = 1 << 15;
    pub const WAVE_OPS: u64 = 1 << 19;
    pub const INT64_OPS: u64 = 1 << 20;
    pub const NATIVE_LOW_PRECISION: u64 = 1 << 23;
    pub const RESOURCE_HEAP_INDEXING: u64 = 1 << 30;
    pub const SAMPLER_HEAP_INDEXING: u64 = 1 << 31;
}

/// DXC's `OP::IsDxilOpWave`: WaveIsFirstLane (110) to QuadOp (123), WaveAllBitCount (135),
/// WavePrefixBitCount (136), WaveMatch (165) and WaveMultiPrefixOp (166).
fn is_wave_op(op: u32) -> bool {
    matches!(op, 110..=123 | 135 | 136 | 165 | 166)
}

impl Sink for BitWriter {
    fn enter(&mut self, block: u32, abbrev_width: u32) {
        self.enter_block(block, abbrev_width);
    }
    fn record(&mut self, code: u32, ops: &[u64]) {
        self.unabbrev_record(code, ops);
    }
    fn exit(&mut self) {
        self.exit_block();
    }
}

impl Module {
    /// `[WaveSize(lanes)]`: the program runs in waves of exactly `lanes` lanes.
    pub fn wave_size(&mut self, lanes: u32) {
        if !lanes.is_power_of_two() || !(4..=128).contains(&lanes) {
            self.refuse("wave-size", None, format!("a wave size of {lanes} (a power of two from 4 to 128)"));
        }
        self.wave_size = Some(lanes);
    }

    /// Calls `f` with each operand of instruction `i`.
    pub(super) fn each_operand(&self, i: u32, mut f: impl FnMut(Val)) {
        match self.body.insts[i as usize] {
            Inst::Binary { a, b, .. } | Inst::Cmp { a, b, .. } => [a, b].into_iter().for_each(f),
            Inst::Select { cond, a, b } => [cond, a, b].into_iter().for_each(f),
            Inst::Cast { value, .. } | Inst::Switch { value, .. } => f(value),
            Inst::ExtractValue { aggregate, .. } => f(aggregate),
            Inst::Phi(list) => self.body.incoming[list as usize].iter().for_each(|&(v, _)| f(v)),
            Inst::Call { args: (start, n), .. } => self.body.operands[start as usize..(start + n) as usize].iter().for_each(|&v| f(v)),
            Inst::BrIf { cond, .. } => f(cond),
            Inst::Gep { base, index, .. } => [base, index].into_iter().for_each(f),
            Inst::Load { pointer } => f(pointer),
            Inst::Store { pointer, value } | Inst::AtomicRmw { pointer, value, .. } => [pointer, value].into_iter().for_each(f),
            Inst::CmpXchg { pointer, expected, new } => [pointer, expected, new].into_iter().for_each(f),
            Inst::Br(_) | Inst::Ret | Inst::Alloca { .. } | Inst::Removed => {}
        }
    }

    /// The `%dx.types.ResourceProperties` a handle was annotated with.
    fn handle_properties(&self, handle: Val) -> Option<(u32, u32)> {
        let Kind::Inst(i) = self.resolve(handle).kind else { return None };
        let Inst::Call { function, args: (start, 3) } = self.body.insts[i as usize] else { return None };
        if self.called_op(function, start) != Some(opcode::AnnotateHandle) {
            return None;
        }
        let Kind::Aggregate(a) = self.body.operands[start as usize + 2].kind else { return None };
        match &self.aggregates[a as usize].1[..] {
            [Val { kind: Kind::Int(basic), .. }, Val { kind: Kind::Int(extended), .. }] => Some((*basic as u32, *extended as u32)),
            _ => None,
        }
    }

    /// The shader flags, and whether the module has native 16-bit types (it uses `half`).
    pub(super) fn shader_flags(&self) -> (u64, bool) {
        let mut flags = 0;
        for (i, inst) in self.body.insts.iter().enumerate().filter(|(_, inst)| **inst != Inst::Removed) {
            // what the instruction's type or an operand's is
            let (mut double, mut int64, mut half) = (false, false, false);
            let mut ty = |t: Ty| {
                double |= t == Ty::F64;
                int64 |= t == Ty::I64;
                half |= t == Ty::F16;
            };
            ty(self.body.tys[i]);
            self.each_operand(i as u32, |v| ty(v.ty));
            if double {
                flags |= flag::DOUBLES;
                let conversion = |op: &Cast| matches!(op, Cast::UiToFp | Cast::SiToFp | Cast::FpToUi | Cast::FpToSi);
                if matches!(inst, Inst::Binary { op: Binary::FDiv, .. }) || matches!(inst, Inst::Cast { op, .. } if conversion(op)) {
                    flags |= flag::DOUBLE_EXTENSIONS;
                }
            }
            if int64 {
                flags |= flag::INT64_OPS;
            }
            if half {
                flags |= flag::LOW_PRECISION | flag::NATIVE_LOW_PRECISION;
            }
            let Inst::Call { function, args: (start, _) } = *inst else { continue };
            let Some(op) = self.called_op(function, start) else { continue };
            let argument = |k: u32| self.body.operands[(start + 1 + k) as usize];
            if is_wave_op(op) {
                flags |= flag::WAVE_OPS;
            }
            match op {
                opcode::CheckAccessFullyMapped => flags |= flag::TILED_RESOURCES,
                opcode::Msad => flags |= flag::MSAD,
                opcode::Fma if double => flags |= flag::DOUBLE_EXTENSIONS,
                opcode::TextureLoad | opcode::BufferLoad => {
                    // a UAV (bit 12) of a typed kind (the textures and the typed buffer, 1 to 10)
                    // with more than one component
                    if let Some((basic, extended)) = self.handle_properties(argument(0)) {
                        if basic & 1 << 12 != 0 && (1..=10).contains(&(basic & 0xFF)) && (extended >> 8) & 0xFF > 1 {
                            flags |= flag::UAV_LOAD_ADDITIONAL_FORMATS;
                        }
                    }
                }
                opcode::CreateHandleFromHeap => {
                    flags |= if argument(1) == Val::bool(true) { flag::SAMPLER_HEAP_INDEXING } else { flag::RESOURCE_HEAP_INDEXING };
                }
                _ => {}
            }
        }
        let uavs = self.resources.iter().filter(|r| r.used && matches!(r.kind, ResKind::RawUav { .. })).count();
        if uavs > 0 {
            flags |= flag::RAW_AND_STRUCTURED_BUFFERS;
        }
        if uavs > 8 {
            flags |= flag::UAVS_64;
        }
        (flags, flags & flag::NATIVE_LOW_PRECISION != 0)
    }

    /// The module's metadata, as DXC writes it for a compute program of shader model 6.6 (without
    /// names: DXC strips them from a program's module).
    fn metadata(&mut self, flags: u64) -> Metadata {
        let mut md = Metadata::default();
        // who wrote the module, where DXC names itself. A module needs one: without it the AMD
        // Windows driver faults in `CreateComputePipelineState` (seen on an RX 7800 XT, #1269)
        let writer = md.string("xcl");
        let ident = md.node(vec![writer]);
        md.name("llvm.ident", &[ident]);
        let (one, six) = (md.i32(1), md.i32(6));
        // DXIL 1.6, and the oldest validator that knows shader model 6.6: a newer one accepts it
        let version = md.node(vec![one, six]);
        md.name("dx.version", &[version]);
        md.name("dx.valver", &[version]);
        let cs = md.string("cs");
        let model = md.node(vec![cs, six, six]);
        md.name("dx.shaderModel", &[model]);

        // a record per resource in use: [id in its class, the resource as a value, name, space, register, range
        // size, ...]; a UAV: [kind, globally coherent, has a counter, is a ROV, no more], a sampler: [kind, no more]
        let (mut uavs, mut samplers) = (Vec::new(), Vec::new());
        for r in self.resources.clone().into_iter().filter(|r| r.used) {
            let (list, name, tail): (&mut Vec<_>, _, Vec<Val>) = match r.kind {
                ResKind::RawUav { coherent } => (&mut uavs, "struct.RWByteAddressBuffer", vec![Val::i32(11), Val::bool(coherent), Val::bool(false), Val::bool(false)]),
                ResKind::Sampler { comparison: false } => (&mut samplers, "struct.SamplerState", vec![Val::i32(0)]),
                ResKind::Sampler { comparison: true } => (&mut samplers, "struct.SamplerComparisonState", vec![Val::i32(1)]),
            };
            let object = self.ty(TypeDef::Struct(Some(name.into()), vec![Ty::I32]));
            let pointer = self.pointer(object, 0);
            let mut record = vec![md.i32(list.len() as u32), md.value(Val::undef(pointer)), md.string(""), md.i32(r.space), md.i32(r.register), md.i32(1)];
            record.extend(tail.into_iter().map(|v| md.value(v)));
            record.push(None);
            list.push(md.node(record));
        }
        let resources = if uavs.is_empty() && samplers.is_empty() {
            None
        } else {
            // SRVs, UAVs, constant buffers, samplers
            let mut lists = |records: Vec<Option<u32>>| if records.is_empty() { None } else { md.node(records) };
            let (uavs, samplers) = (lists(uavs), lists(samplers));
            let resources = md.node(vec![None, uavs, None, samplers]);
            md.name("dx.resources", &[resources]);
            resources
        };

        // the entry point: [function, name, signatures, resources, properties]; the properties are tagged:
        // 0 the shader flags (if any), 4 numthreads, 11 the wave size
        let main = Val { ty: self.pointer(self.functions[0].ty, 0), kind: Kind::Function(0) };
        let (function, name) = (md.value(main), md.string("main"));
        let mut properties = Vec::new();
        if flags != 0 {
            properties.extend([md.i32(0), md.value(Val::i64(flags))]);
        }
        let threads = self.numthreads.map(|n| md.i32(n)).to_vec();
        properties.extend([md.i32(4), md.node(threads)]);
        if let Some(lanes) = self.wave_size {
            let lanes = md.i32(lanes);
            properties.extend([md.i32(11), md.node(vec![lanes])]);
        }
        let properties = md.node(properties);
        let entry = md.node(vec![function, name, None, resources, properties]);
        md.name("dx.entryPoints", &[entry]);
        md
    }

    /// The program as the DXIL container D3D12 takes (`CreateComputePipelineState`), with its
    /// digest, or why it cannot be written. Nothing here needs `dxil.dll`.
    pub fn finish(&mut self) -> Result<Vec<u8>, Refused> {
        let (shader_flags, native_16) = self.shader_flags();
        if self.numthreads.iter().any(|&n| n == 0) || self.numthreads[0] > 1024 || self.numthreads[1] > 1024 || self.numthreads[2] > 64 || self.numthreads.iter().product::<u32>() > 1024 {
            self.refuse("numthreads", None, format!("numthreads {:?}", self.numthreads));
        }
        let metadata = self.metadata(shader_flags);
        let mut w = BitWriter::new();
        // 'BC' 0xC0DE
        w.emit(0xDEC0_4342, 32);
        // a module that is refused is refused before anything is written
        write_module(self, &metadata, native_16, &mut w)?;
        // the resources of `dx.resources`: a raw buffer (11) or a sampler (14) at one register
        let bound = |r: &super::dxop::Resource| Binding { class: r.class(), kind: if r.class() == 1 { 11 } else { 14 }, space: r.space, lower: r.register, upper: r.register, ..Binding::default() };
        let bindings: Vec<Binding> = self.resources.iter().filter(|r| r.used).map(bound).collect();
        let program = ComputeProgram { bitcode: &w.finish(), numthreads: self.numthreads, wave_size: self.wave_size, shader_flags, bindings: &bindings };
        compute_container(&program).map_err(|text| Refused { kind: "container", at: None, text })
    }
}
