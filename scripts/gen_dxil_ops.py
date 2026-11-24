"""Generate crates/xcl/src/gpu/recomp/dxil/ops.rs: the DXIL operations (the `dx.op` functions).

Input: DirectXShaderCompiler's own database of the operations, utils/hct/hctdb.py
(github.com/microsoft/DirectXShaderCompiler, University of Illinois / NCSA licence), at the
tag below. DXC's build generates its C++ tables (lib/DXIL/DxilOperations.cpp,
include/dxc/DXIL/DxilInstructions.h, the enumerations of DxilConstants.h) from it with
utils/hct/hctdb_instrhelp.py; this script reads the same database for the same facts:

- utils/hct/hctdb.py is loaded and its `db_dxil()` built (it reads hctdb_inst_docs.txt beside
  it): each table's id, and each operation's number, name, class, attribute, overloads,
  operands (type code and name), lowest shader model and stages; how many core operations
  each DXIL version has;
- utils/hct/hctdb_instrhelp.py is read, not run, for the tables its generators turn the
  database's codes into C++ with: `op_type_texts` (an operand's type code as OP::GetOpFunc's
  type expression), `attr_dict`, `lower_exceptions` (the class's function name) and
  `shader_stage_to_ShaderKind`.

Usage:  python -I scripts/gen_dxil_ops.py crates/xcl/src/gpu/recomp/dxil/ops.rs [<folder>]

<folder> holds the three files, by file name or by their path in the DXC repository, as the
tag has them (LF line ends). Without it they are downloaded from the tag into a new temporary
folder. hctdb.py is code this script runs, so the files must have the SHA-256 hashes below;
for another tag, read its files, then change TAG and the hashes. The hashes also go into the
header of the generated file.

The first version of this script (after `xtask/src/dxil_ops.rs` of github.com/Kalmarv/dx,
commit 02d837a0d4788eee8eb64ac7682d5cc2160012aa, GPL-3.0-or-later, contributed to XCL under
GPL-2.0-only by its author) read the generated C++ files instead; for this tag both give the
same table.
"""
import ast
import hashlib
import importlib.util
import os
import re
import sys
import tempfile
import urllib.request

TAG = "v1.9.2609"
URL = "https://raw.githubusercontent.com/microsoft/DirectXShaderCompiler/%s/%s"
DATABASE = "utils/hct/hctdb.py"
HELPERS = "utils/hct/hctdb_instrhelp.py"
DOCS = "utils/hct/hctdb_inst_docs.txt"
SHA256 = {
    DATABASE: "07a9bffc87fa4b5e028b39c1b53a57b12e400682b67869e63ecf4b13128580f8",
    HELPERS: "493980906937e96f05cdf1c1a48f2ec53151ce259b2f914086982c0cbb2729bf",
    DOCS: "fcd54db75f43ab40999d7e3f743c688e81a86d6fbe75afdc2d9c6bb9fef34eca",
}

# OP::TypeSlot (include/dxc/DXIL/DxilOperations.h): the bit of an overload mask, in the order of
# the letters of hctdb.py's `dxil_all_user_oload_chars`
SLOTS = [("F16", "h"), ("F32", "f"), ("F64", "d"), ("I1", "1"), ("I8", "8"), ("I16", "w"), ("I32", "i"), ("I64", "l"), ("UDT", "u"), ("OBJECT", "o"), ("VECTOR", "<")]

# DXIL::ShaderKind (include/dxc/DXIL/DxilConstants.h; the database does not have it): the bit
# of a stage mask
KINDS = ["Pixel", "Vertex", "Geometry", "Hull", "Domain", "Compute", "Library", "RayGeneration", "Intersection", "AnyHit", "ClosestHit", "Miss", "Callable", "Mesh", "Amplification", "Node"]

# the types OP::GetOpFunc builds a function type from: its expression (the one `op_type_texts`
# has for a type code of the database), `Ty`'s variant
TYPES = {
    "A(pV)": "Void",
    "A(pI1)": "I1",
    "A(pI8)": "I8",
    "A(pI16)": "I16",
    "A(pI32)": "I32",
    "A(pI64)": "I64",
    "A(pF16)": "F16",
    "A(pF32)": "F32",
    "A(pF64)": "F64",
    "A(pPF32)": "F32Ptr",
    "A(pETy)": "Overload",
    "A(udt)": "Overload",
    "A(obj)": "Overload",
    "A(pVecElt)": "OverloadElement",
    "A(pOlTplI1)": "I1LikeOverload",
    "A(pOlTplI32)": "I32LikeOverload",
    "A(EXT(0))": "Ext(0)",
    "A(EXT(1))": "Ext(1)",
    "A(EXT(2))": "Ext(2)",
    "A(EXT(3))": "Ext(3)",
    "TGSM(EXT(1))": "GroupSharedPtrExt(1)",
    "RRT(pETy)": "ResRet",
    "CBRT(pETy)": "CBufRet",
    "VEC2(pI32)": "I32x2",
    "VEC4(pETy)": "OverloadStruct4",
    "VEC9(pETy)": "OverloadX9",
    "A(pRes)": "Handle",
    "A(pNodeHandle)": "NodeHandle",
    "A(pNodeRecordHandle)": "NodeRecordHandle",
    "A(pHit)": "HitObject",
    "A(pDim)": "Dimensions",
    "A(pPos)": "SamplePos",
    "A(pI32C)": "I32C",
    "A(p2I32)": "TwoI32",
    "A(p4I32)": "FourI32",
    "A(pSDT)": "SplitDouble",
    "A(resProperty)": "ResourceProperties",
    "A(resBind)": "ResBind",
    "A(nodeProperty)": "NodeInfo",
    "A(nodeRecordProperty)": "NodeRecordInfo",
}

ATTRS = ["None", "ReadNone", "ReadOnly", "ArgMemOnly", "NoDuplicate", "NoReturn"]

PRELUDE = '''//! The DXIL operations: what an emitter needs to declare and call each `dx.op` function.
//!
//! An operation is called as `dx.op.<function>[.<overload>](i32 <opcode>, <params>)`: the
//! function is the operation class's, shared by the operations of the class, with the suffix of
//! the overload type it is declared for (`i32`, `f32`, ...; none for an operation with no
//! overload). Every `dx.op` function is `nounwind` and has the attribute in [`Op::attr`].

/// A type in the signature of a `dx.op` function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    Void,
    I1,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    /// `float*`
    F32Ptr,
    /// the overload type the function is declared for
    Overload,
    /// the element type of the overload type, a vector
    OverloadElement,
    /// `i1`, or a vector of them as long as the overload type when that is a vector
    I1LikeOverload,
    /// `i32`, or a vector of them as long as the overload type when that is a vector
    I32LikeOverload,
    /// the nth type of an extended overload (an operation with more than one overload
    /// dimension; DXC wraps the types in an unnamed struct)
    Ext(u8),
    /// a pointer to the nth type of an extended overload in address space 3 (group shared)
    GroupSharedPtrExt(u8),
    /// `%dx.types.ResRet.<overload>`: four of the overload type and an i32 status (for a vector
    /// overload `%dx.types.ResRet.v<length><element>`: the vector and the status)
    ResRet,
    /// `%dx.types.CBufRet.<overload>`
    CBufRet,
    /// `<2 x i32>`
    I32x2,
    /// four of the overload type in a struct: `%dx.types.fouri32` or `%dx.types.fouri16`
    OverloadStruct4,
    /// a vector of nine of the overload type
    OverloadX9,
    /// `%dx.types.Handle`
    Handle,
    /// `%dx.types.NodeHandle`
    NodeHandle,
    /// `%dx.types.NodeRecordHandle`
    NodeRecordHandle,
    /// `%dx.types.HitObject`
    HitObject,
    /// `%dx.types.Dimensions`
    Dimensions,
    /// `%dx.types.SamplePos`
    SamplePos,
    /// `%dx.types.i32c`: the result of a binary operation with carry
    I32C,
    /// `%dx.types.twoi32`
    TwoI32,
    /// `%dx.types.fouri32`
    FourI32,
    /// `%dx.types.splitdouble`
    SplitDouble,
    /// `%dx.types.ResourceProperties`
    ResourceProperties,
    /// `%dx.types.ResBind`
    ResBind,
    /// `%dx.types.NodeInfo`
    NodeInfo,
    /// `%dx.types.NodeRecordInfo`
    NodeRecordInfo,
}

/// The attribute of a `dx.op` function, next to `nounwind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attr {
    None,
    ReadNone,
    ReadOnly,
    ArgMemOnly,
    NoDuplicate,
    NoReturn,
}

/// The overload types (DXC's `TypeSlot`), the bits of [`Op::overloads`].
pub mod slot {
@SLOTS@}

/// The shader stages (DXC's `ShaderKind`), the bits of [`Op::stages`].
pub mod stage {
@STAGES@}

/// One DXIL operation.
#[derive(Debug)]
pub struct Op {
    /// the first argument of the call: the table id in the upper 16 bits, the index in the table
    /// in the lower
    pub opcode: u32,
    pub name: &'static str,
    /// the operation class (DXC's `OpCodeClass`)
    pub class: &'static str,
    /// the class's function is `dx.op.<function>`, with `.<overload>` after it for an operation
    /// with overloads
    pub function: &'static str,
    pub attr: Attr,
    /// For each overload dimension (none: the function has no overload suffix), the [`slot`]s
    /// the type may be, and the slots its elements may be where it may be a vector.
    pub overloads: &'static [(u16, u16)],
    pub ret: Ty,
    /// the parameters after the opcode, by DXC's name; none for a reserved operation
    pub params: &'static [(&'static str, Ty)],
    /// The lowest shader model (major, minor) a program that calls the operation may have, and
    /// the [`stage`]s that may call it: DXC's `OP::GetMinShaderModelAndMask` by opcode. Its
    /// validator has further rules for some calls, by their arguments.
    pub min_sm: (u8, u8),
    pub stages: u16,
}

/// The operation of an opcode, where DXC's tables have it.
pub fn op(opcode: u32) -> Option<&'static Op> {
    let index = (opcode & 0xFFFF) as usize;
    match opcode >> 16 {
@LOOKUP@        _ => None,
    }
}
'''


def literals(source, names):
    """The dictionaries hctdb_instrhelp.py assigns to `names`, read from its text."""
    found = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            name = node.targets[0].id
            if name in found:
                sys.exit("%s: `%s` is assigned twice" % (HELPERS, name))
            found[name] = ast.literal_eval(node.value)
    for name in names:
        if not isinstance(found.get(name), dict):
            sys.exit("%s: no dictionary `%s`" % (HELPERS, name))
    return [found[name] for name in names]


def tables(db, letters, helpers):
    """Each table of the database: its name, id and operations as `render` writes them."""
    type_texts, attr_names, lower_exceptions, stage_kinds = helpers
    if letters != "".join(letter for _, letter in SLOTS):
        sys.exit("%s: the overload letters are `%s`" % (DATABASE, letters))
    if sorted(stage_kinds.values()) != sorted(KINDS):
        sys.exit("%s: the stages are %s" % (HELPERS, sorted(stage_kinds.values())))
    # as hctdb_instrhelp.py's print_opfunc_props_for_table: a letter's bit is its place
    mask = lambda chars: sum(1 << letters.index(c) for c in chars)
    out, classes = [], {}
    for table in db.op_tables:
        ops = []
        for index, i in enumerate(table):
            here = "%s, %s: %s" % (DATABASE, table.name, i.name)
            if i.dxil_opid != table.id << 16 | index or i.dxil_op != i.name:
                sys.exit("%s is the operation %s, number %#x" % (here, i.dxil_op, i.dxil_opid))
            if attr_names.get(i.fn_attr) not in ATTRS:
                sys.exit("%s has the attribute `%s`" % (here, i.fn_attr))
            overloads = []
            if i.num_oloads > 0:
                for dimension in i.oload_types.split(","):
                    scalars, vector, elements = dimension.partition("<")
                    overloads.append((mask(scalars + vector), mask(elements)))
            if len(overloads) != i.num_oloads or (i.oload_types == "v") != (i.num_oloads == 0):
                sys.exit("%s: the overloads `%s`" % (here, i.oload_types))
            types = []
            for o in i.ops:
                text = type_texts.get(o.llvm_type, "").rstrip(";")
                if text not in TYPES:
                    sys.exit("%s: the type `%s` (`%s`), which this script does not know" % (here, o.llvm_type, text))
                types.append(TYPES[text])
            if [o.pos for o in i.ops] != list(range(len(i.ops))) or len(types) < 2 or types[1] != "I32" or i.ops[1].name != "opcode":
                sys.exit("%s: its operands are not a result, the i32 opcode and the rest" % here)
            params = [(o.name, ty) for o, ty in zip(i.ops[2:], types[2:])]
            if i.is_reserved != (i.dxil_class == "Reserved") or i.is_reserved and params:
                sys.exit("%s is reserved but has operands" % here)
            function = lower_exceptions.get(i.dxil_class, i.dxil_class[:1].lower() + i.dxil_class[1:])
            # a class has one function per overload, so its operations with as many overload
            # dimensions have the same types (WavePrefixBitCount, with none, is in WavePrefixOp's)
            shared = classes.setdefault((i.dxil_class, i.num_oloads), (function, i.name, types, i.fn_attr))
            if (function, types, i.fn_attr) != (shared[0], shared[2], shared[3]):
                sys.exit("%s: its function, types or attribute differ from %s's, of the same class %s" % (here, shared[1], i.dxil_class))
            # OP::GetMinShaderModelAndMask: the shader model where the operation is kept (not
            # translated for an older one), and every stage when none is named
            stages = sum(1 << KINDS.index(stage_kinds[s]) for s in i.shader_stages) or (1 << len(KINDS)) - 1
            ops.append((i.dxil_opid, i.name, i.dxil_class, function, attr_names[i.fn_attr], overloads, types[0], params, tuple(i.shader_model), stages))
        out.append((table.name, table.id, ops))
    if not out or out[0][:2] != ("CoreOps", 0):
        sys.exit("%s: the first table is not CoreOps" % DATABASE)
    return out


def versions(table):
    """How many core operations DXIL 1.0, 1.1, ... have."""
    info = table.op_enum.dxil_version_info
    counts = [info[key] for key in sorted(info)]
    if sorted(info) != [(1, n) for n in range(len(info))] or counts != sorted(counts) or counts[-1:] != [len(table.ops)]:
        sys.exit("%s: the core operations by DXIL version: %s of %d" % (DATABASE, info, len(table.ops)))
    return counts


def constant(name):
    """A table's constant: its name in upper snake case (`CoreOps` as `CORE_OPS`)."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


def render(tabs, counts, kinds):
    slots = "".join("    pub const %s: u16 = 1 << %d;\n" % (name, bit) for bit, (name, _) in enumerate(SLOTS))
    lookup = "".join("        %#x => %s.get(index),\n" % (table_id, constant(table)) for table, table_id, _ in tabs)
    stages = "".join("    pub const %s: u16 = 1 << %d;\n" % (constant(name), bit) for bit, name in enumerate(kinds))
    out = [PRELUDE.replace("@SLOTS@", slots).replace("@STAGES@", stages).replace("@LOOKUP@", lookup)]
    for table, table_id, ops in tabs:
        out.append("\n/// The `%s` table (table id %#x), by index.\npub static %s: [Op; %d] = [\n" % (table, table_id, constant(table), len(ops)))
        for opcode, name, cls, function, attr, overloads, ret, params, sm, mask in ops:
            out.append(
                '    Op { opcode: %#x, name: "%s", class: "%s", function: "%s", attr: Attr::%s, overloads: &[%s], ret: Ty::%s, params: &[%s], min_sm: (%d, %d), stages: %#x },\n'
                % (opcode, name, cls, function, attr, ", ".join("(%#x, %#x)" % o for o in overloads), ret, ", ".join('("%s", Ty::%s)' % p for p in params), sm[0], sm[1], mask))
        out.append("];\n")
    out.append("\n/// How many of the core operations DXIL 1.0, 1.1, ... have: DXIL 1.n has the core opcodes\n/// below `CORE_OPS_IN_DXIL_1[n]`.\n")
    out.append("pub const CORE_OPS_IN_DXIL_1: [u32; %d] = [%s];\n" % (len(counts), ", ".join(map(str, counts))))
    out.append("\n/// The opcodes, by DXC's names.\n#[allow(non_upper_case_globals)]\npub mod opcode {\n")
    for _, _, ops in tabs:
        for opcode, name, cls, *_ in ops:
            if cls != "Reserved":
                out.append("    pub const %s: u32 = %#x;\n" % (name, opcode))
    out.append("}\n")
    return "".join(out)


def find(folder, path):
    for candidate in (os.path.join(folder, path), os.path.join(folder, os.path.basename(path))):
        if os.path.isfile(candidate):
            return candidate
    sys.exit("%s: no %s" % (folder, path))


def main(dst, folder=None):
    if folder is None:
        folder = tempfile.mkdtemp(prefix="dxc-%s-" % TAG)
        for path in SHA256:
            urllib.request.urlretrieve(URL % (TAG, path), os.path.join(folder, os.path.basename(path)))
        print("downloaded to %s" % folder)
    files = {path: find(folder, path) for path in SHA256}
    for path, file in files.items():
        got = hashlib.sha256(open(file, "rb").read()).hexdigest()
        if got != SHA256[path]:
            sys.exit("%s is not %s of %s (sha256 %s, not %s): nothing was run" % (file, path, TAG, got, SHA256[path]))
    if os.path.dirname(files[DOCS]) != os.path.dirname(files[DATABASE]):
        sys.exit("%s: hctdb.py reads hctdb_inst_docs.txt beside itself" % folder)
    spec = importlib.util.spec_from_file_location("hctdb", files[DATABASE])
    hctdb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hctdb)
    db = hctdb.db_dxil()
    helpers = literals(open(files[HELPERS], encoding="utf-8").read(), ["op_type_texts", "attr_dict", "lower_exceptions", "shader_stage_to_ShaderKind"])
    tabs = tables(db, hctdb.dxil_all_user_oload_chars, helpers)
    header = "// @generated by scripts/gen_dxil_ops.py from DirectXShaderCompiler %s – do not edit by hand.\n" % TAG
    header += "// Copyright (c) Microsoft Corporation. Licence: University of Illinois Open Source (NCSA).\n"
    for path in SHA256:
        header += "// input %s sha256 %s\n" % (path, SHA256[path])
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + render(tabs, versions(db.op_tables[0]), KINDS))
    print("%s: %s" % (dst, ", ".join("%s %d" % (table, len(ops)) for table, _, ops in tabs)))


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(*sys.argv[1:])
