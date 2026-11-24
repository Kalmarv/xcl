"""Generate crates/xcl/src/gpu/recomp/dxil/ops.rs: the DXIL operations (the `dx.op` functions).

Input: three files of DirectXShaderCompiler (github.com/microsoft/DirectXShaderCompiler,
University of Illinois / NCSA licence) at the tag below, which its build generates from
utils/hct/hctdb.py and commits:

- include/dxc/DXIL/DxilConstants.h, the `OpCodeTableID` enumeration: each table's name and id;
  each table's `OpCode` enumeration: the operations' numbers (checked against their index in
  the table) and how many of the core ones each DXIL version has; the `ShaderKind`
  enumeration: the stages;
- lib/DXIL/DxilOperations.cpp, `<table>_OpCodeProps`: each operation's name, class, function
  attribute and overloads, by index; `OP::GetOpFunc`: each operation's function type;
  `OP::GetMinShaderModelAndMask`: each operation's lowest shader model and its stages;
- include/dxc/DXIL/DxilInstructions.h, `DxilInst_<name>`: the names of the operands after the
  opcode (a reserved operation has none).

Usage:  python -I scripts/gen_dxil_ops.py crates/xcl/src/gpu/recomp/dxil/ops.rs [<folder>]

<folder> holds the three files, by file name or by their path in the DXC repository. Without
it they are downloaded from the tag into a new temporary folder. Their SHA-256 hashes go into
the header of the generated file.

The parsing follows `xtask/src/dxil_ops.rs` of github.com/Kalmarv/dx (commit
02d837a0d4788eee8eb64ac7682d5cc2160012aa, GPL-3.0-or-later, contributed to XCL under
GPL-2.0-only by its author), which reads the names and operands; the class, attribute,
overloads, types, shader models and stages are read here as well.
"""
import hashlib
import os
import re
import sys
import tempfile
import urllib.request

TAG = "v1.9.2609"
CONSTANTS = "include/dxc/DXIL/DxilConstants.h"
OPERATIONS = "lib/DXIL/DxilOperations.cpp"
INSTRUCTIONS = "include/dxc/DXIL/DxilInstructions.h"
URL = "https://raw.githubusercontent.com/microsoft/DirectXShaderCompiler/%s/%s"

# OP::TypeSlot (include/dxc/DXIL/DxilOperations.h): the bit of an overload mask, and the letter
# DxilOperations.cpp's `// Overloads:` comments give it, which the masks are checked against
SLOTS = [("F16", "h"), ("F32", "f"), ("F64", "d"), ("I1", "1"), ("I8", "8"), ("I16", "w"), ("I32", "i"), ("I64", "l"), ("UDT", "u"), ("OBJECT", "o"), ("VECTOR", "<")]

# the types OP::GetOpFunc builds a function type from: its expression, `Ty`'s variant
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


def between(text, begin, end, what):
    """The text from after `begin` to before the first `end` after it."""
    start = text.find(begin)
    if start < 0:
        sys.exit("%s: no `%s`" % (what, begin))
    start += len(begin)
    stop = text.find(end, start)
    if stop < 0:
        sys.exit("%s: `%s` has no `%s`" % (what, begin, end))
    return text[start:stop]


def operands(instructions):
    """Each DxilInst_<operation>'s operands after the opcode, in order (`arg_<name> = <n>`)."""
    found = {}
    for chunk in instructions.split("\nstruct DxilInst_")[1:]:
        op, body = chunk.split(" {", 1)
        body = body[:body.index("\n};")]
        names = []
        if "enum OperandIdx {" in body:
            lines = [l.strip() for l in between(body, "enum OperandIdx {", "};", INSTRUCTIONS).splitlines() if l.strip()]
            for n, line in enumerate(lines):
                m = re.fullmatch(r"arg_(\w+) = (\d+),", line)
                if not m or int(m.group(2)) != n + 1:
                    sys.exit("%s: DxilInst_%s operand %d: `%s`" % (INSTRUCTIONS, op, n + 1, line))
                names.append(m.group(1))
        if op in found:
            sys.exit("%s: DxilInst_%s twice" % (INSTRUCTIONS, op))
        found[op] = names
    return found


def signatures(operations):
    """Each operation's function type in OP::GetOpFunc: the return type, the opcode, the rest."""
    found = {}
    cases = between(operations, "// OPCODE-OLOAD-FUNCS:BEGIN", "// OPCODE-OLOAD-FUNCS:END", OPERATIONS)
    for case in cases.split("case OpCode::")[1:]:
        op, body = case.split(":", 1)
        body = body[:body.index("break;")]
        types = []
        for statement in body.split(";"):
            statement = re.sub(r"//.*", "", statement).strip()
            if statement:
                if statement not in TYPES:
                    sys.exit("%s: GetOpFunc gives %s a type this script does not know: `%s`" % (OPERATIONS, op, statement))
                types.append(TYPES[statement])
        if len(types) < 2 or types[1] != "I32":
            sys.exit("%s: GetOpFunc: %s has no i32 opcode after its return type" % (OPERATIONS, op))
        if op in found:
            sys.exit("%s: GetOpFunc has %s twice" % (OPERATIONS, op))
        found[op] = types
    return found


def masks(text, what):
    """`{{0x63}, {0x400}}` as [0x63, 0x400]."""
    text = "".join(text.split())
    if not re.fullmatch(r"\{(\{0x[0-9a-f]+\},?)*\}", text):
        sys.exit("%s: overload masks `%s`" % (what, text))
    return [int(m, 16) for m in re.findall(r"0x[0-9a-f]+", text)]


def letters(mask, elements):
    """A dimension's masks as the `// Overloads:` comment writes them, each part as a set."""
    part = lambda m: frozenset(l for i, (_, l) in enumerate(SLOTS) if m >> i & 1 and l != "<")
    return (part(mask), mask >> 10 & 1, part(elements))


def tables(constants, operations, instructions):
    names, types = operands(instructions), signatures(operations)
    _, required = requirements(constants, operations)
    out, classes = [], {}
    enumeration = between(constants, "enum class OpCodeTableID : unsigned {", "};", CONSTANTS)
    for line in (l.strip() for l in enumeration.splitlines() if l.strip()):
        m = re.match(r"(\w+) = (\d+),", line)
        if not m:
            sys.exit("%s: table `%s`" % (CONSTANTS, line))
        table, table_id = m.group(1), int(m.group(2))
        what = "%s, %s" % (OPERATIONS, table)
        props = between(operations, "static const OP::OpCodeProperty %s_OpCodeProps[] = {" % table, "\n};", what) + "\n"
        ops = []
        # each entry: {OC::<id>, "<name>", OCC::<class>, "<function>", Attribute::<attr>, <dims>,
        # {<type masks>}, {<vector element masks>}}, // Overloads: <letters>
        entry = re.compile(
            r'\{OC::(\w+),\s*"(\w+)",\s*OCC::(\w+),\s*"(\w+)",\s*Attribute::(\w+),\s*(\d+),\s*(\{(?:\{\w+\},?\s*)*\}),\s*(\{(?:\{\w+\},?\s*)*\})\},\s*// Overloads: (\S+)\n')
        found = entry.findall(props)
        if len(found) != props.count("{OC::"):
            sys.exit("%s: %d of %d entries read" % (what, len(found), props.count("{OC::")))
        for index, (ident, name, cls, function, attr, dims, allowed, elements, comment) in enumerate(found):
            here = "%s: %s" % (what, name)
            if ident != name:
                sys.exit("%s is OC::%s" % (here, ident))
            if attr not in ATTRS:
                sys.exit("%s has the attribute %s" % (here, attr))
            allowed, elements = masks(allowed, here), masks(elements, here)
            if not len(allowed) == len(elements) == int(dims) <= 4:
                sys.exit("%s: %s overload dimensions, %d and %d masks" % (here, dims, len(allowed), len(elements)))
            said = [] if comment == "v" else [(frozenset(a), int("<" in d), frozenset(b)) for d in comment.split(",") for a, _, b in [d.partition("<")]]
            if said != [letters(a, e) for a, e in zip(allowed, elements)]:
                sys.exit("%s: the masks %s / %s are not `%s`" % (here, allowed, elements, comment))
            if name not in types:
                sys.exit("%s: GetOpFunc has no case for %s" % (OPERATIONS, name))
            ret, params = types[name][0], types[name][2:]
            if cls == "Reserved":
                if name in names or params:
                    sys.exit("%s is reserved but has operands" % here)
                params = []
            elif name not in names:
                sys.exit("%s: no DxilInst_%s for %s's %s" % (INSTRUCTIONS, name, table, name))
            elif len(names[name]) != len(params):
                sys.exit("%s: DxilInst_%s names %d operands, GetOpFunc gives %d types" % (here, name, len(names[name]), len(params)))
            else:
                params = list(zip(names[name], params))
            # a class has one function per overload, so its operations with as many overload
            # dimensions have the same types (WavePrefixBitCount, with none, is in WavePrefixOp's)
            shared = classes.setdefault((cls, dims), (function, name, types[name], attr))
            if (function, types[name], attr) != (shared[0], shared[2], shared[3]):
                sys.exit("%s: its function, types or attribute differ from %s's, of the same class %s" % (here, shared[1], cls))
            ops.append((table_id << 16 | index, name, cls, function, attr, list(zip(allowed, elements)), ret, params) + required(table_id << 16 | index, name))
        numbered = numbers(constants, table)
        if numbered.get("NumOpCodes") != len(ops) or any(numbered.get(op[1]) != index for index, op in enumerate(ops)):
            sys.exit("%s: the indices of %s are not the numbers of its OpCode enumeration" % (what, table))
        out.append((table, table_id, ops))
    if not out:
        sys.exit("%s: OpCodeTableID lists no tables" % CONSTANTS)
    return out


def requirements(constants, operations):
    """The stages' names, and OP::GetMinShaderModelAndMask as a function of an opcode and its
    name: the lowest shader model (where the operation is kept, not translated for an older
    one) and the stage mask. The names are checked against the function's comments."""
    kinds = [k.split("=")[0].strip() for k in between(constants, "enum class ShaderKind {", "Invalid,", CONSTANTS).split(",") if k.strip()]
    if kinds[:2] != ["Pixel", "Vertex"] or len(kinds) > 16 or not all(re.fullmatch(r"\w+", k) for k in kinds):
        sys.exit("%s: ShaderKind: %s" % (CONSTANTS, kinds))
    rules = []
    section = between(operations, "// OPCODE-SMMASK:BEGIN", "// OPCODE-SMMASK:END", OPERATIONS)
    for chunk in section.split("// Instructions:")[1:]:
        comment, rest = chunk.split("\n  if (", 1)
        condition, body = rest.split(") {\n", 1)
        named = {int(n): name for name, n in re.findall(r"(\w+)=(\d+)", comment)}
        what = "%s: GetMinShaderModelAndMask, %s" % (OPERATIONS, sorted(named.values()))
        condition = "".join(condition.split())
        ranges = [(int(a or c), int(b or c)) for a, b, c in re.findall(r"(\d+)<=op&&op<=(\d+)|op==(\d+)", condition)]
        if "||".join("(%d<=op&&op<=%d)" % r if r[0] != r[1] else "op==%d" % r[0] for r in ranges) != condition:
            sys.exit("%s: the condition `%s`" % (what, condition))
        if sorted(named) != sorted(n for a, b in ranges for n in range(a, b + 1)):
            sys.exit("%s: `%s` is not the opcodes %s" % (what, condition, sorted(named)))
        if "bWithTranslation" in body:
            body = body.split("} else {", 1)[1]
        sm = re.search(r"major = (\d+);\s*minor = (\d+);", body)
        mask = re.search(r"mask =([^;]*);", body)
        flags = re.findall(r"SFLAG\((\w+)\)", mask.group(1)) if mask else kinds
        if mask and "".join(mask.group(1).split()) != "|".join("SFLAG(%s)" % f for f in flags) or not (sm or mask):
            sys.exit("%s: its shader model and mask" % what)
        rules.append((named, tuple(map(int, sm.groups())) if sm else (6, 0), sum(1 << kinds.index(f) for f in flags)))

    def of(opcode, name):
        for named, sm, mask in rules:
            if opcode in named:
                if named[opcode] != name:
                    sys.exit("%s: GetMinShaderModelAndMask calls opcode %d %s, the table %s" % (OPERATIONS, opcode, named[opcode], name))
                return sm, mask
        return (6, 0), (1 << len(kinds)) - 1

    return kinds, of


def numbers(constants, table):
    """The `OpCode` enumeration of a table: each name's value."""
    block = between(constants, "// Enumeration for %s DXIL operations\nenum class OpCode : unsigned {" % table, "\n};", CONSTANTS)
    block = re.sub(r"//.*", "", block)
    return {name: int(value, 0) for name, value in re.findall(r"(\w+)\s*=\s*(\w+),", block)}


def versions(constants):
    """How many core operations DXIL 1.0, 1.1, ... have (`NumOpCodes_Dxil_1_<n>`)."""
    core = numbers(constants, "CoreOps")
    counts = []
    while "NumOpCodes_Dxil_1_%d" % len(counts) in core:
        counts.append(core["NumOpCodes_Dxil_1_%d" % len(counts)])
    if not counts or counts != sorted(counts) or counts[-1] != core["NumOpCodes"]:
        sys.exit("%s: the core operations by DXIL version: %s of %s" % (CONSTANTS, counts, core.get("NumOpCodes")))
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


def read(folder, path):
    for candidate in (os.path.join(folder, path), os.path.join(folder, os.path.basename(path))):
        if os.path.isfile(candidate):
            return open(candidate, "rb").read()
    sys.exit("%s: no %s" % (folder, path))


def main(dst, folder=None):
    paths = [OPERATIONS, INSTRUCTIONS, CONSTANTS]
    if folder is None:
        folder = tempfile.mkdtemp(prefix="dxc-%s-" % TAG)
        for path in paths:
            urllib.request.urlretrieve(URL % (TAG, path), os.path.join(folder, os.path.basename(path)))
        print("downloaded to %s" % folder)
    data = {path: read(folder, path) for path in paths}
    text = {path: data[path].decode("utf-8").replace("\r\n", "\n") for path in paths}
    tabs = tables(text[CONSTANTS], text[OPERATIONS], text[INSTRUCTIONS])
    header = "// @generated by scripts/gen_dxil_ops.py from DirectXShaderCompiler %s – do not edit by hand.\n" % TAG
    header += "// Copyright (c) Microsoft Corporation. Licence: University of Illinois Open Source (NCSA).\n"
    for path in paths:
        header += "// input %s sha256 %s\n" % (path, hashlib.sha256(data[path]).hexdigest())
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + render(tabs, versions(text[CONSTANTS]), requirements(text[CONSTANTS], text[OPERATIONS])[0]))
    print("%s: %s" % (dst, ", ".join("%s %d" % (table, len(ops)) for table, _, ops in tabs)))


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(*sys.argv[1:])
