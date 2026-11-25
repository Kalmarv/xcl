"""Instrument a separate HLSL preflight; production input and output binaries stay intact."""
import re

# Byte offsets in the diagnostic buffer, shared with lane1912_replay.rs.
limit = 16384
requested = 1024
bound = requested + limit * 4
pages = bound + limit * 4
helpers = f'''
uint probe_tex(uint idx,uint kind) {{
  if(idx < {limit}u) {{
    res.InterlockedOr({requested}u+4u*idx,1u<<kind);
    if(res.Load({bound}u+4u*idx)==kind+1u) return idx;
  }} else res.InterlockedOr(256u,1u);
  return {limit}u+kind;
}}
uint probe_sampler(uint idx) {{
  if(idx>=2048u) {{res.InterlockedOr(256u,2u); return 0u;}}
  return idx;
}}
uint probe_mem(uint a,uint w,uint n) {{
  uint p=a>>16; res.InterlockedOr({pages}u+8192u*w+4u*(p>>5),1u<<(p&31u));
  p=(a+(n-1u))>>16; res.InterlockedOr({pages}u+8192u*w+4u*(p>>5),1u<<(p&31u));
  return a;
}}
'''

def probe(source):
    declaration = "RWByteAddressBuffer res : register(u1);"
    if source.count(declaration) != 1:
        raise ValueError("expected one result buffer")
    source = source.replace(declaration, declaration + "\n" + helpers)
    texture = re.compile(r"((RW)?Texture([23])D<(float4|int4|uint4|uint)> \w+ = ResourceDescriptorHeap\[NonUniformResourceIndex\()([^\]]+?)(\)\];)")

    def record_texture(match):
        kind = (int(match[3]) - 2) * 8 + (4 if match[2] else 0) + ("float4", "int4", "uint4", "uint").index(match[4])
        return match[1] + "probe_tex(" + match[5] + f", {kind}u)" + match[6]

    source, count = texture.subn(record_texture, source)
    if source.count("ResourceDescriptorHeap[") != count:
        raise ValueError("unrecognized heap texture access")
    source, count = re.subn(r"(SamplerDescriptorHeap\[NonUniformResourceIndex\()([^\]]+?)(\)\])", r"\1probe_sampler(\2)\3", source)
    if source.count("SamplerDescriptorHeap[") != count:
        raise ValueError("unrecognized heap sampler access")

    calls = list(re.finditer(r"\bmem([123]?)\.(Load[234]?|Store[234]?|Interlocked\w+)\(", source))
    if len(calls) != len(re.findall(r"\bmem[123]?\.\w+\(", source)):
        raise ValueError("unrecognized memory window operation")
    # Insert from right to left so positions still refer to the original first operand.
    for call in reversed(calls):
        start = end = call.end()
        depth = 0
        while source[end] not in ",)" or depth:
            if source[end] == "(":
                depth += 1
            elif source[end] == ")":
                depth -= 1
            end += 1
        op = call[2]
        size = 8 if op.endswith("64") else 4 * int(op[-1]) if op[-1].isdigit() else 4
        arg = f"probe_mem({source[start:end]}, {call[1] or 0}u, {size}u)"
        source = source[:start] + arg + source[end:]
    return source
