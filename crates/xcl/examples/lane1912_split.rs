//! Export the selected guest programs through the HLSL and per-lane compute entry points.
use std::{collections::BTreeMap, fs, path::Path};
use xcl::gpu::{pack, recomp, shader, xbox};
fn samplers(items: &[([u32; 4], bool)]) -> String {
    items.iter().map(|(w, c)| format!("{:08x} {:08x} {:08x} {:08x} {}\n", w[0], w[1], w[2], w[3], *c as u8)).collect()
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "usage: lane1912_split PACK SELECTION.csv OUTPUT");
    let root = Path::new(&args[3]);
    let selected: BTreeMap<_, _> = fs::read_to_string(&args[2])
        .unwrap()
        .lines()
        .skip(1)
        .map(|row| {
            let w: Vec<_> = row.split(',').collect();
            (w[0].to_owned(), (w[1].parse::<usize>().unwrap(), w[2].parse::<usize>().unwrap(), w[3].parse::<usize>().unwrap()))
        })
        .collect();
    let bytes = fs::read(&args[1]).unwrap();
    let mut seen = std::collections::HashSet::new();
    for record in pack::scan(&bytes, 0x40) {
        let Some(blob) = pack::inflate(&bytes, &record) else { continue };
        let h = format!("{:016x}", shader::fnv64(&blob));
        let Some(&(fi, pi, size)) = selected.get(&h) else { continue };
        if !seen.insert(h.clone()) {
            continue;
        }
        let sh = xbox::parse(&blob).unwrap();
        assert_eq!(sh.stage, "CS");
        let f = &sh.fssm[fi];
        let Some(recomp::Mode::Compute(cs)) = recomp::interface(&blob, &sh, f) else { panic!("interface: {h}") };
        let p = &f.programs[pi];
        let code = &f.data[p.offset..p.offset + p.len];
        assert_eq!(size, code.len());
        let wave64 = recomp::detect_wave64(code);
        let split = wave64 && std::env::var_os("LANE1912_SPLIT").is_some();
        let opt = xcl::gpu::dxil::kernel::KernelOptions { wave64, split, mem_windows: 4, sampler_base: 0 };
        match recomp::lower_dxil::compute_with(code, &cs, opt) {
            Ok(k) => {
                fs::write(root.join(format!("lane/{h}.dxil")), k.dxil).unwrap();
                fs::write(root.join(format!("lane/{h}.samplers")), samplers(&k.static_samplers)).unwrap();
            }
            Err(e) => {
                println!("{h} refused {e}");
                continue;
            }
        }
        let count = cs.user_sgprs;
        let threads = cs.threads;
        let tr =
            recomp::translate(code, &recomp::Options { mode: recomp::Mode::Compute(cs), wave64, split_wave64: split, mem_windows: 4, ..Default::default() })
                .unwrap();
        if !tr.complete() {
            println!("{h} incomplete reference");
            continue;
        }
        fs::write(root.join(format!("ref/{h}.hlsl")), &tr.hlsl).unwrap();
        fs::write(root.join(format!("ref/{h}.samplers")), samplers(&tr.static_samplers)).unwrap();
        println!("{h} ok {} {count} {} {} {}", if wave64 { 64 } else { 32 }, threads[0], threads[1], threads[2]);
    }
    assert_eq!(seen.len(), selected.len());
}
