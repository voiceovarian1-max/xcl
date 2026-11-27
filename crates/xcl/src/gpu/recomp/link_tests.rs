//! Programs linked with their memory accessors (`emit::linked_form`, `dxc::compile_linked`): the
//! linked form does what the whole one does, and keeps what a program declares. They live in
//! their own file because `tests.rs` ends at a place where every merged change adds its own tests.

use super::runner::{self, KernelState};
use super::{translate, Mode, Options};
use crate::gpu::rdna2::encode;

/// Assemble a program from (name, encoding, fields, literal).
fn prog(list: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
    let mut v = Vec::new();
    for (n, e, f, l) in list {
        v.extend(encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}")));
    }
    v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    v
}

/// A GPU and DXC for these tests (the same gate as `tests.rs`: wave64 programs run natively where
/// the adapter has 64-lane waves and in the split form where it has only 32).
fn gpu_ready() -> bool {
    match (runner::adapter(), runner::wave64_programs_supported(), super::dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => true,
        (a, w, d) => {
            println!("GPU/DXC unavailable – skipping ({:?} / {:?} / {:?})", a.as_ref().err(), w.as_ref().ok(), d.as_ref().err());
            runner::skip_test(if a.is_err() || w.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" });
            false
        }
    }
}

/// `o` for a program run on this adapter (the split wave64 form where it is needed).
fn on_gpu(o: Options) -> Options {
    Options { split_wave64: runner::split_for(&o.mode), ..o }
}

/// A program linked with its memory accessors (`emit::linked_form`) does what it does compiled
/// whole: loads and stores of 1..4 dwords, scalar loads, 32- and 64-bit atomics, in two windows,
/// the lanes of a wave in different ones.
#[test]
fn gpu_linked_memory_accessors() {
    if !gpu_ready() {
        return;
    }
    type Ins = (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>);
    let smem = |n: &'static str, data: u64, base: u64, off: u64| -> Ins { (n, "ENC_SMEM", vec![("SDATA", data), ("SBASE", base), ("OFFSET", off), ("SOFFSET", 125)], None) };
    let mubuf = |n: &'static str, data: u64, srsrc: u64, off: u64| -> Ins { (n, "ENC_MUBUF", vec![("VDATA", data), ("VADDR", 0), ("SRSRC", srsrc), ("SOFFSET", 128), ("IDXEN", 1), ("OFFSET", off)], None) };
    let glbl = |n: &'static str, f: Vec<(&'static str, u64)>| -> Ins { (n, "ENC_FLAT_GLBL", f, None) };
    let atomic = |n: &'static str, dst: u64, data: u64, off: u64| -> Ins { (n, "ENC_FLAT_GLBL", vec![("VDST", dst), ("ADDR", 26), ("DATA", data), ("SADDR", 0x7D), ("OFFSET", off), ("GLC", 1)], None) };
    // s[8:11] / s[12:15]: V#s in windows 1 and 0, stride 16; s[2:3]: an address in window 1;
    // v[4:5] / v[6:7] / v[26:27]: addresses per lane, odd lanes in window 1
    let list = vec![
        smem("S_LOAD_DWORD", 20, 1, 4),
        smem("S_LOAD_DWORDX2", 22, 1, 8),
        smem("S_LOAD_DWORDX8", 24, 1, 16),
        smem("S_BUFFER_LOAD_DWORD", 32, 8, 4),
        smem("S_BUFFER_LOAD_DWORDX8", 36, 8, 16),
        mubuf("BUFFER_LOAD_DWORD", 10, 2, 0),
        mubuf("BUFFER_LOAD_DWORDX2", 11, 2, 4),
        mubuf("BUFFER_LOAD_DWORDX3", 13, 2, 0),
        mubuf("BUFFER_LOAD_DWORDX4", 16, 2, 0),
        mubuf("BUFFER_STORE_DWORD", 10, 3, 0),
        mubuf("BUFFER_STORE_DWORDX2", 11, 3, 4),
        mubuf("BUFFER_STORE_DWORDX3", 13, 3, 1024),
        mubuf("BUFFER_STORE_DWORDX4", 16, 3, 2048),
        glbl("GLOBAL_LOAD_DWORD", vec![("VDST", 20), ("ADDR", 4), ("SADDR", 0x7D), ("OFFSET", 0)]),
        glbl("GLOBAL_STORE_DWORD", vec![("ADDR", 6), ("DATA", 20), ("SADDR", 0x7D), ("OFFSET", 0)]),
        atomic("GLOBAL_ATOMIC_ADD", 21, 0, 0),
        atomic("GLOBAL_ATOMIC_SMIN", 22, 2, 4),
        atomic("GLOBAL_ATOMIC_CMPSWAP", 23, 2, 8),
        atomic("GLOBAL_ATOMIC_ADD_X2", 24, 2, 16),
    ];
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let t = translate(&prog(&refs), &on_gpu(Options { mem_windows: 2, ..Default::default() })).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let [(whole, _), (linked, _)] = super::dxc::both_forms(&t.hlsl, "cs_6_6").unwrap_or_else(|e| panic!("{e}"));
    assert!(whole != linked);
    let run = |dxil: &[u8]| {
        let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
        let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
        for (w, seed) in [(w0, 1u32), (w1, 2)] {
            for k in 0..0x4000u64 {
                unsafe { *((w + 4 * k) as *mut u32) = (k as u32).wrapping_mul(0x9E37_79B1).wrapping_add(seed) };
            }
        }
        let hi = |a: u64| (a >> 32) as u32;
        let mut st = KernelState::default();
        st.sgpr[2..4].copy_from_slice(&[(w1 + 0x1000) as u32, hi(w1)]);
        st.sgpr[8..12].copy_from_slice(&[(w1 + 0x2000) as u32, hi(w1) & 0xFFFF | 16 << 16, 64, 0]);
        st.sgpr[12..16].copy_from_slice(&[(w0 + 0x4000) as u32, hi(w0) & 0xFFFF | 16 << 16, 4096, 0]);
        st.sgpr[16..20].copy_from_slice(&[(w1 + 0x3000) as u32, hi(w1) & 0xFFFF, 256, 0]);
        for (l, r) in st.vgpr.iter_mut().enumerate() {
            let w = if l % 2 == 1 { w1 } else { w0 };
            let l = l as u64;
            r[0] = l as u32;
            r[2] = 7 * l as u32;
            r[3] = 3 + l as u32;
            for (v, at) in [(4, 0x9000 + 4 * l), (6, 0xA000 + 4 * l), (26, 0xB000 + 32 * l)] {
                r[v] = (w + at) as u32;
                r[v + 1] = hi(w);
            }
        }
        let r = runner::run_in_arena(dxil, &st, &arena).unwrap();
        let mem: Vec<u32> = [w0, w1].iter().flat_map(|&w| (0..0x4000u64).map(move |k| unsafe { *((w + 4 * k) as *const u32) })).collect();
        // what ran: results against the memory's own contents
        let word = |seed: u32, at: u64| ((at / 4) as u32).wrapping_mul(0x9E37_79B1).wrapping_add(seed);
        assert_eq!(r.sgpr[20], word(2, 0x1004), "s_load from window 1");
        assert_eq!(r.sgpr[24..32], (0..8).map(|k| word(2, 0x1010 + 4 * k)).collect::<Vec<_>>(), "s_load_dwordx8");
        assert_eq!(r.sgpr[36..44], (0..8).map(|k| word(2, 0x3010 + 4 * k)).collect::<Vec<_>>(), "s_buffer_load_dwordx8");
        for l in 0..64usize {
            let at = 0x2000 + 16 * l as u64;
            assert_eq!(r.vgpr[l][16..20], (0..4).map(|k| word(2, at + 4 * k)).collect::<Vec<_>>(), "lane {l}: buffer_load_dwordx4");
            assert_eq!(r.vgpr[l][20], word(1 + (l % 2) as u32, 0x9000 + 4 * l as u64), "lane {l}: global_load from its own window");
        }
        let regs: Vec<u32> = r.vgpr.iter().flat_map(|v| v[..32].to_vec()).chain(r.sgpr[20..48].iter().copied()).collect();
        (regs, mem)
    };
    let (a, b) = (run(&linked), run(&whole));
    // v4..v7, v26 and v27 hold addresses of each run's own arena
    let own = |i: usize| i < 64 * 32 && matches!(i % 32, 4..=7 | 26..=27);
    let differ: Vec<usize> = (0..a.0.len()).filter(|&i| !own(i) && a.0[i] != b.0[i]).collect();
    assert!(differ.is_empty(), "registers differ: {differ:?}");
    assert!(a.1 == b.1, "memory differs at dword {:?}", (0..a.1.len()).find(|&i| a.1[i] != b.1[i]));
}

/// The cost of the memory windows: DXC, the driver's pipeline and a dispatch for programs of
/// 100 / 400 / 1 600 memory accesses with one window, and with four compiled whole and linked.
/// Ignored: minutes.
#[test]
#[ignore]
fn bench_memory_window_accessors() {
    use super::CsInterface;
    if !gpu_ready() {
        return;
    }
    println!("GPU: {}", runner::adapter().unwrap());
    // a different program every time: the driver must not answer from its cache
    let mut nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u32 | 1;
    let program = |n: u64, nonce: u32| {
        let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![];
        for i in 0..n {
            let (a, b) = (1 + i % 24, 1 + (i + 1) % 24);
            match i % 4 {
                0 => list.push(("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", vec![("SDATA", 24 + 4 * (i % 8)), ("SBASE", 8), ("OFFSET", 16 * i), ("SOFFSET", 125)], None)),
                1 => list.push(("BUFFER_STORE_DWORD", "ENC_MUBUF", vec![("VDATA", a), ("VADDR", b), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1), ("OFFSET", 4 * (i % 512))], None)),
                _ => list.push(("BUFFER_LOAD_DWORD", "ENC_MUBUF", vec![("VDATA", b), ("VADDR", a), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1), ("OFFSET", 4 * (i % 512))], None)),
            }
            list.push(("V_XOR_B32", "ENC_VOP2", vec![("VDST", b), ("SRC0", 24 + 4 * (i % 8)), ("VSRC1", b)], None));
        }
        list.push(("V_XOR_B32", "ENC_VOP2", vec![("VDST", 1), ("SRC0", 255), ("VSRC1", 1)], Some(nonce)));
        list.push(("BUFFER_STORE_DWORD", "ENC_MUBUF", vec![("VDATA", 1), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None));
        let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
        prog(&refs)
    };
    // loads stay in range: V#s over the memory, small values in it
    let mem: Vec<u32> = (0..1u32 << 18).map(|k| k & 0xFF).collect();
    let mut sgprs = vec![0u32; 24];
    for (at, base) in [(8usize, 0u32), (12, 0x8_0000), (16, 0)] {
        sgprs[at..at + 4].copy_from_slice(&[base, 4 << 16, 0x1_0000, 0]);
    }
    for n in [100u64, 400, 1600] {
        for round in 0..3 {
            for windows in [1, 4] {
                nonce = nonce.wrapping_mul(0x9E37_79B1) | 1;
                let cs = CsInterface { threads: [64, 1, 1], user_sgprs: 24, tgid: [false; 3], tg_size: false, tid_components: 1, lds_bytes: 0 };
                let t = translate(&program(n, nonce), &on_gpu(Options { mode: Mode::Compute(cs), wave64: true, mem_windows: windows, ..Default::default() })).unwrap();
                assert!(t.complete(), "{:?}", t.unsupported_why);
                let forms = if windows == 1 {
                    let t0 = std::time::Instant::now();
                    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
                    vec![("1 window", dxil, t0.elapsed().as_secs_f64())]
                } else {
                    let [(whole, w), (linked, l)] = super::dxc::both_forms(&t.hlsl, "cs_6_6").unwrap_or_else(|e| panic!("{e}"));
                    vec![("4, whole", whole, w), ("4, linked", linked, l)]
                };
                for (name, dxil, dxc) in forms {
                    let t0 = std::time::Instant::now();
                    runner::dispatch_timed(&dxil, [1, 1, 1], &sgprs, &mem, 1).unwrap();
                    let pipeline = t0.elapsed().as_secs_f64();
                    let mut times = runner::dispatch_timed(&dxil, [2048, 1, 1], &sgprs, &mem, 9).unwrap();
                    times.sort_by(|a, b| a.total_cmp(b));
                    println!(
                        "{n:5} accesses, round {round}, {name:9}: DXC {dxc:6.2} s, DXIL {:4} KB, pipeline {pipeline:5.2} s, 2048 groups: median {:.3} min {:.3}",
                        dxil.len() >> 10,
                        times[times.len() / 2],
                        times[0]
                    );
                }
            }
        }
    }
}

/// What a linked program declares stays: its wave size, thread count, the windows' registers,
/// and no accessor is left a function (a driver may not inline those: 5x slower on NVIDIA).
#[test]
fn linked_programs_keep_their_declarations() {
    if super::dxc::location().is_err() {
        runner::skip_test("no DXC");
        return;
    }
    let buf = |n: &'static str, srsrc: u64, glc: bool| {
        let mut fields = vec![("VDATA", 1u64), ("VADDR", 0), ("SRSRC", srsrc), ("SOFFSET", 128), ("IDXEN", 1)];
        if glc {
            fields.push(("GLC", 1));
        }
        (n, "ENC_MUBUF", fields, None)
    };
    // whether the windows are `globallycoherent` follows the program (a GLC load needs them);
    // linking keeps either
    for (coherent, wave64) in [(false, false), (false, true), (true, false), (true, true)] {
        let list = [buf("BUFFER_LOAD_DWORD", 2, coherent), buf("BUFFER_STORE_DWORD", 3, false)];
        let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
        let p = prog(&refs);
        let cs = super::CsInterface { threads: [8, 4, 2], user_sgprs: 16, tgid: [false; 3], tg_size: false, tid_components: 1, lds_bytes: 0 };
        let t = translate(&p, &Options { mode: Mode::Compute(cs), wave64, mem_windows: 4, ..Default::default() }).unwrap();
        let size = if wave64 { 64 } else { 32 };
        assert!(t.hlsl.contains(&format!("[WaveSize({size})]")), "{}", t.hlsl);
        let [whole, linked] = super::dxc::both_forms(&t.hlsl, "cs_6_6").unwrap_or_else(|e| panic!("{e}")).map(|(d, _)| super::dxc::disassemble(&d).unwrap());
        let declared = |text: &str| -> Vec<String> {
            let lines = text.lines().filter(|l| l.contains("WaveSize") || l.contains("NumThreads") || l.contains("byte") && l.contains("UAV"));
            let mut v: Vec<String> = lines.map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).collect();
            v.sort();
            v
        };
        // how each buffer is accessed (globallycoherent is one of these bits)
        let properties = |text: &str| -> std::collections::BTreeSet<String> {
            text.split("ResourceProperties { ").skip(1).filter_map(|r| r.split_once(" }")).map(|(r, _)| r.to_string()).collect()
        };
        assert_eq!(t.hlsl.contains("globallycoherent RWByteAddressBuffer mem1"), coherent, "{}", t.hlsl);
        assert_eq!(properties(&whole), properties(&linked));
        // the resources' ids may be numbered in another order
        let registers = |text: &str| (2..6).map(|k| declared(text).iter().any(|l| l.contains(&format!(" u{k} ")))).collect::<Vec<_>>();
        assert_eq!(registers(&whole), [true; 4], "{whole}");
        assert_eq!(registers(&linked), [true; 4], "{linked}");
        for text in [&whole, &linked] {
            assert!(text.contains("NumThreads=(8,4,2)"), "{text}");
            assert!(text.contains(&format!("WaveSize={size}")) || text.contains(&format!("{{i32 {size}}}")), "{text}");
        }
        assert_eq!(whole.matches("define ").count(), linked.matches("define ").count(), "{linked}");
    }
}
