//! Checks of the direct compute translator (`kernel`, `lower`): translated
//! programs pass the validator. The GPU differential tests are the
//! recompiler's instruction tests run with `XCL_DIRECT_DXIL=1`.

use super::kernel::{translate, KernelOptions};
use crate::gpu::rdna2::encode;

fn prog(list: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
    let mut v = Vec::new();
    for (n, e, f, l) in list {
        v.extend(encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}")));
    }
    v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    v
}

/// One console wave in two hardware halves (33 to 64 threads a group): the halves of a compute
/// or mesh group exchange through the bounded counters, as those of larger groups do, and meet
/// at no group barrier after the one at the program's start. A barrier at every exchange hung NVIDIA
/// GPUs when several such programs ran at once (#1727; the change is #1741's).
#[test]
fn a_single_split_wave_of_a_compute_group_exchanges_without_a_group_barrier() {
    use crate::gpu::recomp::CsInterface;
    // v1 = lane 32's v0: the lower half needs the upper half's value
    let code = prog(&[("V_READLANE_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 256), ("SRC1", 160)], None)]);
    let o = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
    let dis = |threads: u32, code: &[u8]| {
        let cs = CsInterface { threads: [threads, 1, 1], user_sgprs: 0, tgid: [false; 3], tg_size: false, tid_components: 1, lds_bytes: 0 };
        super::validate::disassemble(&super::kernel::translate_compute(code, o, &cs).unwrap_or_else(|e| panic!("{threads} threads: {e}")).dxil)
    };
    // (barrier mode 9: the thread group's sync with its memory fence, `ops::barrier::GROUP_SYNC`)
    let syncs = |d: &str| d.matches("@dx.op.barrier(i32 80, i32 9)").count();
    let Ok(alone) = dis(32, &code) else {
        crate::gpu::recomp::runner::skip_test("no DXC");
        return;
    };
    assert_eq!(syncs(&alone), 1, "the start's barrier:
{alone}");
    assert!(alone.matches("atomicrmw").count() < 3, "a half without a partner does not wait for one:
{alone}");
    for threads in [33, 64, 128] {
        let d = dis(threads, &code).unwrap();
        assert_eq!(syncs(&d), 1, "{threads} threads: a group barrier at the exchange:
{d}");
        assert!(d.matches("atomicrmw").count() >= 3, "{threads} threads: the exchange's counters:
{d}");
    }
    // the text before the start's barrier, and after it
    let at_start = |d: &str| d.split_at(d.find("@dx.op.barrier(i32 80, i32 9)").expect("the start's barrier")).0.to_string();
    // the value (`%N`) the first line of `pre` that `hit` accepts defines
    fn def(pre: &str, hit: impl Fn(&str) -> bool) -> Option<String> {
        pre.lines().find(|l| hit(l)).and_then(|l| l.trim().split_once(" = ")).map(|(n, _)| n.to_string())
    }
    // a zero stored through that address in `pre`
    fn zeroed(pre: &str, at: Option<String>) -> bool {
        at.is_some_and(|p| pre.contains(&format!("store i32 0, i32 addrspace(3)* {p},")))
    }
    // without LDS: slots and counters of their own (one console wave: 4 slot words, 2 counters),
    // the counters zeroed before the start's barrier
    let d = dis(64, &code).unwrap();
    assert!(d.contains("@xcl_xs = external addrspace(3) global [4 x i32]") && d.contains("@xcl_xc = external addrspace(3) global [2 x i32]"), "the spin storage:
{d}");
    let pre = at_start(&d);
    assert!(zeroed(&pre, def(&pre, |l| l.contains("addrspace(3)* @xcl_xc, i32 0, i32 %"))), "the counters zeroed before the start's barrier:
{d}");
    // a mesh program's compute pass of one console wave (48 or 64 threads) exchanges the same way
    for threads in [48, 64] {
        let ms = crate::gpu::recomp::MsInterface { outputs: vec![], threads, max_vertices: 64, max_primitives: 64 };
        let d = super::validate::disassemble(&super::kernel::translate_mesh(&code, o, &ms).unwrap_or_else(|e| panic!("mesh, {threads} threads: {e}")).dxil).unwrap();
        assert_eq!(syncs(&d), 1, "mesh, {threads} threads: a group barrier at the exchange:
{d}");
        assert!(d.contains("@xcl_xc = external addrspace(3) global [2 x i32]") && d.matches("atomicrmw").count() >= 3, "mesh, {threads} threads: the exchange's counters:
{d}");
        let pre = at_start(&d);
        assert!(zeroed(&pre, def(&pre, |l| l.contains("addrspace(3)* @xcl_xc, i32 0, i32 %"))), "mesh, {threads} threads: the counters zeroed before the start's barrier:
{d}");
    }
    // with the program's LDS the counters live in its top words
    let with_lds = prog(&[
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 0)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 256), ("SRC1", 160)], None),
    ]);
    let d = dis(64, &with_lds).unwrap();
    assert_eq!(syncs(&d), 1, "a group barrier at the exchange (LDS):
{d}");
    assert!(d.matches("atomicrmw").count() >= 3, "the exchange's counters (LDS):
{d}");
    // the HLSL layout: the thread counter in the top word (8191), the two counters below it
    // (8189, 8190: zeroed before the start's barrier), the 4 slot words below them (from
    // 8192 - 4 - 2 - 1 = 8185), and the program's LDS clamped below the slots (8184)
    let pre = at_start(&d);
    let rest = &d[pre.len()..];
    let index = def(&pre, |l| l.ends_with(", 8189"));
    let counter = index.and_then(|i| def(&pre, |l| l.contains("@xcl_lds, i32 0, i32 ") && l.ends_with(&format!(" {i}"))));
    assert!(pre.contains("@xcl_lds, i32 0, i32 8191") && zeroed(&pre, counter), "the top word and the counters zeroed before the start's barrier:
{d}");
    assert!(rest.contains(", 8185"), "the slots from word 8185:
{d}");
    assert_eq!(rest.matches(", 8189").count(), 2, "this half's and the partner's counter:
{d}");
    assert!(rest.contains("i32 8184)"), "the program's LDS below the exchange:
{d}");
}

/// The split form keeps the console wave's order of LDS accesses (`Lw::lds_in_order`): an LDS
/// read after a write, and a write after a read, meet the other half first (the kernel form's
/// group barrier, a compute group's counters); a read with no write since the last meet does
/// not, and a half without a partner and the native form never do. In a loop the meet is
/// conditional on the bits, and still validates.
#[test]
fn split_lds_reads_meet_after_writes() {
    use crate::gpu::recomp::CsInterface;
    type Inst<'a> = (&'a str, &'a str, &'a [(&'a str, u64)], Option<u32>);
    let write: Inst = ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 0)], None);
    let read: Inst = ("DS_READ_B32", "ENC_DS", &[("VDST", 2), ("ADDR", 1)], None);
    let syncs = |d: &str| d.matches("@dx.op.barrier(i32 80, i32 9)").count();
    let kernel = |list: &[Inst], split: bool| {
        let o = KernelOptions { wave64: true, split, mem_windows: 1, sampler_base: 0 };
        super::validate::disassemble(&translate(&prog(list), o).unwrap_or_else(|e| panic!("{e}")))
    };
    let Ok(rw) = kernel(&[write, read], true) else {
        crate::gpu::recomp::runner::skip_test("no DXC");
        return;
    };
    let w = syncs(&kernel(&[write], true).unwrap());
    assert_eq!(syncs(&rw), w + 1, "the read meets after the write:\n{rw}");
    assert_eq!(syncs(&kernel(&[write, write], true).unwrap()), w, "a write after a write");
    assert_eq!(syncs(&kernel(&[read, write], true).unwrap()), w + 1, "the write meets after the read");
    assert_eq!(syncs(&kernel(&[write, read, read], true).unwrap()), w + 1, "the second read: the halves met since the write");
    assert_eq!(syncs(&kernel(&[write, read, write], true).unwrap()), w + 2, "a meet before the read and before the second write");
    let native = |list: &[Inst]| syncs(&kernel(list, false).unwrap());
    assert_eq!(native(&[write, read]), native(&[write]), "the native form");
    // a loop: read, write, three times – each waits for the other from the second pass on
    let looped = |body: &[Inst]| {
        let mut l: Vec<Inst> = vec![("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 131)], None)];
        l.extend_from_slice(body);
        // back to the loop's first instruction: the DS instructions are 2 words each
        let back = 0x1_0000 - (2 * body.len() as u64 + 3);
        l.extend_from_slice(&[
            ("S_SUB_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None),
            ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None),
        ]);
        let mut code = prog(&l);
        code.truncate(code.len() - 4);
        code.extend(encode("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", back)], None).unwrap());
        code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        let o = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
        syncs(&super::validate::disassemble(&translate(&code, o).unwrap_or_else(|e| panic!("{e}"))).unwrap())
    };
    assert!(looped(&[read, write]) >= looped(&[read]) + 2, "the loop's read and write meet when the bits are set");
    // compute groups: the counters' meet (two more atomics), none without a partner half
    let atomics = |threads: u32, list: &[Inst]| {
        let cs = CsInterface { threads: [threads, 1, 1], user_sgprs: 0, tgid: [false; 3], tg_size: false, tid_components: 1, lds_bytes: 0 };
        let o = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
        let d = super::validate::disassemble(&super::kernel::translate_compute(&prog(list), o, &cs).unwrap_or_else(|e| panic!("{threads} threads: {e}")).dxil).unwrap();
        d.matches("atomicrmw").count()
    };
    for threads in [64, 128] {
        assert_eq!(atomics(threads, &[write, read]), atomics(threads, &[write]) + 2, "{threads} threads");
    }
    assert_eq!(atomics(32, &[write, read]), atomics(32, &[write]), "32 threads: no partner half");
}

#[test]
fn straight_line_and_loop_kernels_validate() {
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 138)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 1)], None),
        ("S_SUB_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 0)], None),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 160), ("VSRC1", 0)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", 257), ("VSRC1", 2)], None),
    ]);
    for (wave64, split) in [(false, false), (true, true), (true, false)] {
        translate(&code, KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 }).unwrap_or_else(|e| panic!("wave64 {wave64} split {split}: {e}"));
    }
}

#[test]
fn latch_blocks_preserve_loop_registers_on_the_gpu() {
    use crate::gpu::recomp::{dxc, runner, Options};
    match runner::wave32_supported() {
        Ok(true) => {}
        Ok(false) => return runner::skip_test("no wave32"),
        Err(e) => return runner::skip_test(&e),
    }
    if dxc::location().is_err() {
        return runner::skip_test("no DXC");
    }
    let two_returns = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 2)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 5)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 130)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xfffb)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 256), ("VSRC1", 1)], None),
        ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xfff9)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 0)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 257), ("SRC1", 129)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 3)], None),
    ]);
    // The inner loop's conditional returns to its own head or falls into the outer head.
    let nested = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)], None),
        ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 4)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 256), ("VSRC1", 1)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)], None),
        ("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 131)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xfffc)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 2)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 3)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 128)], None),
        ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xfff7)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 0)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 257), ("SRC1", 129)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 3)], None),
    ]);
    for (name, code) in [("two returns", two_returns), ("nested", nested)] {
        let cfg = super::cfg::Cfg::build(&code).expect("CFG");
        let latches = cfg.blocks.iter().filter(|b| b.insts.is_empty() && matches!(b.term, super::cfg::Term::Jump(_))).count();
        assert_eq!(latches, if name == "nested" { 2 } else { 1 }, "{name}");
        for (wave64, split) in [(false, false), (true, true)] {
            let direct = translate(&code, KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 }).expect("direct DXIL");
            super::validate::validate(&direct).expect("valid direct DXIL");
            let opt = Options { wave64, split_wave64: split, ..Default::default() };
            let hlsl = crate::gpu::recomp::translate(&code, &opt).expect("HLSL").hlsl;
            let reference = dxc::compile(&hlsl, "main", "cs_6_6", &[]).expect("DXC");
            for count in [0u32, 1, 2, 5] {
                let mut state = runner::KernelState::default();
                state.sgpr[2] = count;
                for (lane, regs) in state.vgpr.iter_mut().enumerate() {
                    regs[0] = lane as u32 + 1;
                    regs[1] = 100 + lane as u32 * 7;
                }
                let got = runner::run(&direct, &state, &[]).expect("direct GPU");
                let want = runner::run(&reference, &state, &[]).expect("HLSL GPU");
                let additions = if name == "nested" { 3 * count } else { count - u32::from(count >= 2) };
                for lane in 0..if wave64 { 64 } else { 32 } {
                    let expected = state.vgpr[lane][1] + additions * state.vgpr[lane][0];
                    assert_eq!(got.vgpr[lane][1], expected, "{name}, wave64={wave64}, count={count}, lane={lane}");
                    assert_eq!(got.vgpr[lane][2], count);
                    assert_eq!(got.vgpr[lane][3], state.vgpr[1][1] + additions * state.vgpr[1][0]);
                    assert_eq!(&got.vgpr[lane][1..4], &want.vgpr[lane][1..4], "HLSL: {name}, count={count}, lane={lane}");
                }
            }
        }
    }
}

/// Development aid: `XCL_DUMP_HLSL=<file>` compiles a compute shader with DXC
/// and prints the disassembly (DXC's lowering of an HLSL intrinsic is the
/// reference for the direct path's).
#[test]
#[ignore]
fn dump_dxc_lowering() {
    let Ok(path) = std::env::var("XCL_DUMP_HLSL") else { return };
    let src = std::fs::read_to_string(path).unwrap();
    let c = crate::gpu::recomp::dxc::compile(&src, "main", "cs_6_6", &[]).unwrap();
    println!("{}", super::validate::disassemble(&c).unwrap());
}

#[test]
fn f64_conversions_validate() {
    for (name, enc) in [("V_CVT_F32_F64", "ENC_VOP1"), ("V_CVT_F64_F32", "ENC_VOP1"), ("V_CVT_I32_F64", "ENC_VOP1"), ("V_CVT_F64_I32", "ENC_VOP1")] {
        let code = prog(&[(name, enc, &[("VDST", 2), ("SRC0", 256)], None)]);
        if let Err(e) = translate(&code, KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 }) {
            panic!("{name}: {}", &e[..e.len().min(600)]);
        }
    }
    let code = prog(&[("V_FMA_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", 256), ("SRC1", 258), ("SRC2", 260)], None)]);
    translate(&code, KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 }).map_err(|e| e[..e.len().min(600)].to_string()).unwrap();
    let code = prog(&[("V_SQRT_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", 256)], None)]);
    translate(&code, KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 }).map_err(|e| e[..e.len().min(600)].to_string()).unwrap();
}

#[test]
fn f64_bisect() {
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("add", prog(&[("V_ADD_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", 256), ("SRC1", 258)], None)])),
        ("fma_k", prog(&[("V_FMA_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", 256), ("SRC1", 258), ("SRC2", 242)], None)])),
        ("fma_v", prog(&[("V_FMA_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", 256), ("SRC1", 258), ("SRC2", 260)], None)])),
    ];
    let mut bad = Vec::new();
    for (n, c) in cases {
        if let Err(e) = translate(&c, KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 }) {
            bad.push(format!("{n}: {}", &e[..e.len().min(200)]));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// A long program (thousands of guarded memory accesses between a register's write and
/// its final store) translates, validates and stays reasonably fast.
#[test]
fn long_programs_translate() {
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = Vec::new();
    list.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 7), ("SRC0", 129)], None));
    for i in 0..2000u64 {
        list.push(("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", vec![("VDST", 2 + (i % 4)), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 4 * (i % 64))], None));
        list.push(("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 1), ("SRC0", 256 + 2 + (i % 4)), ("VSRC1", 1)], None));
    }
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let code = prog(&refs);
    let t = std::time::Instant::now();
    translate(&code, KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 }).unwrap_or_else(|e| panic!("{}", &e[..e.len().min(500)]));
    println!("4001 instructions: {:?}", t.elapsed());
}

/// A direct-translated VS links with an HLSL-path PS and vice versa (both declare their
/// signatures as the HLSL entry structs do), drawing the same pixels as two direct stages.
#[test]
fn direct_and_hlsl_graphics_stages_link() {
    use crate::gpu::dxbc::SigElement as S;
    use crate::gpu::recomp::{runner, translate, Mode, Options, PsInterface, VsInterface};
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32_FLOAT};
    if !super::graphics::tests::draw_ready() {
        return;
    }
    if crate::gpu::recomp::dxc::location().is_err() {
        return runner::skip_test("no DXC");
    }
    let el = |name: &str, index: u32, sv: u32, ct: u32, reg: u32, mask: u8| S { name: name.into(), index, system_value: sv, component_type: ct, register: reg, mask, rw_mask: mask, stream: 0 };
    const V: u64 = 256;
    let vs_code = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
        ("EXP", "ENC_EXP", &[("TGT", 33), ("EN", 0x3), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 0), ("VSRC3", 0)], None),
    ]);
    let vsi = VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 3), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF), el("TEXCOORD", 3, 0, 3, 2, 0x3)],
        attr_base: 4,
        params: None,
    };
    let ps_code = prog(&[
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 1)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 1), ("ATTR", 1), ("ATTRCHAN", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
    ]);
    let psi = PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF), el("TEXCOORD", 3, 0, 3, 2, 0x3)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    };
    let (vo, po) = (Options { mode: Mode::Vertex(vsi.clone()), wave64: true, ..Default::default() }, Options { mode: Mode::Pixel(psi.clone()), wave64: true, ..Default::default() });
    let hvs = crate::gpu::recomp::dxc::compile(&translate(&vs_code, &vo).unwrap().hlsl, "main", "vs_6_6", &[]).unwrap();
    let hps = crate::gpu::recomp::dxc::compile(&translate(&ps_code, &po).unwrap().hlsl, "main", "ps_6_6", &[]).unwrap();
    let ko = KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 };
    let dvs = super::kernel::translate_vertex(&vs_code, ko, &vsi).unwrap().dxil;
    let dps = super::kernel::translate_pixel(&ps_code, ko, &psi).unwrap().dxil;
    let c = [0.8f32, 0.6, 0.4, 1.0];
    let verts: Vec<f32> = [[-1.0f32, -1.0], [-1.0, 3.0], [3.0, -1.0]].iter().flat_map(|p| [p[0], p[1], c[0], c[1], c[2], c[3]]).collect();
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "COLOR", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 8 },
    ];
    let draw = |vs: &[u8], ps: &[u8]| runner::draw(vs, ps, &layout, &verts, 24, &[], &[], 8, [0.0; 4]).unwrap();
    let want = draw(&hvs, &hps);
    assert_eq!(draw(&dvs, &dps), want, "direct VS + direct PS");
    assert_eq!(draw(&dvs, &hps), want, "direct VS + HLSL PS");
    assert_eq!(draw(&hvs, &dps), want, "HLSL VS + direct PS");
}

fn sig(name: &str, index: u32, sv: u32, ct: u32, reg: u32, mask: u8) -> crate::gpu::dxbc::SigElement {
    crate::gpu::dxbc::SigElement { name: name.into(), index, system_value: sv, component_type: ct, register: reg, mask, rw_mask: mask, stream: 0 }
}

/// VS / PS with a branch: the blocks after it are created after the exit block, whose
/// output stores read registers they define (once "forward reference to value N",
/// `module.rs` writing blocks in creation order). Also a PS reading SV_IsFrontFace
/// (declared uint, as the validator demands).
#[test]
fn branchy_graphics_stages_validate() {
    use crate::gpu::recomp::{PsInterface, VsInterface};
    let body = |exp: (&'static str, &'static str, &'static [(&'static str, u64)], Option<u32>)| {
        prog(&[
            ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None),
            ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 242)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 256 + 4)], None),
            exp,
        ])
    };
    let vs_code = body(("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 4), ("VSRC3", 5), ("DONE", 1)], None));
    let vsi = VsInterface { inputs: vec![], outputs: vec![sig("SV_Position", 0, 1, 3, 0, 0xF)], attr_base: 4, params: None };
    let ko = KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 };
    for wave64 in [true, false] {
        super::kernel::translate_vertex(&vs_code, KernelOptions { wave64, ..ko }, &vsi).unwrap_or_else(|e| panic!("VS wave64 {wave64}: {}", &e[..e.len().min(800)]));
    }
    // v2 = front face (SPI_PS_INPUT_ENA bit 12 after the two I/J pairs of 0x3)
    let ps_code = body(("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 2), ("VSRC3", 5), ("DONE", 1), ("VM", 1)], None));
    let psi = PsInterface { inputs: vec![sig("SV_Position", 0, 1, 3, 0, 0xF)], outputs: vec![sig("SV_Target", 0, 64, 3, 0, 0xF)], input_ena: 0x1002, attrs: None };
    for wave64 in [true, false] {
        super::kernel::translate_pixel(&ps_code, KernelOptions { wave64, ..ko }, &psi).unwrap_or_else(|e| panic!("PS wave64 {wave64}: {}", &e[..e.len().min(800)]));
    }
}

/// A long pixel shader (guarded memory accesses make thousands of joins between a
/// register's write and the output stores) translates quickly and stays small: the
/// SSA builder drops trivial phis, `Module::optimize` cleans up the rest.
#[test]
fn long_pixel_shader_translates_fast() {
    use crate::gpu::recomp::PsInterface;
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = Vec::new();
    for r in 4..16u64 {
        list.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", r), ("SRC0", 129 + r)], None));
    }
    for i in 0..1500u64 {
        list.push(("BUFFER_LOAD_DWORD", "ENC_MUBUF", vec![("VDATA", 20 + (i % 4)), ("VADDR", 0), ("SRSRC", 1), ("SOFFSET", 128), ("OFFSET", 4 * (i % 64)), ("OFFEN", 1)], None));
        list.push(("V_ADD_F32", "ENC_VOP2", vec![("VDST", 4 + (i % 4)), ("SRC0", 256 + 20 + (i % 4)), ("VSRC1", 4 + (i % 4))], None));
    }
    list.push(("EXP", "ENC_EXP", vec![("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None));
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let code = prog(&refs);
    let psi = PsInterface { inputs: vec![sig("SV_Position", 0, 1, 3, 0, 0xF)], outputs: vec![sig("SV_Target", 0, 64, 3, 0, 0xF)], input_ena: 0x2, attrs: None };
    let t = std::time::Instant::now();
    let k = super::kernel::translate_pixel(&code, KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 }, &psi).unwrap_or_else(|e| panic!("{}", &e[..e.len().min(800)]));
    println!("3000 instructions: {:?}, {} B", t.elapsed(), k.dxil.len());
}

/// A vertex and a pixel shader that branch before their exports translate (their epilogue then
/// uses values of blocks created after the exit block), and draw what the HLSL path's stages
/// draw, whichever way the branch goes.
#[test]
fn graphics_stages_with_a_branch_before_their_exports() {
    use crate::gpu::dxbc::SigElement as S;
    use crate::gpu::recomp::{runner, translate, Mode, Options, PsInterface, VsInterface};
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32_FLOAT};
    let el = |name: &str, index: u32, sv: u32, ct: u32, reg: u32, mask: u8| S { name: name.into(), index, system_value: sv, component_type: ct, register: reg, mask, rw_mask: mask, stream: 0 };
    const V: u64 = 256;
    // s0 != 0 skips the instruction that clears a colour channel
    let vs_code = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 128)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
        ("EXP", "ENC_EXP", &[("TGT", 33), ("EN", 0x3), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 0), ("VSRC3", 0)], None),
    ]);
    let vsi = VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 3), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF), el("TEXCOORD", 3, 0, 3, 2, 0x3)],
        attr_base: 4,
        params: None,
    };
    let ps_code = prog(&[
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 1)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 1), ("ATTR", 1), ("ATTRCHAN", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 242)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 128)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
    ]);
    let psi = PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF), el("TEXCOORD", 3, 0, 3, 2, 0x3)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    };
    let ko = KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 };
    let dvs = super::kernel::translate_vertex(&vs_code, ko, &vsi).unwrap_or_else(|e| panic!("VS: {}", &e[..e.len().min(600)])).dxil;
    let dps = super::kernel::translate_pixel(&ps_code, ko, &psi).unwrap_or_else(|e| panic!("PS: {}", &e[..e.len().min(600)])).dxil;
    if !super::graphics::tests::draw_ready() {
        return;
    }
    if crate::gpu::recomp::dxc::location().is_err() {
        return runner::skip_test("no DXC");
    }
    let (vo, po) = (Options { mode: Mode::Vertex(vsi.clone()), wave64: true, ..Default::default() }, Options { mode: Mode::Pixel(psi.clone()), wave64: true, ..Default::default() });
    let hvs = crate::gpu::recomp::dxc::compile(&translate(&vs_code, &vo).unwrap().hlsl, "main", "vs_6_6", &[]).unwrap();
    let hps = crate::gpu::recomp::dxc::compile(&translate(&ps_code, &po).unwrap().hlsl, "main", "ps_6_6", &[]).unwrap();
    let c = [0.8f32, 0.6, 0.4, 1.0];
    let verts: Vec<f32> = [[-1.0f32, -1.0], [-1.0, 3.0], [3.0, -1.0]].iter().flat_map(|p| [p[0], p[1], c[0], c[1], c[2], c[3]]).collect();
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "COLOR", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 8 },
    ];
    let draw = |vs: &[u8], ps: &[u8], s0: u32| runner::draw(vs, ps, &layout, &verts, 24, &[s0], &[], 8, [0.0; 4]).unwrap();
    let (cleared, kept) = (draw(&hvs, &hps, 0), draw(&hvs, &hps, 1));
    assert_ne!(cleared, kept, "the branch changes the picture");
    assert_eq!(draw(&dvs, &dps, 0), cleared, "branch not taken");
    assert_eq!(draw(&dvs, &dps, 1), kept, "branch taken");
}

/// Geometry mode (ES+GS, the compute pass of `recomp::Mode::Geometry`): an ES that writes
/// its vertex to the LDS ring, the GS that reads its triangle's vertices, writes them to
/// its output area and emits a strip; split and native forms validate (also the ES or GS
/// alone, as the coverage sweep translates them).
#[test]
fn geometry_mode_validates() {
    use crate::gpu::recomp::GsInterface;
    let e = |n: &str, enc: &str, f: &[(&str, u64)]| encode(n, enc, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let mut es = Vec::new();
    // v20 = lane * ES item (16 B), ring[v20] = v9 (a fetched input)
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)]));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 256 + 20)]));
    es.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 20), ("SRC0", 132), ("VSRC1", 20)]));
    es.extend(e("DS_WRITE_B32", "ENC_DS", &[("ADDR", 20), ("DATA0", 9)]));
    es.extend(e("S_BARRIER", "ENC_SOPP", &[]));
    es.extend(e("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)]));
    let mut gs = Vec::new();
    // v0 = vertex 0's dword offset: read it, write 3 output vertices at s2 + lane * 48
    gs.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 130), ("VSRC1", 0)]));
    gs.extend(e("DS_READ_B32", "ENC_DS", &[("ADDR", 30), ("VDST", 31)]));
    gs.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 32), ("SRC0", 193), ("SRC1", 128)]));
    gs.extend(e("V_MUL_U32_U24", "ENC_VOP2", &[("VDST", 32), ("SRC0", 176), ("VSRC1", 32)]));
    gs.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 32), ("SRC0", 2), ("VSRC1", 32)]));
    for k in 0..3u64 {
        gs.extend(e("DS_WRITE_B32", "ENC_DS", &[("ADDR", 32), ("DATA0", 31), ("OFFSET0", 16 * k)]));
        gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x22)]));
    }
    gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x12)]));
    gs.extend(e("S_ENDPGM", "ENC_SOPP", &[]));
    let sig = |name: &str, sv: u32, reg: u32| crate::gpu::dxbc::SigElement { name: name.into(), index: 0, system_value: sv, component_type: 3, register: reg, mask: 0xF, rw_mask: 0xF, stream: 0 };
    let g = GsInterface {
        es_inputs: vec![sig("POSITION", 0, 0)],
        es_outputs: vec![sig("SV_Position", 1, 0)],
        outputs: vec![sig("SV_Position", 1, 0)],
        max_vertex_count: 3,
        gs_start: Some(es.len() as u64),
    };
    let mut code = es.clone();
    code.extend_from_slice(&gs);
    let mut es_end = es;
    es_end.extend(e("S_ENDPGM", "ENC_SOPP", &[]));
    let cases = [
        ("ES+GS", code, g.clone()),
        ("ES alone", es_end, GsInterface { gs_start: None, ..g.clone() }),
        ("GS alone", gs, GsInterface { gs_start: Some(0), ..g }),
    ];
    for (name, code, g) in cases {
        for split in [true, false] {
            let o = KernelOptions { wave64: true, split, mem_windows: 1, sampler_base: 0 };
            match super::kernel::translate_geometry(&code, o, &g) {
                Ok(_) => {}
                // a split pass with an ES phase is left to the HLSL path (the safe barrier):
                // complete there, and DXC compiles it
                Err(e) if e.contains("split wave64 geometry: the ES phase exchanges at group barriers") => {
                    let t = crate::gpu::recomp::translate_geometry(&code, g.clone(), true, 1).unwrap_or_else(|e| panic!("{name} HLSL: {e}"));
                    assert!(t.complete(), "{name}: {:?}", t.unsupported_why);
                    if crate::gpu::recomp::dxc::location().is_ok() {
                        crate::gpu::recomp::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{name} HLSL: {e}"));
                    }
                }
                Err(e) => panic!("{name} split {split}: {}", &e[..e.len().min(3000)]),
            }
        }
    }
}

/// Exhaustive bit-exact check of the IEEE-754 bit-manipulation formula used by
/// `lower_mem::uf_small_bits` for all 10-bit (mb=5) and 11-bit (mb=6) float inputs.
#[test]
fn uf_small_ieee_bit_math_matches_exact_formula() {
    for mb in [5u32, 6u32] {
        let p = 1u32 << mb;
        let mmask = p - 1;
        let mshift = 23 - mb;
        let den_scale = f32::from_bits(if mb == 6 { 0x3580_0000 } else { 0x3600_0000 });
        for v in 0..(1u32 << (mb + 5)) {
            let e = v >> mb;
            let m = v & mmask;
            // Reference formula (matching HLSL `uf_small`)
            let fm = (m as f32) / (p as f32);
            let ref_f = if e == 31 {
                f32::from_bits(if m != 0 { 0x7FC0_0000 } else { 0x7F80_0000 })
            } else if e == 0 {
                fm * (-14.0f32).exp2()
            } else {
                ((e as f32) - 15.0).exp2() * (1.0 + fm)
            };
            // Exact IEEE bit math
            let got_bits = if e == 31 {
                if m != 0 { 0x7FC0_0000 } else { 0x7F80_0000 }
            } else if e == 0 {
                ((m as f32) * den_scale).to_bits()
            } else {
                ((e + 112) << 23) | (m << mshift)
            };
            assert_eq!(got_bits, ref_f.to_bits(), "mb={mb} v={v:#x} (e={e} m={m})");
        }
    }
}

/// Multi-window switch dispatch, vector rawBufferLoad/Store, hoisted FormatLoad
/// descriptors, single/multi-component FormatLoad, and hoisted BVH loads all pass
/// DXIL validation across 1..=4 memory windows and wave configurations.
#[test]
fn multi_window_mem_format_and_bvh_validate() {
    let code = prog(&[
        // Load descriptor s[12:15] once, then reuse across repeated BUFFER_LOAD_FORMAT_X
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 0), ("OFFSET", 0x40), ("SOFFSET", 125)], None),
        ("S_LOAD_DWORDX8", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 0), ("OFFSET", 0x80), ("SOFFSET", 125)], None),
        ("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 24), ("SBASE", 12), ("OFFSET", 0x10), ("SOFFSET", 125)], None),
        // Divergent EXEC so lane guards are active (non-constant `act`)
        ("V_CMPX_LT_U32", "ENC_VOPC", &[("SRC0", 160), ("VSRC1", 0)], None),
        // Repeated single-component format loads sharing s[12:15] (exercises hoisted FmtComp1 + Load2)
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 5), ("VADDR", 1), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        // Sub-4 and full-4 component format loads + D16 + static MTBUF + small byte load
        ("BUFFER_LOAD_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 6), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 8), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_FORMAT_D16_X", "ENC_MUBUF", &[("VDATA", 12), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_FORMAT_D16_HI_X", "ENC_MUBUF", &[("VDATA", 12), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("TBUFFER_LOAD_FORMAT_X", "ENC_MTBUF", &[("VDATA", 13), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("FORMAT", 28), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_UBYTE", "ENC_MUBUF", &[("VDATA", 14), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        // Vector MUBUF & FLAT loads/stores
        ("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", &[("VDATA", 16), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_STORE_DWORDX4", "ENC_MUBUF", &[("VDATA", 16), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("GLOBAL_LOAD_DWORDX4", "ENC_FLAT_GLBL", &[("VDST", 20), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 0)], None),
        ("GLOBAL_STORE_DWORDX4", "ENC_FLAT_GLBL", &[("DATA", 20), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 16)], None),
        ("GLOBAL_LOAD_USHORT", "ENC_FLAT_GLBL", &[("VDST", 24), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 4)], None),
        // BVH node intersection (box32 / box16 / tri hoisted vector loads)
        ("IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", &[("VDATA", 28), ("VADDR", 0), ("SRSRC", 3), ("DMASK", 0xF)], None),
    ]);
    for windows in [1u32, 2, 3, 4] {
        for (wave64, split) in [(true, false), (false, false)] {
            translate(&code, KernelOptions { wave64, split, mem_windows: windows, sampler_base: 0 })
                .unwrap_or_else(|e| panic!("w={windows} wave64={wave64}: {}", &e[..e.len().min(800)]));
            if windows > 1 {
                translate(&code, KernelOptions { wave64, split, mem_windows: windows | crate::gpu::recomp::HEAP_WINDOWS, sampler_base: 0 })
                    .unwrap_or_else(|e| panic!("heap w={windows} wave64={wave64}: {}", &e[..e.len().min(800)]));
            }
        }
    }
}

/// `image_get_lod` with a static sampler (an S# built from constants in the code) translates
/// to `CalculateLevelOfDetail` on the sampler bound at `s<i>, space1`, shared with a sample of
/// the same S#, and validates.
#[test]
fn ps_get_lod_with_a_static_sampler() {
    use crate::gpu::recomp::{Mode, PsInterface};
    let clamp = 2 | 2 << 3 | 2 << 6;
    let linear = [clamp, 0x00FF_F000u32, 1 << 20 | 1 << 22, 0];
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = Vec::new();
    for (k, &w) in linear.iter().enumerate() {
        list.push(if w == 0 { ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 24 + k as u64), ("SSRC0", 128)], None) } else { ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 24 + k as u64), ("SSRC0", 255)], Some(w)) });
    }
    // uv = (I, J) of the first barycentric pair (v0, v1); T# s[16:23], S# s[24:27]
    list.push(("IMAGE_GET_LOD", "ENC_MIMG", vec![("VDATA", 4), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x3), ("DIM", 1)], None));
    list.push(("IMAGE_SAMPLE", "ENC_MIMG", vec![("VDATA", 6), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x3), ("DIM", 1)], None));
    list.push(("EXP", "ENC_EXP", vec![("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None));
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let code = prog(&refs);
    let psi = PsInterface { inputs: vec![sig("SV_Position", 0, 1, 3, 0, 0xF)], outputs: vec![sig("SV_Target", 0, 64, 3, 0, 0xF)], input_ena: 0x2, attrs: None };
    let ko = KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 };
    let k = super::kernel::translate_pixel(&code, ko, &psi).unwrap_or_else(|e| panic!("{}", &e[..e.len().min(800)]));
    assert_eq!(k.static_samplers, vec![(linear, false)], "one static sampler, shared by get_lod and sample");
    if crate::gpu::recomp::dxc::location().is_ok() {
        super::coverage::translate_mode(&code, &Mode::Pixel(psi), ko, true, &[]).unwrap_or_else(|e| panic!("{}", &e[..e.len().min(800)]));
    }
}

/// Without a program-set `MODE[3:2]`, `v_cvt_f16_f32`, `v_mul_f16` and `v_pk_mul_f16`
/// must round to nearest-even (matching RDNA2 and the HLSL path) rather than using
/// `dx.op.legacyF32ToF16` (which rounds toward zero / flushes below the smallest f16 denormal).
#[test]
fn f16_conversions_and_products_default_to_round_to_nearest_even() {
    use crate::gpu::recomp::runner::{self, KernelState};
    const V: u64 = 256;
    let code = prog(&[
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V)], None),
        ("V_MUL_F16", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 1), ("VSRC1", 2)], None),
        ("V_PK_MUL_F16", "ENC_VOP3P", &[("VDST", 12), ("SRC0", V + 1), ("SRC1", V + 2), ("OP_SEL_HI", 3)], None),
    ]);
    let w32_dxil = translate(&code, KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 }).unwrap();
    let split_dxil = translate(&code, KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 }).unwrap();
    let w64_dxil = translate(&code, KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 }).unwrap();
    if let Ok(dis) = super::validate::disassemble(&w32_dxil) {
        assert!(!dis.contains("legacyF32ToF16"), "default RNE f16 ops should not emit legacyF32ToF16:\n{dis}");
    }

    let (has_w32, has_w64) = match (runner::wave32_supported(), runner::wave64_supported()) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => return runner::skip_test(&e),
    };
    if !has_w32 && !has_w64 {
        return runner::skip_test("no wave32 or wave64 GPU");
    }

    let cvt_cases: [(f32, u16); 8] = [
        (f32::from_bits(3075.0f32.to_bits() - 1), 0x6A01),
        (3075.0, 0x6A02),
        (f32::from_bits(3075.0f32.to_bits() + 1), 0x6A02),
        (3081.0, 0x6A04),
        (-3075.0, 0xEA02),
        (65520.0, 0x7C00),
        (f32::from_bits(0x3300_0000), 0x0000),
        (f32::from_bits(0x3300_0001), 0x0001),
    ];
    // Lane 0: 3072 * (1 + 2^-10) = 3075.0 (exact midpoint -> 0x6A02 / 0xEA02).
    // Lane 1: 2^-24 * (0.5 + 2^-11) = 2^-25 + 2^-35 (above half smallest f16 denormal -> 0x0001 / 0x8001).
    let mul_cases: [(u32, u32, u16, u32); 2] = [
        (0xEA00_6A00, 0x3C01_3C01, 0x6A02, 0xEA02_6A02),
        (0x8001_0001, 0x3801_3801, 0x0001, 0x8001_0001),
    ];

    let mut runs: Vec<(&str, &[u8], bool)> = Vec::new();
    if has_w32 {
        runs.push(("wave32", &w32_dxil, false));
        runs.push(("split-wave64", &split_dxil, true));
    }
    if has_w64 {
        runs.push(("native-wave64", &w64_dxil, true));
    }
    for (label, dxil, wave64) in runs {
        let mut st = KernelState::default();
        if !wave64 {
            st.sgpr[127] = 0;
        }
        let bases: &[usize] = if wave64 { &[0, 32] } else { &[0] };
        for &base in bases {
            for (i, &(value, _)) in cvt_cases.iter().enumerate() {
                st.vgpr[base + i][0] = value.to_bits();
            }
            for (i, &(a, b, _, _)) in mul_cases.iter().enumerate() {
                st.vgpr[base + i][1] = a;
                st.vgpr[base + i][2] = b;
            }
        }
        let r = runner::run(dxil, &st, &[]).unwrap_or_else(|e| panic!("{label}: {e}"));
        for &base in bases {
            let actual: [u16; 8] = std::array::from_fn(|i| r.vgpr[base + i][10] as u16);
            let expected = cvt_cases.map(|(_, bits)| bits);
            assert_eq!(actual, expected, "{label} base {base}: v_cvt_f16_f32 {actual:04x?} != {expected:04x?}");
            for (i, &(_, _, want_mul, want_pk)) in mul_cases.iter().enumerate() {
                let got_mul = r.vgpr[base + i][11] as u16;
                let got_pk = r.vgpr[base + i][12];
                assert_eq!(got_mul, want_mul, "{label} lane {}: v_mul_f16 {got_mul:04x} != {want_mul:04x}", base + i);
                assert_eq!(got_pk, want_pk, "{label} lane {}: v_pk_mul_f16 {got_pk:08x} != {want_pk:08x}", base + i);
            }
        }
    }
}

/// `s_setreg_b32 hwreg(HW_REG_MODE, 3, 1), s0` sets bit 3 of `MODE` (offset 3, dp_mask 2):
/// lowering must not subtract `2 - off` before matching `dp_mask`, and the dynamic f16 rounding
/// result must agree with HLSL on the GPU.
#[test]
fn s_setreg_b32_mode_bit3_does_not_underflow() {
    use crate::gpu::recomp::{dxc, runner, Options};
    // HW_REG_MODE (1), offset 3, size-1 = 0 -> 1 | (3 << 6) = 0x00C1
    let code = prog(&[
        ("S_SETREG_B32", "ENC_SOPK", &[("SDST", 0), ("SIMM16", 0x00C1)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 257)], None),
    ]);
    let ko = KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 };
    let dxil = translate(&code, ko).unwrap_or_else(|e| panic!("direct DXIL failed: {e}"));
    if runner::wave32_supported() != Ok(true) {
        return runner::skip_test("no wave32");
    }
    if dxc::location().is_err() {
        return runner::skip_test("no DXC");
    }
    let mut st = runner::KernelState::default();
    st.sgpr[0] = 1; // MODE[3:2] = 0b10 (RTN, round toward -inf)
    st.vgpr[0][1] = 0x3F80_0001; // +1.0 + 2^-23 -> 0x3C00
    st.vgpr[1][1] = 0xBF80_0001; // -1.0 - 2^-23 -> 0xBC01
    let direct = runner::run(&dxil, &st, &[]).unwrap();
    let t = crate::gpu::recomp::translate(&code, &Options { wave64: false, ..Default::default() }).unwrap();
    let hlsl = runner::run(&dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap(), &st, &[]).unwrap();
    assert_eq!((direct.vgpr[0][0], direct.vgpr[1][0]), (hlsl.vgpr[0][0], hlsl.vgpr[1][0]));
    assert_eq!((direct.vgpr[0][0], direct.vgpr[1][0]), (0x3C00, 0xBC01));
}

/// A 16-byte `buffer_store_format_xyzw` (32_32_32_32_FLOAT) at an address that is not
/// dword aligned writes all four components; the bytes around it stay intact.
#[test]
fn gpu_direct_unaligned_multi_dword_format_store() {
    use crate::gpu::recomp::runner::{self, KernelState};
    // V# s[12:15] (SRSRC 3): base 0x100, stride 0, 64 bytes, format 77; lane 0 stores at +2
    let code = prog(&[
        ("V_CMPX_EQ_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 0)], None),
        ("BUFFER_STORE_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 10), ("VADDR", 1), ("SRSRC", 3), ("SOFFSET", 128), ("OFFEN", 1)], None),
    ]);
    let dxil = runner::direct_kernel(&code, true).unwrap_or_else(|e| panic!("direct translation: {e}"));
    super::validate::validate(&dxil).unwrap_or_else(|e| panic!("validator: {e}"));
    if runner::adapter().is_err() || runner::wave64_programs_supported() != Ok(true) || crate::gpu::recomp::dxc::location().is_err() {
        runner::skip_test("no GPU / DXC / wave64");
        return;
    }
    let vals = [1.0f32, 2.0, 3.0, 4.0].map(f32::to_bits);
    let mut st = KernelState::default();
    st.sgpr[12..16].copy_from_slice(&[0x100, 0, 64, (4 | 5 << 3 | 6 << 6 | 7 << 9) | 77 << 12]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        r[1] = 2;
        r[10..14].copy_from_slice(&vals);
    }
    let fill = 0xDEAD_BEEFu32;
    let r = runner::run(&dxil, &st, &vec![fill; 128]).expect("GPU run");
    let mut want: Vec<u8> = std::iter::repeat_n(fill.to_le_bytes(), 128).flatten().collect();
    want[0x102..0x112].copy_from_slice(&vals.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let got: Vec<u8> = r.mem.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(&got[0xFC..0x118], &want[0xFC..0x118], "bytes 0xFC..0x118 after the store at 0x102");
}

#[test]
fn window_at_lowers_to_32bit_carry_without_int64() {
    let ko = KernelOptions { wave64: false, split: false, mem_windows: 4, sampler_base: 0 };
    let with_adds = prog(&[
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 1), ("ADDR", 0), ("SADDR", 2), ("OFFSET", 16)], None),
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 2), ("ADDR", 0), ("SADDR", 2), ("OFFSET", 0xFFC)], None),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 10), ("SBASE", 2), ("OFFSET", 0x1F_FFFC), ("SOFFSET", 4)], None),
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 3), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1), ("OFFEN", 1), ("OFFSET", 4)], None),
    ]);
    let dxil = translate(&with_adds, ko).expect("multi-window with adds translates");
    super::validate::validate(&dxil).expect("validates without Int64Ops");
    if let Ok(dis) = super::validate::disassemble(&dxil) {
        let body = dis.split("define void @main()").nth(1).and_then(|s| s.split("\n}\n").next()).expect("main body");
        assert!(!dis.contains("64-bit integers") && !body.contains("i64"), "window_at with adds must not emit i64:\n{dis}");
        assert!(!dis.contains("binaryWithCarryOrBorrow"), "window_at should avoid uaddc struct overhead:\n{dis}");
        assert!(body.contains("icmp ult i32"), "expected scalar 32-bit carry compare in DXIL:\n{dis}");
        assert!(dis.contains("dx.op.binaryWithTwoOuts.i32"), "expected UMul in DXIL:\n{dis}");
    }

    let zero_off = prog(&[
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 1), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 0)], None),
    ]);
    let dxil0 = translate(&zero_off, ko).expect("zero-offset multi-window translates");
    super::validate::validate(&dxil0).expect("zero-offset validates");
    if let Ok(dis0) = super::validate::disassemble(&dxil0) {
        let body0 = dis0.split("define void @main()").nth(1).and_then(|s| s.split("\n}\n").next()).expect("main body");
        assert!(!dis0.contains("64-bit integers") && !body0.contains("i64"), "zero-offset window_at must not emit i64:\n{dis0}");
        assert!(!body0.contains("icmp ult"), "zero-offset window_at must not emit carry compare:\n{dis0}");
    }
}

/// A loop that only ends when its counter wraps to 0, then global_store_dword [v0 * 4], the
/// counter (the recompiler tests' `endless_loop_program`).
fn endless_loop() -> Vec<u8> {
    prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 10)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 4), ("SADDR", 0x7D)], None),
    ])
}

/// The same loop inside one of three passes (`endless_loop_entered_three_times`).
fn endless_loop_entered_three_times() -> Vec<u8> {
    prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 11), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 12), ("SSRC1", 10)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 11), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 11), ("SSRC1", 131)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFF8)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 12)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 4), ("SADDR", 0x7D)], None),
    ])
}

/// The blocks of the emitted `main`: (label, predecessors, text); the entry block is `%0`.
fn main_blocks(dis: &str) -> Vec<(String, Vec<String>, String)> {
    let body = dis.split("define void @main()").nth(1).and_then(|s| s.split("\n}\n").next()).expect("main body");
    let mut out = vec![("%0".to_string(), Vec::new(), String::new())];
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("; <label>:") {
            let (label, preds) = rest.split_once(';').unwrap_or((rest, ""));
            let preds = preds.trim().strip_prefix("preds = ").unwrap_or("").split(", ").filter(|p| !p.is_empty()).map(str::to_string).collect();
            out.push((format!("%{}", label.trim()), preds, String::new()));
        } else {
            let last = out.last_mut().unwrap();
            last.2.push_str(line);
            last.2.push('\n');
        }
    }
    out
}

/// What a direct compute program's text shows of its loop counts (`kernel::loop_passes`):
/// (loads of `IO_LOOP_BUDGET`, `sub ..., 1` decrements, and with one load the count `cap`
/// made of it, the phis that take it from the entry block, the other blocks they take it from).
fn loop_counts(dis: &str) -> (usize, usize, Option<(String, Vec<String>, Vec<String>)>) {
    let slot = format!("i32 {}, i32 undef, i8 1, i32 4)", crate::gpu::recomp::IO_LOOP_BUDGET * 4);
    let loads: Vec<&str> = dis.lines().filter(|l| l.contains("@dx.op.rawBufferLoad.i32(") && l.contains(&slot)).collect();
    let subs = dis.lines().filter(|l| l.trim_end().ends_with(", 1") && l.contains(" = sub i32 %")).count();
    let [load] = loads[..] else { return (loads.len(), subs, None) };
    let name = |l: &str| l.trim().split_once(" = ").map(|(n, _)| n.to_string()).expect("a definition");
    let def = |hit: &str| name(dis.lines().find(|l| l.contains(hit)).unwrap_or_else(|| panic!("no {hit}:\n{dis}")));
    // %v = extractvalue (the load), %z = icmp eq i32 %v, 0, %cap = select i1 %z, i32 65536, …
    let v = def(&format!("extractvalue %dx.types.ResRet.i32 {}, 0", name(load)));
    let z = def(&format!("= icmp eq i32 {v}, 0"));
    let cap = def(&format!("= select i1 {z}, i32 65536, i32 %"));
    let at_entry = dis.lines().filter(|l| l.contains(" = phi i32 ") && l.contains(&format!("[ {cap}, %0 ]"))).map(name).collect();
    let again = main_blocks(dis).into_iter().filter(|(b, _, _)| b != "%0" && dis.contains(&format!("[ {cap}, {b} ]"))).map(|(b, _, _)| b).collect();
    (1, subs, Some((cap, at_entry, again)))
}

/// Whether the count `n` leaves the program at 0: `%z = icmp eq i32 n, 0` drives a `br i1 %z`
/// whose taken label is the block that returns.
fn leaves_to_ret(dis: &str, n: &str) -> bool {
    let name = |l: &str| l.trim().split_once(" = ").map(|(n, _)| n.to_string());
    let Some(z) = dis.lines().find(|l| l.contains(&format!("= icmp eq i32 {n}, 0"))).and_then(name) else { return false };
    let br = format!("br i1 {z}, label ");
    let Some(to) = dis.lines().find_map(|l| l.trim().strip_prefix(br.as_str())).and_then(|r| r.split(',').next()) else { return false };
    main_blocks(dis).iter().any(|(b, _, text)| b == to && text.contains("ret void"))
}

/// The count phis of `dis` that take `cap` from block `b`.
fn count_phis(dis: &str, cap: &str, b: &str) -> Vec<String> {
    dis.lines().filter(|l| l.contains(" = phi i32 ") && l.contains(&format!("[ {cap}, {b} ]"))).filter_map(|l| l.trim().split_once(" = ").map(|(n, _)| n.to_string())).collect()
}

/// The direct translator counts each entry of a compute program's loop as the HLSL path does
/// (`only_compute_programs_with_a_loop_read_the_dispatch_loop_budget`): `IO_LOOP_BUDGET` is
/// loaded once, every loop has a count besides the lane's budget, and the outer loop's head
/// sets the inner loop's count back. A loop-free compute program and other entries have neither.
#[test]
fn direct_compute_loops_count_each_entry() {
    use crate::gpu::recomp::CsInterface;
    let cs = CsInterface { threads: [8, 4, 2], user_sgprs: 2, tgid: [false; 3], tg_size: false, tid_components: 1, lds_bytes: 0 };
    let straight = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 2)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 4), ("SADDR", 0x7D)], None),
    ]);
    for (wave64, split) in [(false, false), (true, true)] {
        let o = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
        let dis = |code: &[u8]| super::validate::disassemble(&super::kernel::translate_compute(code, o, &cs).unwrap_or_else(|e| panic!("{e}")).dxil);
        let Ok(one) = dis(&endless_loop()) else {
            return crate::gpu::recomp::runner::skip_test("no DXC");
        };
        // one loop: its count, set at the entry, and the lane's budget, both counted down at its head
        let (loads, subs, counts) = loop_counts(&one);
        let (_, at_entry, again) = counts.unwrap_or_default();
        assert_eq!((loads, subs, at_entry.len(), again.len()), (1, 2, 1, 0), "wave64 {wave64}:\n{one}");
        assert!(leaves_to_ret(&one, &at_entry[0]), "wave64 {wave64}: a count of 0 in {} ends the program:\n{one}", at_entry[0]);
        // nested: the outer count set at the entry; the inner one set back to `cap` on a path
        // from the outer loop's head, after the outer count was counted down there
        let nested = dis(&endless_loop_entered_three_times()).unwrap();
        let (loads, subs, counts) = loop_counts(&nested);
        let (cap, at_entry, again) = counts.unwrap_or_default();
        assert_eq!((loads, subs, at_entry.len(), again.len()), (1, 4, 1, 1), "wave64 {wave64}:\n{nested}");
        let inner = count_phis(&nested, &cap, &again[0]);
        assert_eq!(inner.len(), 1, "wave64 {wave64}:\n{nested}");
        for n in [&at_entry[0], &inner[0]] {
            assert!(leaves_to_ret(&nested, n), "wave64 {wave64}: a count of 0 in {n} ends the program:\n{nested}");
        }
        let blocks = main_blocks(&nested);
        let mut b = again[0].clone();
        let after_outer_head = loop {
            let (_, preds, text) = blocks.iter().find(|(l, _, _)| *l == b).expect("the block");
            if text.contains(&format!("= sub i32 {}, 1", at_entry[0])) {
                break true;
            }
            match &preds[..] {
                [p] => b = p.clone(),
                _ => break false,
            }
        };
        assert!(after_outer_head, "wave64 {wave64}: the inner count is set back in {} after {} - 1:\n{nested}", again[0], at_entry[0]);
        // neither for a loop-free compute program nor for a kernel with the loop
        let none = dis(&straight).unwrap();
        assert_eq!(loop_counts(&none), (0, 0, None), "wave64 {wave64}:\n{none}");
        let kernel = super::validate::disassemble(&translate(&endless_loop(), o).unwrap()).unwrap();
        assert_eq!(loop_counts(&kernel), (0, 1, None), "wave64 {wave64}:\n{kernel}");
    }
}

/// The compute entry loads the dispatch's thread limits (`recomp::IO_THREAD_LIMIT`) in one
/// 3-dword load and lets only the lanes below them into the program. Without a group barrier the
/// lanes past them leave at once and the running lanes' vector writes stay unguarded, as before
/// (the one conditional branch is the entry's); with one they stay, out of EXEC, and the store
/// is guarded. Wave32, wave64 and the split form (with and without a partner half).
#[test]
fn the_compute_entry_lets_only_the_lanes_below_the_thread_limits_in() {
    use crate::gpu::recomp::{CsInterface, IO_THREAD_LIMIT};
    let code = |barrier: bool| {
        let mut l: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = vec![
            ("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 4), ("SRC0", 2), ("SRC1", 134), ("SRC2", 256)], None),
            ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 4)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None),
            ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 4), ("SADDR", 0x7D)], None),
        ];
        if barrier {
            l.insert(0, ("S_BARRIER", "ENC_SOPP", &[], None));
        }
        prog(&l)
    };
    let load = format!("i32 {}, i32 undef, i8 7, i32 4)", IO_THREAD_LIMIT[0] * 4);
    for (wave64, split, threads) in [(false, false, 32), (true, false, 64), (true, true, 64), (true, true, 32)] {
        let o = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
        let cs = CsInterface { threads: [threads, 1, 1], user_sgprs: 2, tgid: [true, false, false], tg_size: false, tid_components: 1, lds_bytes: 0 };
        // (a split wave64 of 64 threads is two hardware waves: its barrier is the HLSL path's)
        for barrier in [false, true].into_iter().filter(|&b| !(b && split && threads == 64)) {
            let what = format!("wave64 {wave64}, split {split}, {threads} threads, barrier {barrier}");
            let k = super::kernel::translate_compute(&code(barrier), o, &cs).unwrap_or_else(|e| panic!("{what}: {e}"));
            let Ok(dis) = super::validate::disassemble(&k.dxil) else {
                return crate::gpu::recomp::runner::skip_test("no DXC");
            };
            assert_eq!(dis.matches(&load).count(), 1, "{what}: the limits in one load\n{dis}");
            if !split || threads == 32 {
                assert_eq!(dis.matches("br i1 ").count(), if barrier { 2 } else { 1 }, "{what}\n{dis}");
                // the entry's branch: on the group (a folded count) and, without a barrier, the
                // lane's place inside the limits too
                let cond = dis.lines().find_map(|l| l.trim().strip_prefix("br i1 ")).and_then(|r| r.split(',').next()).unwrap();
                let def = dis.lines().find(|l| l.trim().starts_with(&format!("{cond} = "))).unwrap();
                assert_eq!(def.contains("= and i1 "), !barrier, "{what}: {def}\n{dis}");
            }
        }
    }
}

/// A loop of `passes` passes (0: one that only ends when its counter wraps to 0) inside one
/// of `entries` passes; each pass stores the inner passes run so far to [v0 * 4], so what a
/// thread stored last is what it ran before its loops (or its program) ended.
fn counted_loops(passes: u32, entries: u32) -> Vec<u8> {
    prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 11), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 128)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None),
        // the outer loop
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)], None),
        // the inner loop
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 12), ("SSRC1", 129)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 12)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 4), ("SADDR", 0x7D)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 255)], Some(passes)),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFF8)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 11), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 11), ("SSRC1", 255)], Some(entries)),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFF3)], None),
    ])
}

/// One loop that only ends when its counter wraps to 0, storing its passes so far to [v0 * 4]
/// in each.
fn endless_storing_loop() -> Vec<u8> {
    prog(&[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 10)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 4), ("SADDR", 0x7D)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFA)], None),
    ])
}

/// The direct translator's compute loops on the GPU (the HLSL path's
/// `gpu_compute_loop_passes_follow_the_dispatch`): an entry of a loop runs the passes the
/// dispatch's share gives a thread (`IO_LOOP_BUDGET` among the group's threads, between 1024
/// and 65 536; 0: the lane's 65 536), the same in every thread, in wave32 and split wave64. A
/// loop that runs out ends the thread's program (the HLSL path goes on after it): the last
/// store shows where. Each entry of an inner loop has a fresh count: three entries of 1024
/// passes run under a share of 1024, and a 1025th pass in one entry ends the program.
#[test]
fn gpu_direct_compute_loop_passes_follow_the_dispatch() {
    use crate::gpu::recomp::{runner, CsInterface, IO_LOOP_BUDGET, KERNEL_SGPRS};
    match runner::wave32_supported() {
        Ok(true) => {}
        Ok(false) => return runner::skip_test("no wave32"),
        Err(e) => return runner::skip_test(&e),
    }
    let threads = 64u32;
    let cs = CsInterface { threads: [threads, 1, 1], user_sgprs: 2, tgid: [false; 3], tg_size: false, tid_components: 1, lds_bytes: 0 };
    for wave64 in [false, true] {
        let o = KernelOptions { wave64, split: wave64, mem_windows: 1, sampler_base: 0 };
        let run = |code: &[u8], slot: u32| -> u32 {
            let k = super::kernel::translate_compute(code, o, &cs).unwrap_or_else(|e| panic!("{e}"));
            let mut io = vec![0u32; KERNEL_SGPRS];
            io[IO_LOOP_BUDGET] = slot;
            let mem = runner::dispatch(&k.dxil, [1, 1, 1], &io, &vec![0xFFFF_FFFFu32; threads as usize]).expect("GPU run");
            for (t, &got) in mem.iter().take(threads as usize).enumerate() {
                assert_eq!(got, mem[0], "wave64 {wave64}, slot {slot}: thread {t}");
            }
            mem[0]
        };
        // (the slot, passes): not known; 5000 a thread; shares below and above the bounds
        for (slot, passes) in [(0u32, 65_536u32), (threads * 5000, 5000), (1, 1024), (u32::MAX, 65_536)] {
            assert_eq!(run(&endless_storing_loop(), slot), passes, "wave64 {wave64}, slot {slot}");
        }
        assert_eq!(run(&counted_loops(1024, 3), 1), 3 * 1024, "wave64 {wave64}: three entries of 1024");
        assert_eq!(run(&counted_loops(1025, 3), 1), 1024, "wave64 {wave64}: a 1025th pass");
        assert_eq!(run(&counted_loops(1025, 3), 0), 3 * 1025, "wave64 {wave64}: no share");
        assert_eq!(run(&counted_loops(0, 1), 1), 1024, "wave64 {wave64}: an endless inner loop");
    }
}

/// Constant data after the program (`kernel::translate_entry_rodata`), as the HLSL kernel's
/// `gpu_loads_read_the_constant_data_after_the_program`: `s_getpc_b64` plus the compiler's
/// offset is its address, and `s_load` / `global_load` (with windows `s_buffer_load` too) read
/// it from the module's table – with one window, two windows (switch) and heap windows, where
/// window 0 holds other bytes at the table's offset. Any other address still reads the memory.
#[test]
fn gpu_direct_loads_read_the_constant_data_after_the_program() {
    use super::kernel::{translate_entry_rodata, Entry};
    use crate::gpu::recomp::{runner, HEAP_WINDOWS};
    let body = |table: u32| {
        prog(&[
            ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 0)], None), // s[0:1] = 4
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 255)], Some(table.wrapping_sub(4))),
            ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 128)], None),
            ("GLOBAL_LOAD_DWORDX3", "ENC_FLAT_GLBL", &[("VDST", 5), ("ADDR", 10), ("SADDR", 0), ("OFFSET", 0)], None),
            ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 0), ("OFFSET", 4), ("SOFFSET", 125)], None),
            ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 24), ("SBASE", 0), ("OFFSET", 8), ("SOFFSET", 125)], None),
            // memory, not the data: v[12:13] and s[2:3]
            ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 8), ("ADDR", 12), ("SADDR", 0x7D), ("OFFSET", 0)], None),
            ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 28), ("SBASE", 1), ("OFFSET", 0), ("SOFFSET", 125)], None),
            // the V# s[4:7]
            ("S_BUFFER_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 30), ("SBASE", 2), ("OFFSET", 4), ("SOFFSET", 125)], None),
        ])
    };
    // as in retail data: program, guard s_endpgm, block (u32 N, N dwords, s_endpgm)
    let block = body(0).len() as u32 + 4;
    let code = body(block + 4);
    let table: Vec<u32> = (0..8).map(|i| 0x7AB1_E000 + i).collect();
    let mut data = vec![table.len() as u32];
    data.extend(&table);
    data.push(0xBF81_0000);
    let rodata = vec![(block, data)];
    let split = runner::split_wave64();
    let direct = |windows: u32, rodata: &[(u32, Vec<u32>)]| {
        let o = KernelOptions { wave64: true, split, mem_windows: windows, sampler_base: 0 };
        translate_entry_rodata(&code, o, Entry::Kernel, rodata).unwrap_or_else(|e| panic!("windows {windows:#x}: {}", &e[..e.len().min(800)]))
    };
    // (translations are validated in test builds)
    let one = direct(1, &rodata).dxil;
    assert_eq!(direct(1, &[]).dxil, translate(&code, KernelOptions { wave64: true, split, mem_windows: 1, sampler_base: 0 }).unwrap(), "no data, no table");
    assert_ne!(one, direct(1, &[]).dxil);
    match (runner::adapter(), runner::wave64_programs_supported()) {
        (Ok(_), Ok(true)) => {}
        (a, w) => return runner::skip_test(if a.is_err() || w.is_err() { "no GPU" } else { "no wave64" }),
    }
    let check = |r: &runner::KernelResult, buffer: [u32; 2], what: &str| {
        assert_eq!(r.sgpr[20], table[1], "{what}: s_load_dword of the table");
        assert_eq!(r.sgpr[24..28], table[2..6], "{what}: s_load_dwordx4 of the table");
        assert_eq!(r.sgpr[28..30], [4242, 4343], "{what}: s_load from memory");
        assert_eq!(r.sgpr[30..32], buffer, "{what}: s_buffer_load");
        for l in 0..64usize {
            let k = l % 6;
            assert_eq!(r.vgpr[l][5..8], table[k..k + 3], "{what}, lane {l}: three dwords of the table from {k}");
            assert_eq!(r.vgpr[l][8], 7000 + l as u32, "{what}, lane {l}: memory outside the data");
        }
    };
    // one window: nothing of the code is in the memory (and a V# is always memory)
    let mut st = runner::KernelState::default();
    st.sgpr[2] = 0x900;
    st.sgpr[4..7].copy_from_slice(&[0x900, 0, 64]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[10] = 4 * (l as u32 % 6);
        r[12] = 0x800 + 4 * l as u32;
    }
    let mut mem = vec![0u32; 1024];
    mem[0x240..0x243].copy_from_slice(&[4242, 4343, 4444]);
    for l in 0..64 {
        mem[0x200 + l] = 7000 + l as u32;
    }
    check(&runner::run(&one, &st, &mem).unwrap(), [4343, 4444], "one window");
    // several windows: a code address is in none of them, and window 0 holds other bytes at
    // the table's offset; the V# is on the table
    let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    let word = |a: u64| unsafe { &mut *(a as *mut u32) };
    for k in 0..16u64 {
        *word(w0 + block as u64 + 4 * k) = 0xBAD0_0000 + k as u32;
    }
    *word(w1 + 0x900) = 4242;
    *word(w1 + 0x904) = 4343;
    st.sgpr[2..4].copy_from_slice(&[(w1 + 0x900) as u32, (w1 >> 32) as u32]);
    st.sgpr[4..7].copy_from_slice(&[block + 4, crate::gpu::recomp::CODE_ADDRESS_HI, 64]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        *word(w1 + 0x800 + 4 * l as u64) = 7000 + l as u32;
        r[12] = (w1 + 0x800 + 4 * l as u64) as u32;
        r[13] = (w1 >> 32) as u32;
    }
    for (windows, what) in [(2, "two windows"), (2 | HEAP_WINDOWS, "heap windows")] {
        check(&runner::run_in_arena(&direct(windows, &rodata).dxil, &st, &arena).unwrap(), [table[1], table[2]], what);
    }
}

#[test]
fn dot8_u32_u4_unclamped_lowers_to_dot4_add_u8_packed() {
    use crate::gpu::recomp::{dxc, runner};
    const V: u64 = 256;
    let unclamped = prog(&[("V_DOT8_U32_U4", "ENC_VOP3P", &[("VDST", 3), ("SRC0", V), ("SRC1", V + 1), ("SRC2", V + 2)], None)]);
    let clamped = prog(&[("V_DOT8_U32_U4", "ENC_VOP3P", &[("VDST", 3), ("SRC0", V), ("SRC1", V + 1), ("SRC2", V + 2), ("CLAMP", 1)], None)]);
    let ko = KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 };
    let dxil = translate(&unclamped, ko).unwrap_or_else(|e| panic!("unclamped translation: {e}"));
    let dxil_clamped = translate(&clamped, ko).unwrap_or_else(|e| panic!("clamped translation: {e}"));
    super::validate::validate(&dxil).unwrap_or_else(|e| panic!("validator (unclamped): {e}"));
    super::validate::validate(&dxil_clamped).unwrap_or_else(|e| panic!("validator (clamped): {e}"));
    if dxc::location().is_err() {
        return runner::skip_test("no DXC");
    }
    let dis = super::validate::disassemble(&dxil).unwrap();
    assert_eq!(dis.matches("dx.op.dot4AddPacked.i32(i32 164,").count(), 2, "expected 2 Dot4AddU8Packed calls:\n{dis}");
    let dis_clamped = super::validate::disassemble(&dxil_clamped).unwrap();
    assert!(!dis_clamped.contains("dot4AddPacked"), "clamped dot8 should stay 64-bit saturating:\n{dis_clamped}");
}

#[test]
fn raw_loads_in_root_windows_are_branchless() {
    let loads = prog(&[
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 0), ("OFFSET", 0x40), ("SOFFSET", 125)], None),
        ("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 6), ("OFFSET", 0x10), ("SOFFSET", 125)], None),
        ("V_CMPX_LT_U32", "ENC_VOPC", &[("SRC0", 160), ("VSRC1", 0)], None),
        ("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 8), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_UBYTE", "ENC_MUBUF", &[("VDATA", 14), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("GLOBAL_LOAD_DWORDX4", "ENC_FLAT_GLBL", &[("VDST", 16), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 0)], None),
        ("GLOBAL_LOAD_USHORT", "ENC_FLAT_GLBL", &[("VDST", 20), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 4)], None),
    ]);
    for (wave64, split) in [(false, false), (true, true), (true, false)] {
        let dxil = translate(&loads, KernelOptions { wave64, split, mem_windows: 4, sampler_base: 0 })
            .unwrap_or_else(|e| panic!("wave64={wave64} split={split}: {e}"));
        super::validate::validate(&dxil).expect("valid DXIL");
        if let Ok(dis) = super::validate::disassemble(&dxil) {
            let body = dis.split("define void @main()").nth(1).and_then(|s| s.split("\n}\n").next()).expect("main body");
            assert!(dis.contains("@dx.op.rawBufferLoad"), "expected rawBufferLoad in DXIL:\n{dis}");
            assert!(!body.contains("switch i32"), "multi-window root-UAV loads must not emit switch (wave64={wave64}, split={split}):\n{dis}");
        }
    }
    let heap_dxil = translate(&loads, KernelOptions { wave64: false, split: false, mem_windows: 4 | crate::gpu::recomp::HEAP_WINDOWS, sampler_base: 0 })
        .expect("heap windows translate");
    super::validate::validate(&heap_dxil).expect("heap windows validate");

    let writes = prog(&[
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 0), ("DATA", 2), ("SADDR", 0x7D), ("OFFSET", 0)], None),
        ("GLOBAL_ATOMIC_ADD", "ENC_FLAT_GLBL", &[("VDST", 3), ("ADDR", 4), ("DATA", 5), ("SADDR", 0x7D), ("OFFSET", 0), ("GLC", 1)], None),
    ]);
    let wdxil = translate(&writes, KernelOptions { wave64: false, split: false, mem_windows: 4, sampler_base: 0 }).expect("writes translate");
    super::validate::validate(&wdxil).expect("writes validate");
    if let Ok(dis) = super::validate::disassemble(&wdxil) {
        let body = dis.split("define void @main()").nth(1).and_then(|s| s.split("\n}\n").next()).expect("main body");
        assert!(body.contains("switch i32"), "stores and atomics must keep multi-window switch:\n{dis}");
    }

    let ps_code = prog(&[
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 4), ("ADDR", 0), ("SADDR", 0), ("OFFSET", 0)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 4), ("VSRC2", 4), ("VSRC3", 4), ("DONE", 1), ("VM", 1)], None),
    ]);
    let psi = crate::gpu::recomp::PsInterface {
        inputs: vec![sig("SV_Position", 0, 1, 3, 0, 0xF)],
        outputs: vec![sig("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x2,
        attrs: None,
    };
    let ps_k = super::kernel::translate_pixel(&ps_code, KernelOptions { wave64: false, split: false, mem_windows: 4, sampler_base: 0 }, &psi)
        .expect("pixel shader multi-window load translates");
    super::validate::validate(&ps_k.dxil).expect("pixel shader validates");
    if let Ok(dis) = super::validate::disassemble(&ps_k.dxil) {
        let body = dis.split("define void @main()").nth(1).and_then(|s| s.split("\n}\n").next()).expect("main body");
        assert!(body.contains("switch i32"), "pixel shader multi-window loads must keep switch:\n{dis}");
    }
}

#[test]
fn raw_loads_in_root_windows_respect_exec_and_bounds() {
    use crate::gpu::recomp::runner::{self, KernelState};
    let code = prog(&[
        ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 1), ("OFFSET", 0)], None),
        ("V_CMPX_EQ_U32", "ENC_VOPC", &[("SRC0", 129), ("VSRC1", 15)], None),
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 20), ("ADDR", 2), ("SADDR", 0x7D), ("OFFSET", 0)], None),
        ("GLOBAL_LOAD_DWORDX2", "ENC_FLAT_GLBL", &[("VDST", 22), ("ADDR", 4), ("SADDR", 0x7D), ("OFFSET", 16)], None),
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 24), ("VADDR", 10), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 25), ("VADDR", 10), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1), ("OFFSET", 2)], None),
    ]);
    let w32_dxil = translate(&code, KernelOptions { wave64: false, split: false, mem_windows: 4, sampler_base: 0 }).expect("wave32 translates");
    let split_dxil = translate(&code, KernelOptions { wave64: true, split: true, mem_windows: 4, sampler_base: 0 }).expect("split-wave64 translates");
    let w64_dxil = translate(&code, KernelOptions { wave64: true, split: false, mem_windows: 4, sampler_base: 0 }).expect("wave64 translates");
    for dxil in [&w32_dxil, &split_dxil, &w64_dxil] {
        super::validate::validate(dxil).expect("valid DXIL");
    }
    let (has_w32, has_w64) = match (runner::wave32_supported(), runner::wave64_supported()) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => return runner::skip_test(&e),
    };
    if !has_w32 && !has_w64 {
        return runner::skip_test("no GPU");
    }
    let arena = crate::gpu::uma::Uma::with_windows(4, 1 << 20).expect("4 UMA windows");
    let ws: [u64; 4] = std::array::from_fn(|k| arena.window(k).0 as u64);
    let word = |a: u64| unsafe { &mut *(a as *mut u32) };
    // Non-zero canary at offset 0 of every window so leaked unselected reads fail immediately.
    for (k, &wk) in ws.iter().enumerate() {
        *word(wk) = 0xBAD0_0000 | k as u32;
        *word(wk + 0x1000) = 0xA000_0000 | (k as u32 * 0x11);
        *word(wk + 0x1004) = 0xB000_0000 | (k as u32 * 0x11);
    }
    *word(ws[1] + 8) = 0xCA44_0001;
    *word(ws[1] + 12) = 0xCA44_0002;
    *word(ws[2] + 0x1000) = 0x510A_D001;
    *word(ws[2] + 0x1004) = 0x510A_D002;
    for l in 0..64u64 {
        *word(ws[3] + 0x2000 + 4 * l) = 0x3000_0000 | l as u32;
        *word(ws[2] + 0x2000 + 4 * l) = ((0xF000 | (l as u32 + 1)) << 16) | 0xBEEF;
    }
    let hi = |a: u64| (a >> 32) as u32;
    let ident = 4 | (5 << 3) | (6 << 6) | (7 << 9);
    let mut st = KernelState::default();
    st.sgpr[2..4].copy_from_slice(&[(ws[2] + 0x1000) as u32, hi(ws[2])]);
    st.sgpr[12..16].copy_from_slice(&[(ws[3] + 0x2000) as u32, (hi(ws[3]) & 0xFFFF) | (4 << 16), 64, 0]);
    st.sgpr[16..20].copy_from_slice(&[(ws[2] + 0x2000) as u32, (hi(ws[2]) & 0xFFFF) | (4 << 16), 64, ident | (11 << 12)]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let slot = l % 16;
        let active = slot < 8;
        r[15] = u32::from(active);
        r[10] = if active { l as u32 } else { 0xFFFF_FFFF };
        r[20] = 0x5EED_0000 | l as u32;
        r[22] = 0x5EED_1000 | l as u32;
        r[23] = 0x5EED_2000 | l as u32;
        r[24] = 0x5EED_3000 | l as u32;
        r[25] = 0x5EED_4000 | l as u32;
        if !active {
            r[2] = 0x7FFF_FFE0;
            r[3] = 0xDEAD_0000;
            r[4] = 0x7FFF_FFE0;
            r[5] = 0xDEAD_0000;
            continue;
        }
        // Dynamic window choices: 0, 1, 2, 3, w>=4, 0xFFFF_FFFF, hi==0 (-> w0), carry w0->w1.
        match slot {
            0..=3 => {
                r[2] = (ws[slot] + 0x1000) as u32;
                r[3] = hi(ws[slot]);
                r[4] = (ws[slot] + 0xFF0) as u32;
                r[5] = hi(ws[slot]);
            }
            4 => {
                r[2] = 0x1000;
                r[3] = hi(ws[0]) + 4;
                r[4] = 0xFF0;
                r[5] = hi(ws[0]) + 4;
            }
            5 => {
                r[2] = 0x1000;
                r[3] = 0xFFFF_FFFF;
                r[4] = 0xFF0;
                r[5] = 0xFFFF_FFFF;
            }
            6 => {
                r[2] = 0x1000;
                r[3] = 0;
                r[4] = 0xFF0;
                r[5] = 0;
            }
            7 => {
                r[2] = 0x1000;
                r[3] = hi(ws[0]);
                // 0xFFFF_FFF8 + OFFSET 16 carries from window 0 into window 1 at byte offset 8.
                r[4] = 0xFFFF_FFF8;
                r[5] = hi(ws[0]);
            }
            _ => unreachable!(),
        }
    }
    let mut runs: Vec<(&str, &[u8], usize)> = Vec::new();
    if has_w32 {
        runs.push(("wave32", &w32_dxil, 32));
        runs.push(("split-wave64", &split_dxil, 64));
    }
    if has_w64 {
        runs.push(("native-wave64", &w64_dxil, 64));
    }
    for (label, dxil, lanes) in runs {
        let r = runner::run_in_arena(dxil, &st, &arena).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!((r.sgpr[20], r.sgpr[21]), (0x510A_D001, 0x510A_D002), "{label}: s_load_dwordx2 from window 2");
        for l in 0..lanes {
            let slot = l % 16;
            if slot >= 8 {
                assert_eq!(r.vgpr[l][20], 0x5EED_0000 | l as u32, "{label} lane {l}: inactive global_load preserved v20");
                assert_eq!(r.vgpr[l][22], 0x5EED_1000 | l as u32, "{label} lane {l}: inactive global_loadx2 preserved v22");
                assert_eq!(r.vgpr[l][23], 0x5EED_2000 | l as u32, "{label} lane {l}: inactive global_loadx2 preserved v23");
                assert_eq!(r.vgpr[l][24], 0x5EED_3000 | l as u32, "{label} lane {l}: inactive buffer_load preserved v24");
                assert_eq!(r.vgpr[l][25], 0x5EED_4000 | l as u32, "{label} lane {l}: inactive buffer_load_format preserved v25");
                continue;
            }
            let (want20, want22, want23) = match slot {
                0..=3 => {
                    let w = slot as u32;
                    let w20 = if w == 2 { 0x510A_D001 } else { 0xA000_0000 | (w * 0x11) };
                    let w22 = if w == 2 { 0x510A_D001 } else { 0xA000_0000 | (w * 0x11) };
                    let w23 = if w == 2 { 0x510A_D002 } else { 0xB000_0000 | (w * 0x11) };
                    (w20, w22, w23)
                }
                4 | 5 => (0, 0, 0),
                6 => (0xA000_0000, 0xA000_0000, 0xB000_0000),
                7 => (0xA000_0000, 0xCA44_0001, 0xCA44_0002),
                _ => unreachable!(),
            };
            assert_eq!(r.vgpr[l][20], want20, "{label} lane {l} (slot {slot}): global_load_dword");
            assert_eq!((r.vgpr[l][22], r.vgpr[l][23]), (want22, want23), "{label} lane {l} (slot {slot}): global_load_dwordx2");
            assert_eq!(r.vgpr[l][24], 0x3000_0000 | l as u32, "{label} lane {l}: buffer_load_dword from window 3");
            assert_eq!(r.vgpr[l][25], 0xF000 | (l as u32 + 1), "{label} lane {l}: buffer_load_format_x unaligned from window 2");
        }
    }
}
