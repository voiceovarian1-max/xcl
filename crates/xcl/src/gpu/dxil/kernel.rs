//! Direct translation of an RDNA2 program in the recompiler's kernel ABI
//! (`recomp::runner`): u0 `io` = 128 SGPRs (SCC in slot 125) then 64 lanes ×
//! 256 VGPRs; u1 `res` = the same layout after the program (+ exports);
//! u2 `mem` = flat memory. As the HLSL kernel does, the registers the program
//! touches are loaded at entry and stored at the end (SGPRs + SCC by lane 0,
//! VGPRs by every lane); the rest of `res` stays 0.
//!
//! [`translate_compute`] is the recompiler's compute mode (the real CS ABI,
//! `recomp::CsInterface`): user SGPRs from `io`, then the enabled group ids
//! and TG_SIZE; v0..v2 = thread id in the group; results only through memory.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::cfg::{Cfg, Term};
use super::container::Stage;
use super::lower::{ComputeInit, Init, Lw, Partner, LOOP_BUDGET, VAR_BUDGET, VAR_LOOP_PASSES, VAR_SCC, VAR_VGPR};
use super::module::{Block, Pred, Value};
use super::shader::{RawUav, Shader};
use super::ssa::Ssa;
use crate::gpu::rdna2::Decoded;
use crate::gpu::recomp::{CsInterface, GsInterface, MsInterface, PsInterface, VsInterface, KERNEL_SCC_SLOT, KERNEL_SGPRS, KERNEL_VGPRS, MESH_HEADER};

#[derive(Debug, Clone, Copy)]
pub struct KernelOptions {
    /// wave64 program (64-bit EXEC / VCC) or wave32
    pub wave64: bool,
    /// the GPU runs only 32-lane waves: a wave64 program runs as two halves
    pub split: bool,
    /// 4 GiB memory windows the program may address (`recomp::Options::mem_windows`):
    /// `mem` at u2, then `mem1`.. at u3.. (with `recomp::HEAP_WINDOWS`: descriptors
    /// `MEM_HEAP_BASE`.. of the resource heap)
    pub mem_windows: u32,
    /// first register (space 1) of the static samplers (another stage may come first)
    pub sampler_base: u32,
}

/// Version of the direct translator's output, part of the on-disk translation cache's
/// identity (`recomp::pso`): bump it when `pso::direct_compute` / `direct_graphics` change the
/// containers. An edit of `gpu/dxil` needs no bump, `XCL_DIRECT_DXIL_SOURCE_ID` is its hash
/// (the `gpu/dxil` sources are not in build.rs's translator source hash).
pub const TRANSLATOR_VERSION: u32 = 12;

/// `XCL_DIRECT_DXIL=1`: console compute shaders (and the kernel tests) go through this
/// translator, with the HLSL path as the fallback for what it cannot translate yet.
pub fn enabled() -> bool {
    std::env::var("XCL_DIRECT_DXIL").is_ok_and(|v| v == "1" || v == "all")
}

/// `XCL_DIRECT_DXIL=all`: console vertex and pixel shaders too (each stage falls back to
/// the HLSL path on its own; the signatures are declared as the HLSL entries declare them,
/// so a direct stage links with an HLSL one).
pub fn enabled_graphics() -> bool {
    std::env::var("XCL_DIRECT_DXIL").is_ok_and(|v| v == "all")
}

/// Translates `code` into a signed DXIL compute container, or names the first
/// instruction it cannot translate.
pub fn translate(code: &[u8], o: KernelOptions) -> Result<Vec<u8>, String> {
    translate_full(code, o).map(|k| k.dxil)
}

/// A translated kernel: the container and the static samplers it samples with
/// (S# built from constants in the code, register `s<i>, space1` — the HLSL
/// kernel's `Translation::static_samplers`).
pub struct Kernel {
    pub dxil: Vec<u8>,
    pub static_samplers: Vec<([u32; 4], bool)>,
    pub dispatch_chain: Option<std::sync::Arc<DispatchChain>>,
}

pub use super::module::split::DispatchChain;

pub fn dispatch_piece_limit() -> usize {
    // (local experiment) `XCL_DXIL_VARIANTS` `split`: this program in chained pieces
    if super::module::variant::current() & super::module::variant::SPLIT != 0 {
        return 12_500;
    }
    std::env::var("XCL_DIRECT_SPLIT_INSTRUCTIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(12_500)
}

/// `XCL_DIRECT_COMPACT`: `1` forces the compact form, `0` forbids it, unset uses automatic selection.
pub fn compact_mode() -> Option<bool> {
    // (local experiment) `XCL_DXIL_VARIANTS` `nocompact`
    if super::module::variant::current() & super::module::variant::NOCOMPACT != 0 {
        return Some(false);
    }
    match std::env::var("XCL_DIRECT_COMPACT").as_deref() {
        Ok("1") => Some(true),
        Ok("0") => Some(false),
        _ => None,
    }
}

/// Non-default `XCL_DIRECT_COMPACT_NEST` cache tag for the translated cache identity.
pub fn compact_nest_tag() -> Option<String> {
    super::compact::nest_mode().cache_tag()
}

/// `translate_full` in the compact form (`compact = true`) or the plain one.
#[cfg(test)]
pub fn translate_full_as(code: &[u8], o: KernelOptions, compact: bool) -> Result<Kernel, String> {
    translate_entry(code, o, Entry::Kernel, Some(compact), None, super::compact::TABLE_WORDS_WAVE32)
}

/// `translate_full` in the compact form with an explicit dispatch nesting policy.
#[cfg(test)]
pub fn translate_full_nested(code: &[u8], o: KernelOptions, nest: super::compact::Nest) -> Result<Kernel, String> {
    translate_entry(code, o, Entry::Kernel, Some(true), Some(nest), super::compact::TABLE_WORDS_WAVE32)
}

pub fn translate_full(code: &[u8], o: KernelOptions) -> Result<Kernel, String> {
    translate_entry(code, o, Entry::Kernel, compact_mode(), None, super::compact::TABLE_WORDS_WAVE32)
}

/// A mesh-mode (NGG) program: the first pass that writes one record per group into
/// `res` (the recompiler's `Mode::Mesh`; the pass-through mesh shader draws them).
pub fn translate_mesh(code: &[u8], o: KernelOptions, ms: &MsInterface) -> Result<Kernel, String> {
    let hw_wave = if o.wave64 && !o.split { 64 } else { 32 };
    if ms.threads > hw_wave
        && std::env::var("XCL_SAFE_BARRIER").map_or(true, |v| v.trim() != "0")
        && super::lower::has_non_uniform_barrier(code)
    {
        return Err("non-uniform s_barrier in multi-wave mesh shader".into());
    }
    translate_entry(code, o, Entry::Mesh(ms.clone()), None, None, 0)
}

/// (local) A mesh-mode (NGG) program as one D3D12 mesh shader, in place of `translate_mesh`'s
/// compute pass and the pass-through (after the VIctory family's one-pass mesh lowering,
/// https://github.com/DoubleGhosty/cool_fox by codec_d, commit f4b48ad, crates/victory-shader/src/emit/lower.rs
/// `mesh_prologue`, `export`, `emit_indices`; GPL-3.0-or-later): the maxima as the output counts
/// at entry and every primitive culled first, then each thread's exports into its own vertex and
/// primitive. Its outputs are the pass-through's for the same pixel shader, so the pair links as
/// before. `io` is u1 (u0 is the pixel shader's).
#[derive(Debug, Clone)]
pub struct MeshOne {
    pub ms: MsInterface,
    /// the pixel shader's inputs (`dxbc::signatures(ps).0`); None: every output of the program
    pub ps_inputs: Option<Vec<crate::gpu::dxbc::SigElement>>,
    /// the LDS the program allocates (its stage record, `recomp::pso::ms_lds_bytes`); 0: none or
    /// not known (a program that uses LDS is then refused: a mesh shader has 28 KB, not 32)
    pub lds_bytes: u32,
}

/// The `MeshOne` mesh shader of `code` (ms_6_6 DXIL), or why not.
pub fn translate_mesh_one(code: &[u8], o: KernelOptions, one: &MeshOne, rodata: &[(u32, Vec<u32>)]) -> Result<Kernel, String> {
    let hw_wave = if o.wave64 && !o.split { 64 } else { 32 };
    if one.ms.threads > hw_wave && super::lower::has_non_uniform_barrier(code) {
        return Err("non-uniform s_barrier in multi-wave mesh shader".into());
    }
    translate_entry_with(code, o, Entry::MeshOnePass(one.clone()), None, None, 0, rodata.to_vec())
}

/// A vertex shader (the recompiler's `Mode::Vertex`; wave-size independent).
pub fn translate_vertex(code: &[u8], o: KernelOptions, vs: &VsInterface) -> Result<Kernel, String> {
    translate_entry(code, o, Entry::Vertex(vs.clone()), None, None, 0)
}

/// A pixel shader (the recompiler's `Mode::Pixel`; wave-size independent).
pub fn translate_pixel(code: &[u8], o: KernelOptions, ps: &PsInterface) -> Result<Kernel, String> {
    translate_entry(code, o, Entry::Pixel(ps.clone()), None, None, 0)
}

/// A geometry-mode (ES+GS) program: the compute pass that writes one mesh record per
/// group of 21 triangles into `res` (the recompiler's `Mode::Geometry`). Always wave64.
pub fn translate_geometry(code: &[u8], o: KernelOptions, g: &GsInterface) -> Result<Kernel, String> {
    // The split form's ES phase runs in both hardware halves (`Partner::Present`): every ballot
    // and lane read meets at a group barrier, the pattern that hung NVIDIA GPUs (#1727). The
    // HLSL path's exchange spins by default: such a pass is left to it (`XCL_SAFE_BARRIER=0`:
    // as before). The GS alone runs in the lower half without a partner.
    if o.split && g.gs_start != Some(0) && crate::gpu::recomp::emit::safe_barrier_wanted() {
        return Err("split wave64 geometry: the ES phase exchanges at group barriers".into());
    }
    translate_geometry_unrefused(code, o, g)
}

/// `translate_geometry` without the refusal of the split form's ES phase: the translation
/// `XCL_SAFE_BARRIER=0` takes (and what tests of that form check whatever the environment).
pub(crate) fn translate_geometry_unrefused(code: &[u8], o: KernelOptions, g: &GsInterface) -> Result<Kernel, String> {
    translate_entry(code, KernelOptions { wave64: true, ..o }, Entry::Geometry(g.clone()), None, None, 0)
}

/// The entry ABI.
#[derive(Clone)]
pub enum Entry {
    Kernel,
    Compute(CsInterface),
    Mesh(MsInterface),
    /// (local) `translate_mesh_one`
    MeshOnePass(MeshOne),
    Vertex(VsInterface),
    Pixel(PsInterface),
    Geometry(GsInterface),
}

/// A compute-mode program (the CS ABI of `cs`).
pub fn translate_compute(code: &[u8], o: KernelOptions, cs: &CsInterface) -> Result<Kernel, String> {
    translate_compute_on(code, o, cs, true)
}

/// `translate_compute` for a GPU: one that runs only 32-lane waves takes larger compact tables.
pub fn translate_compute_on(code: &[u8], o: KernelOptions, cs: &CsInterface, wave32_only: bool) -> Result<Kernel, String> {
    translate_compute_rodata(code, o, cs, wave32_only, &[])
}

/// `translate_compute_on` with the constant data after the program (`translate_entry_rodata`).
pub fn translate_compute_rodata(code: &[u8], o: KernelOptions, cs: &CsInterface, wave32_only: bool, rodata: &[(u32, Vec<u32>)]) -> Result<Kernel, String> {
    let tables = if wave32_only { super::compact::TABLE_WORDS_WAVE32 } else { super::compact::TABLE_WORDS };
    translate_entry_with(code, o, Entry::Compute(cs.clone()), compact_mode(), None, tables, rodata.to_vec())
}

fn translate_entry(
    code: &[u8],
    o: KernelOptions,
    entry: Entry,
    compact: Option<bool>,
    nest: Option<super::compact::Nest>,
    tables: usize,
) -> Result<Kernel, String> {
    translate_entry_with(code, o, entry, compact, nest, tables, Vec::new())
}

/// A program at `entry` with the constant data the console compiler put after it in the code
/// region (`xbox::Fssm::rodata`, the recompiler's `Options::rodata`): `s_getpc_b64` plus the
/// compiler's offset is its address, and dword loads at such an address read the table the
/// module carries (`Lw::rodata_loads`). The compact tables as `translate_full` /
/// `translate_compute` take them (a compute program for a GPU with 64-lane waves:
/// `translate_compute_rodata`).
pub fn translate_entry_rodata(code: &[u8], o: KernelOptions, entry: Entry, rodata: &[(u32, Vec<u32>)]) -> Result<Kernel, String> {
    let (compact, tables) = if matches!(entry, Entry::Kernel | Entry::Compute(_)) { (compact_mode(), super::compact::TABLE_WORDS_WAVE32) } else { (None, 0) };
    translate_entry_with(code, o, entry, compact, None, tables, rodata.to_vec())
}

fn translate_entry_with(
    code: &[u8],
    o: KernelOptions,
    entry: Entry,
    compact: Option<bool>,
    nest: Option<super::compact::Nest>,
    tables: usize,
    rodata: Vec<(u32, Vec<u32>)>,
) -> Result<Kernel, String> {
    // SSA construction recurses along the CFG (one level per guarded memory access between
    // a register's definition and its use): a thread with a large stack for big programs
    let code = code.to_vec();
    let variant = super::module::variant::current();
    std::thread::Builder::new()
        .name("xcl-dxil-translate".into())
        .stack_size(256 << 20)
        .spawn(move || super::module::variant::with(variant, || translate_on_this_thread(&code, o, entry, compact, nest, tables, rodata)))
        .map_err(|e| format!("translator thread: {e}"))?
        .join()
        .map_err(|e| {
            let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            format!("the direct DXIL translator panicked: {msg}")
        })?
}

/// The memory windows of `mem_windows` and whether they come from the descriptor heap
/// (`recomp::HEAP_WINDOWS` in the count, `XCL_HEAP_WINDOWS=1`; Some(globallycoherent), and
/// `mems` then only counts them). Compute work across groups relies on coherent memory
/// (as the HLSL compute mode's `globallycoherent` windows).
fn mem_windows(s: &mut Shader, mem_windows: u32, coherent: bool) -> (Vec<RawUav>, Option<bool>) {
    let heap = mem_windows & crate::gpu::recomp::HEAP_WINDOWS != 0;
    let max = if heap { crate::gpu::uma::MAX_HEAP_WINDOWS as u32 } else { crate::gpu::uma::MAX_WINDOWS as u32 };
    let n = (mem_windows & !crate::gpu::recomp::HEAP_WINDOWS).clamp(1, max);
    if heap && n > 1 {
        let z = s.m.i32c(0);
        return (vec![RawUav { handle: z, binding: None }; n as usize], Some(coherent));
    }
    ((0..n).map(|k| if coherent { s.raw_uav_coherent(2 + k, 0) } else { s.raw_uav(2 + k, 0) }).collect(), None)
}

fn translate_on_this_thread(
    code: &[u8],
    o: KernelOptions,
    entry: Entry,
    compact: Option<bool>,
    nest: Option<super::compact::Nest>,
    tables: usize,
    rodata: Vec<(u32, Vec<u32>)>,
) -> Result<Kernel, String> {
    let geo = match &entry {
        Entry::Geometry(g) => Some(g.clone()),
        _ => None,
    };
    let cfg = match &geo {
        Some(g) => Cfg::build_geometry(code, g.gs_start)?,
        None => Cfg::build(code)?,
    };
    let kernel = matches!(entry, Entry::Kernel);
    let follow_loads = matches!(entry, Entry::Kernel | Entry::Compute(_));
    let gfx = matches!(entry, Entry::Vertex(_) | Entry::Pixel(_));
    let compute = matches!(entry, Entry::Compute(_));
    let split = o.wave64 && o.split && !gfx;
    let leaders: BTreeSet<u64> = cfg.blocks.iter().filter_map(|b| b.insts.first().map(|&i| cfg.insts[i].pc)).collect();
    let mut lw = Lw::new(&cfg.insts, &leaders, o, &entry);
    lw.rodata = rodata;
    let uses_lds = lw.lds.is_some();
    let res = lw.res;
    let io = lw.init.io.expect("the I/O buffer");
    let mut split_tid = None;
    // a compute program: whether this group runs (`compute_init`: not past a folded dispatch's count)
    let mut group_runs = None;
    let numthreads = match &entry {
        Entry::Kernel => {
            lw.setup_lanes();
            let lane = lw.lane;
            let k = lw.c((KERNEL_VGPRS * 4) as u32);
            let b = lw.mul(lane, k);
            let base = lw.addc(b, (KERNEL_SGPRS * 4) as u32);
            lw.init.vgpr_base = Some(base);
            [if o.wave64 { 64 } else { 32 }, 1, 1]
        }
        Entry::Compute(cs) => {
            let (tid, gtid) = lw.setup_lanes_compute(cs.threads, uses_lds);
            if split && !uses_lds && cs.threads == [64, 1, 1] { split_tid = Some(tid); }
            group_runs = Some(compute_init(&mut lw, cs, tid, gtid));
            cs.threads
        }
        Entry::Mesh(ms) => {
            let (tid, _) = lw.setup_lanes_compute([ms.threads, 1, 1], uses_lds);
            mesh_init(&mut lw, ms, tid, io);
            [ms.threads, 1, 1]
        }
        Entry::MeshOnePass(one) => {
            let (tid, _) = lw.setup_lanes_compute([one.ms.threads, 1, 1], uses_lds);
            mesh_init(&mut lw, &one.ms, tid, io);
            let out = lw.mesh_outputs(&one.ms, one.ps_inputs.as_deref());
            lw.mesh.as_mut().expect("mesh").one = Some(out);
            [one.ms.threads, 1, 1]
        }
        Entry::Geometry(g) => {
            let (tid, _) = lw.setup_lanes_compute([64, 1, 1], uses_lds);
            let used: BTreeSet<u64> = cfg.insts.iter().flat_map(|d| {
                let (r, w) = crate::gpu::rdna2::vgpr_uses(d);
                r.into_iter().chain(w)
            }).collect();
            geom_init(&mut lw, g, tid, io, &used);
            [64, 1, 1]
        }
        Entry::Vertex(vs) => {
            lw.setup_lanes_gfx();
            let used: BTreeSet<u64> = cfg.insts.iter().flat_map(|d| {
                let (r, w) = crate::gpu::rdna2::vgpr_uses(d);
                r.into_iter().chain(w)
            }).collect();
            lw.setup_vertex(vs, &used);
            [1, 1, 1]
        }
        Entry::Pixel(ps) => {
            lw.setup_lanes_gfx();
            let modes = crate::gpu::recomp::emit::ps_attr_modes(&cfg.insts, ps.input_ena);
            lw.setup_pixel(ps, &modes);
            [1, 1, 1]
        }
    };
    // the HLSL kernel loads and stores every register its emitted instructions name: all of
    // them in a program with branches, those up to the first s_endpgm in straight-line code
    let branchy = cfg.insts.iter().any(|d| crate::gpu::rdna2::branch_target(d).is_some());
    (lw.dyn_rm, lw.has_f64_round_op) = super::lower::round_modes(&cfg.insts, !branchy);
    lw.f32_rm = super::lower_round::f32_round_writes(&cfg.insts);
    for d in cfg.insts.iter().filter(|_| kernel) {
        if !branchy && d.inst.name.starts_with("S_ENDPGM") {
            break;
        }
        let (sr, sw) = crate::gpu::rdna2::sgpr_uses(d);
        let (vr, vw) = crate::gpu::rdna2::vgpr_uses(d);
        for r in sr.into_iter().chain(sw) {
            lw.init.touched.insert(r as u32);
        }
        for r in vr.into_iter().chain(vw).filter(|&r| r < 256) {
            lw.init.touched.insert(VAR_VGPR + r as u32);
        }
    }

    let plan = match compact {
        _ if !compute && !kernel => None,
        Some(false) => None,
        Some(true) => super::compact::plan(&cfg, &lw),
        None => super::compact::automatic_plan(&cfg, &lw, compute && split, tables),
    };
    if let Some(p) = &plan {
        crate::xlog_info!("[SHADER] direct DXIL compact form: {} blocks of {} kinds", p.order.len(), p.rep.len());
    }
    let n = cfg.blocks.len();
    let mut heads: Vec<Option<Block>> = vec![None; n];
    let entry = *cfg.rpo.first().ok_or("empty program")?;
    // split geometry mode: the GS's first block is entered from the phase switch (one edge)
    // instead of from the ES blocks that jump there
    if let (Some(g), true) = (&geo, split) {
        let gs_block = (0..n).find(|&b| g.gs_start.is_some_and(|s| s > 0 && cfg.blocks[b].pc == s));
        if let Some(t) = gs_block {
            let gs_preds = cfg.blocks[t].preds.iter().filter(|&&p| cfg.blocks[p].pc >= g.gs_start.unwrap_or(0)).count();
            let es = cfg.blocks[t].preds.len() > gs_preds;
            heads[t] = Some(lw.block(Some(gs_preds + es as usize)));
        }
    }
    let mut head = |lw: &mut Lw, b: usize| -> Block {
        if let Some(h) = heads[b] {
            return h;
        }
        let preds = cfg.blocks[b].preds.len() + (b == entry) as usize;
        let h = lw.block(Some(preds));
        heads[b] = Some(h);
        h
    };
    // geometry mode: which blocks are GS code; the split form runs the ES in both hardware
    // waves, then (`switch`: a group barrier) the GS in the lower one with fresh registers
    let gs_start = geo.as_ref().and_then(|g| g.gs_start);
    let in_gs = |b: usize| gs_start.is_some_and(|s| cfg.blocks[b].pc >= s);
    let two_phase = split && gs_start.is_some_and(|s| s > 0);
    let gs_only = split && gs_start == Some(0);
    let es_xn = lw.xn;
    if gs_only {
        lw.partner = Partner::Missing;
        lw.xn = lw.c(0);
    }
    let h0 = if plan.is_some() { lw.block(None) } else { head(&mut lw, entry) };
    let exit = lw.block(None);
    // where `Block(0)` goes
    let mut first = h0;
    if gs_only {
        // the GS alone: in the lower hardware wave only
        let gate = lw.block(Some(1));
        lw.edge(Block(0), gate);
        lw.s.m.set_block(gate);
        let up = lw.upper;
        lw.cond_br(up, exit, h0);
    } else if let Some(st) = group_runs {
        // a folded dispatch's groups past the console's count leave at once (the whole group:
        // no barrier waits on them), and the lanes past a thread limit (`compute_gate`) unless
        // the program has a group barrier
        let gate = lw.block(Some(1));
        lw.edge(Block(0), gate);
        lw.s.m.set_block(gate);
        let leave = !cfg.insts.iter().any(|d| d.inst.name == "S_BARRIER");
        let runs = compute_gate(&mut lw, &st, leave);
        lw.cond_br(runs, h0, exit);
        first = gate;
    } else if lw.mesh.as_ref().is_some_and(|m| m.one.is_some()) {
        // (local) one-pass mesh shader: its output counts and culled primitives before the program
        // (the entry block stays open for the registers' first values until the end)
        let pro = lw.block(Some(1));
        lw.edge(Block(0), pro);
        lw.s.m.set_block(pro);
        lw.mesh_prologue()?;
        lw.br(h0);
        first = pro;
    } else {
        lw.edge(Block(0), h0);
    }
    let switch = if two_phase { Some(lw.block(None)) } else { None };
    // a compute program's loops: each entry has its own count of passes besides the lane's budget
    let passes = if compute && plan.is_none() && !cfg.headers.is_empty() { Some(loop_passes(&mut lw, &cfg, numthreads.iter().product())) } else { None };
    // scalar loads the program repeats keep their first values (compute programs and kernels:
    // the other stages enter their code by edges the CFG does not hold)
    let reloads = if follow_loads { super::reload::analyze(&cfg) } else { Default::default() };
    let mut gs_entry: Option<usize> = None;
    if let Some(p) = &plan {
        match nest {
            Some(n) => super::compact::emit_with_nest(&mut lw, &cfg, p, h0, exit, n)?,
            None => super::compact::emit(&mut lw, &cfg, p, h0, exit)?,
        }
    }
    for &b in cfg.rpo.iter().filter(|_| plan.is_none()) {
        let h = head(&mut lw, b);
        lw.s.m.set_block(h);
        let es_block = two_phase && !in_gs(b);
        if two_phase {
            if in_gs(b) {
                lw.partner = Partner::Missing;
                lw.xn = lw.c(0);
            } else {
                lw.partner = Partner::Present;
                lw.xn = es_xn;
            }
        }
        let end = if es_block { switch.unwrap() } else { exit };
        if cfg.headers.contains(&b) {
            if let Some((cap, inside)) = &passes {
                // this entry of the loop has a pass less (`Em::loop_guard`'s `xcl_n<k>`), and the
                // loops directly inside it start afresh. In a reducible CFG every path that enters
                // a loop, first or again, passes the head of the loop around it, which lies outside
                // the inner one: a reset there is one on each entry. A loop that runs out leaves to
                // `end` and ends the lane's program, where the HLSL path's break goes on after the
                // loop: both end a loop that ran away, neither gives a right result.
                let var = VAR_LOOP_PASSES + b as u32;
                let n = lw.rd(var);
                let none = lw.cmpc(Pred::Eq, n, 0);
                let cont = lw.block(Some(1));
                lw.cond_br(none, end, cont);
                lw.s.m.set_block(cont);
                let one = lw.c(1);
                let n = lw.sub(n, one);
                lw.wr(var, n);
                for &c in inside.get(&b).into_iter().flatten() {
                    lw.wr(VAR_LOOP_PASSES + c as u32, *cap);
                }
            }
            // a lane's loop iterations are capped (the recompiler's LOOP_GUARD)
            let left = lw.rd(VAR_BUDGET);
            let none = lw.cmpc(Pred::Eq, left, 0);
            let cont = lw.block(Some(1));
            lw.cond_br(none, end, cont);
            lw.s.m.set_block(cont);
            let one = lw.c(1);
            let l = lw.sub(left, one);
            lw.wr(VAR_BUDGET, l);
        }
        for &i in &cfg.blocks[b].insts {
            lw.reload = reloads.get(b, i);
            lw.inst(&cfg.insts[i])?;
        }
        lw.reload = None;
        match cfg.blocks[b].term {
            Term::Jump(t) if es_block && in_gs(t) => {
                // the ES's s_setpc_b64 into the GS: through the phase switch
                gs_entry = Some(t);
                lw.br(end);
            }
            Term::Jump(t) => {
                let t = head(&mut lw, t);
                lw.br(t);
            }
            Term::Cond(i, t, f) => {
                let c = lw.branch_cond(&cfg.insts[i])?;
                let (t, f) = (head(&mut lw, t), head(&mut lw, f));
                lw.cond_br(c, t, f);
            }
            Term::End => lw.br(end),
        }
    }
    if let Some(sw) = switch {
        lw.s.m.set_block(sw);
        lw.seal(sw);
        lw.partner = Partner::Present;
        lw.xn = es_xn;
        lw.s.barrier(super::ops::barrier::GROUP_SYNC);
        match gs_entry {
            Some(t) => {
                let gs = lw.block(Some(1));
                let up = lw.upper;
                lw.cond_br(up, exit, gs);
                lw.s.m.set_block(gs);
                let vars: Vec<u32> = lw.init.touched.iter().copied().filter(|&v| v < VAR_BUDGET && v != VAR_SCC && v != 126 && v != 127).collect();
                lw.geom_gs_phase_registers(&vars);
                let h = head(&mut lw, t);
                lw.br(h);
            }
            None => lw.br(exit),
        }
    }
    lw.s.m.set_block(exit);
    lw.seal(exit);
    // the epilogue uses values of the program's blocks, which were created after `exit`: it goes
    // into a block of its own after them (only a phi may refer to a later block's value)
    let tail = lw.block(Some(1));
    lw.br(tail);
    lw.s.m.set_block(tail);
    if let (Some(res), true) = (res, kernel) {
        store_registers(&mut lw, res);
    }
    if gfx {
        lw.finish_gfx();
    }
    if geo.is_some() {
        if split {
            // the record comes from the lower hardware wave (the GS phase's)
            let epi = lw.block(Some(1));
            let done = lw.block(Some(2));
            let up = lw.upper;
            lw.cond_br(up, done, epi);
            lw.s.m.set_block(epi);
            lw.geom_epilogue();
            lw.br(done);
            lw.s.m.set_block(done);
        } else {
            lw.geom_epilogue();
        }
    }
    lw.s.m.ret();
    lw.s.m.set_block(Block(0));
    lw.s.m.br(first);
    let hw = if split || !o.wave64 { 32 } else { 64 };
    let limit = if compute && (!split || split_tid.is_some()) { dispatch_piece_limit() } else { 0 };
    lw.finish_dispatch(numthreads, hw, limit, split_tid)
}

/// True when a pixel shader uses RDNA2 Primitive Ordered Pixel Shading (POPS):
/// reads `src_pops_exiting_wave_id` (operand 239) or accesses `HW_REG_POPS_PACKER`
/// (hardware register 25 via `S_GETREG_B32` / `S_SETREG*`).
fn uses_pops(insts: &[Decoded]) -> bool {
    const SRC_POPS_EXITING_WAVE_ID: u64 = 239;
    const HW_REG_POPS_PACKER: u64 = 25;
    insts.iter().any(|d| {
        if matches!(d.inst.name, "S_GETREG_B32" | "S_SETREG_B32" | "S_SETREG_IMM32_B32")
            && crate::gpu::rdna2::field_value(d.enc, &d.words, "SIMM16").is_some_and(|v| (v & 0x3F) == HW_REG_POPS_PACKER)
        {
            return true;
        }
        d.ienc.operands.iter().any(|o| {
            let Some(f) = o.field else { return false };
            o.input
                && !o.implicit
                && (o.ty.starts_with("OPR_SRC") || o.ty.starts_with("OPR_SSRC"))
                && crate::gpu::rdna2::field_value(d.enc, &d.words, f) == Some(SRC_POPS_EXITING_WAVE_ID)
        })
    })
}

impl Lw {
    /// The translator's state for a program (`insts`, its blocks starting at `leaders`) at
    /// `entry`: the shader with its buffers,
    /// the SSA's entry block, the program's LDS and the split form's exchange storage and
    /// partner. The caller then sets up the lanes for its entry (`setup_lanes*`), lowers the
    /// program's blocks, branches from `Block(0)` to the first of them and ends with
    /// [`Lw::finish`]. #1426's block walk (`translate_on_this_thread`) and the recompiler's
    /// per-lane lowering (`recomp::lower_dxil`) both build their `Lw` here.
    pub fn new(insts: &[Decoded], leaders: &BTreeSet<u64>, o: KernelOptions, entry: &Entry) -> Lw {
        let split = o.wave64 && o.split;
        let kernel = matches!(entry, Entry::Kernel);
        let cs = match entry {
            Entry::Compute(cs) => Some(cs.clone()),
            _ => None,
        };
        let ms = match entry {
            Entry::Mesh(ms) => Some(ms.clone()),
            _ => None,
        };
        let one = matches!(entry, Entry::MeshOnePass(_));
        let gfx = matches!(entry, Entry::Vertex(_) | Entry::Pixel(_));
        let split = split && !gfx;
        let stage = match entry {
            Entry::Vertex(_) => Stage::Vertex,
            Entry::Pixel(_) => Stage::Pixel,
            Entry::MeshOnePass(_) => Stage::Mesh,
            _ => Stage::Compute,
        };
        let mut s = Shader::new(stage, (6, 6));
        // (one-pass mesh: u0 is the pixel shader's `io`, the mesh shader's is u1, where the record was)
        let io = s.raw_uav(if one { 1 } else { 0 }, 0);
        let res = if kernel || ms.is_some() || matches!(entry, Entry::Geometry(_)) { Some(s.raw_uav(1, 0)) } else { None };
        if stage == Stage::Pixel && uses_pops(insts) {
            s.enable_rov();
        }
        let coherent = cs.is_some() && crate::gpu::recomp::needs_coherent_mem(insts);
        let (mems, heap_mem) = mem_windows(&mut s, o.mem_windows, coherent);
        let i32 = s.m.i32();
        let z = s.m.i32c(0);
        let mut lw = Lw {
            s,
            ssa: Ssa::new(i32),
            init: Init { io: Some(io), compute: None, vgpr_base: None, entry: Block(0), touched: BTreeSet::new(), slots: HashMap::new(), mem: None },
            wave64: o.wave64,
            split,
            lane: z,
            lane31: z,
            upper: z,
            half: z,
            mems,
            heap_mem,
            res,
            fmt_table: None,
            rodata: Vec::new(),
            rodata_table: None,
            fmt_desc_cache: HashMap::new(),
            fmt_comp1_cache: HashMap::new(),
            reload: None,
            uma_base_val: None,
            ntmp: 0,
            scratch: None,
            static_samplers: Default::default(),
            static_used: Vec::new(),
            sampler_base: o.sampler_base,
            dyn_rm: false,
            f32_rm: false,
            track_frexp_denorm: crate::gpu::recomp::emit::tracks_frexp_f32_denorm(insts),
            has_f64_round_op: false,
            img_writes: crate::gpu::recomp::emit::image_write_srsrcs(insts),
            exec_cache: HashMap::new(),
            xs: None,
            xw: None,
            xc: None,
            barrier_waves: None,
            waves: 1,
            cw: z,
            xn: z,
            live_exec: None,
            xslot: 1,
            lds: None,
            lds_top: 8191,
            dpp_enable: None,
            partner: Partner::Present,
            mesh: None,
            geom: None,
            gfx: None,
            gfx_half: None,
            whole_live: Default::default(),
            guard_override: None,
            mask_override: None,
            local_masks: false,
            writes_lds: split && crate::gpu::recomp::emit::lds_order_wanted() && crate::gpu::recomp::emit::writes_lds(insts),
            exchange_error: None,
            pc: 0,
        };
        {
            let Lw { s, ssa, init, .. } = &mut lw;
            ssa.expect_preds(s, Block(0), 0, &mut |s, v| init.load(s, v));
        }
        // LDS and the split form's exchange slot size (16 words for a coherent s_load_dwordx16,
        // 32 for ds_bpermute), as the HLSL kernel sizes them
        // geometry mode always has the LDS (the ES ring and the GS output)
        let mut uses_lds = matches!(entry, Entry::Geometry(_));
        let mut slot = 1u32;
        for d in insts {
            let n = d.inst.name;
            if d.enc.name == "ENC_DS" && !matches!(n, "DS_SWIZZLE_B32" | "DS_BPERMUTE_B32" | "DS_PERMUTE_B32" | "DS_NOP") {
                uses_lds = true;
            }
            if n == "DS_BPERMUTE_B32" {
                slot = slot.max(32);
            }
            if d.enc.name == "ENC_SMEM" && ["GLC", "DLC"].iter().any(|f| crate::gpu::rdna2::field_value(d.enc, &d.words, f) == Some(1)) {
                let w = match n.rsplit('_').next().unwrap_or("") {
                    "DWORDX2" => 2,
                    "DWORDX4" => 4,
                    "DWORDX8" => 8,
                    "DWORDX16" => 16,
                    _ => 1,
                };
                slot = slot.max(w);
            }
        }
        lw.static_samplers = crate::gpu::recomp::constant_samplers(insts, leaders, matches!(entry, Entry::Vertex(_) | Entry::Pixel(_)));
        let group_threads: Option<u32> = match entry {
            Entry::Vertex(_) | Entry::Pixel(_) => None,
            Entry::Kernel => None,
            Entry::Compute(cs) => Some(cs.threads.iter().product()),
            Entry::Mesh(ms) => Some(ms.threads),
            Entry::MeshOnePass(m) => Some(m.ms.threads),
            Entry::Geometry(_) => Some(64),
        };
        if let (Some(t), true) = (group_threads, split) {
            lw.waves = t.div_ceil(64).max(1);
        }
        // PORT-CHECK: HEAD's barrier refusal here and its Spin partner below leave our MeshOnePass out (as the snapshot's code did)
        if let (Some(t), Entry::Compute(_) | Entry::Mesh(_)) = (group_threads, entry) {
            let hw_lanes = if split || !o.wave64 { 32 } else { 64 };
            lw.barrier_waves = Some(t.div_ceil(hw_lanes).max(1));
        }
        if split && std::env::var_os("XCL_X64_ALL_MASKS").is_none() {
            lw.whole_live = crate::gpu::recomp::x64_masks::by_pc(insts, kernel);
        }
        // (before the storage: the spin exchange has its own slots and counters)
        if let (Some(t), true) = (group_threads, split) {
            lw.partner = match t {
                0..=32 => Partner::Missing,
                // One console wave in two hardware halves. The halves of a compute or mesh
                // group meet through the bounded counters, as those of larger groups do: a
                // group barrier at every exchange hung NVIDIA GPUs when several such programs
                // ran at once (device hung, driver event 153: #1727; the change is #1741's).
                // A geometry program sets its partner by phase (`translate_on_this_thread`).
                33..=64 if !matches!(entry, Entry::Compute(_) | Entry::Mesh(_)) => Partner::Present,
                _ => Partner::Spin,
            };
        }
        // The LDS array sized to what the program allocates (RSRC2 LDS_SIZE), not the whole 32 KB:
        // groups that ask for little LDS fit many to a GPU core (a retail game's busiest program
        // allocates 512 B: 32 KB let ~3 of its groups onto an NVIDIA SM, -34 % GPU time per call
        // with the fit). `XCL_LDS_FIT=0`: always 32 KB, as before (the `ldsfit` variant still fits).
        let on = crate::gpu::recomp::lds_fit_enabled() || super::module::variant::current() & super::module::variant::LDSFIT != 0;
        let fit = match entry {
            Entry::Compute(cs) if cs.lds_bytes > 0 && on => Some(cs.lds_bytes / 4),
            // (a mesh shader has 28 KB of groupshared memory: always fitted)
            Entry::MeshOnePass(m) if m.lds_bytes > 0 => Some(m.lds_bytes / 4),
            _ => None,
        };
        lw.setup_storage(uses_lds, if split { slot } else { 1 }, fit);
        if let Entry::MeshOnePass(m) = entry {
            // what `setup_storage` declared, at most (the split form's exchange words included)
            let extra = if split { 4 * lw.xslot * lw.waves + 2 * lw.waves + 1 } else { 0 };
            let words = if uses_lds { fit.map_or(8192, |w| (w.max(1) + extra).min(8192)) } else { extra };
            if uses_lds && m.lds_bytes == 0 {
                lw.exchange_error = Some("one-pass mesh shader: the program's LDS size is not known".into());
            } else if words > 7168 {
                lw.exchange_error = Some(format!("one-pass mesh shader: {} B of groupshared memory, over 28 KB", 4 * words));
            }
        }
        lw
    }

    /// The finished container and the static samplers the program bound, once every block
    /// is closed: `numthreads` and the hardware wave size `hw` for a compute-like entry, the
    /// stage's I/O for a VS / PS.
    pub fn finish(self, numthreads: [u32; 3], hw: u32) -> Result<Kernel, String> {
        self.finish_dispatch(numthreads, hw, 0, None)
    }

    fn finish_dispatch(mut self, numthreads: [u32; 3], hw: u32, limit: usize, split_tid: Option<Value>) -> Result<Kernel, String> {
        if let Some(e) = self.exchange_error.take() {
            return Err(e);
        }
        let stage = self.s.stage;
        let mut dispatch_chain = None;
        let c = if matches!(stage, Stage::Vertex | Stage::Pixel) {
            let g = self.gfx.take().ok_or("graphics entry without graphics I/O")?;
            let info = if stage == Stage::Pixel { super::graphics::StageInfo::Pixel { early_depth_stencil: false } } else { super::graphics::StageInfo::Vertex };
            self.s.try_finish_graphics(g.io, info)
        } else if stage == Stage::Mesh {
            let o = self.mesh.take().and_then(|m| m.one).ok_or("mesh entry without mesh outputs")?;
            let info = super::graphics::StageInfo::Mesh {
                numthreads,
                max_vertices: o.max_vertices,
                max_primitives: o.max_primitives,
                topology: super::graphics::Topology::Triangle,
                payload_bytes: 0,
            };
            self.s.try_finish_graphics(o.io, info)
        } else if limit > 0 && (numthreads == [32, 1, 1] || split_tid.is_some()) && hw == 32 {
            self.s.m.optimize();
            // A single split wave64 meets its other half through group-shared counters its entry
            // zeroes (`Partner::Spin`), and the exchange's running number counts from the
            // program's start: a later piece of a chain, a new dispatch, would wait for counts
            // its own counters never reach. Such a program stays whole.
            if split_tid.is_some() && self.partner == Partner::Spin {
                let size: usize = (0..self.s.m.block_count()).map(|b| self.s.m.block_insts(Block(b as u32)).len()).sum();
                if size > limit {
                    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if n < 4 || n.is_power_of_two() {
                        crate::xlog_info!("compute split: a split wave64 compute program of {size} instructions stays whole: its halves exchange through group-shared counters a later dispatch would start again at zero ({} so far)", n + 1);
                    }
                }
            } else {
                dispatch_chain = self.s.split_compute(numthreads, split_tid, limit).map(std::sync::Arc::new);
            }
            self.s.finish_optimized(numthreads, Some(hw))
        } else {
            self.s.try_finish(numthreads, Some(hw))
        }
        .map_err(|e| format!("direct DXIL builder error: {e}"))?;
        if cfg!(any(test, debug_assertions)) {
            if let Err(e) = super::validate::validate(&c) {
                let dis = super::validate::disassemble(&c).unwrap_or_else(|d| d);
                return Err(format!("direct DXIL failed validation: {e}\n--- disassembly ---\n{dis}"));
            }
        }
        Ok(Kernel { dxil: c, static_samplers: self.static_used.iter().map(|&(w, c, _)| (w, c)).collect(), dispatch_chain })
    }
}

#[cfg(test)]
#[test]
fn dispatch_chain_native_wave64_builds_whole() {
    use crate::gpu::recomp::runner;

    let threads = [64, 1, 1];
    let cs = CsInterface { threads, ..Default::default() };
    let options = KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 };
    let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
    let (tid, _) = lw.setup_lanes_compute(threads, false);
    let offset = lw.shlc(tid, 2);
    let width = lw.s.wave_lane_count();
    let mem = lw.mems[0];
    // conditional updates: blocks the optimiser cannot merge into one
    for _ in 0..40 {
        let update = lw.s.m.new_block();
        let join = lw.s.m.new_block();
        let taken = lw.cmp(Pred::ULt, tid, width);
        lw.s.m.cond_br(taken, update, join);
        lw.s.m.set_block(update);
        let old = lw.s.raw_load(mem, offset, 1)[0];
        let new = lw.add(old, width);
        lw.s.raw_store(mem, offset, &[new]);
        lw.s.m.br(join);
        lw.s.m.set_block(join);
    }
    lw.s.m.ret();
    lw.s.m.optimize();
    // Exercise the native-wave admission gate with an otherwise partitionable CFG.
    assert!(lw.s.m.dispatch_plan(64).is_some());
    let program = lw.finish_dispatch(threads, 64, 64, None).unwrap();
    assert!(program.dispatch_chain.is_none());
    super::validate::validate(&program.dxil).expect("valid native wave64 whole shader");

    if !runner::wave64_supported().unwrap_or(false) {
        runner::skip_test("no wave64");
        return;
    }
    let initial: Vec<u32> = (1..=64).collect();
    let expected: Vec<_> = initial.iter().map(|v| v + 40 * 64).collect();
    let result = runner::dispatch(&program.dxil, [1, 1, 1], &[], &initial).unwrap();
    assert_eq!(result, expected);
}

/// A single split wave64 of a compute group meets its other half through group-shared counters
/// its entry zeroes (`Partner::Spin`): its program stays whole above the piece limit too.
#[cfg(test)]
#[test]
fn a_split_wave64_compute_program_stays_whole() {
    let threads = [64, 1, 1];
    let cs = CsInterface { threads, ..Default::default() };
    let options = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
    let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
    assert!(lw.partner == Partner::Spin);
    let (tid, _) = lw.setup_lanes_compute(threads, false);
    let offset = lw.shlc(tid, 2);
    let mem = lw.mems[0];
    let half = lw.half;
    let early = lw.exchange(half);
    // conditional updates: blocks the optimiser cannot merge into one (their edges go to the
    // SSA too: the spin exchange reads its running number through it)
    for _ in 0..40 {
        let update = lw.block(Some(1));
        let join = lw.block(Some(2));
        let taken = lw.cmpc(Pred::ULt, tid, 48);
        lw.cond_br(taken, update, join);
        lw.s.m.set_block(update);
        let old = lw.s.raw_load(mem, offset, 1)[0];
        let new = lw.add(old, early);
        lw.s.raw_store(mem, offset, &[new]);
        lw.br(join);
        lw.s.m.set_block(join);
    }
    let late = lw.exchange(tid);
    lw.s.raw_store(mem, offset, &[late]);
    lw.s.m.ret();
    lw.s.m.optimize();
    assert!(lw.s.m.dispatch_plan(64).is_some(), "the program could be cut");
    let program = lw.finish_dispatch(threads, 32, 64, Some(tid)).unwrap();
    assert!(program.dispatch_chain.is_none());
    super::validate::validate(&program.dxil).expect("valid split wave64 whole shader");
}

/// A compute program cut into a dispatch chain applies the thread limits (`compute_gate`) once,
/// in its first piece: a lane past one leaves there with the continuation tag 0, which the later
/// pieces take to their end, and EXEC reaches them as any other value. Each thread adds to its
/// word in 40 blocks the optimiser cannot merge, then stores EXEC's count after them.
#[cfg(test)]
#[test]
fn a_dispatch_chain_leaves_the_lanes_past_a_thread_limit_out() {
    use crate::gpu::recomp::{runner, IO_THREAD_LIMIT};
    let threads = [32, 1, 1];
    let cs = CsInterface { threads, user_sgprs: 2, tgid: [true, false, false], tg_size: false, tid_components: 1, lds_bytes: 0 };
    let options = KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 };
    let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs.clone()));
    let (tid, gtid) = lw.setup_lanes_compute(threads, false);
    let st = compute_init(&mut lw, &cs, tid, gtid);
    let gate = lw.block(Some(1));
    lw.edge(Block(0), gate);
    lw.s.m.set_block(gate);
    let runs = compute_gate(&mut lw, &st, true);
    let exec = lw.rd(126);
    let count = lw.s.bit_count(super::ops::BitCount::Countbits, exec);
    let (body, out) = (lw.s.m.new_block(), lw.s.m.new_block());
    lw.s.m.cond_br(runs, body, out);
    lw.s.m.set_block(out);
    lw.s.m.ret();
    lw.s.m.set_block(body);
    let group = lw.s.group_id(0);
    let k = lw.c(32);
    let first = lw.mul(group, k);
    let index = lw.add(first, tid);
    let offset = lw.shlc(index, 2);
    let mem = lw.mems[0];
    for _ in 0..40 {
        let (update, join) = (lw.s.m.new_block(), lw.s.m.new_block());
        let taken = lw.cmpc(Pred::ULt, tid, 32);
        lw.s.m.cond_br(taken, update, join);
        lw.s.m.set_block(update);
        let old = lw.s.raw_load(mem, offset, 1)[0];
        let new = lw.addc(old, 1);
        lw.s.raw_store(mem, offset, &[new]);
        lw.s.m.br(join);
        lw.s.m.set_block(join);
    }
    let at = lw.addc(offset, 4 * 32 * 4);
    lw.s.raw_store(mem, at, &[count]);
    lw.s.m.ret();
    lw.s.m.set_block(Block(0));
    lw.s.m.br(gate);
    lw.s.m.optimize();
    let chain = lw.s.split_compute(threads, None, 64).expect("a chain of pieces");
    assert!(chain.pieces.len() >= 2);
    // the limits are read once, by the first piece
    let load = format!("i32 {}, i32 undef, i8 7, i32 4)", IO_THREAD_LIMIT[0] * 4);
    for (i, piece) in chain.pieces.iter().enumerate() {
        super::validate::validate(piece).unwrap_or_else(|e| panic!("piece {i}: {e}"));
        let Ok(dis) = super::validate::disassemble(piece) else {
            return runner::skip_test("no DXC");
        };
        assert_eq!(dis.matches(&load).count(), (i == 0) as usize, "piece {i}\n{dis}");
    }
    if runner::wave32_supported() != Ok(true) {
        return runner::skip_test("no wave32 GPU");
    }
    // 4 groups: 128 counters, then 128 counts
    for l in [3 * 32 + 9, 0] {
        let mut io = vec![0u32; 128];
        io[IO_THREAD_LIMIT[0]] = l;
        let (mem, _) = runner::dispatch_split_samples(&chain, [4, 1, 1], &io, &[7u32; 256], u64::MAX, 1, false).unwrap();
        let l = if l == 0 { 128 } else { l };
        for t in 0..128u32 {
            let ran = t < l;
            assert_eq!(mem[t as usize], if ran { 47 } else { 7 }, "limit {l}: thread {t}");
            let n = (l.saturating_sub(t / 32 * 32)).min(32);
            assert_eq!(mem[128 + t as usize], if ran { n } else { 7 }, "limit {l}: thread {t}'s EXEC");
        }
    }
}

/// The kernel's epilogue: SGPRs and SCC from lane 0, every touched VGPR per lane.
fn store_registers(lw: &mut Lw, res: super::shader::RawUav) {
    let touched: Vec<u32> = lw.init.touched.iter().copied().collect();
    let mut sg: BTreeSet<u32> = touched.iter().copied().filter(|&v| v < VAR_VGPR && v != VAR_SCC).collect();
    sg.extend([106, 107, 124, 126, 127]);
    let vg: Vec<u32> = touched.iter().copied().filter(|&v| (VAR_VGPR..VAR_BUDGET).contains(&v)).collect();
    let svals: Vec<(u32, Value)> = sg.iter().map(|&r| (r, lw.rd(r))).collect();
    let scc = lw.rd(VAR_SCC);
    let vvals: Vec<(u32, Value)> = vg.iter().map(|&r| (r - VAR_VGPR, lw.rd(r))).collect();
    let lane = lw.lane;
    let first = lw.cmpc(Pred::Eq, lane, 0);
    let _ = lw.guarded(first, &mut |lw| {
        for &(r, v) in &svals {
            let off = lw.c(r * 4);
            lw.s.raw_store(res, off, &[v]);
        }
        let off = lw.c((KERNEL_SCC_SLOT * 4) as u32);
        lw.s.raw_store(res, off, &[scc]);
        Ok(())
    });
    let base = lw.init.vgpr_base.expect("vgpr base");
    for (r, v) in vvals {
        let k = lw.c(r * 4);
        let off = lw.add(base, k);
        lw.s.raw_store(res, off, &[v]);
    }
}

/// What the compute entry leaves to the block after the entry block (`compute_gate`).
pub struct ComputeStart {
    /// this group runs (not past a folded dispatch's count)
    pub runs: Value,
    /// this lane's thread is below every thread limit of the dispatch (`IO_THREAD_LIMIT`)
    pub inside: Value,
    /// split form: the other half's lanes inside the limits
    pub partner_lanes: Option<Value>,
}

/// The dispatch's thread limits (`recomp::IO_THREAD_LIMIT`, in one load), each less 1: the last
/// thread of an axis that runs. A count of 0 (no limit) wraps around and passes every thread.
pub fn thread_limits(lw: &mut Lw) -> [Value; 3] {
    let io = lw.init.io.expect("the I/O buffer");
    let o = lw.c((crate::gpu::recomp::IO_THREAD_LIMIT[0] * 4) as u32);
    let l = lw.s.raw_load(io, o, 0b111);
    [0, 1, 2].map(|k| lw.addc(l[k], u32::MAX))
}

/// Whether the thread `coords` of the group (x, y, z) is inside the thread limits `last`
/// (`thread_limits`): along each axis the console's group (`groups`, unfolded) times the group
/// size plus its coordinate is at most the last thread that runs.
pub fn inside_limits(lw: &mut Lw, threads: [u32; 3], groups: [Value; 3], last: [Value; 3], coords: [Value; 3]) -> Value {
    let mut inside = lw.b(true);
    for k in 0..3 {
        let n = lw.c(threads[k]);
        let t = lw.mul(groups[k], n);
        let t = lw.add(t, coords[k]);
        let ok = lw.cmp(Pred::ULe, t, last[k]);
        inside = lw.land(inside, ok);
    }
    inside
}

/// Split form: the other half's word of its threads inside the thread limits, from this half
/// alone (an exchange would meet the other half while its lanes are still all there, and count
/// them): lane j of this half tests the other half's j-th thread – in rounds of `r` lanes when
/// this half may have only `r` (a group's last console wave's upper half of 32 + `r` threads)
/// and the other half 32. `tid`: this lane's console thread in the group.
pub fn partner_word(lw: &mut Lw, threads: [u32; 3], groups: [Value; 3], last: [Value; 3], tid: Value) -> Value {
    let [x, y, _] = threads;
    let total: u32 = threads.iter().product();
    let r = match total % 64 {
        m if m > 32 => m - 32,
        _ => 32,
    };
    // the other half's first thread and its count (`Lw::xn`)
    let other = lw.andc(tid, !31);
    let k32 = lw.c(32);
    let other = lw.xor(other, k32);
    let (xn, lane) = (lw.xn, lw.lane31);
    let mut word = lw.c(0);
    for round in 0..32u32.div_ceil(r) {
        let i = lw.addc(lane, round * r);
        let t = lw.add(other, i);
        let (kx, kxy, ky) = (lw.c(x), lw.c(x * y), lw.c(y));
        let cx = lw.bin(super::module::BinOp::URem, t, kx);
        let q = lw.bin(super::module::BinOp::UDiv, t, kx);
        let cy = lw.bin(super::module::BinOp::URem, q, ky);
        let cz = lw.bin(super::module::BinOp::UDiv, t, kxy);
        let inside = inside_limits(lw, threads, groups, last, [cx, cy, cz]);
        let there = lw.cmp(Pred::ULt, i, xn);
        let bit = lw.land(inside, there);
        let w = lw.s.wave_ballot(bit)[0];
        let w = lw.shlc(w, round * r);
        word = lw.or(word, w);
    }
    word
}

/// The compute entry's fixed registers (the HLSL compute mode's prologue): group ids
/// after the user SGPRs, TG_SIZE, SCC = 0, EXEC = the wave's threads below the dispatch's thread
/// limits (`thread_limits`), v0..v2 = thread id. Returns whether this group runs: a dispatch
/// folded into Y (`recomp::IO_GROUP_LIMIT` in `io` not 0, the console's group count in X) takes
/// its group in X from both ids, and the groups past that count do not run; and which lanes are
/// inside the thread limits (`compute_gate` leaves the others out).
pub fn compute_init(lw: &mut Lw, cs: &CsInterface, tid: Value, gtid: [Value; 3]) -> ComputeStart {
    let io = lw.init.io.expect("the I/O buffer");
    let o = lw.c((crate::gpu::recomp::IO_GROUP_LIMIT * 4) as u32);
    let limit = lw.s.raw_load(io, o, 1)[0];
    let (gx, gy) = (lw.s.group_id(0), lw.s.group_id(1));
    let k = lw.c(crate::gpu::recomp::FOLD_X);
    let rows = lw.mul(gy, k);
    let flat = lw.add(gx, rows);
    let folded = lw.cmpc(Pred::Ne, limit, 0);
    let zero = lw.c(0);
    let ids = [lw.sel(folded, flat, gx), lw.sel(folded, zero, gy)];
    let inside = lw.cmp(Pred::ULt, flat, limit);
    let t = lw.b(true);
    let runs = lw.sel(folded, inside, t);
    let groups = [ids[0], ids[1], lw.s.group_id(2)];
    let last = thread_limits(lw);
    let inside = inside_limits(lw, cs.threads, groups, last, gtid);
    let mut sgpr = HashMap::new();
    let mut next = cs.user_sgprs;
    for k in 0..3 {
        if cs.tgid[k] {
            let g = if k < 2 { ids[k] } else { lw.s.group_id(2) };
            sgpr.insert(next, g);
            next += 1;
        }
    }
    let lanes: u32 = if lw.wave64 { 64 } else { 32 };
    let total: u32 = cs.threads.iter().product();
    if cs.tg_size {
        // the group's wave count (a partial last wave counts) | (gi / lanes) << 6
        let w = lw.c(total.div_ceil(lanes));
        let l = lw.c(lanes);
        let i = lw.bin(super::module::BinOp::UDiv, tid, l);
        let i = lw.shlc(i, 6);
        let v = lw.or(w, i);
        sgpr.insert(next, v);
    }
    let zero = lw.c(0);
    sgpr.insert(VAR_SCC, zero);
    let (lo, hi) = entry_exec(lw, total, tid);
    // the wave's threads inside the thread limits: this hardware wave's ballot (all its lanes
    // are here yet), in the split form this half's word and the other half's (`partner_word`)
    let w = lw.s.wave_ballot(inside);
    let (lo, hi, partner_lanes) = if lw.split {
        let partner = partner_word(lw, cs.threads, groups, last, tid);
        let n = lw.s.bit_count(super::ops::BitCount::Countbits, partner);
        let up = lw.upper;
        (lw.sel(up, partner, w[0]), lw.sel(up, w[0], partner), Some(n))
    } else if lw.wave64 {
        (lw.and(lo, w[0]), lw.and(hi, w[1]), None)
    } else {
        (lw.and(lo, w[0]), hi, None)
    };
    sgpr.insert(126, lo);
    sgpr.insert(127, hi);
    let mut vgpr = HashMap::new();
    for (k, &g) in gtid.iter().enumerate() {
        if k == 0 || cs.tid_components as usize > k {
            vgpr.insert(k as u32, g);
        }
    }
    lw.init.compute = Some(ComputeInit { sgpr, vgpr, zero });
    ComputeStart { runs, inside, partner_lanes }
}

/// The block after a compute entry's: with `leave` the lanes outside the thread limits go there
/// (the caller branches on the returned condition): they take part in nothing after, as threads
/// the console never started – no memory, no LDS, no ballot, no lane read. EXEC is then the
/// running lanes' bits: every running lane has its bit set (`Lw::live_exec`: the lane bit needs
/// no test, the program's vector writes no guard where they had none before), and a split
/// half meets the other one counting the lanes that run there (`Lw::xn`). Without `leave` (a
/// program with a group barrier, which a lane that left would not reach) they run on with their
/// EXEC bits clear. Returns whether this lane goes on into the program.
pub fn compute_gate(lw: &mut Lw, st: &ComputeStart, leave: bool) -> Value {
    if !leave {
        return st.runs;
    }
    if let Some(n) = st.partner_lanes {
        lw.xn = n;
    }
    let lo = lw.rd(126);
    let hi = if lw.wave64 { lw.rd(127) } else { lw.c(0) };
    lw.live_exec = Some((lo, hi));
    lw.land(st.runs, st.inside)
}

/// A compute program's count of passes for each entry of a loop (the HLSL compute mode's
/// `xcl_cap`): the dispatch's share for a group (`IO_LOOP_BUDGET`, loaded once in the entry
/// block; 0 = not known: `LOOP_BUDGET`) among its `threads`, between `DISPATCH_LOOP_PASSES_MIN`
/// and `LOOP_BUDGET`, set there for every loop. Returns it and the loops directly inside each
/// loop (`cfg::loop_parents`), whose counts that loop's head sets back to it.
fn loop_passes(lw: &mut Lw, cfg: &Cfg, threads: u32) -> (Value, BTreeMap<usize, Vec<usize>>) {
    use crate::gpu::recomp::{DISPATCH_LOOP_PASSES_MIN, IO_LOOP_BUDGET};
    let cur = lw.cur();
    let before = lw.s.m.set_insert_before_term(true);
    lw.s.m.set_block(lw.init.entry);
    let io = lw.init.io.expect("io");
    let off = lw.c(IO_LOOP_BUDGET as u32 * 4);
    let share = lw.s.raw_load(io, off, 1)[0];
    let k = lw.c(threads.max(1));
    let each = lw.bin(super::module::BinOp::UDiv, share, k);
    let lo = lw.c(DISPATCH_LOOP_PASSES_MIN);
    let each = lw.umax(each, lo);
    let hi = lw.c(LOOP_BUDGET);
    let each = lw.umin(each, hi);
    let unknown = lw.cmpc(Pred::Eq, share, 0);
    let cap = lw.sel(unknown, hi, each);
    let mut inside: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (h, around) in super::cfg::loop_parents(cfg) {
        lw.wr(VAR_LOOP_PASSES + h as u32, cap);
        if let Some(p) = around {
            inside.entry(p).or_default().push(h);
        }
    }
    lw.s.m.set_block(cur);
    lw.s.m.set_insert_before_term(before);
    (cap, inside)
}

/// The mesh entry's registers (the HLSL mesh mode's prologue): SGPRs from `io` with the
/// wave index in s3[27:24], SCC 0, EXEC = the wave's threads, v5 = group · threads + slot,
/// other VGPRs 0; and the record slots.
pub fn mesh_init(lw: &mut Lw, ms: &MsInterface, tid: Value, io: super::shader::RawUav) {
    let lanes: u32 = if lw.wave64 { 64 } else { 32 };
    let l = lw.c(lanes);
    let wave = lw.bin(super::module::BinOp::UDiv, tid, l);
    let lane = lw.lane;
    let base = lw.mul(wave, l);
    let slot = lw.add(base, lane);
    let gid = lw.s.group_id(0);
    let kr = lw.c(ms.record_dwords() as u32);
    let rec = lw.mul(gid, kr);
    let kv = lw.c(ms.vert_dwords() as u32);
    let sv = lw.mul(slot, kv);
    let vtx = lw.addc(rec, MESH_HEADER as u32);
    let vtx = lw.add(vtx, sv);
    let prm = lw.addc(rec, (MESH_HEADER + ms.max_vertices as usize * ms.vert_dwords()) as u32);
    let prm = lw.add(prm, slot);
    // A record has room for `max_vertices` vertices and `max_primitives` primitives; a thread
    // past either exports nothing (the console's hardware drops an export past what the program
    // asked room for). `res` is a root UAV: nothing bounds a store there, and one past the last
    // record of the buffer is a store to an unmapped address (a retail title's first frames
    // with mesh programs lost the device to a shader's access to unmapped memory).
    let vok = lw.cmpc(Pred::ULt, slot, ms.max_vertices);
    let pok = lw.cmpc(Pred::ULt, slot, ms.max_primitives);
    let zero = lw.c(0);
    let mut sgpr = HashMap::new();
    let o3 = lw.c(12);
    let s3 = lw.s.raw_load(io, o3, 1)[0];
    let s3 = lw.andc(s3, 0xF0FF_FFFF);
    let w24 = lw.shlc(wave, 24);
    let s3 = lw.or(s3, w24);
    sgpr.insert(3, s3);
    sgpr.insert(VAR_SCC, zero);
    let (lo, hi) = entry_exec(lw, ms.threads, tid);
    sgpr.insert(126, lo);
    sgpr.insert(127, hi);
    let kt = lw.c(ms.threads);
    let g = lw.mul(gid, kt);
    let v5 = lw.add(g, slot);
    let mut vgpr = HashMap::new();
    vgpr.insert(5, v5);
    lw.init.compute = Some(ComputeInit { sgpr, vgpr, zero });
    lw.mesh = Some(super::lower::MeshCtx { rec, vtx, prm, vok, pok, outputs: ms.outputs.clone(), slot, one: None });
}

/// EXEC at entry: the console wave's threads (static per wave in the split form: no
/// exchange); a constant when every wave of the group is full, so vector writes need no
/// lane guard.
fn entry_exec(lw: &mut Lw, total: u32, tid: Value) -> (Value, Value) {
    let zero = lw.c(0);
    let wave_threads: u32 = if lw.wave64 { 64 } else { 32 };
    let (lo, hi) = if total % wave_threads == 0 {
        let all = lw.c(u32::MAX);
        (all, if lw.wave64 { all } else { zero })
    } else if lw.split {
        let base = lw.andc(tid, !63);
        let t = lw.c(total);
        let n = lw.sub(t, base);
        let k64 = lw.c(64);
        let n = lw.umin(n, k64);
        let all = lw.c(u32::MAX);
        let one = lw.c(1);
        let n31 = lw.andc(n, 31);
        let part = lw.bin(super::module::BinOp::Shl, one, n31);
        let part = lw.sub(part, one);
        let ge32 = lw.cmpc(Pred::UGe, n, 32);
        let lo = lw.sel(ge32, all, part);
        let k32 = lw.c(32);
        let nh = lw.sub(n, k32);
        let nh31 = lw.andc(nh, 31);
        let parth = lw.bin(super::module::BinOp::Shl, one, nh31);
        let parth = lw.sub(parth, one);
        let ge64 = lw.cmpc(Pred::UGe, n, 64);
        let gt32 = lw.cmpc(Pred::UGt, n, 32);
        let hi = lw.sel(gt32, parth, zero);
        let hi = lw.sel(ge64, all, hi);
        (lo, hi)
    } else {
        let t = lw.s.m.i1c(true);
        let w = lw.s.wave_ballot(t);
        // HLSL ballot(): (m.x, m.y) — the second word is 0 in a 32-lane wave
        (w[0], if lw.wave64 { w[1] } else { zero })
    };
    (lo, hi)
}

/// The geometry entry's registers (the HLSL geometry mode's prologue): the group's
/// primitives from `io`, s2 = the GS output area, s3 = ES vertices | GS primitives << 8,
/// SCC 0, EXEC = every lane; v0 / v1 = LDS offsets of each GS lane's triangle vertices,
/// v2 = primitive id, v5 = vertex id, the fetch shader's VGPRs from the pre-fetched
/// vertices; and the GS state (`lower_geom`).
fn geom_init(lw: &mut Lw, g: &GsInterface, _tid: Value, io: super::shader::RawUav, used: &BTreeSet<u64>) {
    use crate::gpu::recomp::{fetched_registers, GS_PRIMS_PER_GROUP, IO_GS_PRIMS, IO_GS_VERTICES};
    let pg = GS_PRIMS_PER_GROUP;
    let rec_i = g.record();
    let o = lw.c((IO_GS_PRIMS * 4) as u32);
    let total = lw.s.raw_load(io, o, 1)[0];
    let o = lw.c((IO_GS_VERTICES * 4) as u32);
    let vin = lw.s.raw_load(io, o, 1)[0];
    let gid = lw.s.group_id(0);
    let kp = lw.c(pg);
    let p0 = lw.mul(gid, kp);
    let more = lw.cmp(Pred::UGt, total, p0);
    let left = lw.sub(total, p0);
    let left = lw.umin(left, kp);
    let zero = lw.c(0);
    let np = lw.sel(more, left, zero);
    let k3 = lw.c(3);
    let nv = lw.mul(np, k3);
    let kr = lw.c(rec_i.record_dwords() as u32);
    let rec = lw.mul(gid, kr);
    let mut sgpr = HashMap::new();
    let s2 = lw.c(g.es_ring_bytes());
    sgpr.insert(2, s2);
    let np8 = lw.shlc(np, 8);
    let s3 = lw.or(nv, np8);
    sgpr.insert(3, s3);
    sgpr.insert(VAR_SCC, zero);
    if lw.split && g.gs_start == Some(0) {
        // the GS alone runs in the lower hardware wave only: no partner half
        lw.partner = Partner::Missing;
        lw.xn = zero;
    }
    let t = lw.b(true);
    let (lo, hi) = lw.ballot(t);
    sgpr.insert(126, lo);
    sgpr.insert(127, hi);
    let lane = lw.lane;
    let item = g.es_item_bytes() / 4;
    let in_np = lw.cmp(Pred::ULt, lane, np);
    let l3 = lw.mul(lane, k3);
    let ki = lw.c(item);
    let a0 = lw.mul(l3, ki);
    let l31 = lw.addc(l3, 1);
    let a1 = lw.mul(l31, ki);
    let a1 = lw.shlc(a1, 16);
    let v0 = lw.or(a0, a1);
    let v0 = lw.sel(in_np, v0, zero);
    let l32 = lw.addc(l3, 2);
    let v1 = lw.mul(l32, ki);
    let v1 = lw.sel(in_np, v1, zero);
    let v2 = lw.add(p0, lane);
    let p03 = lw.mul(p0, k3);
    let v5 = lw.add(p03, lane);
    let mut vgpr = HashMap::new();
    vgpr.insert(0, v0);
    vgpr.insert(1, v1);
    vgpr.insert(2, v2);
    vgpr.insert(5, v5);
    let nreg = g.input_registers();
    let fetched = fetched_registers(&g.es_inputs);
    let in_nv = lw.cmp(Pred::ULt, lane, nv);
    let mut mem = None;
    for &v in used.iter().filter(|&&v| (9..256).contains(&v)) {
        let k = ((v - 9) / 4) as usize;
        if k >= fetched.len() || fetched[k] >= nreg {
            continue;
        }
        let mem = *mem.get_or_insert_with(|| match lw.heap_mem {
            Some(coherent) => {
                let idx = lw.c(crate::gpu::recomp::MEM_HEAP_BASE);
                lw.s.heap_raw_buffer_uav(idx, coherent)
            }
            None => lw.mems[0],
        });
        let (r, c) = (fetched[k], ((v - 9) % 4) as u32);
        // mem.Load(vin + ((p0 * 3 + lane) * nreg + r) * 16 + 4c), 0 past the group's vertices
        let kn = lw.c(nreg);
        let x = lw.mul(v5, kn);
        let x = lw.addc(x, r);
        let x = lw.shlc(x, 4);
        let x = lw.add(vin, x);
        let x = lw.addc(x, 4 * c);
        let val = lw.s.raw_load(mem, x, 1)[0];
        let val = lw.sel(in_nv, val, zero);
        vgpr.insert(v as u32, val);
    }
    lw.init.compute = Some(ComputeInit { sgpr, vgpr, zero });
    let maxt = g.max_vertex_count.saturating_sub(2).max(1);
    let i32 = lw.s.m.i32();
    let arr = lw.s.m.array(maxt as u64, i32);
    let tri = lw.s.m.alloca(arr, 4);
    let (emitted, strip, ntri, base) = (lw.tmp_var(), lw.tmp_var(), lw.tmp_var(), lw.tmp_var());
    for v in [emitted, strip, ntri] {
        lw.wr(v, zero);
    }
    let none = lw.c(u32::MAX);
    lw.wr(base, none);
    lw.geom = Some(super::lower_geom::GeomCtx { g: g.clone(), emitted, strip, ntri, base, tri, maxt, rec });
}
