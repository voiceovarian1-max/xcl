//! The counts of an ExecuteIndirect, looked at before the GPU runs them.
//!
//! The counts come from memory the GPU (or an older copy) wrote. Where they are not what the
//! title meant, they can be anything: a float's bits are a billion groups, and a dispatch of a
//! billion groups or a draw of a billion vertices holds the GPU for seconds or until Windows
//! resets the driver (a retail title's world on an RTX 5090: culling passes that ran for 1.4 to
//! 30 s, and an indirect draw that ended in a driver error). So a program of XCL's own reads the
//! counts first and hands them to the ExecuteIndirect through a buffer of XCL's own: zeros where
//! they are out of range. It also writes them where the CPU reads them: the log names the first
//! ones out of range, and the profile (`XCL_GPU_PROFILE=1`) shows what an ExecuteIndirect ran.
//!
//! `XCL_INDIRECT_GUARD=0` turns it off (`=dispatch`: only dispatches are checked);
//! `XCL_INDIRECT_MAX_GROUPS` is the most groups one ExecuteIndirect may run (4 Mi: a 4K target
//! in 8x8 groups has 130 thousand).

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::{transmute_copy, ManuallyDrop};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use super::real;

/// ExecuteIndirects whose counts are kept (a ring). A title can submit a recording again seconds
/// later, when its slot has been given to another: then one of the two runs the other's counts
/// (in range, but another program's), so the ring holds minutes of them.
const SLOTS: u32 = 1 << 18;
/// bytes of a slot in `out` and in `seen`: the arguments (5 words at most), in `seen` then 1
const SLOT: u32 = 32;
/// the word of a `seen` slot that says the GPU read it
const SEEN_WORD: u32 = 5;
/// `seen`: how many counts were out of range, how many the title's buffer showed differently, how
/// many of either kind, the first of these, then the slots
const BAD_AT: u32 = 32;
const BAD_SIZE: u32 = 64;
const BAD_KEPT: u32 = 128;
const SEEN_AT: u32 = BAD_AT + BAD_KEPT * BAD_SIZE;

/// vertices (indices) and instances one draw may have, and their product: far above what a
/// frame draws
const MAX_VERTICES: u32 = 1 << 28;
const MAX_INSTANCES: u32 = 1 << 24;

/// `XCL_INDIRECT_GUARD=compare`: what the title's buffer shows is read as well and compared with
/// the memory window (a diagnostic: the two are one memory).
fn compare() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var("XCL_INDIRECT_GUARD").is_ok_and(|v| v.trim() == "compare"));
    *ON
}

/// What a command signature of one argument runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Counts {
    /// groups x, y, z
    Dispatch = 0,
    /// vertices, instances, first vertex, first instance
    Draw = 1,
    /// indices, instances, first index, base vertex, first instance
    DrawIndexed = 2,
}

fn on(kind: Counts) -> bool {
    static MODE: LazyLock<u32> = LazyLock::new(|| match std::env::var("XCL_INDIRECT_GUARD").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("dispatch") => 1,
        _ => 2,
    });
    *MODE == 2 || (*MODE == 1 && kind == Counts::Dispatch)
}

fn repeat() -> u32 {
    static N: LazyLock<u32> = LazyLock::new(|| std::env::var("XCL_INDIRECT_GUARD_REPEAT").ok().and_then(|v| v.trim().parse().ok()).filter(|&n| (1..=8).contains(&n)).unwrap_or(1));
    *N
}

/// The groups in all a dispatch whose counts XCL has to check, fold or clamp may have
/// (`XCL_INDIRECT_MAX_GROUPS`, 2^22 unless set).
pub(super) fn max_groups() -> u32 {
    static MAX: LazyLock<u32> = LazyLock::new(|| std::env::var("XCL_INDIRECT_MAX_GROUPS").ok().and_then(|v| v.trim().parse().ok()).filter(|&n| n > 0).unwrap_or(1 << 22));
    *MAX
}

/// command signatures of one dispatch or draw and nothing else (its counts come first)
static PLAIN: Mutex<Option<HashMap<usize, Counts>>> = Mutex::new(None);

/// A command signature was made from arguments of the types `types` (at each creation: one at a
/// freed one's address does not keep what that was).
pub(super) fn note_signature(sig: usize, types: &[i32]) {
    let counts = match types {
        [t] if *t == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH.0 => Some(Counts::Dispatch),
        [t] if *t == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW.0 => Some(Counts::Draw),
        [t] if *t == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED.0 => Some(Counts::DrawIndexed),
        _ => None,
    };
    let mut g = PLAIN.lock().unwrap();
    let g = g.get_or_insert_with(Default::default);
    match counts {
        Some(c) => {
            g.insert(sig, c);
        }
        None => {
            g.remove(&sig);
        }
    }
}

/// Whether `sig` is a signature of one dispatch and nothing else (its group counts first).
pub(super) fn plain_dispatch(sig: usize) -> bool {
    PLAIN.lock().unwrap().as_ref().and_then(|g| g.get(&sig).copied()) == Some(Counts::Dispatch)
}

fn program() -> String {
    format!(
        r#"
cbuffer C : register(b0) {{ uint max_groups, slot, id_lo, id_hi, at_lo, at_hi, kind; }};
RWByteAddressBuffer args : register(u0);
RWByteAddressBuffer outp : register(u1);
RWByteAddressBuffer seen : register(u2);
// the same memory as the title's buffer shows it
RWByteAddressBuffer shown : register(u3);
[numthreads(1, 1, 1)]
void main() {{
    // (what lies behind the arguments may be no memory at all)
    uint4 a = uint4(args.Load3(0u), 0u);
    uint4 b = uint4(shown.Load3(0u), 0u);
    uint a4 = 0u;
    uint b4 = 0u;
    [branch] if (kind != 0u) {{ a.w = args.Load(12u); b.w = shown.Load(12u); }}
    [branch] if (kind == 2u) {{ a4 = args.Load(16u); b4 = shown.Load(16u); }}
    // (as floats: the product of the counts does not fit 32 bits; a count of 0 runs nothing)
    bool ok;
    if (kind == 0u)
        ok = a.x == 0u || a.y == 0u || a.z == 0u || (a.x <= max_groups && a.y <= 65535u && a.z <= 65535u && (float)a.x * (float)a.y * (float)a.z <= (float)max_groups);
    else
        ok = a.x == 0u || a.y == 0u || (a.x <= {MAX_VERTICES}u && a.y <= {MAX_INSTANCES}u && (float)a.x * (float)a.y <= (float){MAX_VERTICES}u);
    uint at = slot * {SLOT}u;
    outp.Store4(at, ok ? a : uint4(0u, 0u, 0u, 0u));
    outp.Store(at + 16u, ok ? a4 : 0u);
    seen.Store4({SEEN_AT}u + at, a);
    seen.Store2({SEEN_AT}u + at + 16u, uint2(a4, 1u));
    bool differs = any(a != b) || a4 != b4;
    if (differs) {{
        uint m;
        seen.InterlockedAdd(4u, 1u, m);
    }}
    if (!ok) {{
        uint m;
        seen.InterlockedAdd(0u, 1u, m);
    }}
    if (!ok || differs) {{
        uint n;
        seen.InterlockedAdd(8u, 1u, n);
        if (n < {BAD_KEPT}u) {{
            uint r = {BAD_AT}u + n * {BAD_SIZE}u;
            seen.Store4(r, a);
            seen.Store4(r + 16u, uint4(id_lo, id_hi, at_lo, at_hi));
            seen.Store4(r + 32u, b);
            // (the last word says that the record is there)
            seen.Store4(r + 48u, uint4(kind, (differs ? 1u : 0u) | (ok ? 0u : 2u), slot, 1u));
        }}
    }}
}}
"#
    )
}

struct Guard {
    dev: usize,
    rs: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// the arguments the ExecuteIndirects read: COMMON between its uses
    out: ID3D12Resource,
    out_va: u64,
    /// what the GPU found, on the CPU
    _seen: ID3D12Resource,
    seen_va: u64,
    seen_at: usize,
    next: AtomicU32,
}

// SAFETY: the mapping lives as long as `_seen`; D3D12 objects are free-threaded.
unsafe impl Send for Guard {}
unsafe impl Sync for Guard {}

impl Guard {
    fn word(&self, byte: u32) -> &AtomicU32 {
        unsafe { &*((self.seen_at + byte as usize) as *const AtomicU32) }
    }
}

static GUARD: Mutex<Option<Arc<Guard>>> = Mutex::new(None);

/// The device's guard, made on first use; None: not possible (logged once)
unsafe fn guard_of(list: &ID3D12GraphicsCommandList) -> Option<Arc<Guard>> {
    static FAILED: AtomicU32 = AtomicU32::new(0);
    if FAILED.load(Ordering::Relaxed) != 0 {
        return None;
    }
    let mut dev: Option<ID3D12Device> = None;
    list.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let mut g = GUARD.lock().unwrap();
    if let Some(g) = g.as_ref().filter(|g| g.dev == dev.as_raw() as usize) {
        return Some(g.clone());
    }
    match super::title_textures::own_work(|| make(&dev)) {
        Ok(n) => {
            let n = Arc::new(n);
            *g = Some(n.clone());
            Some(n)
        }
        Err(e) => {
            FAILED.store(1, Ordering::Relaxed);
            crate::xlog_warn!("D3D12.X: ExecuteIndirect: the counts are not checked ({e})");
            None
        }
    }
}

unsafe fn make(dev: &ID3D12Device) -> Result<Guard, String> {
    let cs = real::compile(&program(), "cs_5_0");
    if cs.is_empty() {
        return Err("its program does not compile".into());
    }
    let root = |ty, anonymous| D3D12_ROOT_PARAMETER { ParameterType: ty, Anonymous: anonymous, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL };
    let uav = |k| root(D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: k, RegisterSpace: 0 } });
    let params = [
        root(D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 7 } }),
        uav(0),
        uav(1),
        uav(2),
        uav(3),
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: params.len() as u32, pParameters: params.as_ptr(), ..Default::default() };
    let mut blob: Option<ID3DBlob> = None;
    D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).map_err(|e| format!("root signature: {e}"))?;
    let blob = blob.ok_or("root signature")?;
    let rs: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())).map_err(|e| format!("CreateRootSignature: {e}"))?;
    let code = D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as *const c_void, BytecodeLength: cs.len() };
    let pso: ID3D12PipelineState = dev
        .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&rs), CS: code, ..Default::default() })
        .map_err(|e| format!("CreateComputePipelineState: {e}"))?;
    let buffer = |heap: D3D12_HEAP_PROPERTIES, width: u64| -> Result<ID3D12Resource, String> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: width,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut r: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).map_err(|e| format!("a buffer of {width} bytes: {e}"))?;
        r.ok_or_else(|| "a buffer".into())
    };
    let out = buffer(D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, SLOTS as u64 * SLOT as u64)?;
    let cpu = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
    let bytes = SEEN_AT as u64 + SLOTS as u64 * SLOT as u64;
    let seen = buffer(cpu, bytes)?;
    let mut p = std::ptr::null_mut();
    seen.Map(0, None, Some(&mut p)).map_err(|e| format!("Map: {e}"))?;
    std::ptr::write_bytes(p.cast::<u8>(), 0, bytes as usize);
    Ok(Guard {
        dev: dev.as_raw() as usize,
        out_va: out.GetGPUVirtualAddress(),
        seen_va: seen.GetGPUVirtualAddress(),
        seen_at: p as usize,
        rs,
        pso,
        out,
        _seen: seen,
        next: AtomicU32::new(0),
    })
}

/// bytes of an argument record the program reads (`check`)
const COUNT_BYTES: usize = 5 * 4;

/// The GPU address of the arguments at UMA address `t` through their memory window, when all
/// `COUNT_BYTES` of them are in one window. The program reads them through a root UAV, which
/// has no bounds: a record across a view window's end would be read past that window's buffer
/// (the view windows are separate 4 GiB resources). None: the caller reads the title's buffer.
fn counts_va(t: usize) -> Option<u64> {
    counts_va_in(t, |p, n| real::uma_window_buffer_gpu(p, n).is_some(), real::uma_va_of)
}

/// `counts_va` with the window lookups given: whether `n` bytes at an address are in one
/// window, and an address's GPU address.
fn counts_va_in(t: usize, in_one_window: impl Fn(usize, usize) -> bool, va_of: impl Fn(usize) -> Option<u64>) -> Option<u64> {
    in_one_window(t, COUNT_BYTES).then(|| va_of(t)).flatten()
}

/// An ExecuteIndirect whose counts XCL's program reads: what it runs instead of the title's.
pub(super) struct Checked {
    guard: Arc<Guard>,
    pub(super) slot: u32,
}

/// Record the check of an ExecuteIndirect's counts (`args` + `args_off`, the PC buffer and the
/// offset in it) on `list`; None: not one this checks, it runs as it is. Binds a compute root
/// signature and a pipeline: the caller binds the dispatch's or draw's after it.
pub(super) unsafe fn check(list: *mut c_void, pso: usize, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize) -> Option<Checked> {
    if max != 1 || count_buf != 0 {
        return None;
    }
    let kind = *PLAIN.lock().unwrap().as_ref()?.get(&sig)?;
    if !on(kind) {
        return None;
    }
    let ty = super::list_x::tracked_list_type(list)?;
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT && ty != D3D12_COMMAND_LIST_TYPE_COMPUTE {
        return None;
    }
    let l = ID3D12GraphicsCommandList::from_raw_borrowed(&list)?;
    let res = args as *mut c_void;
    let va = ID3D12Resource::from_raw_borrowed(&res)?.GetGPUVirtualAddress();
    if va == 0 || (va + args_off) % 4 != 0 {
        return None;
    }
    let shown = va + args_off;
    // The memory window is what the title's shaders write: the counts are read there. (The
    // title's buffer is a resource of its own over the same video memory.)
    let delta = real::buffer_delta(args);
    let title = real::console_buffer(args).and_then(|(p, _)| Some(p + args_off.checked_sub(delta)? as usize));
    // (in a GPU view: its window, which has the arguments once their tiles are asked for; the
    // program reads them on the GPU only)
    if let Some(t) = title {
        super::bind::want_view(super::view_want::Path::MemoryOp, t as u64, COUNT_BYTES as u64);
    }
    let va = title.and_then(counts_va).filter(|v| v % 4 == 0).unwrap_or(shown);
    let shown = if compare() { shown } else { va };
    let g = guard_of(l)?;
    let slot = g.next.fetch_add(1, Ordering::Relaxed) % SLOTS;
    g.word(SEEN_AT + slot * SLOT + 4 * SEEN_WORD).store(0, Ordering::Release);
    let id = real::pso_shaders(pso).map_or(0, |(a, _)| a);
    // the counts were written as UAVs: they are read after those writes
    let written = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }) },
    };
    l.ResourceBarrier(&[written]);
    l.SetComputeRootSignature(&g.rs);
    // (past the hook of a title's call: what the title bound stays the list's pipeline)
    super::list_x::set_compute_pipeline(list, g.pso.as_raw());
    // (the title's address, for the log; a buffer that is not in unified memory: the GPU's)
    let at = title.map_or(va, |t| t as u64);
    let constants = [max_groups(), slot, id as u32, (id >> 32) as u32, at as u32, (at >> 32) as u32, kind as u32];
    l.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr() as *const c_void, 0);
    l.SetComputeRootUnorderedAccessView(1, va);
    l.SetComputeRootUnorderedAccessView(2, g.out_va);
    l.SetComputeRootUnorderedAccessView(3, g.seen_va);
    l.SetComputeRootUnorderedAccessView(4, shown);
    l.Dispatch(1, 1, 1);
    report(&g);
    Some(Checked { guard: g, slot })
}

impl Checked {
    /// `op(buffer, offset)`: the ExecuteIndirect with the checked counts.
    pub(super) unsafe fn run(&self, list: *mut c_void, op: impl FnOnce(usize, u64)) {
        let out = self.guard.out.as_raw() as usize;
        // (the driver reads the counts after the program's writes with this transition only,
        // `bind::fresh_indirect_arguments`)
        super::list_x::transition(list, out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        op(out, self.slot as u64 * SLOT as u64);
        super::list_x::transition(list, out, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
    }

    /// Where the checked counts are (buffer, offset), for a pass that reads them as a UAV in place
    /// of `run` (`clamp::before`); `read` after it.
    pub(super) fn arguments(&self) -> (usize, u64) {
        (self.guard.out.as_raw() as usize, self.slot as u64 * SLOT as u64)
    }

    /// After a pass read the counts at `arguments`: the buffer back to COMMON (its write and that
    /// read left it in UNORDERED_ACCESS).
    pub(super) unsafe fn read(&self, list: *mut c_void) {
        super::list_x::transition(list, self.guard.out.as_raw() as usize, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    }
}

/// The counts slot `slot`'s ExecuteIndirect was asked to run (groups; vertices or indices,
/// instances, the first of them), once the GPU has read them.
pub(super) fn counts(slot: u32) -> Option<[u32; 3]> {
    let g = GUARD.lock().unwrap().clone()?;
    let at = SEEN_AT + slot * SLOT;
    (g.word(at + 4 * SEEN_WORD).load(Ordering::Acquire) != 0).then(|| std::array::from_fn(|k| g.word(at + 4 * k as u32).load(Ordering::Relaxed)))
}

/// Counts out of range so far.
fn bad_so_far(g: &Guard) -> u32 {
    g.word(0).load(Ordering::Acquire)
}

/// A counter that stood at `last` when it was last logged and stands at `now`: is it logged again
/// (the first few, then at each doubling)? It is looked at now and then, so it jumps.
fn logged_again(last: u64, now: u64) -> bool {
    now > last && (last < 4 || now.ilog2() > last.ilog2())
}

/// `report` from elsewhere (the GPU wait watchdog: nothing may be recorded any more).
pub(super) fn report_now() {
    let g = GUARD.lock().unwrap().clone();
    if let Some(g) = g {
        report(&g);
    }
}

/// Log the counts found out of range since the last time (and how often the window and the
/// title's buffer held different ones).
fn report(g: &Guard) {
    static DIFFERED: AtomicU64 = AtomicU64::new(0);
    static BAD: AtomicU64 = AtomicU64::new(0);
    static RECORDS: Mutex<u32> = Mutex::new(0);
    let (bad, d, n) = (bad_so_far(g) as u64, g.word(4).load(Ordering::Acquire) as u64, g.word(8).load(Ordering::Acquire));
    let records = RECORDS.try_lock();
    let Ok(mut records) = records else { return };
    while *records < n.min(BAD_KEPT) {
        let at = BAD_AT + *records * BAD_SIZE;
        // (a record the GPU has counted but not written yet is logged the next time)
        if g.word(at + 60).load(Ordering::Acquire) == 0 {
            break;
        }
        *records += 1;
        let w: [u32; 16] = std::array::from_fn(|i| g.word(at + 4 * i as u32).load(Ordering::Relaxed));
        let id = w[4] as u64 | (w[5] as u64) << 32;
        let title = w[6] as u64 | (w[7] as u64) << 32;
        let what = match w[12] {
            0 => format!("CS {id:016x}: {:#x} x {:#x} x {:#x} groups", w[0], w[1], w[2]),
            _ => format!("VS {id:016x}: {:#x} vertices or indices x {:#x} instances", w[0], w[1]),
        };
        let shown = if w[13] & 1 != 0 { format!(" (the title's buffer shows {:#x}, {:#x}, {:#x})", w[8], w[9], w[10]) } else { String::new() };
        let verdict = if w[13] & 2 != 0 { "out of range, not run" } else { "run" };
        crate::xlog_warn!("D3D12.X: ExecuteIndirect of {what}, arguments at {title:#x}{shown} – {verdict} ({bad} out of range so far)");
    }
    drop(records);
    for (counter, now, what) in [(&BAD, bad, "counts out of range so far, not run"), (&DIFFERED, d, "time(s) the title's buffer did not show the counts the memory window holds")] {
        let last = counter.load(Ordering::Relaxed);
        if logged_again(last, now) && counter.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_ok() && last >= BAD_KEPT as u64 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect: {now} {what}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::list_x;
    use super::*;
    use crate::gpu::{fence, recomp::runner::skip_test};
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

    type Close = unsafe extern "system" fn(*mut c_void) -> windows::core::HRESULT;

    /// the guard (and its count of what was out of range) is the device's: one test at a time
    static ONE: Mutex<()> = Mutex::new(());

    unsafe fn device() -> Option<ID3D12Device> {
        let mut dev: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            skip_test("no GPU");
            return None;
        }
        dev
    }

    /// A buffer the CPU and the GPU see, and its mapping.
    unsafe fn cpu_buffer(dev: &ID3D12Device, width: u64) -> (ID3D12Resource, *mut u32) {
        let cpu = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: width,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut r: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&cpu, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
        let r = r.unwrap();
        let mut p = std::ptr::null_mut();
        r.Map(0, None, Some(&mut p)).unwrap();
        (r, p as *mut u32)
    }

    unsafe fn signature(dev: &ID3D12Device, ty: D3D12_INDIRECT_ARGUMENT_TYPE, stride: u32) -> ID3D12CommandSignature {
        let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: ty, ..Default::default() };
        let mut sig: Option<ID3D12CommandSignature> = None;
        dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: stride, NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 }, None, &mut sig).unwrap();
        let sig = sig.unwrap();
        note_signature(sig.as_raw() as usize, &[ty.0]);
        sig
    }

    /// A list of `ty` as XCL has a title's.
    unsafe fn title_list(dev: &ID3D12Device, ty: D3D12_COMMAND_LIST_TYPE) -> (ID3D12CommandAllocator, ID3D12GraphicsCommandList) {
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
        real::adopt(list.as_raw(), real::Kind::CommandList);
        list_x::on_create(list.as_raw(), alloc.as_raw() as usize, 0, ty);
        (alloc, list)
    }

    unsafe fn run(dev: &ID3D12Device, queue: &ID3D12CommandQueue, list: &ID3D12GraphicsCommandList) {
        let close: Close = std::mem::transmute(real::orig_fn(list.as_raw(), 9));
        close(list.as_raw()).unwrap();
        let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        queue.Signal(&done, 1).unwrap();
        fence::wait(&done, 1, dev, None, 10_000).unwrap();
    }

    /// The program reads the arguments through their memory window only when the whole record is
    /// in that window: one across a window's end is read through the title's buffer instead.
    #[test]
    fn arguments_across_a_window_end_are_not_read_through_the_window() {
        // windows of 0x1000 bytes from 0x1_0000, each at its own GPU address (separate buffers)
        let in_one_window = |p: usize, n: usize| p >= 0x1_0000 && (p - 0x1_0000) % 0x1000 + n <= 0x1000;
        let va_of = |p: usize| (p >= 0x1_0000).then(|| 0x7000_0000 + ((p - 0x1_0000) / 0x1000) as u64 * 0x10_0000 + ((p - 0x1_0000) % 0x1000) as u64);
        assert_eq!(counts_va_in(0x1_0100, in_one_window, va_of), Some(0x7000_0100));
        assert_eq!(counts_va_in(0x1_1000 - COUNT_BYTES, in_one_window, va_of), Some(0x7000_1000 - COUNT_BYTES as u64), "the last record in a window");
        assert_eq!(counts_va_in(0x1_1000 - 8, in_one_window, va_of), None, "a record across the window's end");
        assert_eq!(counts_va_in(0x100, in_one_window, va_of), None, "not in a window");
    }

    #[test]
    fn a_counter_is_logged_at_its_first_values_and_at_each_doubling_however_it_jumps() {
        assert!(logged_again(0, 1) && logged_again(0, 7) && logged_again(3, 5));
        assert!(logged_again(7, 8) && logged_again(7, 100) && logged_again(64, 129));
        assert!(!logged_again(7, 7) && !logged_again(8, 15) && !logged_again(100, 127));
    }

    /// Garbage counts run 0 groups and are kept for the log; counts in range run as they are. Each
    /// case takes the slot the one before it ran from, so that a guard that writes nothing would
    /// run that one's counts.
    #[test]
    fn an_execute_indirect_with_garbage_counts_runs_no_groups() {
        // the counts the memory holds, and the groups that may run
        let cases: [([u32; 3], u32); 8] = [
            ([7, 1, 1], 7),
            ([3, 4, 5], 60),
            ([0, 1, 1], 0),
            ([7, 1, 1], 7),
            // (garbage beside a 0 runs nothing as it is)
            ([0x3F80_0000, 0, 1], 0),
            // the bits of 1.0f, a count below 2^32 that is none, and a product too large
            ([0x3F80_0000, 1, 1], 0),
            ([1, 0xFFFF_FFFF, 1], 0),
            ([60_000, 60_000, 2], 0),
        ];
        let src = r#"
RWByteAddressBuffer c : register(u0);
[numthreads(1, 1, 1)]
void main() { c.InterlockedAdd(0, 1u); }
"#;
        unsafe {
            let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
            let Some(dev) = device() else { return };
            let cs = real::compile(src, "cs_5_0");
            assert!(!cs.is_empty());
            let param = D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: 1, pParameters: &param, ..Default::default() };
            let mut blob: Option<ID3DBlob> = None;
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).unwrap();
            let blob = blob.unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())).unwrap();
            let code = D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as _, BytecodeLength: cs.len() };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&rs), CS: code, ..Default::default() }).unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let sig = signature(&dev, D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, 12);
            let ((args, args_at), (groups, groups_at)) = (cpu_buffer(&dev, 256), cpu_buffer(&dev, 256));
            let s = sig.as_raw() as usize;
            let a = args.as_raw() as usize;
            for (k, (want, runs)) in cases.iter().enumerate() {
                // (at an offset: the title's arguments are anywhere in its buffer)
                let off = 4 * k as u64;
                for (i, w) in want.iter().enumerate() {
                    *args_at.add(k + i) = *w;
                }
                *groups_at = 0;
                let (_alloc, list) = title_list(&dev, ty);
                let raw = list.as_raw();
                type Ei = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
                let ei: Ei = std::mem::transmute(real::orig_fn(raw, 59));
                let before = GUARD.lock().unwrap().as_ref().map_or(0, |g| bad_so_far(g));
                if let Some(g) = GUARD.lock().unwrap().as_ref() {
                    g.next.store(0, Ordering::Relaxed);
                }
                let checked = check(raw, 0, s, 1, a, off, 0).expect("checked");
                assert!(k == 0 || checked.slot == 0);
                list.SetComputeRootSignature(&rs);
                list.SetPipelineState(&pso);
                list.SetComputeRootUnorderedAccessView(0, groups.GetGPUVirtualAddress());
                checked.run(raw, |b, o| ei(raw, s, 1, b, o, 0, 0));
                run(&dev, &queue, &list);
                assert_eq!(std::ptr::read_volatile(groups_at), *runs, "{want:x?}: groups run");
                assert_eq!(counts(checked.slot), Some(*want), "{want:x?}: what the CPU sees");
                let bad = bad_so_far(&checked.guard) - before;
                assert_eq!(bad, (*runs == 0 && !want.contains(&0)) as u32, "{want:x?}: counted as out of range");
            }
            // a count buffer or several commands: not checked; nor a signature with more arguments
            let (_alloc, list) = title_list(&dev, ty);
            let raw = list.as_raw();
            assert!(check(raw, 0, s, 2, a, 0, 0).is_none());
            assert!(check(raw, 0, s, 1, a, 0, a).is_none());
            note_signature(s, &[D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT.0, D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH.0]);
            assert!(check(raw, 0, s, 1, a, 0, 0).is_none());
        }
    }

    /// The arguments a draw gets: all of them as they are, or zeros where a count is garbage.
    #[test]
    fn an_execute_indirect_with_garbage_counts_draws_nothing() {
        let cases: [(bool, [u32; 5], bool); 6] = [
            (false, [3, 2, 7, 9, 0xDEAD], true),
            (true, [36, 100_000, 5, 0xFFFF_FFF0, 3], true),
            (true, [0, 0, 0, 0, 0], true),
            (true, [0x3F80_0000, 1, 0, 0, 0], false),
            (false, [3, 0xFFFF_FFFF, 0, 0, 0], false),
            (true, [1 << 20, 1 << 20, 0, 0, 0], false),
        ];
        unsafe {
            let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
            let Some(dev) = device() else { return };
            let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let plain = signature(&dev, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, 16);
            let indexed = signature(&dev, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, 20);
            let ((args, args_at), (got, got_at)) = (cpu_buffer(&dev, 256), cpu_buffer(&dev, 256));
            for (is_indexed, want, ok) in cases {
                let words = if is_indexed { 5 } else { 4 };
                for (i, w) in want.iter().enumerate() {
                    *args_at.add(i) = *w;
                    *got_at.add(i) = 0xAAAA_AAAA;
                }
                let (_alloc, list) = title_list(&dev, ty);
                let raw = list.as_raw();
                let sig = if is_indexed { &indexed } else { &plain };
                let checked = check(raw, 0, sig.as_raw() as usize, 1, args.as_raw() as usize, 0, 0).expect("checked");
                // (the arguments, copied out instead of drawn)
                checked.run(raw, |b, o| {
                    list_x::transition(raw, b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    let b_raw = b as *mut c_void;
                    let buf = ID3D12Resource::from_raw_borrowed(&b_raw).unwrap();
                    list.CopyBufferRegion(&got, 0, buf, o, 20);
                    list_x::transition(raw, b, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
                });
                run(&dev, &queue, &list);
                let drawn: Vec<u32> = (0..5).map(|i| std::ptr::read_volatile(got_at.add(i))).collect();
                let mut expect = if ok { want } else { [0; 5] };
                if words == 4 {
                    expect[4] = 0;
                }
                assert_eq!(drawn, expect, "{want:x?}");
                assert_eq!(counts(checked.slot), Some([want[0], want[1], want[2]]), "{want:x?}: what the CPU sees");
            }
        }
    }
}
