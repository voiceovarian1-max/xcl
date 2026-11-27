//! `XCL_GPU_PROFILE=1`: the GPU time each console program takes.
//!
//! Each draw, dispatch and ExecuteIndirect of console code gets a timestamp query on its PC
//! list right before it and one right after it, and the pair is resolved into a readback buffer
//! right behind them. The resolve waits on the GPU for the op to finish, so with the profile on
//! a queue runs its ops one after the other. Nothing on the CPU waits for them: a reporter thread looks at the pairs
//! every 100 ms, adds the ones the GPU has written to their program, and every 2 s logs the 15
//! programs that took the most GPU time in those 2 s (every 20 s: since the start), with their
//! calls, longest call, group counts and, for compute, thread group size, wave size and code size.
//!
//! A pair measures the GPU clock between the two stamps (ticks of a title queue's
//! `GetTimestampFrequency`): the time the op held its queue. Queues run on the GPU at the same
//! time, so the time of ops that overlap is counted for each of them, and the sum over programs
//! can exceed the wall time. The driver decides when a stamp is taken relative to the work
//! around it. Bundles get no stamps (no queries in bundles), and a recording that runs again
//! without a Reset is counted once.

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::{offset_of, size_of, transmute};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use windows::core::{Interface, HRESULT};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use super::real;

/// begin / end pairs in flight (a ring): two queries and 16 readback bytes each
const PAIRS: usize = 1 << 17;
/// programs a log names
/// rows per list (local: `XCL_GPU_PROFILE_TOP`, 1..=200)
fn top_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("XCL_GPU_PROFILE_TOP").ok().and_then(|v| v.trim().parse().ok()).filter(|n| (1..=200).contains(n)).unwrap_or(15))
}
const POLL: Duration = Duration::from_millis(100);
const WINDOW: Duration = Duration::from_secs(2);
/// windows between two logs of the totals since the start
const SINCE_START_EVERY: u32 = 10;

pub(super) fn on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var("XCL_GPU_PROFILE").is_ok_and(|v| v.trim() == "1"));
    *ON
}

/// The op a pair times.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum Kind {
    Dispatch,
    /// an ExecuteIndirect of dispatches with a translated pipeline
    IndirectDispatch,
    Draw,
    /// an ExecuteIndirect of draws with a translated pipeline
    IndirectDraw,
    /// an ExecuteIndirect with a PC pipeline
    PcIndirect,
    /// (local) GPU work of XCL's own around the title's (`begin_own`): which, in the key
    Own,
    /// (local) a two-pass mesh draw: its compute pass, barriers and mesh pass (`bind::dispatch_mesh`)
    TwoPass,
    /// (local) an ExecuteIndirect of two-pass mesh draws
    IndirectTwoPass,
    /// (local) a one-pass mesh draw (`XCL_MESH_ONE_PASS`), and an ExecuteIndirect of them (its
    /// argument rewrite included)
    OnePass,
    IndirectOnePass,
}

/// (local) XCL's own GPU work that `begin_own` times.
pub(super) const OWN_COUNT_CHECK: u64 = 1;
pub(super) const OWN_UPLOAD: u64 = 2;
pub(super) const OWN_COPY_BACK: u64 = 3;

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Dispatch => "Dispatch",
            Kind::IndirectDispatch => "ExecuteIndirect (dispatches)",
            Kind::Draw => "draw",
            Kind::IndirectDraw => "ExecuteIndirect (draws)",
            Kind::PcIndirect => "ExecuteIndirect with a PC pipeline",
            Kind::Own => "XCL's own",
            Kind::TwoPass => "two-pass mesh draw (compute pass, barriers, mesh pass)",
            Kind::IndirectTwoPass => "ExecuteIndirect of two-pass mesh draws",
            Kind::OnePass => "one-pass mesh draw",
            Kind::IndirectOnePass => "ExecuteIndirect of one-pass mesh draws (argument rewrite, draws)",
        }
    }
}

/// A program: the kind of op and its console shaders, (CS, 0) or (VS, PS); (0, 0): none known.
type Key = (Kind, u64, u64);

struct Op {
    key: Key,
    /// a dispatch's groups, a draw's (vertices or indices, instances, 0), an ExecuteIndirect's
    /// (max count, 0, 0)
    size: [u32; 3],
    /// an ExecuteIndirect whose counts XCL reads (`indirect_guard`): where they are
    counts: Option<u32>,
}

impl Op {
    /// its size, with the counts the GPU read where they are known
    fn size_now(&self) -> [u32; 3] {
        self.counts.and_then(super::indirect_guard::counts).unwrap_or(self.size)
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
struct Total {
    calls: u64,
    ticks: u64,
    /// ticks of the longest call, and its size
    max: u64,
    size: [u32; 3],
}

/// What a pair of readback words holds.
#[derive(Debug, PartialEq)]
pub(super) enum Stamps {
    /// not both written yet
    Pending,
    /// the end before the begin: the GPU wrote an earlier op's pair over the slot
    Bad,
    Span(u64),
}

pub(super) fn stamps(begin: u64, end: u64) -> Stamps {
    match (begin, end) {
        (0, _) | (_, 0) => Stamps::Pending,
        (b, e) if e < b => Stamps::Bad,
        (b, e) => Stamps::Span(e - b),
    }
}

fn add(totals: &mut HashMap<Key, Total>, key: Key, ticks: u64, size: [u32; 3]) {
    let t = totals.entry(key).or_default();
    t.calls += 1;
    t.ticks += ticks;
    if ticks >= t.max {
        t.max = ticks;
        t.size = size;
    }
}

/// The `n` programs of `totals` with the most ticks, most first.
fn top(totals: &HashMap<Key, Total>, n: usize) -> Vec<(Key, Total)> {
    let mut v: Vec<(Key, Total)> = totals.iter().map(|(k, t)| (*k, t.clone())).collect();
    v.sort_by(|a, b| b.1.ticks.cmp(&a.1.ticks).then(b.1.calls.cmp(&a.1.calls)).then((a.0 .1, a.0 .2).cmp(&(b.0 .1, b.0 .2))));
    v.truncate(n);
    v
}

/// A console compute program as its pipeline was made.
#[derive(Clone, Copy)]
struct Shape {
    threads: [u32; 3],
    wave64: bool,
    /// run as two 32-lane halves (`docs/wave64-on-wave32.md`)
    split: bool,
    code: usize,
    dxil: usize,
}

/// console CS hash → its shape
static SHAPES: Mutex<Option<HashMap<u64, Shape>>> = Mutex::new(None);

/// A translated compute pipeline of console shader `blob` (hash `hash`, `dxil` bytes of DXIL)
/// was made, on a device that runs wave64 programs split (`split_device`).
pub(super) fn note_compute(hash: u64, blob: &[u8], dxil: usize, split_device: bool) {
    if !on() && !super::gpu_times::enabled() {
        return;
    }
    let mut g = SHAPES.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if m.contains_key(&hash) {
        return;
    }
    if let Some((threads, wave64, code)) = crate::gpu::recomp::pso::compute_shape(blob) {
        m.insert(hash, Shape { threads, wave64, split: wave64 && split_device, code, dxil });
    }
}

/// The wave size console compute program `hash` runs with (None: not noted).
pub(super) fn wave_of(hash: u64) -> Option<&'static str> {
    let s = *SHAPES.lock().unwrap().as_ref()?.get(&hash)?;
    Some(if s.split { "wave64 split in halves" } else if s.wave64 { "wave64" } else { "wave32" })
}

/// ticks a second of the title queues' timestamps (0: no queue seen yet)
static FREQUENCY: AtomicU64 = AtomicU64::new(0);

/// `queue` (a title queue) executes lists: its timestamp frequency converts the pairs' ticks.
pub(super) unsafe fn note_queue(queue: *mut c_void) {
    if !on() || queue.is_null() || FREQUENCY.load(Ordering::Relaxed) != 0 {
        return;
    }
    const SLOT: usize = offset_of!(ID3D12CommandQueue_Vtbl, GetTimestampFrequency) / size_of::<usize>();
    let f: unsafe extern "system" fn(*mut c_void, *mut u64) -> HRESULT = transmute(pc_fn(queue, SLOT));
    let mut hz = 0u64;
    if f(queue, &mut hz).is_ok() && hz != 0 && FREQUENCY.swap(hz, Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: XCL_GPU_PROFILE: GPU timestamps at {hz} Hz");
    }
}

struct Profile {
    _heap: ID3D12QueryHeap,
    _buf: ID3D12Resource,
    heap: usize,
    buf: usize,
    /// the readback buffer's mapping: two u64 a pair
    words: usize,
    /// two dwords a pair the GPU sets to 1: [everything before the op has finished, the op has]
    _marks: ID3D12Resource,
    marks_va: u64,
    marks: usize,
    /// ops seen begun and not ended, and since when
    running: HashMap<usize, Instant>,
    next: usize,
    ops: Vec<Option<Op>>,
    in_flight: usize,
    window: HashMap<Key, Total>,
    since_start: HashMap<Key, Total>,
    /// pairs taken again before the GPU wrote them, and pairs the GPU wrote out of order
    lost: u64,
    bad: u64,
    /// (local) the GPU clock at the first begin and the last end of the window's ops (0: none yet)
    span: (u64, u64),
    /// (local) the window's ops as (begin, end): their union is the time the GPU ran timed work
    spans: Vec<(u64, u64)>,
}

// SAFETY: the readback mapping lives as long as `_buf`; D3D12 objects are free-threaded.
unsafe impl Send for Profile {}

static STATE: Mutex<Option<Profile>> = Mutex::new(None);

unsafe fn init(dev: &ID3D12Device) -> Option<Profile> {
    let mut heap: Option<ID3D12QueryHeap> = None;
    let desc = D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2 * PAIRS as u32, NodeMask: 0 };
    if let Err(e) = dev.CreateQueryHeap(&desc, &mut heap) {
        crate::xlog_warn!("D3D12.X: XCL_GPU_PROFILE: no timestamp query heap ({e})");
        return None;
    }
    let heap = heap?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: (PAIRS * 16) as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let mut buf: Option<ID3D12Resource> = None;
    if let Err(e) = dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buf) {
        crate::xlog_warn!("D3D12.X: XCL_GPU_PROFILE: no readback buffer ({e})");
        return None;
    }
    let buf = buf?;
    let mut p = std::ptr::null_mut();
    buf.Map(0, None, Some(&mut p)).ok()?;
    std::ptr::write_bytes(p.cast::<u8>(), 0, PAIRS * 16);
    let mut md = desc;
    md.Width = (PAIRS * 8) as u64;
    let mut marks: Option<ID3D12Resource> = None;
    if let Err(e) = dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &md, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut marks) {
        crate::xlog_warn!("D3D12.X: XCL_GPU_PROFILE: no buffer for the marks ({e})");
        return None;
    }
    let marks = marks?;
    let mut mp = std::ptr::null_mut();
    marks.Map(0, None, Some(&mut mp)).ok()?;
    std::ptr::write_bytes(mp.cast::<u8>(), 0, PAIRS * 8);
    let marks_va = marks.GetGPUVirtualAddress();
    let _ = std::thread::Builder::new().name("xcl-gpu-profile".into()).spawn(|| {
        let (mut last, mut windows) = (Instant::now(), 0u32);
        loop {
            std::thread::sleep(POLL);
            poll();
            if last.elapsed() >= WINDOW && report(last.elapsed(), (windows + 1) % SINCE_START_EVERY == 0) {
                windows += 1;
                last = Instant::now();
            }
        }
    });
    crate::xlog_info!("D3D12.X: XCL_GPU_PROFILE: draws, dispatches and ExecuteIndirects of console code are timed on the GPU; the programs that take the most GPU time are logged every 2 s");
    Some(Profile {
        heap: heap.as_raw() as usize,
        buf: buf.as_raw() as usize,
        _heap: heap,
        _buf: buf,
        words: p as usize,
        _marks: marks,
        marks_va,
        marks: mp as usize,
        running: HashMap::new(),
        next: 0,
        ops: (0..PAIRS).map(|_| None).collect(),
        in_flight: 0,
        window: HashMap::new(),
        since_start: HashMap::new(),
        lost: 0,
        bad: 0,
        span: (0, 0),
        spans: Vec::new(),
    })
}

/// The profile, made on first use with `dev` (outside the lock: the device calls go through
/// XCL's hooks).
fn with_state<R>(dev: impl FnOnce() -> Option<ID3D12Device>, f: impl FnOnce(&mut Profile) -> R) -> Option<R> {
    static INIT: std::sync::Once = std::sync::Once::new();
    if !INIT.is_completed() {
        let dev = dev()?;
        INIT.call_once(|| {
            let s = unsafe { init(&dev) };
            *STATE.lock().unwrap() = s;
        });
    }
    STATE.lock().unwrap().as_mut().map(f)
}

impl Profile {
    fn word(&self, i: usize) -> &AtomicU64 {
        unsafe { &*(self.words as *const AtomicU64).add(i) }
    }

    fn mark(&self, i: usize) -> &std::sync::atomic::AtomicU32 {
        unsafe { &*(self.marks as *const std::sync::atomic::AtomicU32).add(i) }
    }
}

type WriteImmediateFn = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER, *const D3D12_WRITEBUFFERIMMEDIATE_MODE);

/// The GPU writes 1 at `va` once everything recorded on `list` before this has finished.
unsafe fn mark_done(list: *mut c_void, va: u64) {
    const WRITE: usize = offset_of!(ID3D12GraphicsCommandList2_Vtbl, WriteBufferImmediate) / size_of::<usize>();
    let f: WriteImmediateFn = transmute(pc_fn(list, WRITE));
    let p = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: va, Value: 1 };
    let m = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
    f(list, 1, &p, &m);
}

/// The PC method in `slot` of `obj` (an adopted object's own, not XCL's console-layout entry).
pub(super) unsafe fn pc_fn(obj: *mut c_void, slot: usize) -> usize {
    if real::is_adopted(obj) {
        real::orig_fn(obj, slot)
    } else {
        *(*(obj as *const *const usize)).add(slot)
    }
}

type EndQueryFn = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32);
type ResolveFn = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32, u32, usize, u64);

/// The begin stamp of an op on a list: `end` records the end stamp and the resolve.
pub(super) struct Pair {
    k: u32,
    heap: usize,
    buf: usize,
    marks_va: u64,
    end_query: EndQueryFn,
    resolve: ResolveFn,
}

/// Before the PC call of an op of console code on `list` (`pso`: its pipeline as bound).
pub(super) unsafe fn begin(list: *mut c_void, pso: usize, kind: Kind, size: [u32; 3]) -> Option<Pair> {
    // (every draw and dispatch with a translated pipeline comes by here: the ring of the last ones)
    super::watch::note(pso, kind as u8 + 1, size);
    super::gpu_times::before_op(list, pso, kind, size);
    if !on() || list.is_null() {
        return None;
    }
    let (a, b) = real::pso_shaders(real::ab::bound(super::pso_async::resolve(pso))).unwrap_or((0, 0));
    begin_key(list, (kind, a, b), size)
}

/// (local) Before GPU work of XCL's own on `list` (`what`: `OWN_…`; `n`: how much of it).
pub(super) unsafe fn begin_own(list: *mut c_void, what: u64, n: u32) -> Option<Pair> {
    if !on() || list.is_null() {
        return None;
    }
    begin_key(list, (Kind::Own, what, 0), [n, 0, 0])
}

unsafe fn begin_key(list: *mut c_void, key: Key, size: [u32; 3]) -> Option<Pair> {
    const GET_TYPE: usize = offset_of!(ID3D12CommandList_Vtbl, GetType) / size_of::<usize>();
    const END_QUERY: usize = offset_of!(ID3D12GraphicsCommandList_Vtbl, EndQuery) / size_of::<usize>();
    const RESOLVE: usize = offset_of!(ID3D12GraphicsCommandList_Vtbl, ResolveQueryData) / size_of::<usize>();
    let get_type: unsafe extern "system" fn(*mut c_void) -> D3D12_COMMAND_LIST_TYPE = transmute(pc_fn(list, GET_TYPE));
    let ty = get_type(list);
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT && ty != D3D12_COMMAND_LIST_TYPE_COMPUTE {
        return None;
    }
    let (k, heap, buf, marks_va) = with_state(
        || {
            let mut dev: Option<ID3D12Device> = None;
            ID3D12GraphicsCommandList::from_raw_borrowed(&list)?.GetDevice(&mut dev).ok()?;
            dev
        },
        |s| {
            let k = s.next;
            s.next = (k + 1) % PAIRS;
            if s.ops[k].is_some() {
                s.lost += 1;
            } else {
                s.in_flight += 1;
            }
            s.word(2 * k).store(0, Ordering::Release);
            s.word(2 * k + 1).store(0, Ordering::Release);
            s.mark(2 * k).store(0, Ordering::Release);
            s.mark(2 * k + 1).store(0, Ordering::Release);
            s.running.remove(&k);
            s.ops[k] = Some(Op { key, size, counts: None });
            (k as u32, s.heap, s.buf, s.marks_va)
        },
    )?;
    let pair = Pair { k, heap, buf, marks_va, end_query: transmute(pc_fn(list, END_QUERY)), resolve: transmute(pc_fn(list, RESOLVE)) };
    mark_done(list, marks_va + 8 * k as u64);
    (pair.end_query)(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * k);
    Some(pair)
}

/// (local) The union of `spans` in ticks, and the gaps between its pieces by length: counts and
/// ticks under 20 us, 20-200 us, 0.2-2 ms, over 2 ms; [4].1 the longest.
fn union_and_gaps(spans: &mut [(u64, u64)], hz: u64) -> (u64, [(u64, u64); 5]) {
    spans.sort_unstable();
    let us = |n: u64| n * hz / 1_000_000;
    let (mut busy, mut gaps) = (0u64, [(0u64, 0u64); 5]);
    let mut cur: Option<(u64, u64)> = None;
    for &(b, e) in spans.iter() {
        match cur {
            Some((cb, ce)) if b <= ce => cur = Some((cb, ce.max(e))),
            Some((cb, ce)) => {
                busy += ce - cb;
                let g = b - ce;
                let k = if g < us(20) { 0 } else if g < us(200) { 1 } else if g < us(2000) { 2 } else { 3 };
                gaps[k].0 += 1;
                gaps[k].1 += g;
                gaps[4].1 = gaps[4].1.max(g);
                cur = Some((b, e));
            }
            None => cur = Some((b, e)),
        }
    }
    if let Some((cb, ce)) = cur {
        busy += ce - cb;
    }
    (busy, gaps)
}

/// The op of `pair` runs the counts in slot `slot` (`indirect_guard`).
pub(super) fn note_counts(pair: &Option<Pair>, slot: u32) {
    let Some(p) = pair else { return };
    if let Some(op) = STATE.lock().unwrap().as_mut().and_then(|s| s.ops[p.k as usize].as_mut()) {
        op.counts = Some(slot);
    }
}

/// After the PC call of the op `begin` gave `pair` for.
pub(super) unsafe fn end(list: *mut c_void, pair: Option<Pair>) {
    super::gpu_times::after(list);
    let Some(p) = pair else { return };
    (p.end_query)(list, p.heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * p.k + 1);
    mark_done(list, p.marks_va + 8 * p.k as u64 + 4);
    (p.resolve)(list, p.heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * p.k, 2, p.buf, 16 * p.k as u64);
}

/// Add the pairs the GPU has written to their programs.
fn poll() {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    let mut left = s.in_flight;
    for k in 0..PAIRS {
        if left == 0 {
            break;
        }
        let Some(op) = &s.ops[k] else { continue };
        left -= 1;
        let (b, e) = (s.word(2 * k).load(Ordering::Acquire), s.word(2 * k + 1).load(Ordering::Acquire));
        if b != 0 && e >= b {
            s.span = (if s.span.0 == 0 { b } else { s.span.0.min(b) }, s.span.1.max(e));
            s.spans.push((b, e));
        }
        let span = match stamps(b, e) {
            Stamps::Pending => {
                if s.mark(2 * k).load(Ordering::Acquire) != 0 && s.mark(2 * k + 1).load(Ordering::Acquire) == 0 {
                    s.running.entry(k).or_insert_with(Instant::now);
                } else {
                    s.running.remove(&k);
                }
                continue;
            }
            Stamps::Bad => None,
            Stamps::Span(t) => Some(t),
        };
        let (key, size) = (op.key, op.size_now());
        match span {
            Some(t) => {
                add(&mut s.window, key, t, size);
                add(&mut s.since_start, key, t, size);
            }
            None => s.bad += 1,
        }
        s.ops[k] = None;
        s.running.remove(&k);
        s.in_flight -= 1;
    }
}

fn program(key: Key, shapes: &HashMap<u64, Shape>) -> String {
    let (kind, a, b) = key;
    match kind {
        Kind::Dispatch | Kind::IndirectDispatch => {
            let shape = shapes.get(&a).map_or(String::new(), |s| {
                format!(
                    "; {}x{}x{} threads, {}, {} B code, {} B DXIL",
                    s.threads[0],
                    s.threads[1],
                    s.threads[2],
                    if s.split { "wave64 split in halves" } else if s.wave64 { "wave64" } else { "wave32" },
                    s.code,
                    s.dxil
                )
            });
            format!("CS {a:016x} {}{shape}", kind.name())
        }
        Kind::Draw | Kind::IndirectDraw | Kind::TwoPass | Kind::IndirectTwoPass | Kind::OnePass | Kind::IndirectOnePass if a != 0 => format!("VS {a:016x} PS {b:016x} {}", kind.name()),
        Kind::Draw | Kind::IndirectDraw | Kind::TwoPass | Kind::IndirectTwoPass | Kind::OnePass | Kind::IndirectOnePass => format!("{} with no console shader known", kind.name()),
        Kind::PcIndirect => kind.name().to_owned(),
        Kind::Own => match a {
            OWN_COUNT_CHECK => "XCL's own: the count check before an ExecuteIndirect (barrier, pipeline, dispatch)",
            OWN_UPLOAD => "XCL's own: an upload of CPU writes (scatter dispatch)",
            OWN_COPY_BACK => "XCL's own: a copy back for a CPU read",
            _ => "XCL's own",
        }
        .to_owned(),
    }
}

fn size_text(kind: Kind, size: [u32; 3]) -> String {
    match kind {
        Kind::Dispatch => format!("{}x{}x{} groups", size[0], size[1], size[2]),
        Kind::Draw => format!("{} vertices or indices x {} instances", size[0], size[1]),
        Kind::IndirectDispatch if size[1..] != [0, 0] => format!("{}x{}x{} groups, the GPU's", size[0], size[1], size[2]),
        Kind::IndirectDraw if size[1..] != [0, 0] => format!("{} vertices or indices x {} instances, the GPU's", size[0], size[1]),
        Kind::IndirectDispatch | Kind::IndirectDraw | Kind::PcIndirect => format!("max count {} (its counts are the GPU's)", size[0]),
        Kind::Own => format!("{} of them", size[0]),
        Kind::TwoPass | Kind::OnePass => format!("{} groups", size[0]),
        Kind::IndirectTwoPass | Kind::IndirectOnePass => format!("max count {}", size[0]),
    }
}

/// Calls and ticks of `totals`, summed.
fn sums(totals: &HashMap<Key, Total>) -> (u64, u64) {
    totals.values().fold((0, 0), |(n, ticks), t| (n + t.calls, ticks + t.ticks))
}

/// Log the programs that took the most GPU time in the last `elapsed` (and since the start).
/// False: nothing logged (no queue's timestamp frequency yet); the window goes on.
fn report(elapsed: Duration, since_start: bool) -> bool {
    let hz = FREQUENCY.load(Ordering::Relaxed);
    if hz == 0 {
        return false;
    }
    let (window, all, in_flight, lost, bad, running) = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return false };
        let all = since_start.then(|| (top(&s.since_start, top_rows()), sums(&s.since_start)));
        let mut running: Vec<(u128, Key, [u32; 3])> = s
            .running
            .iter()
            .filter_map(|(&k, t)| s.ops[k].as_ref().map(|op| (t.elapsed().as_millis(), op.key, op.size_now())))
            .filter(|r| r.0 >= 500)
            .collect();
        running.sort_by(|a, b| b.0.cmp(&a.0));
        running.truncate(6);
        let span = std::mem::take(&mut s.span);
        if span.0 != 0 {
            let (_, ticks) = sums(&s.window);
            let own: u64 = s.window.iter().filter(|(k, _)| k.0 == Kind::Own).map(|(_, t)| t.ticks).sum();
            crate::xlog_info!(
                "D3D12.X: XCL_GPU_PROFILE: the window's ops lie within {:.1} ms on the GPU's clock: {:.1} ms in the title's ops, {:.1} ms in XCL's own that are timed, the rest in work that is not timed or idle",
                (span.1 - span.0) as f64 * 1000.0 / hz as f64,
                (ticks - own) as f64 * 1000.0 / hz as f64,
                own as f64 * 1000.0 / hz as f64
            );
            let (busy, gaps) = union_and_gaps(&mut s.spans, hz);
            let ms = |t: u64| t as f64 * 1000.0 / hz as f64;
            crate::xlog_info!(
                "D3D12.X: XCL_GPU_PROFILE: (local) timed ops cover {:.1} ms (overlaps counted once); gaps between them: {} under 20 us = {:.1} ms, {} of 20-200 us = {:.1} ms, {} of 0.2-2 ms = {:.1} ms, {} over 2 ms = {:.1} ms (longest {:.1} ms)",
                ms(busy), gaps[0].0, ms(gaps[0].1), gaps[1].0, ms(gaps[1].1), gaps[2].0, ms(gaps[2].1), gaps[3].0, ms(gaps[3].1), ms(gaps[4].1)
            );
        }
        s.spans.clear();
        (std::mem::take(&mut s.window), all, s.in_flight, s.lost, s.bad, running)
    };
    let ms = |ticks: u64| ticks as f64 * 1000.0 / hz as f64;
    let shapes = SHAPES.lock().unwrap().clone().unwrap_or_default();
    let lines = |what: &str, totals: Vec<(Key, Total)>, (ops, ticks): (u64, u64)| {
        crate::xlog_info!(
            "D3D12.X: XCL_GPU_PROFILE: {what}: {:.1} ms of GPU time in {ops} op(s) of console code ({in_flight} in flight, {lost} lost, {bad} out of order); the {} that took most:",
            ms(ticks),
            totals.len()
        );
        for (i, ((kind, a, b), t)) in totals.into_iter().enumerate() {
            crate::xlog_info!(
                "D3D12.X: XCL_GPU_PROFILE:   {:>2}. {:>9.1} ms  {}: {} call(s), longest {:.1} ms ({})",
                i + 1,
                ms(t.ticks),
                program((kind, a, b), &shapes),
                t.calls,
                ms(t.max),
                size_text(kind, t.size)
            );
        }
    };
    if !window.is_empty() {
        lines(&format!("last {:.1} s", elapsed.as_secs_f64()), top(&window, top_rows()), sums(&window));
    }
    for (ms_running, key, size) in running {
        crate::xlog_warn!("D3D12.X: XCL_GPU_PROFILE: RUNNING for {ms_running} ms and not finished: {} ({})", program(key, &shapes), size_text(key.0, size));
    }
    if let Some((all, sum)) = all {
        lines("since the start", all, sum);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_ops_count_once_and_gaps_are_binned() {
        // 1 MHz clock: ticks are microseconds. [0,100] and [50,150] overlap; a 10 us gap, a 500 us gap, a 3 ms gap
        let mut s = vec![(1160, 1200), (50, 150), (0, 100), (160, 200), (700, 1000), (4000, 4100)];
        let (busy, g) = union_and_gaps(&mut s, 1_000_000);
        assert_eq!(busy, 150 + 40 + 300 + 40 + 100);
        assert_eq!(g[0], (1, 10));
        assert_eq!(g[2], (1, 500));
        assert_eq!(g[3], (1, 2800));
        assert_eq!((g[1].0, g[4].1), (1, 2800));
    }

    #[test]
    fn a_pair_counts_once_both_stamps_are_written_and_in_order() {
        assert_eq!(stamps(0, 0), Stamps::Pending);
        // (the resolve's copy can be seen half done)
        assert_eq!(stamps(1_000, 0), Stamps::Pending);
        assert_eq!(stamps(0, 1_000), Stamps::Pending);
        assert_eq!(stamps(2_000, 1_000), Stamps::Bad);
        assert_eq!(stamps(1_000, 1_000), Stamps::Span(0));
        assert_eq!(stamps(1_000, 4_500), Stamps::Span(3_500));
    }

    #[test]
    fn programs_are_ranked_by_their_total_and_keep_their_longest_call() {
        let mut t = HashMap::new();
        let (a, b, c) = ((Kind::Dispatch, 0xA, 0), (Kind::IndirectDispatch, 0xB, 0), (Kind::Draw, 0xC, 0xD));
        add(&mut t, a, 10, [1, 1, 1]);
        add(&mut t, a, 30, [8, 4, 1]);
        add(&mut t, a, 20, [2, 1, 1]);
        add(&mut t, b, 100, [1, 0, 0]);
        add(&mut t, c, 5, [3, 1, 0]);
        let r = top(&t, 2);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0], (b, Total { calls: 1, ticks: 100, max: 100, size: [1, 0, 0] }));
        assert_eq!(r[1], (a, Total { calls: 3, ticks: 60, max: 30, size: [8, 4, 1] }));
        // the same op with the same program under two kinds stays two programs
        add(&mut t, (Kind::IndirectDispatch, 0xA, 0), 1, [1, 0, 0]);
        assert_eq!(top(&t, 10).len(), 4);
    }
}
