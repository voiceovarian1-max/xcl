//! `XCL_GPU_TIMES=<frames>[,<after_s>]`: the GPU time each program of a few frames takes.
//!
//! From the first present `after_s` seconds (default 0) after the first present on, every draw,
//! dispatch, ExecuteIndirect and two-pass DispatchMesh of console code recorded on a direct or
//! compute list gets a timestamp query after it (the first of a list's recording one before it
//! too), resolved into a readback buffer before the list closes. A call's interval is its stamp
//! minus the one before it on its list: one query between two calls, not two a call. A list's
//! calls count for the frame of its submit (the frame ends at the next PresentX). The first frame
//! (its lists began before the stamps did) and frames with fewer than `MIN_CALLS` calls measured
//! do not count, as the dump switches skip frames with little work (`targets`, `capture`). After
//! `frames` frames (16 at most) one table is logged (`D3D12.X: GPU times:`): each frame's wall
//! time, measured sum and per-submit totals, then the `TOP` programs by GPU time per frame.
//!
//! What it does not tell: on a desktop GPU a timestamp is written once the work before it is done,
//! and the work after it may already run, so intervals bracket calls coarsely – fine for ranking,
//! not exact per call. An interval also holds what was recorded between the two calls (barriers,
//! copies, clears, XCL's own passes). The direct and compute queues run side by side, so their
//! sums can pass the wall time. Bundles and copy lists get no stamps; a list submitted twice
//! counts once; a list recorded before the measuring began is not measured. Unset, no query heap
//! is made and each hook checks one cached value.

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::{offset_of, size_of, transmute};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::core::{Interface, HRESULT};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use super::gpu_profile::{pc_fn, stamps, Stamps};
use super::sharded::Sharded;

/// timestamps the query heap has for each measured frame (two frames more: the first, which does
/// not count, and one light frame)
const PER_FRAME: u32 = 1 << 16;
const FRAMES_MOST: u32 = 16;
/// queries a list takes at a time, in a row (one resolve each)
const BLOCK: u32 = 64;
/// calls measured a frame needs to count
const MIN_CALLS: u64 = 100;
/// programs the table lists
const TOP: usize = 60;
/// how long the report waits for the GPU to write the frames' stamps
const READ_WAIT: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Debug, PartialEq)]
struct Config {
    frames: u32,
    /// none before this long after the first present
    after: Duration,
}

/// `<frames>[,<after_s>]` (frames: 1 ..= `FRAMES_MOST`)
fn parse(v: &str) -> Option<Config> {
    let mut it = v.split(',').map(str::trim);
    let frames = it.next()?.parse::<u32>().ok().filter(|&n| n > 0)?.min(FRAMES_MOST);
    let after = match it.next() {
        Some(s) => Duration::try_from_secs_f64(s.parse::<f64>().ok()?).ok()?,
        None => Duration::ZERO,
    };
    it.next().is_none().then_some(Config { frames, after })
}

fn config() -> Option<Config> {
    static C: OnceLock<Option<Config>> = OnceLock::new();
    *C.get_or_init(|| {
        let v = std::env::var("XCL_GPU_TIMES").ok()?;
        let c = parse(&v);
        if c.is_none() {
            crate::xlog_warn!("D3D12.X: XCL_GPU_TIMES={v:?} is not <frames>[,<seconds after the first present>]: nothing measured");
        }
        c
    })
}

const UNSET: u8 = 0;
const OFF: u8 = 1;
/// the switch is on, the measuring has not begun
const WAITING: u8 = 2;
const RECORDING: u8 = 3;
/// the frames were taken (or the heap could not be made)
const DONE: u8 = 4;
static STATE: AtomicU8 = AtomicU8::new(UNSET);

fn state() -> u8 {
    let s = STATE.load(Ordering::Relaxed);
    if s != UNSET {
        return s;
    }
    let _ = STATE.compare_exchange(UNSET, if config().is_some() { WAITING } else { OFF }, Ordering::Relaxed, Ordering::Relaxed);
    STATE.load(Ordering::Relaxed)
}

/// The switch is on and the frames are not measured yet: pipelines note their console shaders.
pub(super) fn enabled() -> bool {
    matches!(state(), WAITING | RECORDING)
}

#[inline]
fn recording() -> bool {
    state() == RECORDING
}

/// What a call ran.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub(super) enum Kind {
    Compute,
    Graphics,
    /// a pipeline XCL built from a console MS or GS ("MS" / "GS"): a compute pass and a mesh pass
    Mesh(&'static str),
    /// an ExecuteIndirect with a PC pipeline
    Pc,
    /// the argument rewrites and compute passes of a run of deferred two-pass draws
    /// (`XCL_TWO_PASS_BATCH=2`), all programs together; their mesh passes are `Mesh`
    MeshBatch,
}

/// A program: the kind of call and its console shaders' dump hashes – CS, or VS / MS / GS and
/// PS (0: none) – or, where they are not known, the pipeline.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
struct Program {
    kind: Kind,
    a: u64,
    b: u64,
    pso: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Op {
    program: Program,
    /// a dispatch's groups, a draw's (vertices or indices, instances, 0), an ExecuteIndirect's
    /// (max count, 0, 0)
    size: [u32; 3],
    indirect: bool,
}

/// A measured call: the queries before and after it.
#[derive(Clone, Copy, Debug)]
struct Span {
    start: u32,
    end: u32,
    op: Op,
}

/// A PC list's recording.
#[derive(Default)]
struct Rec {
    /// the stamp after the last call (or the opening one): the next call's start
    last: Option<u32>,
    /// the call begun and not ended
    open: Option<(u32, Op)>,
    spans: Vec<Span>,
    /// its queries: blocks (first, used)
    runs: Vec<(u32, u32)>,
    /// the calls of the recording closed last, resolved, waiting for its submit
    closed: Option<Vec<Span>>,
}

static RECS: Sharded<Rec> = Sharded::new();

struct Heap {
    _heap: ID3D12QueryHeap,
    _buf: ID3D12Resource,
    heap: usize,
    buf: usize,
    /// the readback buffer's mapping: a u64 a query
    words: usize,
    count: u32,
    next: AtomicU32,
}

// SAFETY: the mapping lives as long as `_buf`; D3D12 objects are free-threaded.
unsafe impl Send for Heap {}
unsafe impl Sync for Heap {}

impl Heap {
    unsafe fn make(dev: &ID3D12Device, count: u32) -> Result<Heap, String> {
        let mut heap: Option<ID3D12QueryHeap> = None;
        let desc = D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: count, NodeMask: 0 };
        dev.CreateQueryHeap(&desc, &mut heap).map_err(|e| format!("no timestamp query heap of {count} ({e})"))?;
        let heap = heap.ok_or("no timestamp query heap")?;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 8 * count as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
        let mut buf: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buf).map_err(|e| format!("no readback buffer ({e})"))?;
        let buf = buf.ok_or("no readback buffer")?;
        let mut p = std::ptr::null_mut();
        buf.Map(0, None, Some(&mut p)).map_err(|e| format!("the readback buffer does not map ({e})"))?;
        // (0: not written yet)
        std::ptr::write_bytes(p.cast::<u8>(), 0, 8 * count as usize);
        Ok(Heap { heap: heap.as_raw() as usize, buf: buf.as_raw() as usize, _heap: heap, _buf: buf, words: p as usize, count, next: AtomicU32::new(0) })
    }

    /// `n` queries in a row (None: the heap is used up)
    fn take(&self, n: u32) -> Option<u32> {
        if self.full() {
            return None;
        }
        let first = self.next.fetch_add(n, Ordering::Relaxed);
        (first.checked_add(n)? <= self.count).then_some(first)
    }

    fn full(&self) -> bool {
        self.next.load(Ordering::Relaxed) >= self.count
    }

    fn word(&self, q: u32) -> u64 {
        assert!(q < self.count);
        unsafe { (*(self.words as *const AtomicU64).add(q as usize)).load(Ordering::Acquire) }
    }
}

static HEAP: OnceLock<Heap> = OnceLock::new();

/// The next query of a recording whose blocks are `runs`: the next of its last block, or the
/// first of a block `take` gives (None: no more).
fn next_query(runs: &mut Vec<(u32, u32)>, take: impl FnOnce() -> Option<u32>) -> Option<u32> {
    match runs.last_mut() {
        Some((first, used)) if *used < BLOCK => {
            *used += 1;
            Some(*first + *used - 1)
        }
        _ => {
            let first = take()?;
            runs.push((first, 1));
            Some(first)
        }
    }
}

type EndQueryFn = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32);
type ResolveFn = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32, u32, usize, u64);
type GetTypeFn = unsafe extern "system" fn(*mut c_void) -> D3D12_COMMAND_LIST_TYPE;
const END_QUERY: usize = offset_of!(ID3D12GraphicsCommandList_Vtbl, EndQuery) / size_of::<usize>();
const RESOLVE: usize = offset_of!(ID3D12GraphicsCommandList_Vtbl, ResolveQueryData) / size_of::<usize>();
const GET_TYPE: usize = offset_of!(ID3D12CommandList_Vtbl, GetType) / size_of::<usize>();

unsafe fn list_type(list: *mut c_void) -> D3D12_COMMAND_LIST_TYPE {
    let f: GetTypeFn = transmute(pc_fn(list, GET_TYPE));
    f(list)
}

/// A timestamp on `list` (the PC's own EndQuery) in the next query of its recording.
unsafe fn stamp(heap: &Heap, r: &mut Rec, list: *mut c_void) -> Option<u32> {
    let q = next_query(&mut r.runs, || heap.take(BLOCK))?;
    let f: EndQueryFn = transmute(pc_fn(list, END_QUERY));
    f(list, heap.heap, D3D12_QUERY_TYPE_TIMESTAMP, q);
    Some(q)
}

/// Before call `op`: it starts at the stamp after the list's last call, or at one made now.
/// False: the heap is used up.
unsafe fn open(heap: &Heap, r: &mut Rec, list: *mut c_void, op: Op) -> bool {
    let Some(start) = r.last.or_else(|| stamp(heap, r, list)) else { return false };
    r.last = Some(start);
    r.open = Some((start, op));
    true
}

/// After the call `open` began: its end stamp, the next call's start. False: the heap is used up.
unsafe fn shut(heap: &Heap, r: &mut Rec, list: *mut c_void) -> bool {
    let Some((start, op)) = r.open.take() else { return true };
    let end = stamp(heap, r, list);
    if let Some(end) = end {
        r.spans.push(Span { start, end, op });
    }
    r.last = end;
    end.is_some()
}

/// Before `list` closes: its stamps resolved into the readback buffer, its calls kept for its
/// submit. Returns the calls of an earlier recording that was never submitted.
unsafe fn resolve(heap: &Heap, r: &mut Rec, list: *mut c_void) -> usize {
    if !r.runs.is_empty() {
        let f: ResolveFn = transmute(pc_fn(list, RESOLVE));
        for &(first, used) in &r.runs {
            f(list, heap.heap, D3D12_QUERY_TYPE_TIMESTAMP, first, used, heap.buf, 8 * first as u64);
        }
    }
    r.runs.clear();
    r.last = None;
    r.open = None;
    let spans = std::mem::take(&mut r.spans);
    r.closed.replace(spans).map_or(0, |old| old.len())
}

/// calls not measured since the last present (the heap was used up), and calls of recordings
/// closed again before their submit
static UNMEASURED: AtomicU64 = AtomicU64::new(0);
static NOT_SUBMITTED: AtomicU64 = AtomicU64::new(0);

fn program_of(pso: usize, kind: Kind) -> Program {
    match super::real::pso_shaders(super::pso_async::resolve(pso)) {
        Some((a, b)) if a != 0 && kind != Kind::Pc => Program { kind, a, b, pso: 0 },
        _ => Program { kind, a: 0, b: 0, pso },
    }
}

/// Before the PC call of a call of `kind` with pipeline `pso` (as bound) on `list`.
pub(super) unsafe fn before(list: *mut c_void, pso: usize, kind: Kind, size: [u32; 3], indirect: bool) {
    if !recording() || list.is_null() {
        return;
    }
    let Some(heap) = HEAP.get() else { return };
    let ty = list_type(list);
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT && ty != D3D12_COMMAND_LIST_TYPE_COMPUTE {
        return;
    }
    let op = Op { program: program_of(pso, kind), size, indirect };
    if !RECS.with(list as usize, |r| open(heap, r, list, op)) {
        UNMEASURED.fetch_add(1, Ordering::Relaxed);
    }
}

/// `before` for a call `gpu_profile::begin` sees.
pub(super) unsafe fn before_op(list: *mut c_void, pso: usize, kind: super::gpu_profile::Kind, size: [u32; 3]) {
    if !recording() {
        return;
    }
    use super::gpu_profile::Kind as P;
    let (kind, indirect) = match kind {
        P::Dispatch => (Kind::Compute, false),
        P::IndirectDispatch => (Kind::Compute, true),
        P::Draw => (Kind::Graphics, false),
        P::IndirectDraw => (Kind::Graphics, true),
        P::PcIndirect => (Kind::Pc, true),
        // (local) the kinds our GPU profile adds: XCL's own work is not the title's
        P::Own => return,
        P::TwoPass | P::OnePass => (Kind::Mesh("MS"), false),
        P::IndirectTwoPass | P::IndirectOnePass => (Kind::Mesh("MS"), true),
    };
    before(list, pso, kind, size, indirect);
}

/// After the PC call `before` saw on `list`.
pub(super) unsafe fn after(list: *mut c_void) {
    if !recording() || list.is_null() {
        return;
    }
    let Some(heap) = HEAP.get() else { return };
    if !RECS.with(list as usize, |r| shut(heap, r, list)) {
        UNMEASURED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Before the PC Close of `list`.
pub(super) unsafe fn before_close(list: *mut c_void) {
    if !recording() || list.is_null() {
        return;
    }
    let Some(heap) = HEAP.get() else { return };
    let dropped = RECS.with(list as usize, |r| resolve(heap, r, list));
    NOT_SUBMITTED.fetch_add(dropped as u64, Ordering::Relaxed);
}

/// A PC submit of `lists`.
struct Submit {
    ty: D3D12_COMMAND_LIST_TYPE,
    /// the queue's timestamp ticks a second
    hz: u64,
    /// lists in it, and of them the ones with no call measured
    lists: usize,
    bare: usize,
    spans: Vec<Span>,
}

struct Frame {
    first: bool,
    began: Instant,
    submits: Vec<Submit>,
}

impl Frame {
    fn new(first: bool) -> Frame {
        Frame { first, began: Instant::now(), submits: Vec::new() }
    }
}

/// A frame taken for the report.
struct Taken {
    /// the present that ended it
    present: u64,
    wall: Duration,
    unmeasured: u64,
    dropped: u64,
    submits: Vec<Submit>,
}

static FRAME: Mutex<Option<Frame>> = Mutex::new(None);
static TAKEN: Mutex<Vec<Taken>> = Mutex::new(Vec::new());

unsafe fn frequency(queue: *mut c_void) -> u64 {
    const SLOT: usize = offset_of!(ID3D12CommandQueue_Vtbl, GetTimestampFrequency) / size_of::<usize>();
    let f: unsafe extern "system" fn(*mut c_void, *mut u64) -> HRESULT = transmute(pc_fn(queue, SLOT));
    let mut hz = 0u64;
    if f(queue, &mut hz).is_ok() {
        hz
    } else {
        0
    }
}

/// `queue` (the PC queue) executes `lists` (PC lists).
pub(super) unsafe fn executed(queue: *mut c_void, lists: &[*mut c_void]) {
    if !recording() || queue.is_null() || lists.is_empty() {
        return;
    }
    let ty = list_type(lists[0]);
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT && ty != D3D12_COMMAND_LIST_TYPE_COMPUTE {
        return;
    }
    let (mut spans, mut bare) = (Vec::new(), 0);
    for &l in lists {
        match RECS.with(l as usize, |r| r.closed.take()) {
            Some(s) if !s.is_empty() => spans.extend(s),
            _ => bare += 1,
        }
    }
    let s = Submit { ty, hz: frequency(queue), lists: lists.len(), bare, spans };
    if let Some(f) = FRAME.lock().unwrap().as_mut() {
        f.submits.push(s);
    }
}

/// The title presents (the `present`th PresentX): a frame ends.
pub(super) unsafe fn present(present: u64) {
    let s = state();
    if s != WAITING && s != RECORDING {
        return;
    }
    let Some(c) = config() else { return };
    static T0: OnceLock<Instant> = OnceLock::new();
    let t0 = *T0.get_or_init(Instant::now);
    if s == WAITING {
        if t0.elapsed() < c.after {
            return;
        }
        let count = (c.frames + 2) * PER_FRAME;
        let raw = super::real::device_raw();
        let made = match ID3D12Device::from_raw_borrowed(&raw) {
            Some(dev) => Heap::make(dev, count),
            None => Err("no device".into()),
        };
        match made {
            Ok(h) => {
                let _ = HEAP.set(h);
            }
            Err(e) => {
                STATE.store(DONE, Ordering::Relaxed);
                crate::xlog_warn!("D3D12.X: GPU times: {e}: nothing measured");
                return;
            }
        }
        *FRAME.lock().unwrap() = Some(Frame::new(true));
        STATE.store(RECORDING, Ordering::Relaxed);
        crate::xlog_info!(
            "D3D12.X: GPU times: measuring {} frame(s) from present {present} on, in a query heap of {count} timestamps ({PER_FRAME} for each frame)",
            c.frames
        );
        return;
    }
    let Some(f) = std::mem::replace(&mut *FRAME.lock().unwrap(), Some(Frame::new(false))) else { return };
    let (unmeasured, dropped) = (UNMEASURED.swap(0, Ordering::Relaxed), NOT_SUBMITTED.swap(0, Ordering::Relaxed));
    let calls: usize = f.submits.iter().map(|s| s.spans.len()).sum();
    let full = HEAP.get().is_some_and(Heap::full);
    let mut taken = TAKEN.lock().unwrap();
    if !f.first && calls as u64 >= MIN_CALLS {
        taken.push(Taken { present, wall: f.began.elapsed(), unmeasured, dropped, submits: f.submits });
    }
    if (taken.len() as u32) < c.frames && !full {
        return;
    }
    STATE.store(DONE, Ordering::Relaxed);
    *FRAME.lock().unwrap() = None;
    let frames = std::mem::take(&mut *taken);
    let _ = std::thread::Builder::new().name("xcl-gpu-times".into()).spawn(move || report(frames, c, full));
}

/// Waits for the GPU to write the frames' stamps (`READ_WAIT` at most) and logs the table.
fn report(frames: Vec<Taken>, c: Config, full: bool) {
    let Some(heap) = HEAP.get() else { return };
    let deadline = Instant::now() + READ_WAIT;
    let all = || frames.iter().flat_map(|f| &f.submits).flat_map(|s| &s.spans).all(|s| heap.word(s.start) != 0 && heap.word(s.end) != 0);
    while !all() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    let reads: Vec<FrameRead> = frames
        .iter()
        .map(|f| FrameRead {
            present: f.present,
            wall: f.wall,
            unmeasured: f.unmeasured,
            dropped: f.dropped,
            submits: f
                .submits
                .iter()
                .map(|s| SubmitRead { ty: s.ty, hz: s.hz, lists: s.lists, bare: s.bare, calls: s.spans.iter().map(|sp| (sp.op, heap.word(sp.start), heap.word(sp.end))).collect() })
                .collect(),
        })
        .collect();
    for line in lines(&reads, &c, full, heap.count, &super::gpu_profile::wave_of) {
        crate::xlog_info!("D3D12.X: GPU times: {line}");
    }
}

// ---------------------------------------------------------------------------------------
// the table (no GPU)
// ---------------------------------------------------------------------------------------

struct SubmitRead {
    ty: D3D12_COMMAND_LIST_TYPE,
    hz: u64,
    lists: usize,
    bare: usize,
    /// each call and its two stamps as read back
    calls: Vec<(Op, u64, u64)>,
}

struct FrameRead {
    present: u64,
    wall: Duration,
    unmeasured: u64,
    dropped: u64,
    submits: Vec<SubmitRead>,
}

#[derive(Default, Clone, Debug, PartialEq)]
struct Total {
    calls: u64,
    ns: u64,
    /// the longest call, and its size
    max: u64,
    size: [u32; 3],
    indirect: bool,
}

#[derive(Default, Debug, PartialEq)]
struct FrameSum {
    present: u64,
    wall: Duration,
    calls: u64,
    ns: u64,
    /// first stamp to last, where every submit's queue has the same clock
    span: Option<u64>,
    unmeasured: u64,
    dropped: u64,
    /// stamps not written by the time the report read them, and written out of order
    pending: u64,
    bad: u64,
    bare: usize,
    /// (queue type, lists, calls, ns) of each submit
    submits: Vec<(i32, usize, u64, u64)>,
}

fn ticks_ns(ticks: u64, hz: u64) -> u64 {
    (ticks as u128 * 1_000_000_000 / hz.max(1) as u128) as u64
}

/// The programs' totals over all `frames`, and each frame's sums.
fn summarize(frames: &[FrameRead]) -> (HashMap<Program, Total>, Vec<FrameSum>) {
    let mut totals: HashMap<Program, Total> = HashMap::new();
    let mut sums = Vec::new();
    for f in frames {
        let mut s = FrameSum { present: f.present, wall: f.wall, unmeasured: f.unmeasured, dropped: f.dropped, ..Default::default() };
        let one_clock = f.submits.windows(2).all(|w| w[0].hz == w[1].hz);
        let (mut lo, mut hi) = (u64::MAX, 0u64);
        for sub in &f.submits {
            s.bare += sub.bare;
            let (mut calls, mut ns) = (0, 0);
            for &(op, start, end) in &sub.calls {
                let t = match stamps(start, end) {
                    Stamps::Pending => {
                        s.pending += 1;
                        continue;
                    }
                    Stamps::Bad => {
                        s.bad += 1;
                        continue;
                    }
                    Stamps::Span(t) if sub.hz != 0 => ticks_ns(t, sub.hz),
                    Stamps::Span(_) => {
                        s.bad += 1;
                        continue;
                    }
                };
                (lo, hi) = (lo.min(start), hi.max(end));
                calls += 1;
                ns += t;
                let total = totals.entry(op.program).or_default();
                total.calls += 1;
                total.ns += t;
                if t >= total.max {
                    (total.max, total.size, total.indirect) = (t, op.size, op.indirect);
                }
            }
            s.calls += calls;
            s.ns += ns;
            s.submits.push((sub.ty.0, sub.lists, calls, ns));
        }
        if one_clock && hi > lo {
            s.span = f.submits.first().map(|sub| ticks_ns(hi - lo, sub.hz));
        }
        sums.push(s);
    }
    (totals, sums)
}

/// The `n` programs of `totals` with the most GPU time, most first (ties: more calls first).
fn ranked(totals: &HashMap<Program, Total>, n: usize) -> Vec<(Program, Total)> {
    let mut v: Vec<(Program, Total)> = totals.iter().map(|(p, t)| (*p, t.clone())).collect();
    v.sort_by(|a, b| b.1.ns.cmp(&a.1.ns).then(b.1.calls.cmp(&a.1.calls)).then(a.0.cmp(&b.0)));
    v.truncate(n);
    v
}

/// A shader as the target dump names it: its `real::shader_key` and its dump hash.
fn shader_text(stage: &str, hash: u64) -> String {
    format!("{stage} {:016x} (dump hash {hash:016x})", super::real::shader_key_of_hash(hash))
}

fn program_text(p: &Program, wave: &dyn Fn(u64) -> Option<&'static str>) -> String {
    let ps = |b: u64| if b != 0 { shader_text("ps", b) } else { "no ps".into() };
    match p.kind {
        Kind::Compute if p.a != 0 => format!("compute {}{}", shader_text("cs", p.a), wave(p.a).map_or(String::new(), |w| format!(", {w}"))),
        Kind::Graphics if p.a != 0 => format!("graphics {}, {}", shader_text("vs", p.a), ps(p.b)),
        Kind::Mesh(k) if p.a != 0 => format!("mesh two-pass {}, {}", shader_text(&k.to_lowercase(), p.a), ps(p.b)),
        Kind::Compute => format!("compute pipeline {:#x} (its CS not known)", p.pso),
        Kind::Graphics => format!("graphics pipeline {:#x} (its shaders not known)", p.pso),
        Kind::Mesh(k) => format!("mesh two-pass {k} pipeline {:#x} (its shaders not known)", p.pso),
        Kind::Pc => format!("PC pipeline {:#x}", p.pso),
        Kind::MeshBatch => "mesh two-pass compute passes of deferred draws (all programs)".into(),
    }
}

fn size_text(kind: Kind, size: [u32; 3], indirect: bool) -> String {
    match (kind, indirect) {
        (Kind::MeshBatch, _) => format!("{} draws", size[0]),
        (_, true) => format!("ExecuteIndirect, max count {}", size[0]),
        (Kind::Graphics, false) => format!("{} vertices or indices x {} instances", size[0], size[1]),
        _ => format!("{}x{}x{} groups", size[0], size[1], size[2]),
    }
}

fn queue_name(ty: i32) -> &'static str {
    match D3D12_COMMAND_LIST_TYPE(ty) {
        D3D12_COMMAND_LIST_TYPE_DIRECT => "direct",
        D3D12_COMMAND_LIST_TYPE_COMPUTE => "compute",
        _ => "other",
    }
}

/// The report's lines (each logged after `D3D12.X: GPU times: `).
fn lines(frames: &[FrameRead], c: &Config, full: bool, heap_count: u32, wave: &dyn Fn(u64) -> Option<&'static str>) -> Vec<String> {
    let (totals, sums) = summarize(frames);
    let n = frames.len().max(1) as f64;
    let ms = |ns: u64| ns as f64 / 1e6;
    let mut out = vec![
        format!(
            "{} frame(s) measured (of {} asked for; a frame counts with {MIN_CALLS} calls measured or more, not before {} s after the first present); a call's time is its timestamp minus the one before it on its list (one query between two calls)",
            frames.len(),
            c.frames,
            c.after.as_secs_f64()
        ),
        "caveats: a desktop GPU writes a timestamp once the work before it is done while later work may already run, so a call's time is coarse (fine for ranking) and holds what was recorded between the two calls (barriers, copies, clears, XCL's own passes); the direct and compute queues overlap, so sums can pass the wall time; bundles and copy lists get no stamps, a list submitted twice counts once, lists recorded before the measuring began are not measured".into(),
    ];
    let unmeasured: u64 = sums.iter().map(|s| s.unmeasured).sum();
    if full || unmeasured > 0 {
        out.push(format!("the query heap ({heap_count} timestamps, {PER_FRAME} for each frame) ran out: {unmeasured} call(s) in the frames counted were not measured"));
    }
    for s in &sums {
        out.push(format!(
            "frame ending at present {}: wall {:.2} ms, GPU first to last stamp {}, measured {:.2} ms in {} calls; not measured: {} (heap full), {} not read back, {} out of order, {} of a recording never submitted; {} list(s) submitted with no call measured",
            s.present,
            s.wall.as_secs_f64() * 1e3,
            s.span.map_or("n/a (queues with different clocks)".into(), |ns| format!("{:.2} ms", ms(ns))),
            ms(s.ns),
            s.calls,
            s.unmeasured,
            s.pending,
            s.bad,
            s.dropped,
            s.bare
        ));
        let subs: Vec<String> = s.submits.iter().enumerate().map(|(k, &(ty, lists, calls, ns))| format!("#{k} {} {lists} lists {calls} calls {:.3} ms", queue_name(ty), ms(ns))).collect();
        out.push(format!("frame ending at present {}: {} submit(s): {}", s.present, subs.len(), subs.join(" | ")));
    }
    let top = ranked(&totals, TOP);
    out.push(format!("the {} of {} programs with the most GPU time, ms per frame averaged over {} frame(s):", top.len(), totals.len(), frames.len()));
    for (i, (p, t)) in top.iter().enumerate() {
        out.push(format!(
            "{:>2}. {:>8.3} ms/frame  {}: {} calls ({:.1}/frame), longest {:.3} ms ({})",
            i + 1,
            ms(t.ns) / n,
            program_text(p, wave),
            t.calls,
            t.calls as f64 / n,
            ms(t.max),
            size_text(p.kind, t.size, t.indirect)
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_takes_frames_and_seconds() {
        assert_eq!(parse("3"), Some(Config { frames: 3, after: Duration::ZERO }));
        assert_eq!(parse(" 3 , 200 "), Some(Config { frames: 3, after: Duration::from_secs(200) }));
        assert_eq!(parse("2,2.5"), Some(Config { frames: 2, after: Duration::from_millis(2500) }));
        assert_eq!(parse("100"), Some(Config { frames: FRAMES_MOST, after: Duration::ZERO }), "16 frames at most");
        for bad in ["0", "", "x", "3,x", "3,-1", "3,inf", "3,1,2", "-3"] {
            assert_eq!(parse(bad), None, "{bad:?}");
        }
    }

    /// A list takes its queries a block at a time; a call needs one query, a list's first call two.
    #[test]
    fn queries_come_in_blocks_until_the_heap_is_used_up() {
        let mut runs = Vec::new();
        let mut blocks = vec![Some(128), Some(512), None].into_iter();
        let mut take = || blocks.next().unwrap();
        let got: Vec<Option<u32>> = (0..2 * BLOCK + 1).map(|_| next_query(&mut runs, &mut take)).collect();
        assert_eq!(got[0], Some(128));
        assert_eq!(got[BLOCK as usize - 1], Some(128 + BLOCK - 1));
        assert_eq!(got[BLOCK as usize], Some(512), "a new block");
        assert_eq!(got[2 * BLOCK as usize - 1], Some(512 + BLOCK - 1));
        assert_eq!(got[2 * BLOCK as usize], None, "the heap is used up");
        assert_eq!(runs, vec![(128, BLOCK), (512, BLOCK)]);
    }

    fn op(kind: Kind, a: u64, size: [u32; 3]) -> Op {
        Op { program: Program { kind, a, b: 0, pso: 0 }, size, indirect: false }
    }

    /// Stamps become ns by their queue's clock; programs add up over the frames and keep their
    /// longest call; stamps not written or out of order do not count.
    #[test]
    fn intervals_add_up_by_program_and_by_submit() {
        let (cs, other) = (op(Kind::Compute, 0xc5, [80, 45, 1]), op(Kind::Graphics, 0x75, [3, 1, 0]));
        let big = Op { size: [160, 90, 1], ..cs };
        // 10 MHz: a tick is 100 ns
        let frame = |present: u64, calls: Vec<(Op, u64, u64)>, more: Vec<(Op, u64, u64)>| FrameRead {
            present,
            wall: Duration::from_millis(200),
            unmeasured: 0,
            dropped: 0,
            submits: vec![
                SubmitRead { ty: D3D12_COMMAND_LIST_TYPE_DIRECT, hz: 10_000_000, lists: 2, bare: 1, calls },
                SubmitRead { ty: D3D12_COMMAND_LIST_TYPE_COMPUTE, hz: 10_000_000, lists: 1, bare: 0, calls: more },
            ],
        };
        let frames = vec![
            // one query between two calls: [1000, 1100] [1100, 1400]
            frame(10, vec![(cs, 1000, 1100), (big, 1100, 1400), (other, 1400, 1450)], vec![(cs, 2000, 2010), (cs, 0, 2020), (cs, 2030, 2025)]),
            frame(11, vec![(cs, 5000, 5100)], vec![]),
        ];
        let (totals, sums) = summarize(&frames);
        let t = &totals[&cs.program];
        assert_eq!((t.calls, t.ns, t.max, t.size), (4, (100 + 300 + 10 + 100) * 100, 30_000, [160, 90, 1]));
        assert_eq!(totals[&other.program].ns, 5_000);
        let s = &sums[0];
        assert_eq!((s.calls, s.ns, s.pending, s.bad, s.bare), (4, 46_000, 1, 1, 1));
        assert_eq!(s.submits, vec![(D3D12_COMMAND_LIST_TYPE_DIRECT.0, 2, 3, 45_000), (D3D12_COMMAND_LIST_TYPE_COMPUTE.0, 1, 1, 1_000)]);
        assert_eq!(s.span, Some((2010 - 1000) * 100), "first to last stamp");
        assert_eq!(sums[1].ns, 10_000);
        // a queue of another clock: no span across them
        let mut odd = frame(12, vec![(cs, 1, 2)], vec![(cs, 1, 2)]);
        odd.submits[1].hz = 1_000;
        assert_eq!(summarize(&[odd]).1[0].span, None);
    }

    /// The table: by ms per frame, most first (ties: more calls); each row names the shader as
    /// the target dump does and the costliest call's size.
    #[test]
    fn the_table_ranks_programs_by_time() {
        let mut t = HashMap::new();
        let p = |kind: Kind, a: u64, pso: usize| Program { kind, a, b: if kind == Kind::Graphics { 0x99 } else { 0 }, pso };
        let (x, y, z, w) = (p(Kind::Compute, 1, 0), p(Kind::Graphics, 2, 0), p(Kind::Mesh("GS"), 3, 0), p(Kind::Pc, 0, 0x40));
        t.insert(x, Total { calls: 2, ns: 500, max: 400, size: [8, 4, 1], indirect: false });
        t.insert(y, Total { calls: 9, ns: 900, max: 100, size: [3, 2, 0], indirect: false });
        t.insert(z, Total { calls: 3, ns: 500, max: 200, size: [64, 0, 0], indirect: true });
        t.insert(w, Total { calls: 1, ns: 50, max: 50, size: [1, 0, 0], indirect: true });
        let order: Vec<Program> = ranked(&t, 10).into_iter().map(|r| r.0).collect();
        assert_eq!(order, vec![y, z, x, w]);
        assert_eq!(ranked(&t, 2).len(), 2);
        let wave = |h: u64| (h == 1).then_some("wave64 split in halves");
        assert_eq!(program_text(&x, &wave), format!("compute {}, wave64 split in halves", shader_text("cs", 1)));
        assert!(shader_text("cs", 1).starts_with(&format!("cs {:016x} (dump hash 0000000000000001)", super::super::real::shader_key_of_hash(1))));
        assert_eq!(program_text(&y, &wave), format!("graphics {}, {}", shader_text("vs", 2), shader_text("ps", 0x99)));
        assert_eq!(program_text(&z, &wave), format!("mesh two-pass {}, no ps", shader_text("gs", 3)));
        assert_eq!(program_text(&w, &wave), "PC pipeline 0x40");
        assert_eq!(size_text(Kind::Compute, [8, 4, 1], false), "8x4x1 groups");
        assert_eq!(size_text(Kind::Graphics, [3, 2, 0], false), "3 vertices or indices x 2 instances");
        assert_eq!(size_text(Kind::Mesh("GS"), [64, 0, 0], true), "ExecuteIndirect, max count 64");
        assert_eq!(size_text(Kind::MeshBatch, [12, 0, 0], false), "12 draws");
        assert_eq!(program_text(&p(Kind::MeshBatch, 0, 0), &wave), "mesh two-pass compute passes of deferred draws (all programs)");
        let c = Config { frames: 2, after: Duration::ZERO };
        let frames = vec![FrameRead {
            present: 7,
            wall: Duration::from_millis(5),
            unmeasured: 3,
            dropped: 0,
            submits: vec![SubmitRead { ty: D3D12_COMMAND_LIST_TYPE_DIRECT, hz: 1_000_000_000, lists: 1, bare: 0, calls: vec![(op(Kind::Compute, 1, [2, 1, 1]), 10, 2_000_010)] }],
        }];
        let l = lines(&frames, &c, false, 1 << 17, &wave);
        assert!(l.iter().any(|s| s.contains("ran out: 3 call(s)")), "{l:#?}");
        assert!(l.iter().any(|s| s.contains("#0 direct 1 lists 1 calls 2.000 ms")), "{l:#?}");
        let row = l.last().unwrap();
        assert!(row.starts_with(" 1.    2.000 ms/frame  compute cs ") && row.ends_with(": 1 calls (1.0/frame), longest 2.000 ms (2x1x1 groups)"), "{row}");
    }

    /// On a D3D12 device (WARP if no other): two dispatches on a list stamped as the switch
    /// stamps them get two intervals that share their middle query, both read back in order.
    #[test]
    fn two_dispatches_get_two_intervals() {
        use crate::gpu::{fence, recomp::runner::skip_test};
        use std::mem::transmute_copy;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        if crate::gpu::recomp::dxc::location().is_err() {
            skip_test("no DXC");
            return;
        }
        let src = "[RootSignature(\"RootFlags(0)\")]\n[numthreads(64, 1, 1)]\nvoid main() {}\n";
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let dxil = crate::gpu::recomp::dxc::compile(src, "main", "cs_6_0", &[]).unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).unwrap();
            let cs = D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&rs), CS: cs, ..Default::default() }).unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, &pso).unwrap();
            list.SetComputeRootSignature(&rs);
            let heap = Heap::make(&dev, 4 * BLOCK).unwrap();
            let raw = list.as_raw();
            let mut r = Rec::default();
            let c = op(Kind::Compute, 0xc5, [1024, 1, 1]);
            for _ in 0..2 {
                assert!(open(&heap, &mut r, raw, c));
                list.Dispatch(1024, 1, 1);
                assert!(shut(&heap, &mut r, raw));
            }
            assert_eq!(r.spans.len(), 2);
            assert_eq!(r.spans[0].end, r.spans[1].start, "one query between the two calls");
            assert_eq!(r.runs, vec![(0, 3)]);
            assert_eq!(resolve(&heap, &mut r, raw), 0);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&done, 1).unwrap();
            fence::wait(&done, 1, &dev, None, 10_000).unwrap();
            let hz = frequency(queue.as_raw());
            assert!(hz > 0);
            let spans = r.closed.take().unwrap();
            let read = FrameRead {
                present: 1,
                wall: Duration::ZERO,
                unmeasured: 0,
                dropped: 0,
                submits: vec![SubmitRead { ty, hz, lists: 1, bare: 0, calls: spans.iter().map(|s| (s.op, heap.word(s.start), heap.word(s.end))).collect() }],
            };
            for s in &spans {
                assert!(matches!(stamps(heap.word(s.start), heap.word(s.end)), Stamps::Span(_)), "{} {}", heap.word(s.start), heap.word(s.end));
            }
            let (totals, sums) = summarize(&[read]);
            assert_eq!((sums[0].calls, sums[0].pending, sums[0].bad), (2, 0, 0));
            assert_eq!(totals[&c.program].calls, 2);
            println!("two dispatches: {} ns in all at {hz} Hz", totals[&c.program].ns);
        }
    }
}
