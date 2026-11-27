//! Root arguments of console code, bound when it dispatches with a translated pipeline.
//!
//! A translated console shader reads its root arguments the console's way (user SGPRs s4…,
//! the rest from a table at s[0:1]: `console_list::user_data`) and memory through XCL's
//! shader ABI (u0 io, u1 res, u2… unified memory). So on a command list of console code the
//! compute root signature and root arguments are kept here – packed by the root signature's
//! layout – instead of going to the PC list, and each Dispatch with a translated pipeline
//! binds the ABI root signature, writes the user SGPRs (`io`) and the argument table to
//! unified memory and dispatches. That memory stays until the list's allocator is reset.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::console_list::{self, Layout, Param};

#[derive(Default)]
struct State {
    layout: Option<Arc<Layout>>,
    args: Vec<u8>,
    /// the graphics root signature's layout and arguments
    layout_gfx: Option<Arc<Layout>>,
    args_gfx: Vec<u8>,
    /// the title's graphics root signature (a PC object), bound before a draw with an
    /// untranslated pipeline – created with it – since SetGraphicsRootSignature binds nothing
    rs_gfx: usize,
    pso: usize,
    /// 32-bit addresses of the view heap and the sampler heap set (0: none)
    heaps: [u32; 2],
    /// the PC allocator this recording is on (`alloc_pool`: not always the title's allocator)
    allocator: usize,
    /// unified memory of this recording (io, argument tables)
    uma: Vec<usize>,
    /// where the next draw's io and argument table go (blocks in `uma`)
    arena: crate::gpu::uma::Arena,
    /// a heap of views is bound (`list_x::ensure_view_heap`)
    heap_bound: bool,
    res_buf: Option<TwoPassBuf>,
    res_retired: Vec<TwoPassBuf>,
    /// rewritten indirect arguments (`scratch_args`): the block (and its bytes), its bytes in
    /// use, and the bytes of blocks this recording took
    scratch: Option<(ID3D12Resource, u64)>,
    scratch_used: u64,
    scratch_taken: u64,
    /// `scratch` was left in INDIRECT_ARGUMENT by the last two-pass ExecuteIndirect (only with
    /// `two_pass_batch`; otherwise it goes back to COMMON after each) – (local) or by a one-pass
    /// ExecuteIndirect or an argument batch (`batched_args`), always
    scratch_in_args: bool,
    /// the record regions a mesh pass may still be reading, with no UAV barrier on the records
    /// since (`two_pass_plan`; only with `two_pass_batch`)
    record_open: Option<u8>,
    /// two-pass draws deferred to `flush_two_pass` (`two_pass_defer`) and the title's pipeline
    /// sets between them, in the title's order
    deferred: Vec<RunItem>,
    /// the records of deferred draws: a region each from the start (`batch_region`), and whether
    /// the mesh passes of the last run may still read them (no barrier after them)
    batch_buf: Option<TwoPassBuf>,
    batch_used: u64,
    batch_open: bool,
    /// the next dispatch's thread limits (`IO_THREAD_LIMIT`; a `DispatchX` with thread
    /// dimensions), taken by its `bind_abi`
    limits: [u32; 3],
    /// the count check's buffer was left INDIRECT_ARGUMENT in this recording (`indirect_guard`)
    guard_out_ia: bool,
    /// (local) the open batch of one-pass ExecuteIndirect arguments (`batched_args`)
    arg_batch: Option<ArgBatch>,
}

/// (local) A pass that rewrites the arguments of many one-pass ExecuteIndirects (`batched_args`):
/// its table of jobs (UMA address), the jobs in it, and its block of rewritten arguments.
struct ArgBatch {
    table: usize,
    jobs: u32,
    dst: ID3D12Resource,
    base: u64,
    used: u64,
}

/// A recording's blocks of rewritten indirect arguments, and most bytes of them (past it, its
/// two-pass ExecuteIndirects are skipped)
const SCRATCH_BLOCK: u64 = 1 << 20;
const SCRATCH_CAP: u64 = 16 << 20;
/// Most bytes of buffers kept for reuse (`FREE_RES`)
const FREE_RES_CAP: u64 = 1 << 30;
/// How long a buffer no recording took again stays in `FREE_RES`. Recordings take them again
/// within a frame or so, and the GPU is done with one there already (its allocator was reset):
/// only video memory waits on this.
const FREE_RES_IDLE: std::time::Duration = std::time::Duration::from_secs(2);

static ALLOC_RES: Mutex<Option<HashMap<usize, Vec<TwoPassBuf>>>> = Mutex::new(None);
static FREE_RES: Mutex<FreePool<TwoPassBuf>> = Mutex::new(FreePool::new(FREE_RES_CAP));

/// A two-pass buffer (records or argument scratch), its bytes and its device: counted from when
/// it is made until it is let go (`TWO_PASS`).
struct TwoPassBuf {
    res: ID3D12Resource,
    bytes: u64,
    dev: usize,
}

impl TwoPassBuf {
    /// A new buffer of `bytes`, if the two-pass buffers alive stay within `two_pass_cap` with it
    /// (the free ones in `FREE_RES` go first to make room, when they are enough). Past it: an
    /// error, and the draw that asked is skipped.
    unsafe fn make(dev: &ID3D12Device, bytes: u64, what: &str) -> Result<Self, String> {
        let cap = two_pass_cap();
        if !reserve(&TWO_PASS[TP_ALIVE], bytes, cap) {
            // (counted with `FREE_RES` locked, so its bytes are in the load; none go when they
            // cannot make the room: the draw is skipped either way, and the recordings after it
            // can still take them)
            let gone = {
                let mut free = FREE_RES.lock().unwrap();
                let over = TWO_PASS[TP_ALIVE].load(Ordering::Relaxed).saturating_add(bytes).saturating_sub(cap);
                free.take_room(over)
            };
            // (the GPU is done with those: let go outside the lock)
            let made_room = gone.is_some();
            drop(gone);
            if !made_room || !reserve(&TWO_PASS[TP_ALIVE], bytes, cap) {
                note_two_pass(TP_REFUSED);
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::xlog_warn!(
                        "D3D12.X: a {what} of {} KiB would take the two-pass buffers alive past {} MiB (XCL_TWO_PASS_MB): its draw is skipped",
                        bytes >> 10,
                        cap >> 20
                    );
                }
                return Err(format!("the two-pass buffers alive are at {} MiB (XCL_TWO_PASS_MB)", cap >> 20));
            }
        }
        let res = match default_uav_buffer(dev, bytes, what) {
            Ok(r) => r,
            Err(e) => {
                TWO_PASS[TP_ALIVE].fetch_sub(bytes, Ordering::Relaxed);
                return Err(e);
            }
        };
        note_two_pass(TP_MADE);
        Ok(TwoPassBuf { res, bytes, dev: dev.as_raw() as usize })
    }
}

/// Most bytes of two-pass buffers alive at once (`XCL_TWO_PASS_MB`, 1024 MiB by default): those
/// recordings hold until the GPU is done with them, and the free ones. Each is a committed buffer
/// in video memory, on a card that may be nearly full.
fn two_pass_cap() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let mb = std::env::var("XCL_TWO_PASS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(1024);
        mb.saturating_mul(1 << 20)
    })
}

/// `bytes` more counted in `alive` if it stays within `cap` (whether it did).
fn reserve(alive: &AtomicU64, bytes: u64, cap: u64) -> bool {
    let mut a = alive.load(Ordering::Relaxed);
    loop {
        let Some(n) = a.checked_add(bytes).filter(|&n| n <= cap) else { return false };
        match alive.compare_exchange_weak(a, n, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(now) => a = now,
        }
    }
}

impl Drop for TwoPassBuf {
    fn drop(&mut self) {
        TWO_PASS[TP_ALIVE].fetch_sub(self.bytes, Ordering::Relaxed);
        note_two_pass(TP_GONE);
    }
}

/// Two-pass buffers made, taken again from `FREE_RES`, let go, the bytes of those alive, and
/// buffers not made for `two_pass_cap`.
static TWO_PASS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
const TP_MADE: usize = 0;
const TP_REUSED: usize = 1;
const TP_GONE: usize = 2;
const TP_ALIVE: usize = 3;
const TP_REFUSED: usize = 4;

/// One more of `TWO_PASS[k]`. Logged at most every 2 s while they change (always: each buffer
/// made or let go is the driver's memory manager at work, in video memory that may be short).
fn note_two_pass(k: usize) {
    use std::sync::atomic::AtomicI64;
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    static LAST: AtomicI64 = AtomicI64::new(i64::MIN);
    TWO_PASS[k].fetch_add(1, Ordering::Relaxed);
    let now = START.get_or_init(std::time::Instant::now).elapsed().as_millis() as i64;
    let last = LAST.load(Ordering::Relaxed);
    if now < last.saturating_add(2000) {
        return;
    }
    // (a buffer is let go while `FREE_RES` is locked on this thread too: no line then, the next
    // change logs)
    let Ok(free) = FREE_RES.try_lock() else { return };
    let (kept, kept_bytes) = (free.free.len(), free.bytes);
    drop(free);
    // (what waits for allocator resets, and for time: bytes that stay there hold the cap; tried
    // only, as `FREE_RES`)
    let Ok(held) = ALLOC_RES.try_lock() else { return };
    let held_bytes: u64 = held.as_ref().map_or(0, |m| m.values().flatten().map(|b| b.bytes).sum());
    drop(held);
    let Ok(unknown) = UNKNOWN_RES.try_lock() else { return };
    let unknown_bytes = unknown.bytes;
    drop(unknown);
    if LAST.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let [made, reused, gone, alive, refused] = std::array::from_fn(|i| TWO_PASS[i].load(Ordering::Relaxed));
    crate::xlog_info!(
        "D3D12.X: two-pass buffers: {made} made, {reused} taken again, {gone} let go; {} alive ({} MiB), {kept} of them kept for reuse ({} MiB), {} MiB waiting for allocator resets, {} MiB for no known allocator; {refused} not made past {} MiB (XCL_TWO_PASS_MB)",
        made.saturating_sub(gone),
        alive >> 20,
        kept_bytes >> 20,
        held_bytes >> 20,
        unknown_bytes >> 20,
        two_pass_cap() >> 20
    );
}

fn hold_res_until_reset(allocator: usize, mut bufs: Vec<TwoPassBuf>) {
    if bufs.is_empty() {
        return;
    }
    if allocator == 0 {
        let over = UNKNOWN_RES.lock().unwrap().hold(std::time::Instant::now(), bufs.into_iter().map(|b| (b.bytes, b)).collect());
        if over > 0 {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: two-pass buffers of recordings with no known allocator past {} MiB: {over} of the oldest let go early", UNKNOWN_RES_CAP >> 20);
            }
        }
        return;
    }
    ALLOC_RES.lock().unwrap().get_or_insert_with(HashMap::new).entry(allocator).or_default().append(&mut bufs);
}

pub(super) fn release_allocator_res(allocator: usize) {
    let aged = UNKNOWN_RES.lock().unwrap().take_older(std::time::Instant::now(), UNKNOWN_RES_AGE);
    let bufs = ALLOC_RES.lock().unwrap().as_mut().and_then(|m| m.remove(&allocator));
    // (let go outside the lock)
    let _gone = {
        let mut free = FREE_RES.lock().unwrap();
        let now = std::time::Instant::now();
        let mut gone = free.take_idle(now, FREE_RES_IDLE);
        for b in bufs.into_iter().flatten().chain(aged) {
            gone.extend(free.put(now, b.bytes, b.dev, b));
        }
        gone
    };
}

/// A buffer of `FREE_RES` on the device `dev` of `min..=max` bytes, for a recording to take.
fn take_free(dev: usize, min: u64, max: u64) -> Option<TwoPassBuf> {
    let (gone, picked) = {
        let mut free = FREE_RES.lock().unwrap();
        (free.take_idle(std::time::Instant::now(), FREE_RES_IDLE), free.take(dev, min, max))
    };
    // (let go outside the lock)
    drop(gone);
    if picked.is_some() {
        note_two_pass(TP_REUSED);
    }
    picked
}

/// How long the two-pass buffers of a recording with no known allocator are kept: no fence or
/// allocator says when the GPU is done with them (a list XCL never saw reset, a segment reset
/// onto none), so time does – far past any frame in flight.
const UNKNOWN_RES_AGE: std::time::Duration = std::time::Duration::from_secs(10);
/// Most bytes of them kept; past it the oldest go early (logged).
const UNKNOWN_RES_CAP: u64 = 1 << 30;

/// Things of `bytes` each kept from when they were handed over until they are old enough.
struct Aging<T> {
    held: std::collections::VecDeque<(std::time::Instant, u64, T)>,
    bytes: u64,
    cap: u64,
}

impl<T> Aging<T> {
    const fn new(cap: u64) -> Self {
        Aging { held: std::collections::VecDeque::new(), bytes: 0, cap }
    }

    /// Keep `items` (bytes, item) from `now`. Past the cap the oldest go (dropped here): how many.
    fn hold(&mut self, now: std::time::Instant, items: Vec<(u64, T)>) -> usize {
        for (b, t) in items {
            self.bytes += b;
            self.held.push_back((now, b, t));
        }
        let mut over = 0;
        while self.bytes > self.cap && self.held.len() > 1 {
            let (_, b, _) = self.held.pop_front().unwrap();
            self.bytes -= b;
            over += 1;
        }
        over
    }

    /// What has been kept `age` or longer at `now`.
    fn take_older(&mut self, now: std::time::Instant, age: std::time::Duration) -> Vec<T> {
        let mut out = Vec::new();
        while self.held.front().is_some_and(|&(at, _, _)| now.saturating_duration_since(at) >= age) {
            let (_, b, t) = self.held.pop_front().unwrap();
            self.bytes -= b;
            out.push(t);
        }
        out
    }
}

/// the two-pass buffers of recordings with no known allocator (`hold_res_until_reset`)
static UNKNOWN_RES: Mutex<Aging<TwoPassBuf>> = Mutex::new(Aging::new(UNKNOWN_RES_CAP));

/// Buffers the GPU is done with, kept for the recordings that need one next (`FREE_RES`): each
/// with when it was handed back, its bytes and its device, oldest first. What it keeps is what
/// recordings took again in the last `FREE_RES_IDLE`, and at most `cap` bytes, so a steady frame
/// takes back what the one before it handed back and makes none.
struct FreePool<T> {
    free: Vec<(std::time::Instant, u64, usize, T)>,
    bytes: u64,
    cap: u64,
}

impl<T> FreePool<T> {
    const fn new(cap: u64) -> Self {
        FreePool { free: Vec::new(), bytes: 0, cap }
    }

    /// Keep `item` of `bytes` on the device `dev` from `now`. Past the cap the oldest go (what
    /// is returned).
    fn put(&mut self, now: std::time::Instant, bytes: u64, dev: usize, item: T) -> Vec<T> {
        self.bytes += bytes;
        self.free.push((now, bytes, dev, item));
        let mut over = Vec::new();
        while self.bytes > self.cap && !self.free.is_empty() {
            let (_, b, _, t) = self.free.remove(0);
            self.bytes -= b;
            over.push(t);
        }
        over
    }

    /// The smallest kept on `dev` of `min..=max` bytes, of those the last handed back (so the
    /// ones no recording needs grow old and go).
    fn take(&mut self, dev: usize, min: u64, max: u64) -> Option<T> {
        let i = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, &(_, b, d, _))| d == dev && b >= min && b <= max)
            .min_by_key(|&(i, &(_, b, _, _))| (b, std::cmp::Reverse(i)))?
            .0;
        let (_, b, _, t) = self.free.remove(i);
        self.bytes -= b;
        Some(t)
    }

    /// The oldest kept, of any size and device, until they come to `bytes`. None when all of
    /// them come to less: then all stay.
    fn take_room(&mut self, bytes: u64) -> Option<Vec<T>> {
        if self.bytes < bytes {
            return None;
        }
        let mut freed = 0;
        let n = self
            .free
            .iter()
            .take_while(|&&(_, b, ..)| {
                let more = freed < bytes;
                freed += b;
                more
            })
            .count();
        Some(
            self.free
                .drain(..n)
                .map(|(_, b, _, t)| {
                    self.bytes -= b;
                    t
                })
                .collect(),
        )
    }

    /// What was kept `age` or longer at `now`, no recording having taken it.
    fn take_idle(&mut self, now: std::time::Instant, age: std::time::Duration) -> Vec<T> {
        let n = self.free.iter().take_while(|&&(at, ..)| now.saturating_duration_since(at) >= age).count();
        self.free
            .drain(..n)
            .map(|(_, b, _, t)| {
                self.bytes -= b;
                t
            })
            .collect()
    }
}

/// the state of each list's recording
static STATES: super::sharded::Sharded<State> = super::sharded::Sharded::new();

/// draws / dispatches of console code: with a translated pipeline, skipped, with a PC one
static COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// draws and dispatches of console code so far (`XCL_ACTIVITY`)
pub fn activity() -> (u64, u64) {
    (COUNTS[DRAWS].load(Ordering::Relaxed), COUNTS[DISPATCHES].load(Ordering::Relaxed))
}
const DRAWS: usize = 0;
const DRAWS_SKIPPED: usize = 1;
const DRAWS_PC: usize = 2;
const DISPATCHES: usize = 3;
const DISPATCHES_SKIPPED: usize = 4;

fn count(k: usize) {
    use super::capture::Outcome;
    super::capture::outcome(match k {
        DRAWS | DISPATCHES => Outcome::Ran,
        DRAWS_PC => Outcome::Pc,
        _ => Outcome::Skipped,
    });
    let n = COUNTS[k].fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() && n >= 64 {
        let c: Vec<u64> = COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        let (allocators, blocks) = console_list::pending_blocks();
        let (coherent_used, coherent_total, sparse_used) = crate::gpu::uma::get()
            .map_or((0, 0, 0), |u| (u.used() >> 20, u.size() >> 20, u.sparse_used() >> 20));
        crate::xlog_info!(
            "D3D12.X: console code so far: {} draws with translated pipelines ({} skipped, {} with PC pipelines), {} dispatches ({} skipped); coherent {coherent_used}/{coherent_total} MB, sparse {sparse_used} MB, {blocks} blocks held for {allocators} allocators",
            c[DRAWS], c[DRAWS_SKIPPED], c[DRAWS_PC], c[DISPATCHES], c[DISPATCHES_SKIPPED],
        );
    }
}

fn with<R>(list: *mut c_void, f: impl FnOnce(&mut State) -> R) -> R {
    STATES.with(list as usize, f)
}

/// The list was created or reset onto the PC allocator `allocator` with `initial_pso`: the
/// previous recording's memory is the previous allocator's until the GPU is done with it, and
/// pipeline and root signature state reset per D3D12 `CreateCommandList` / `Reset` semantics.
pub(super) fn reset(list: *mut c_void, allocator: usize, initial_pso: usize) {
    let (old, blocks, res_bufs) = with(list, |s| {
        let old = std::mem::replace(&mut s.allocator, allocator);
        s.arena = Default::default();
        s.heap_bound = false;
        s.pso = initial_pso;
        s.layout = None;
        s.args.clear();
        s.layout_gfx = None;
        s.args_gfx.clear();
        s.rs_gfx = 0;
        s.heaps = [0; 2];
        // (the scratch blocks are in `res_retired` already)
        (s.scratch, s.scratch_used, s.scratch_taken, s.scratch_in_args, s.record_open) = (None, 0, 0, false, None);
        (s.guard_out_ia, s.arg_batch) = (false, None);
        let mut res_bufs = std::mem::take(&mut s.res_retired);
        if let Some(b) = s.res_buf.take() {
            res_bufs.push(b);
        }
        res_bufs.extend(s.end_batch());
        (old, std::mem::take(&mut s.uma), res_bufs)
    });
    unmark_deferring(list);
    console_list::hold_until_reset(old, blocks);
    hold_res_until_reset(old, res_bufs);
}

impl State {
    /// The deferred draws of a recording that ends without them (a Reset without a Close, a
    /// list let go): dropped, and the buffer of their records for the caller to hold.
    fn end_batch(&mut self) -> Option<TwoPassBuf> {
        self.deferred.clear();
        (self.batch_used, self.batch_open) = (0, false);
        self.batch_buf.take()
    }
}

/// `ClearState(pso)`: resets pipeline and root-argument state back to creation defaults
/// with `pso` bound, keeping the current recording's allocator and UMA arena intact.
pub(super) fn clear_state(list: *mut c_void, pso: usize) {
    with(list, |s| {
        s.heap_bound = false;
        s.pso = pso;
        s.layout = None;
        s.args.clear();
        s.layout_gfx = None;
        s.args_gfx.clear();
        s.rs_gfx = 0;
        s.heaps = [0; 2];
    });
}

/// Carry bound pipeline and root-signature / root-argument state from `from` to `to` across a
/// deferred mid-list split (`to` records into `allocator` with its own fresh UMA arena while
/// `from` retains its previously allocated blocks until `allocator` resets).
pub(super) fn inherit_for_split(from: *mut c_void, to: *mut c_void, allocator: usize) {
    let (layout, args, layout_gfx, args_gfx, rs_gfx, pso, heaps) = with(from, |s| {
        (s.layout.clone(), s.args.clone(), s.layout_gfx.clone(), s.args_gfx.clone(), s.rs_gfx, s.pso, s.heaps)
    });
    let (old, blocks, res_bufs) = with(to, |s| {
        let old = std::mem::replace(&mut s.allocator, allocator);
        s.arena = Default::default();
        s.heap_bound = false;
        s.layout = layout;
        s.args = args;
        s.layout_gfx = layout_gfx;
        s.args_gfx = args_gfx;
        s.rs_gfx = rs_gfx;
        s.pso = pso;
        s.heaps = heaps;
        // the two-pass record buffers and argument scratch of `to`'s previous recording, as in
        // `reset`: the GPU may still read them, so the new one takes fresh ones
        (s.scratch, s.scratch_used, s.scratch_taken, s.scratch_in_args, s.record_open) = (None, 0, 0, false, None);
        (s.guard_out_ia, s.arg_batch) = (false, None);
        let mut res_bufs = std::mem::take(&mut s.res_retired);
        if let Some(b) = s.res_buf.take() {
            res_bufs.push(b);
        }
        res_bufs.extend(s.end_batch());
        (old, std::mem::take(&mut s.uma), res_bufs)
    });
    unmark_deferring(to);
    console_list::hold_until_reset(old, blocks);
    hold_res_until_reset(old, res_bufs);
}

/// The title changed what descriptor heaps the list has: they are looked at again.
pub(super) fn heaps_changed(list: *mut c_void) {
    with(list, |s| s.heap_bound = false);
}

/// A submit is about to wait for the pipeline `this` has bound.
pub(super) fn want_bound(this: *mut c_void) {
    super::pso_async::want(with(this, |s| s.pso));
}

/// The pipeline the title has bound on `this` (a stand-in while its build goes on).
pub(super) fn bound_pso(this: *mut c_void) -> usize {
    with(this, |s| s.pso)
}

/// `XCL_DUMP_TARGETS`: the resources the first 16 slots of each descriptor table of `this`'s
/// graphics arguments name, each once (as `trace_reads_of_watched` reads them).
pub(super) fn table_resources(this: *mut c_void) -> Vec<usize> {
    table_resources_of(this, true, 16)
}

/// The resources the first `slots` slots of each descriptor table of `this`'s graphics (`gfx`)
/// or compute arguments name, each once.
fn table_resources_of(this: *mut c_void, gfx: bool, slots: usize) -> Vec<usize> {
    let Some(u) = crate::gpu::uma::get() else { return Vec::new() };
    let (layout, args) = with(this, |s| if gfx { (s.layout_gfx.clone(), s.args_gfx.clone()) } else { (s.layout.clone(), s.args.clone()) });
    let Some(layout) = layout else { return Vec::new() };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut out = Vec::new();
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..slots {
            if let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k).filter(|r| !out.contains(r)) {
                out.push(r);
            }
        }
    }
    out
}

/// `XCL_CAPTURE_FRAME`: a draw or dispatch of the title's on `this` (`gfx`: with the graphics
/// root arguments), with the root arguments, heaps and pipeline it has, until the call returns.
pub(super) unsafe fn capture_work(this: *mut c_void, kind: impl FnOnce() -> super::capture::WorkKind, gfx: bool) -> super::capture::Guard {
    use super::capture::{RootArg, RootVal};
    super::capture::op(this, || {
        let (root, heaps, pso) = with(this, |s| {
            let (layout, args) = if gfx { (&s.layout_gfx, &s.args_gfx) } else { (&s.layout, &s.args) };
            let word = |at: usize| args.get(at..at + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
            let root: Vec<RootArg> = layout.as_ref().map_or(Vec::new(), |l| {
                l.params
                    .iter()
                    .enumerate()
                    .map(|(i, p)| RootArg {
                        i: i as u32,
                        v: match *p {
                            Param::Constants { at, count } => RootVal::Constants((0..count as usize).map(|k| word(at as usize + 4 * k)).collect()),
                            Param::Descriptor { at } => RootVal::View(word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32),
                            Param::Table { at } => RootVal::Table(word(at as usize)),
                        },
                    })
                    .collect()
            });
            (root, s.heaps, s.pso)
        });
        let views = table_resources_of(this, gfx, super::capture::TABLE_SLOTS);
        Some(super::capture::work(this, kind(), pso, gfx, root, heaps, views))
    })
}

/// A put-off dispatch that could not be recorded.
pub(super) fn dispatch_dropped() {
    count(DISPATCHES_SKIPPED);
}

/// `to` goes on in the arena `from` took its pieces from (`from` takes no more).
pub(super) fn pass_arena(from: *mut c_void, to: *mut c_void) {
    let arena = with(from, |s| std::mem::take(&mut s.arena));
    with(to, |s| s.arena = arena);
}

/// The list is gone.
pub(super) fn forget(list: *mut c_void) {
    unmark_deferring(list);
    if let Some(mut s) = STATES.remove(list as usize) {
        let batch = s.end_batch();
        console_list::hold_until_reset(s.allocator, s.uma);
        if let Some(b) = s.res_buf.take() {
            s.res_retired.push(b);
        }
        s.res_retired.extend(batch);
        hold_res_until_reset(s.allocator, s.res_retired);
    }
}

#[cfg(test)]
pub(crate) fn seed_reset_state_for_test(list: *mut c_void, allocator: usize, pso: usize, uma: Vec<usize>) {
    with(list, |s| {
        s.allocator = allocator;
        s.pso = pso;
        s.layout = Some(Arc::new(Layout::default()));
        s.args = vec![1, 2, 3, 4];
        s.layout_gfx = Some(Arc::new(Layout::default()));
        s.args_gfx = vec![5, 6, 7, 8];
        s.uma = uma;
    });
}

#[cfg(test)]
pub(crate) fn reset_state_for_test(list: *mut c_void) -> (usize, usize, Vec<u8>, Vec<u8>, Vec<usize>) {
    with(list, |s| (s.allocator, s.pso, s.args.clone(), s.args_gfx.clone(), s.uma.clone()))
}

#[cfg(test)]
pub(crate) fn remove_state_for_test(list: *mut c_void) {
    STATES.remove(list as usize);
}

/// SetDescriptorHeaps: the console addresses of `heaps`, for shaders that index them themselves
/// (`Layout::heaps`).
pub(super) fn set_heaps(list: *mut c_void, heaps: &[usize]) {
    let mut bases = [0u32; 2];
    for (sampler, base) in heaps.iter().filter_map(|&h| super::console_heap::base(h)) {
        bases[sampler as usize] = base;
    }
    with(list, |s| s.heaps = bases);
}

/// bytes of parameter `i` at constant `offset` (of the compute or the graphics arguments)
fn put_in(list: *mut c_void, gfx: bool, i: u32, offset: u32, bytes: &[u8]) {
    with(list, |s| {
        let (layout, args) = if gfx { (&s.layout_gfx, &mut s.args_gfx) } else { (&s.layout, &mut s.args) };
        let Some(at) = layout.as_ref().and_then(|l| l.params.get(i as usize)).map(|p| match *p {
            Param::Constants { at, .. } => at as usize + 4 * offset as usize,
            Param::Descriptor { at } | Param::Table { at } => at as usize,
        }) else {
            return;
        };
        if let Some(t) = args.get_mut(at..at + bytes.len()) {
            t.copy_from_slice(bytes);
        }
    })
}

fn put(list: *mut c_void, i: u32, offset: u32, bytes: &[u8]) {
    put_in(list, false, i, offset, bytes)
}

/// #34 SetGraphicsRootSignature (PC #30)
pub(super) unsafe extern "system" fn set_graphics_root_signature(this: *mut c_void, rs: *mut c_void) {
    let _c = super::capture::op(this, || Some(super::capture::Op::RootSig { gfx: true, rs: rs as usize }));
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        s.args_gfx = vec![0u8; layout.as_ref().map_or(0, |l| l.size as usize)];
        s.layout_gfx = layout;
        s.rs_gfx = rs as usize;
    });
}

/// #36 SetGraphicsRootDescriptorTable (PC #32)
pub(super) unsafe extern "system" fn set_graphics_root_table(this: *mut c_void, i: u32, handle: u64) {
    put_in(this, true, i, 0, &(handle as u32).to_le_bytes());
}

/// #38 SetGraphicsRoot32BitConstant (PC #34)
pub(super) unsafe extern "system" fn set_graphics_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    put_in(this, true, i, offset, &value.to_le_bytes());
}

/// #40 SetGraphicsRoot32BitConstants (PC #36)
pub(super) unsafe extern "system" fn set_graphics_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    if !data.is_null() {
        let bytes: Vec<u8> = std::slice::from_raw_parts(data, n as usize).iter().flat_map(|w| w.to_le_bytes()).collect();
        put_in(this, true, i, offset, &bytes);
    }
}

/// #42 / #44 / #46 SetGraphicsRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #38 /
/// #40 / #42)
pub(super) unsafe extern "system" fn set_graphics_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    note_address(va);
    put_in(this, true, i, 0, &va.to_le_bytes());
}

/// `XCL_TRACE_ADDRESSES=1`: root descriptor addresses of console code – in unified memory
/// (translated shaders see them) or not – counted and logged
fn note_address(va: u64) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static IN: AtomicU64 = AtomicU64::new(0);
    static OUT: AtomicU64 = AtomicU64::new(0);
    if !*ON.get_or_init(|| std::env::var_os("XCL_TRACE_ADDRESSES").is_some()) {
        return;
    }
    let inside = crate::gpu::uma::get().is_some_and(|u| u.contains(va as usize));
    let (i, o) = if inside { (IN.fetch_add(1, Ordering::Relaxed) + 1, OUT.load(Ordering::Relaxed)) } else { (IN.load(Ordering::Relaxed), OUT.fetch_add(1, Ordering::Relaxed) + 1) };
    if (i + o).is_power_of_two() {
        crate::xlog_info!("D3D12.X: root descriptor addresses: {i} in unified memory, {o} outside (latest {va:#x})");
    }
}

/// Bytes asked for at a root view's address in a GPU view (`view_want`). A root view names an
/// address and no size, and no lookup by address says which of the title's buffers it lies in.
const ROOT_VIEW_BYTES: u64 = 64 << 10;

/// The view windows of unified memory as one range (start, bytes; 0 bytes: none, as in every
/// layout below 16 windows). The layout is fixed when unified memory is made, so an address the
/// title gives costs one compare to be told apart from a GPU view.
pub(super) fn view_span() -> (u64, u64) {
    static SPAN: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *SPAN.get_or_init(|| crate::gpu::uma::get().map_or((0, 0), view_span_of))
}

/// `view_span` of `u` (its view windows follow each other).
pub(super) fn view_span_of(u: &crate::gpu::uma::Uma) -> (u64, u64) {
    let mut views = (0..u.shader_windows()).filter(|&k| u.is_view_window(k)).map(|k| u.window(k));
    let Some((start, bytes)) = views.next() else { return (0, 0) };
    let end = views.last().map_or(start + bytes, |(at, bytes)| at + bytes);
    (start as u64, (end - start) as u64)
}

/// Whether `va` lies in `span` (a `view_span`).
pub(super) fn in_span((start, bytes): (u64, u64), va: u64) -> bool {
    va.wrapping_sub(start) < bytes
}

/// Whether `va` lies in a view window.
pub(super) fn in_views(va: u64) -> bool {
    in_span(view_span(), va)
}

/// `view_want::want` for an address the title gave the GPU: one compare for any other address.
pub(super) fn want_view(path: super::view_want::Path, va: u64, len: u64) -> bool {
    in_views(va) && super::view_want::want(path, va, len)
}

/// The root views of `args` (packed by `layout`) in the view windows `span`, and the bytes asked
/// for at each.
fn root_views_wanted<'a>(layout: &'a Layout, args: &'a [u8], span: (u64, u64)) -> impl Iterator<Item = (u64, u64)> + 'a {
    console_list::root_views(layout, args).filter(move |&va| in_span(span, va)).map(|va| (va, ROOT_VIEW_BYTES))
}

/// A bind of the root arguments `args`: the tiles of the root views among them that lie in a GPU
/// view are asked for (on `path`), before the list that reads them is submitted. Without view
/// windows one compare a bind.
pub(super) fn want_root_views(layout: Option<&Layout>, args: &[u8], path: super::view_want::Path) {
    let span = view_span();
    if span.1 == 0 {
        return;
    }
    let Some(l) = layout else { return };
    for (va, len) in root_views_wanted(l, args, span) {
        super::view_want::want(path, va, len);
    }
}

/// command signatures that change root arguments (they need the title's root signature)
static ROOT_SIGNATURES: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DispatchMeshSig {
    companion_dispatch: usize,
    byte_stride: u32,
    mesh_offset: u32,
}

static DISPATCH_MESH_SIGS: Mutex<Option<HashMap<usize, DispatchMeshSig>>> = Mutex::new(None);

/// (local) title DispatchMesh signature → (Dispatch, DispatchMesh) signatures with the command
/// argument alone, for the two-pass ExecuteIndirect (`device_x::create_command_signature_x`)
static PLAIN_TWO_PASS_SIGS: Mutex<Option<HashMap<usize, (usize, usize)>>> = Mutex::new(None);

pub(super) fn note_plain_two_pass_signatures(sig: usize, dispatch: usize, mesh: usize) {
    if dispatch != 0 && mesh != 0 {
        PLAIN_TWO_PASS_SIGS.lock().unwrap().get_or_insert_with(HashMap::new).insert(sig, (dispatch, mesh));
    }
}

/// `XCL_TWO_PASS_PLAIN=0`: the two-pass ExecuteIndirect uses the title's signature as upstream does.
fn plain_two_pass_signatures(sig: usize) -> Option<(usize, usize)> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("XCL_TWO_PASS_PLAIN").map_or(true, |v| v.trim() != "0")) {
        return None;
    }
    PLAIN_TWO_PASS_SIGS.lock().unwrap().as_ref()?.get(&sig).copied()
}

pub(super) fn note_root_changing_signature(sig: usize) {
    ROOT_SIGNATURES.lock().unwrap().get_or_insert_with(Default::default).insert(sig);
}

/// command signatures with a Dispatch argument (noted at every creation, so a signature made
/// at a freed one's address does not keep its kind)
static DISPATCH_SIGNATURES: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

pub(super) fn note_signature_dispatches(sig: usize, dispatches: bool) {
    let mut s = DISPATCH_SIGNATURES.lock().unwrap();
    let s = s.get_or_insert_with(Default::default);
    if dispatches {
        s.insert(sig);
    } else {
        s.remove(&sig);
    }
}

fn signature_dispatches(sig: usize) -> bool {
    DISPATCH_SIGNATURES.lock().unwrap().as_ref().is_some_and(|s| s.contains(&sig))
}

pub(super) fn note_dispatch_mesh_signature(sig: usize, companion_dispatch: usize, byte_stride: u32, mesh_offset: u32) {
    DISPATCH_MESH_SIGS
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(sig, DispatchMeshSig { companion_dispatch, byte_stride, mesh_offset });
}

/// A command signature was made at `sig`: what a freed one there was noted as goes (a mesh
/// dispatch signature is noted again after this).
pub(super) fn forget_dispatch_mesh_signature(sig: usize) {
    if let Some(m) = DISPATCH_MESH_SIGS.lock().unwrap().as_mut() {
        m.remove(&sig);
    }
}

fn dispatch_mesh_sig_of(sig: usize) -> Option<DispatchMeshSig> {
    DISPATCH_MESH_SIGS.lock().unwrap().as_ref()?.get(&sig).copied()
}

unsafe fn emit_uav_barrier(list: &ID3D12GraphicsCommandList, res: Option<&ID3D12Resource>) {
    let mut b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(res.cloned()),
            }),
        },
    };
    list.ResourceBarrier(&[std::ptr::read(&b)]);
    std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.UAV).pResource);
}

/// The barrier between two passes of a two-pass ExecuteIndirect: a UAV barrier on the records.
/// (local) With the title's signatures (`titles`), which vkd3d-proton runs as device generated
/// commands when they set more than the command, a global aliasing barrier goes with it under
/// Wine: on NVIDIA's driver a UAV barrier alone (on the records or on all) did not order such
/// commands – a mesh pass read its records before its compute pass wrote them, or the next
/// one's – and with it every run was right (`tests::gpu_two_pass_indirect_probe`).
unsafe fn emit_two_pass_barrier(list: &ID3D12GraphicsCommandList, records: &ID3D12Resource, titles: bool) {
    if !titles || !crate::util::under_wine() {
        return emit_uav_barrier(list, Some(records));
    }
    let mut b = [
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(Some(records.clone())) }) },
        },
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER { pResourceBefore: std::mem::ManuallyDrop::new(None), pResourceAfter: std::mem::ManuallyDrop::new(None) }),
            },
        },
    ];
    list.ResourceBarrier(&[std::ptr::read(&b[0]), std::ptr::read(&b[1])]);
    std::mem::ManuallyDrop::drop(&mut (*b[0].Anonymous.UAV).pResource);
}

unsafe fn emit_transition(list: &ID3D12GraphicsCommandList, res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) {
    let mut b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::ManuallyDrop::new(Some(res.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    list.ResourceBarrier(&[std::ptr::read(&b)]);
    std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.Transition).pResource);
}

/// A UAV barrier on `res` (None: on all resources) and a transition of `res` from `before` to
/// `after`, in one ResourceBarrier call.
unsafe fn emit_uav_and_transition(list: &ID3D12GraphicsCommandList, uav: Option<&ID3D12Resource>, res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) {
    let mut b = [
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(uav.cloned()) }) },
        },
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(Some(res.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        },
    ];
    list.ResourceBarrier(&b);
    std::mem::ManuallyDrop::drop(&mut (*b[0].Anonymous.UAV).pResource);
    std::mem::ManuallyDrop::drop(&mut (*b[1].Anonymous.Transition).pResource);
}

unsafe fn default_uav_buffer(dev: &ID3D12Device, bytes: u64, what: &str) -> Result<ID3D12Resource, String> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };
    let mut res: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        None,
        &mut res,
    )
    .map_err(|e| format!("{what} ({bytes:#x} B): {e}"))?;
    // (a name for what a device removal reports about an address, `real::log_device_removed`;
    // and where the first ones lie: an address just past one's end has no allocation to name)
    if let Some(r) = &res {
        let _ = r.SetName(&windows::core::HSTRING::from(format!("xcl {what}")));
        static MADE: AtomicU64 = AtomicU64::new(0);
        let n = MADE.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 48 || n.is_power_of_two() {
            let va = r.GetGPUVirtualAddress();
            crate::xlog_info!("D3D12.X: {what} #{n}: GPU addresses {va:#x} .. {:#x}", va + bytes);
        }
    }
    res.ok_or_else(|| format!("null {what}"))
}

/// Groups of a two-pass dispatch that have a record: one record a group in `cap_bytes` of
/// records of `record_bytes`, and every group in X, the mesh pass's `split` times as many (at most
/// 65 535 a dimension). The compute pass writes record `gid.x` with no bound of its own.
fn record_groups(cap_bytes: u64, record_bytes: u64, split: u32) -> u32 {
    (cap_bytes / record_bytes.max(1)).min(65_535 / split.max(1) as u64) as u32
}

/// The ExecuteIndirects of a two-pass ExecuteIndirect of `max` commands, each of one command and
/// followed by a UAV barrier on the records: (command, compute pass). Each command's compute pass
/// writes its records from record 0, so the next one goes only after the mesh pass drew them.
fn two_pass_steps(max: u32) -> impl Iterator<Item = (u32, bool)> {
    (0..max).flat_map(|k| [(k, true), (k, false)])
}

/// `XCL_TWO_PASS_BATCH=1`: two-pass draws record fewer barriers (read once). Their records go
/// in the two halves of the record buffer in turn (`Records::Halves`), so a draw's mesh pass
/// needs no UAV barrier after it: the next compute pass writes the other half, and the barrier
/// after that one orders the mesh pass before the half is written again. An ExecuteIndirect's
/// rewritten arguments stay in INDIRECT_ARGUMENT until the next one rewrites into the block, with
/// the UAV barrier before the rewrite in the same ResourceBarrier call as that transition.
/// `XCL_TWO_PASS_BATCH=2` also defers runs of two-pass draws (`flush_two_pass`); the draws it
/// does not defer go as with 1.
fn two_pass_batch() -> bool {
    two_pass_mode() >= 1
}

/// `XCL_TWO_PASS_BATCH=2`: runs of two-pass draws are deferred (`flush_two_pass`).
pub(super) fn two_pass_defer() -> bool {
    two_pass_mode() == 2
}

/// `XCL_TWO_PASS_BATCH`: 0 (unset or anything else), 1 or 2 (read once).
fn two_pass_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("XCL_TWO_PASS_BATCH").as_deref().map(str::trim) {
        Ok("1") => 1,
        Ok("2") => 2,
        _ => 0,
    })
}

/// (local, to tell them apart on a GPU that hangs) `XCL_TWO_PASS`: `0` skips every two-pass
/// mesh draw, `direct` skips the ExecuteIndirect ones only. 2: all run, 1: direct only, 0: none.
fn two_pass_switch() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("XCL_TWO_PASS").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("direct") => 1,
        _ => 2,
    })
}

/// Record regions of a two-pass pass (`TwoPassStep::Pass`): the first half, the second half, or
/// the whole buffer (both).
const HALF_0: u8 = 0b01;
const HALF_1: u8 = 0b10;
const WHOLE: u8 = HALF_0 | HALF_1;

/// How a two-pass draw lays out its records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Records {
    /// every command from the start of the buffer, a UAV barrier after each pass
    Whole,
    /// commands in the buffer's two halves in turn, no UAV barrier after a mesh pass
    Halves,
}

/// A step of a two-pass draw on the PC list (`two_pass_plan`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TwoPassStep {
    /// the compute pass (writes the records of `region`) or the mesh pass (reads them) of command `k`
    Pass { k: u32, compute: bool, region: u8 },
    /// a UAV barrier on the records
    Barrier,
}

/// The steps of a two-pass draw of `commands` commands (one for a DispatchMesh) with `records`,
/// after draws that left a mesh pass reading `open` with no barrier since. `Whole` is the order
/// the draws had before `Halves`: `two_pass_steps`, a barrier after each.
fn two_pass_plan(commands: u32, records: Records, open: Option<u8>) -> Vec<TwoPassStep> {
    let mut v = Vec::with_capacity(commands as usize * 4 + 1);
    let first = match (records, open) {
        (_, None) => HALF_0,
        (Records::Halves, Some(HALF_0)) => HALF_1,
        (Records::Halves, Some(HALF_1)) => HALF_0,
        _ => {
            v.push(TwoPassStep::Barrier);
            HALF_0
        }
    };
    for (k, compute) in two_pass_steps(commands) {
        match records {
            Records::Whole => {
                v.push(TwoPassStep::Pass { k, compute, region: WHOLE });
                v.push(TwoPassStep::Barrier);
            }
            Records::Halves => {
                let region = if k % 2 == 0 { first } else { first ^ WHOLE };
                v.push(TwoPassStep::Pass { k, compute, region });
                if compute {
                    v.push(TwoPassStep::Barrier);
                }
            }
        }
    }
    v
}

/// What a mesh pass may still be reading after `step`, with `open` before it.
fn open_after(open: Option<u8>, step: TwoPassStep) -> Option<u8> {
    match step {
        TwoPassStep::Barrier => None,
        TwoPassStep::Pass { compute: false, region, .. } => Some(open.unwrap_or(0) | region),
        TwoPassStep::Pass { compute: true, .. } => open,
    }
}

/// `plan` on `list` with the records in `res` at `res_va`: `pass(k, compute, records)` records a
/// pass with its region at GPU address `records` (the second half `half` bytes in). What each
/// step leaves a mesh pass reading is kept for the next draw's plan (`State::record_open`).
/// (local) `titles`: the passes are ExecuteIndirects with the title's signatures
/// (`emit_two_pass_barrier`).
#[allow(clippy::too_many_arguments)]
unsafe fn run_two_pass_plan(
    this: *mut c_void,
    list: &ID3D12GraphicsCommandList,
    plan: &[TwoPassStep],
    res: &ID3D12Resource,
    res_va: u64,
    half: u64,
    titles: bool,
    mut pass: impl FnMut(u32, bool, u64) -> Result<(), String>,
) -> Result<(), String> {
    for &step in plan {
        match step {
            TwoPassStep::Barrier => emit_two_pass_barrier(list, res, titles),
            TwoPassStep::Pass { k, compute, region } => pass(k, compute, res_va + if region == HALF_1 { half } else { 0 })?,
        }
        with(this, |s| s.record_open = open_after(s.record_open, step));
    }
    Ok(())
}

/// The record buffer of a two-pass draw that needs `need` bytes of records: buffer, address, the
/// bytes of a region and the layout. `Halves` (with `two_pass_batch`) when twice the buffer the
/// draw would have had whole fits `RECORD_MAX` and the two-pass buffers: each half holds what the
/// whole one would.
unsafe fn records_for(this: *mut c_void, need: u64, batch: bool) -> Result<(ID3D12Resource, u64, u64, Records), String> {
    if let Some(both) = halves_bytes(need, batch) {
        if let Ok((buf, va, cap)) = ensure_res_buffer(this, both) {
            return Ok((buf, va, cap / 2, Records::Halves));
        }
    }
    let (buf, va, cap) = ensure_res_buffer(this, need)?;
    Ok((buf, va, cap, Records::Whole))
}

/// The bytes of a record buffer in halves for `need` (`records_for`), if it can have them.
fn halves_bytes(need: u64, batch: bool) -> Option<u64> {
    let both = record_buffer_bytes(need) * 2;
    (batch && both <= RECORD_MAX).then_some(both)
}

// ---------------------------------------------------------------------------------------
// deferred two-pass draws (`XCL_TWO_PASS_BATCH=2`)
//
// A two-pass draw on a direct list is not recorded at once: what both of its passes read is
// written to unified memory as the unbatched draw would write it (its io blocks with the title's
// root arguments of that moment, the argument rewrite's constants), and the draw waits in the
// list's `State::deferred` with a region of its own in the list's batch record buffer. The run
// goes out (`flush_two_pass`) before any title call on the list that is not a root-argument set
// or another such draw (`before_list_call`, in front of every slot of the list's vtable), at the
// list's Close, at `XCL_TWO_PASS_BATCH_MOST` draws and when the next draw's records do not fit:
// the argument rewrites, all compute passes, one barrier, the mesh passes in the title's order.
// Between the draws of a run nothing reaches the PC list, so each mesh pass draws with the PC
// state (render targets, viewports, …) its draw had.
// ---------------------------------------------------------------------------------------

/// Most draws in a run (`XCL_TWO_PASS_BATCH_MOST`, 64 by default; read once).
fn batch_most() -> usize {
    static MOST: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| std::env::var("XCL_TWO_PASS_BATCH_MOST").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(64).max(1))
}

/// Bytes of a list's batch record buffer (`XCL_TWO_PASS_BATCH_MB`, 128 MiB by default; read
/// once). An ExecuteIndirect's region is what its draw would have had alone, up to 32 MiB.
fn batch_records_bytes() -> u64 {
    static BYTES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| std::env::var("XCL_TWO_PASS_BATCH_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(128).clamp(1, 1024) << 20)
}

/// Lists with deferred draws, and how many (the hook in front of every list call looks at the
/// count first).
static DEFERRING: super::sharded::Sharded<u8> = super::sharded::Sharded::new();
static DEFERRING_LISTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn mark_deferring(list: *mut c_void) {
    if DEFERRING.insert(list as usize, 1).is_none() {
        DEFERRING_LISTS.fetch_add(1, Ordering::Relaxed);
    }
}

fn unmark_deferring(list: *mut c_void) {
    if DEFERRING.remove(list as usize).is_some() {
        DEFERRING_LISTS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// What a deferred draw runs besides its two stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeferredWork {
    /// DispatchMesh: the compute pass's groups, the mesh pass's
    Mesh { groups: u32, mesh_groups: u32 },
    /// ExecuteIndirect of one command: its argument rewrite into `block` (in `before` until the
    /// run), the companion Dispatch signature and the DispatchMesh one, where each pass's command is
    Indirect { rewrite: Rewrite, block: usize, before: D3D12_RESOURCE_STATES, companion: usize, sig: usize, compute_at: u64, mesh_at: u64 },
}

/// What a deferred draw records: everything it reads was fixed when the title made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DrawSnap {
    compute: StageBind,
    mesh: StageBind,
    work: DeferredWork,
}

/// What a run holds (`State::deferred`): a deferred draw, or a SetPipelineState of the title to a
/// two-pass pipeline between them, recorded (`real::bind_pipeline`: the PC pipeline and what
/// `list_x::on_set_pipeline_state` sets with it, such as the depth bias) where it came among the
/// mesh passes. The reference keeps the pipeline until then.
enum RunItem {
    Draw(DeferredDraw),
    Bind { pso: usize, _held: Option<ID3D12PipelineState> },
}

/// A run item as `batch_commands` takes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Draw(DrawSnap),
    Bind(usize),
}

impl RunItem {
    fn step(&self) -> Step {
        match self {
            RunItem::Draw(d) => Step::Draw(d.snap),
            RunItem::Bind { pso, .. } => Step::Bind(*pso),
        }
    }
}

/// A deferred two-pass draw (`RunItem::Draw`).
struct DeferredDraw {
    snap: DrawSnap,
    /// (what the snapshot names stays alive until the run is recorded)
    _tp: Arc<super::real::TwoPassPso>,
    _mesh_pso: Option<ID3D12PipelineState>,
    _block: Option<ID3D12Resource>,
    /// the pipeline the title bound, for the counts; the kind and size for `XCL_GPU_TIMES`
    raw_pso: usize,
    kind: &'static str,
    size: [u32; 3],
}

/// Why a two-pass draw is recorded at once rather than deferred.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NoDefer {
    /// `XCL_TWO_PASS_BATCH` is not 2
    Off,
    /// not a direct list (a bundle cannot hold the barriers; a compute list draws nothing)
    NotDirect,
    /// `XCL_CAPTURE_FRAME` records each call where it is made
    Capture,
    /// an ExecuteIndirect of more than one command (each runs its passes in turn)
    Commands,
    /// its records are more than a batch record buffer
    TooBig,
}

/// Whether a two-pass draw of `commands` commands whose records take `region` bytes is deferred.
fn defer_check(on: bool, ty: Option<D3D12_COMMAND_LIST_TYPE>, capture: bool, commands: u32, region: u64, batch_bytes: u64) -> Result<(), NoDefer> {
    if !on {
        return Err(NoDefer::Off);
    }
    if ty != Some(D3D12_COMMAND_LIST_TYPE_DIRECT) {
        return Err(NoDefer::NotDirect);
    }
    if capture {
        return Err(NoDefer::Capture);
    }
    if commands != 1 {
        return Err(NoDefer::Commands);
    }
    if region > batch_bytes {
        return Err(NoDefer::TooBig);
    }
    Ok(())
}

/// Where in a batch record buffer of `bytes` with `used` taken a draw's `region` goes: None, the
/// run so far is recorded first and it goes at the start.
fn batch_place(used: u64, region: u64, bytes: u64) -> Option<u64> {
    (used.saturating_add(region) <= bytes).then_some(used)
}

/// Bytes of a draw's region: its records, 256-aligned.
fn batch_region_bytes(need: u64) -> u64 {
    need.max(1).next_multiple_of(256)
}

/// Console list slots a run of deferred draws goes on across (`before_list_call`): QueryInterface,
/// AddRef, Release and the private data / name / device / type ones (they record nothing), the
/// compute and graphics root signature and root argument sets (kept by XCL, `put_in`: nothing
/// reaches the PC list), SetPipelineState, ExecuteIndirect and DispatchMesh (at their slots of
/// the title's layout, `real::NEWER_LIST_SLOTS`; they go into the run or record it first
/// themselves: `set_pipeline_state`). Everything else – barriers, copies, clears, other draws,
/// dispatches, render targets, viewports, heaps, queries, predication, bundles, Close, Reset,
/// the console's own methods – records the run first.
fn keeps_two_pass_batch(slot: usize, newer_layout: bool) -> bool {
    let (ei, dm) = if newer_layout { (61, 129) } else { (59, 127) };
    matches!(slot, 0..=8 | 25 | 33..=46) || slot == ei || slot == dm
}

/// Whether a list call through `slot` records the list's deferred draws first.
fn flush_wanted(slot: usize, newer_layout: bool, from_title: bool, deferring: bool) -> bool {
    deferring && from_title && !keeps_two_pass_batch(slot, newer_layout)
}

/// What ended a run (`flush_two_pass`), for the log: a title call through a console list slot
/// (`before_list_call`), or one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RunEnd {
    Call(usize),
    /// the list's Close
    Close,
    /// a PC Close of a segment of the list (`list_x`)
    Segment,
    /// `XCL_TWO_PASS_BATCH_MOST` draws
    Most,
    /// the next draw's records did not fit
    Full,
    /// a DispatchMesh / ExecuteIndirect not deferred
    Draw,
    /// a SetPipelineState to a pipeline that is not two-pass
    Pipeline,
}

/// Console slots counted apart in `RUN_ENDS`; the other ends after them.
const RUN_END_SLOTS: usize = 640;
static RUN_ENDS: [AtomicU64; RUN_END_SLOTS + 6] = [const { AtomicU64::new(0) }; RUN_END_SLOTS + 6];

fn run_end_index(e: RunEnd) -> usize {
    match e {
        RunEnd::Call(s) => s.min(RUN_END_SLOTS - 1),
        RunEnd::Close => RUN_END_SLOTS,
        RunEnd::Segment => RUN_END_SLOTS + 1,
        RunEnd::Most => RUN_END_SLOTS + 2,
        RunEnd::Full => RUN_END_SLOTS + 3,
        RunEnd::Draw => RUN_END_SLOTS + 4,
        RunEnd::Pipeline => RUN_END_SLOTS + 5,
    }
}

/// The name of `RUN_ENDS[i]` (a slot by the PC method the title's layout has there).
fn run_end_name(i: usize, newer_layout: bool) -> String {
    match i.checked_sub(RUN_END_SLOTS) {
        None if newer_layout && i >= 59 => format!("console list slot #{i} (newer layout)"),
        None => super::real::list_slot_name(i),
        Some(0) => "Close".into(),
        Some(1) => "a Close of XCL's (list segment, KickoffX)".into(),
        Some(2) => "XCL_TWO_PASS_BATCH_MOST draws".into(),
        Some(3) => "the batch record buffer full".into(),
        Some(4) => "a DispatchMesh / ExecuteIndirect not deferred".into(),
        _ => "SetPipelineState to a pipeline that is not two-pass".into(),
    }
}

/// The run ends counted so far, most first: "name count, …" (the first `n`).
fn run_ends_text(n: usize) -> String {
    let mut v: Vec<(u64, usize)> = RUN_ENDS.iter().enumerate().map(|(i, c)| (c.load(Ordering::Relaxed), i)).filter(|e| e.0 > 0).collect();
    v.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let newer = super::real::QUEUE_SHIFTED.load(Ordering::Acquire);
    v.iter().take(n).map(|&(c, i)| format!("{} {c}", run_end_name(i, newer))).collect::<Vec<_>>().join(", ")
}

/// In front of every command list slot with `XCL_TWO_PASS_BATCH=2` (`real::class_for`): a title
/// call on a list with deferred draws records them first, unless the slot keeps the run
/// (`keeps_two_pass_batch`). Calls from XCL and the runtime go on as they are.
pub(super) unsafe extern "system" fn before_list_call(caller: usize, slot: u64, frame: *const u64) -> usize {
    if DEFERRING_LISTS.load(Ordering::Relaxed) == 0 {
        return 0;
    }
    let this = *frame.add(0x80 / 8) as *mut c_void;
    let deferring = DEFERRING.contains(this as usize);
    let newer = super::real::QUEUE_SHIFTED.load(Ordering::Acquire);
    if flush_wanted(slot as usize, newer, deferring && crate::modules::image_base_for_addr(caller).is_some(), deferring) {
        flush_two_pass(this, RunEnd::Call(slot as usize));
    }
    0
}

/// A UAV barrier on all resources or on `uav`, and transitions, in one ResourceBarrier call
/// (`Cmd::Barriers`).
unsafe fn emit_barrier_set(list: &ID3D12GraphicsCommandList, all: bool, uav: Option<&ID3D12Resource>, transitions: &[(&ID3D12Resource, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES)]) {
    let mut b: Vec<D3D12_RESOURCE_BARRIER> = Vec::with_capacity(transitions.len() + 2);
    for r in [all.then_some(None), uav.map(Some)].into_iter().flatten() {
        b.push(D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(r.cloned()) }) },
        });
    }
    for &(r, before, after) in transitions {
        b.push(D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(Some(r.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        });
    }
    if b.is_empty() {
        return;
    }
    list.ResourceBarrier(&b);
    for x in &mut b {
        if x.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV {
            std::mem::ManuallyDrop::drop(&mut (*x.Anonymous.UAV).pResource);
        } else {
            std::mem::ManuallyDrop::drop(&mut (*x.Anonymous.Transition).pResource);
        }
    }
}

/// A command of a run of deferred draws (`batch_commands`); objects by address.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Cmd {
    /// one ResourceBarrier call: a UAV barrier on all resources (`all`), one on `uav`, transitions
    Barriers { all: bool, uav: Option<usize>, transitions: Vec<(usize, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES)> },
    ComputeRs(usize),
    GraphicsRs(usize),
    Pso(usize),
    ComputeView(u32, u64),
    GraphicsView(u32, u64),
    Dispatch(u32),
    DispatchMesh(u32),
    /// ExecuteIndirect of one command: signature, argument buffer, offset
    Indirect(usize, usize, u64),
    /// the title's SetPipelineState to the pipeline `p`, as without a run (`real::bind_pipeline`)
    Bind(usize),
    /// an `XCL_GPU_TIMES` span: the rewrites and compute passes (None), the mesh pass of draw `d`
    TimesBegin(Option<usize>),
    TimesEnd,
}

/// The commands of a run of `steps` with their records in `records`, after a run whose mesh
/// passes may still read them (`open`): the argument rewrites (after a UAV barrier on all
/// resources – the title's writes of the arguments, `fresh_indirect_arguments` – and the
/// argument blocks to UNORDERED_ACCESS in the same call), the blocks to INDIRECT_ARGUMENT, every
/// compute pass, one UAV barrier on the records, every mesh pass in the title's order with the
/// title's pipeline sets between them where they came. A root signature or pipeline is set where
/// it changes (and a pipeline after each of the title's); every pass sets all its root views.
fn batch_commands(steps: &[Step], records: usize, open: bool) -> Vec<Cmd> {
    use D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT as ARGS;
    use D3D12_RESOURCE_STATE_UNORDERED_ACCESS as UAV;
    let draws: Vec<DrawSnap> = steps.iter().filter_map(|s| if let Step::Draw(d) = s { Some(*d) } else { None }).collect();
    let mut v = Vec::with_capacity(steps.len() * 20 + 8);
    let mut blocks: Vec<(usize, D3D12_RESOURCE_STATES)> = Vec::new();
    for d in &draws {
        if let DeferredWork::Indirect { block, before, .. } = d.work {
            if !blocks.iter().any(|b| b.0 == block) {
                blocks.push((block, before));
            }
        }
    }
    if !blocks.is_empty() {
        v.push(Cmd::Barriers { all: true, uav: None, transitions: blocks.iter().map(|&(b, s)| (b, s, UAV)).collect() });
    } else if open {
        v.push(Cmd::Barriers { all: false, uav: Some(records), transitions: Vec::new() });
    }
    v.push(Cmd::TimesBegin(None));
    let (mut crs, mut grs, mut pso) = (0usize, 0usize, 0usize);
    let set = |v: &mut Vec<Cmd>, now: &mut usize, to: usize, cmd: fn(usize) -> Cmd| {
        if *now != to {
            v.push(cmd(to));
            *now = to;
        }
    };
    for d in &draws {
        if let DeferredWork::Indirect { rewrite, .. } = d.work {
            set(&mut v, &mut crs, rewrite.rs, Cmd::ComputeRs);
            set(&mut v, &mut pso, rewrite.pso, Cmd::Pso);
            v.extend(rewrite.views.iter().map(|&(slot, va)| Cmd::ComputeView(slot, va)));
            v.push(Cmd::Dispatch(rewrite.groups));
        }
    }
    if !blocks.is_empty() {
        v.push(Cmd::Barriers { all: false, uav: None, transitions: blocks.iter().map(|&(b, _)| (b, UAV, ARGS)).collect() });
    }
    for d in &draws {
        set(&mut v, &mut crs, d.compute.rs, Cmd::ComputeRs);
        set(&mut v, &mut pso, d.compute.pso, Cmd::Pso);
        v.extend(d.compute.views.iter().enumerate().map(|(k, &va)| Cmd::ComputeView(k as u32, va)));
        v.push(match d.work {
            DeferredWork::Mesh { groups, .. } => Cmd::Dispatch(groups),
            DeferredWork::Indirect { block, companion, compute_at, .. } => Cmd::Indirect(companion, block, compute_at),
        });
    }
    v.push(Cmd::TimesEnd);
    v.push(Cmd::Barriers { all: false, uav: Some(records), transitions: Vec::new() });
    let mut i = 0;
    for s in steps {
        let d = match s {
            Step::Bind(p) => {
                v.push(Cmd::Bind(*p));
                // (it set the PC pipeline)
                pso = 0;
                continue;
            }
            Step::Draw(d) => d,
        };
        v.push(Cmd::TimesBegin(Some(i)));
        i += 1;
        set(&mut v, &mut grs, d.mesh.rs, Cmd::GraphicsRs);
        set(&mut v, &mut pso, d.mesh.pso, Cmd::Pso);
        v.extend(d.mesh.views.iter().enumerate().map(|(k, &va)| Cmd::GraphicsView(k as u32, va)));
        v.push(match d.work {
            DeferredWork::Mesh { mesh_groups, .. } => Cmd::DispatchMesh(mesh_groups),
            DeferredWork::Indirect { block, sig, mesh_at, .. } => Cmd::Indirect(sig, block, mesh_at),
        });
        v.push(Cmd::TimesEnd);
    }
    v
}

/// `cmds` (`batch_commands` of `draws`) on the list `this`.
unsafe fn emit_commands(this: *mut c_void, list: &ID3D12GraphicsCommandList, cmds: &[Cmd], draws: &[&DeferredDraw]) {
    type Ei = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
    type Dm = unsafe extern "system" fn(*mut c_void, u32, u32, u32);
    let pc_ei: Ei = std::mem::transmute(super::real::orig_fn(this, 59));
    let pc_dm: Dm = std::mem::transmute(super::real::orig_fn(this, 79));
    // (every object here is held by `draws` or the list's state until this returns)
    let res = |raw: usize| {
        let p = raw as *mut c_void;
        ID3D12Resource::from_raw_borrowed(&p).cloned()
    };
    let rs = |raw: usize| {
        let p = raw as *mut c_void;
        ID3D12RootSignature::from_raw_borrowed(&p).cloned()
    };
    for c in cmds {
        match c {
            Cmd::Barriers { all, uav, transitions } => {
                let uav = uav.and_then(res);
                let held: Vec<(ID3D12Resource, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES)> = transitions.iter().filter_map(|&(r, a, b)| Some((res(r)?, a, b))).collect();
                let t: Vec<_> = held.iter().map(|(r, a, b)| (r, *a, *b)).collect();
                emit_barrier_set(list, *all, uav.as_ref(), &t);
            }
            Cmd::ComputeRs(r) => list.SetComputeRootSignature(rs(*r).as_ref()),
            Cmd::GraphicsRs(r) => list.SetGraphicsRootSignature(rs(*r).as_ref()),
            Cmd::Pso(p) => {
                let p = *p as *mut c_void;
                list.SetPipelineState(ID3D12PipelineState::from_raw_borrowed(&p));
            }
            Cmd::ComputeView(slot, va) => list.SetComputeRootUnorderedAccessView(*slot, *va),
            Cmd::GraphicsView(slot, va) => list.SetGraphicsRootUnorderedAccessView(*slot, *va),
            Cmd::Dispatch(n) => list.Dispatch(*n, 1, 1),
            Cmd::DispatchMesh(n) => pc_dm(this, *n, 1, 1),
            Cmd::Indirect(sig, buf, at) => pc_ei(this, *sig, 1, *buf, *at, 0, 0),
            Cmd::Bind(p) => super::real::bind_pipeline(this, *p as *mut c_void),
            Cmd::TimesBegin(None) => super::gpu_times::before(this, 0, super::gpu_times::Kind::MeshBatch, [draws.len() as u32, 0, 0], false),
            Cmd::TimesBegin(Some(d)) => {
                let d = &draws[*d];
                let indirect = matches!(d.snap.work, DeferredWork::Indirect { .. });
                super::gpu_times::before(this, d.snap.mesh.pso, super::gpu_times::Kind::Mesh(d.kind), d.size, indirect);
            }
            Cmd::TimesEnd => super::gpu_times::after(this),
        }
    }
}

/// The deferred draws of the list `this`, recorded now (`batch_commands`), for `why`; nothing
/// when it has none.
pub(super) unsafe fn flush_two_pass(this: *mut c_void, why: RunEnd) {
    if DEFERRING_LISTS.load(Ordering::Relaxed) == 0 || !DEFERRING.contains(this as usize) {
        return;
    }
    unmark_deferring(this);
    let (items, records, open) = with(this, |s| (std::mem::take(&mut s.deferred), s.batch_buf.as_ref().map(|b| b.res.clone()), s.batch_open));
    let draws: Vec<&DeferredDraw> = items.iter().filter_map(|i| if let RunItem::Draw(d) = i { Some(d) } else { None }).collect();
    if draws.is_empty() {
        // (pipeline sets only: as they would have gone)
        for i in &items {
            if let RunItem::Bind { pso: p, .. } = i {
                super::real::bind_pipeline(this, *p as *mut c_void);
            }
        }
        return;
    }
    let (Some(list), Some(records)) = (ID3D12GraphicsCommandList::from_raw_borrowed(&this), records) else { return };
    RUN_ENDS[run_end_index(why)].fetch_add(1, Ordering::Relaxed);
    let steps: Vec<Step> = items.iter().map(RunItem::step).collect();
    let cmds = batch_commands(&steps, records.as_raw() as usize, open);
    emit_commands(this, list, &cmds, &draws);
    with(this, |s| (s.batch_used, s.batch_open) = (0, true));
    for d in &draws {
        mesh_work(this, super::targets::RAN);
        let (what, size) = match d.snap.work {
            DeferredWork::Mesh { .. } => (super::watch::MESH, [d.size[0].saturating_mul(d.size[1]).saturating_mul(d.size[2]), 0, 0]),
            DeferredWork::Indirect { .. } => (super::watch::INDIRECT_MESH, [1, 0, 0]),
        };
        super::watch::note(d.raw_pso, what, size);
        count(DRAWS);
    }
    static RUNS: AtomicU64 = AtomicU64::new(0);
    static DRAWN: AtomicU64 = AtomicU64::new(0);
    let n = RUNS.fetch_add(1, Ordering::Relaxed) + 1;
    let drawn = DRAWN.fetch_add(draws.len() as u64, Ordering::Relaxed) + draws.len() as u64;
    if n <= 8 || n.is_power_of_two() {
        let binds = items.len() - draws.len();
        crate::xlog_info!("D3D12.X: deferred two-pass draws: run #{n} of {} with {binds} pipeline sets ({drawn} in all, {:.1} a run); runs ended by: {}", draws.len(), drawn as f64 / n as f64, run_ends_text(8));
    }
}

/// A region of `region` bytes in the batch record buffer of `this` for the next deferred draw
/// (the run so far is recorded first when it does not fit): its GPU address.
unsafe fn batch_region(this: *mut c_void, region: u64) -> Result<u64, String> {
    let bytes = batch_records_bytes();
    if with(this, |s| s.batch_buf.is_none()) {
        let dev = super::real::list_device(this).ok_or("no device for two-pass buffer")?;
        let buf = match take_free(dev.as_raw() as usize, bytes, bytes) {
            Some(b) => b,
            // (a new one only while the two-pass buffers alive leave half the budget: past it the
            // draws go at once, and the records they need alone are still made)
            None if TWO_PASS[TP_ALIVE].load(Ordering::Relaxed).saturating_add(bytes) > two_pass_cap() / 2 => return Err("the two-pass buffers alive are at half their budget".into()),
            None => TwoPassBuf::make(&dev, bytes, "two-pass batch records")?,
        };
        with(this, |s| (s.batch_buf, s.batch_used, s.batch_open) = (Some(buf), 0, false));
    }
    let (used, cap) = with(this, |s| (s.batch_used, s.batch_buf.as_ref().map_or(0, |b| b.bytes)));
    let at = match batch_place(used, region, cap) {
        Some(at) => at,
        None => {
            flush_two_pass(this, RunEnd::Full);
            0
        }
    };
    if at + region > cap {
        return Err("its records are more than the batch record buffer".into());
    }
    let va = with(this, |s| {
        s.batch_used = at + region;
        s.batch_buf.as_ref().map(|b| b.res.GetGPUVirtualAddress())
    });
    va.map(|va| va + at).ok_or_else(|| "no batch record buffer".into())
}

/// `draw` deferred on `this`; the run is recorded when it has `batch_most` draws.
unsafe fn push_deferred(this: *mut c_void, draw: DeferredDraw) {
    let n = with(this, |s| {
        s.deferred.push(RunItem::Draw(draw));
        s.deferred.iter().filter(|i| matches!(i, RunItem::Draw(_))).count()
    });
    mark_deferring(this);
    if n >= batch_most() {
        flush_two_pass(this, RunEnd::Most);
    }
}

/// What SetPipelineState does on a list: with no deferred draws, what it always did; with them,
/// a two-pass pipeline goes into the run (`RunItem::Bind`) and any other records the run first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OnPipeline {
    Bind,
    Queue,
    FlushThenBind,
}

fn on_pipeline(deferring: bool, two_pass: bool) -> OnPipeline {
    match (deferring, two_pass) {
        (false, _) => OnPipeline::Bind,
        (true, true) => OnPipeline::Queue,
        (true, false) => OnPipeline::FlushThenBind,
    }
}

/// A two-pass DispatchMesh of `dims` groups with `tp` (`pso`: its mesh pass's pipeline) deferred
/// on `this` (`XCL_TWO_PASS_BATCH=2`). False: not deferred – the caller records the run so far
/// and this draw at once, as without.
unsafe fn defer_dispatch_mesh(this: *mut c_void, tp: &Arc<super::real::TwoPassPso>, pso: usize, raw_pso: usize, dims: [u32; 3]) -> bool {
    let groups = dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]);
    let record_bytes = tp.record.record_dwords() as u64 * 4;
    let region = batch_region_bytes((groups as u64).saturating_mul(record_bytes));
    let ty = super::list_x::tracked_list_type(this);
    if groups == 0 || defer_check(true, ty, super::capture::on(), 1, region, batch_records_bytes()).is_err() {
        return false;
    }
    // (as many groups as the draw alone would have: its region holds them all)
    let n = groups.min(record_groups(region, record_bytes, tp.mesh_groups_per_record));
    if n == 0 {
        return false;
    }
    let Ok(records) = batch_region(this, region) else { return false };
    super::list_x::ensure_view_heap(this);
    let (Ok(compute), Ok(mesh)) = (prepare_two_pass_stage(this, pso, tp, records, true), prepare_two_pass_stage(this, pso, tp, records, false)) else { return false };
    let work = DeferredWork::Mesh { groups: n, mesh_groups: n * tp.mesh_groups_per_record };
    push_deferred(this, DeferredDraw { snap: DrawSnap { compute, mesh, work }, _tp: tp.clone(), _mesh_pso: held_pso(pso), _block: None, raw_pso, kind: tp.kind, size: dims });
    true
}

/// A two-pass ExecuteIndirect of one command deferred (`defer_dispatch_mesh`): `sig` with the
/// DispatchMesh signature `msig`, its command at UMA address `src`, its count at `count`.
#[allow(clippy::too_many_arguments)]
unsafe fn defer_execute_indirect(this: *mut c_void, tp: &Arc<super::real::TwoPassPso>, pso: usize, raw_pso: usize, sig: usize, msig: DispatchMeshSig, max: u32, src: usize, count: Option<usize>) -> bool {
    // PORT-CHECK: the plain signatures (ours) carried over to HEAD's deferred ExecuteIndirect
    // (local) the plain signatures, as the one recorded at once takes them; with the title's under
    // Wine it is recorded at once, where its barriers get the aliasing one (`emit_two_pass_barrier`)
    let plain = plain_two_pass_signatures(sig);
    if plain.is_none() && crate::util::under_wine() {
        return false;
    }
    let record_bytes = tp.record.record_dwords() as u64 * 4;
    let region = batch_region_bytes(indirect_record_need(record_bytes, tp.mesh_groups_per_record));
    let ty = super::list_x::tracked_list_type(this);
    if defer_check(true, ty, super::capture::on(), max, region, batch_records_bytes()).is_err() {
        return false;
    }
    // (the groups a command keeps: as the draw alone, `two_pass_record_buffers_fit_what_their_draws_need`)
    let cap = record_groups(region, record_bytes, tp.mesh_groups_per_record);
    if cap == 0 {
        return false;
    }
    let Ok(records) = batch_region(this, region) else { return false };
    super::list_x::ensure_view_heap(this);
    let (Ok(compute), Ok(mesh)) = (prepare_two_pass_stage(this, pso, tp, records, true), prepare_two_pass_stage(this, pso, tp, records, false)) else { return false };
    let Ok((rewrite, block, at)) = prepare_scaled_args(this, src, count, 1, msig.byte_stride, msig.mesh_offset, tp.mesh_groups_per_record, cap) else { return false };
    // (the block is in INDIRECT_ARGUMENT after the run, `scratch_in_args`)
    let before = scratch_state(with(this, |s| std::mem::replace(&mut s.scratch_in_args, true)));
    // (a plain signature's command is the argument alone, where it is in the command)
    let (companion, sig, arg) = match plain {
        Some((dispatch, mesh)) => (dispatch, mesh, msig.mesh_offset as u64),
        None => (msig.companion_dispatch, sig, 0),
    };
    let work = DeferredWork::Indirect { rewrite, block: block.as_raw() as usize, before, companion, sig, compute_at: at + arg, mesh_at: at + msig.byte_stride as u64 + arg };
    push_deferred(this, DeferredDraw { snap: DrawSnap { compute, mesh, work }, _tp: tp.clone(), _mesh_pso: held_pso(pso), _block: Some(block), raw_pso, kind: tp.kind, size: [max, 0, 0] });
    true
}

/// A reference to the pipeline `pso` for as long as a deferred draw names it.
unsafe fn held_pso(pso: usize) -> Option<ID3D12PipelineState> {
    let p = pso as *mut c_void;
    if !super::real::is_adopted(p) {
        return None;
    }
    ID3D12PipelineState::from_raw_borrowed(&p).cloned()
}

/// Smallest and largest record buffer. The smallest is an argument scratch block's size
/// (`SCRATCH_BLOCK`), so a freed one of either serves both.
const RECORD_MIN: u64 = 1 << 20;
const RECORD_MAX: u64 = 128 << 20;
/// Record bytes an ExecuteIndirect of two-pass commands asks for: a command's groups are known on
/// the GPU only (`SCALE_ARGS_HLSL` keeps them to what fits).
const INDIRECT_RECORDS: u64 = 32 << 20;

/// Bytes of a record buffer for `need` bytes of records: the power of two at or above it, from
/// `RECORD_MIN` to `RECORD_MAX`. A few sizes go from recording to recording through `FREE_RES`; a
/// buffer made to each need would seldom fit the next.
fn record_buffer_bytes(need: u64) -> u64 {
    need.clamp(RECORD_MIN, RECORD_MAX).next_power_of_two()
}

/// Record bytes a two-pass ExecuteIndirect asks for: `INDIRECT_RECORDS`, or less when a command's
/// groups (at most 65 535 / `split` in X, `record_groups`) need less – the same groups fit.
fn indirect_record_need(record_bytes: u64, split: u32) -> u64 {
    indirect_records().min(record_bytes.saturating_mul(65_535 / split.max(1) as u64))
}

/// (local) `XCL_TWO_PASS_RES_MB` (1..=128, read once): `INDIRECT_RECORDS` in MiB (upstream: 32
/// MiB for each recording that has one; about 100 recordings of a retail title held one each, on
/// a card that was full already)
fn indirect_records() -> u64 {
    static RES_MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *RES_MB.get_or_init(|| std::env::var("XCL_TWO_PASS_RES_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|m| (1..=128).contains(m)).map_or(INDIRECT_RECORDS, |m| m << 20))
}

/// The record buffer of this list for `need_bytes` (at most `RECORD_MAX`): buffer, address, bytes.
unsafe fn ensure_res_buffer(this: *mut c_void, need_bytes: u64) -> Result<(ID3D12Resource, u64, u64), String> {
    let need = record_buffer_bytes(need_bytes);
    if let Some((buf, cap)) = with(this, |s| s.res_buf.as_ref().filter(|b| b.bytes >= need).map(|b| (b.res.clone(), b.bytes))) {
        let va = buf.GetGPUVirtualAddress();
        return Ok((buf, va, cap));
    }
    let dev = super::real::list_device(this).ok_or("no device for two-pass buffer")?;
    let dev_key = dev.as_raw() as usize;
    let picked = take_free(dev_key, need, u64::MAX);
    let entry = match picked {
        Some(e) => e,
        None => TwoPassBuf::make(&dev, need, "two-pass res buffer")?,
    };
    let (buf, cap) = (entry.res.clone(), entry.bytes);
    let va = buf.GetGPUVirtualAddress();
    with(this, |s| {
        if let Some(old) = s.res_buf.replace(entry) {
            s.res_retired.push(old);
        }
    });
    Ok((buf, va, cap))
}

/// The `io` block of a two-pass stage: its user SGPRs (`compute_pass`: the MS layout) with the
/// argument table at `table_at`, and the high dword of unified memory's base, as the other binds
/// write it (the translated pair picks each access's memory window from it). After the
/// arguments, the addresses of the view heap and the sampler heap set (`heaps`) when the root
/// signature has its shaders index them (`console_list::with_heaps`): the console's mesh
/// stages take them there as its compute and graphics stages do (s23 / s24 for a mesh stage
/// with three tables, a root view and two constants).
fn two_pass_io(layout: Option<&Layout>, args: &[u8], table_at: u64, compute_pass: bool, uma_base: usize, heaps: [u32; 2]) -> Vec<u32> {
    let mut io = match layout {
        Some(l) if compute_pass => console_list::with_heaps(l, None, console_list::user_data_ms(l, args, table_at), heaps),
        Some(l) => console_list::with_heaps(l, None, console_list::user_data_gfx(l, args, table_at), heaps),
        None if compute_pass => {
            let mut v = vec![0u32; 16];
            v[8] = table_at as u32;
            v[9] = (table_at >> 32) as u32;
            v
        }
        None => vec![table_at as u32, (table_at >> 32) as u32],
    };
    io.resize(crate::gpu::recomp::KERNEL_SGPRS, 0);
    io[crate::gpu::recomp::IO_UMA_BASE_HI] = (uma_base as u64 >> 32) as u32;
    io
}

unsafe fn bind_two_pass_stage(
    this: *mut c_void,
    list: &ID3D12GraphicsCommandList,
    real_mesh_pso: usize,
    tp: &super::real::TwoPassPso,
    res_va: u64,
    compute_pass: bool,
) -> Result<(), String> {
    // the pair is translated with heap windows when they are on (`pso::mem_windows`): as `bind_abi`
    super::list_x::ensure_view_heap(this);
    let stage = prepare_two_pass_stage(this, real_mesh_pso, tp, res_va, compute_pass)?;
    emit_two_pass_stage(list, &stage, compute_pass);
    Ok(())
}

/// What a two-pass stage binds: its root signature, pipeline and root views – the io block made
/// from the title's root arguments as they are when it is prepared, the records, the memory
/// windows (`prepare_two_pass_stage`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StageBind {
    rs: usize,
    pso: usize,
    views: [u64; 6],
}

/// `stage` on `list`: root signature, root views, pipeline.
unsafe fn emit_two_pass_stage(list: &ID3D12GraphicsCommandList, stage: &StageBind, compute_pass: bool) {
    let (rs_raw, pso_raw) = (stage.rs as *mut c_void, stage.pso as *mut c_void);
    let (Some(rs), Some(pc)) = (ID3D12RootSignature::from_raw_borrowed(&rs_raw), ID3D12PipelineState::from_raw_borrowed(&pso_raw)) else { return };
    if compute_pass {
        list.SetComputeRootSignature(rs);
        for (k, &v) in stage.views.iter().enumerate() {
            list.SetComputeRootUnorderedAccessView(k as u32, v);
        }
    } else {
        list.SetGraphicsRootSignature(rs);
        for (k, &v) in stage.views.iter().enumerate() {
            list.SetGraphicsRootUnorderedAccessView(k as u32, v);
        }
    }
    list.SetPipelineState(pc);
}

/// The binding of a two-pass stage with the title's root arguments as they are now, written to
/// unified memory (its io block and argument table): nothing is recorded on the list.
unsafe fn prepare_two_pass_stage(this: *mut c_void, real_mesh_pso: usize, tp: &super::real::TwoPassPso, res_va: u64, compute_pass: bool) -> Result<StageBind, String> {
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let first = first_window(super::real::uma_window_va(0), u.shadowed())?;
    let (layout, args, heaps) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone(), s.heaps));
    want_root_views(layout.as_deref(), &args, super::view_want::Path::MeshStage);
    let copy = |bytes: &[u8]| -> Result<usize, String> {
        let p = with(this, |s| s.arena.take(u, bytes.len().max(16), 256, &mut s.uma)).ok_or("unified memory is full")?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
        Ok(p as usize)
    };
    let inline = layout
        .as_ref()
        .map_or(0, |l| if compute_pass { l.inline_ms } else { l.inline_gfx } as usize)
        .min(args.len());
    let table_at = copy(&args[inline..])?;
    let io = two_pass_io(layout.as_deref(), &args, table_at as u64, compute_pass, u.base(), heaps);
    let io_at = copy(&io.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>())?;
    let va = |p: usize| super::real::uma_va_of(p).unwrap_or(first);
    let rs = if compute_pass { tp.compute_rs } else { tp.mesh_rs };
    if rs == 0 {
        return Err("two-pass root signature".into());
    }
    let pso = if compute_pass { tp.compute } else { real_mesh_pso };
    if pso == 0 {
        return Err(if compute_pass { "two-pass compute pipeline" } else { "two-pass mesh pipeline" }.into());
    }
    let w = |k: usize| super::real::uma_window_va(k).unwrap_or(first);
    Ok(StageBind { rs, pso, views: [va(io_at), res_va, w(0), w(1), w(2), w(3)] })
}

/// (local, `XCL_MESH_TRACE=1`) The one-pass mesh trace: a buffer in system memory (a custom heap,
/// mapped), `TRACE_SLICES` slices of `emit::MESH_TRACE_BYTES`, one for each draw in turn (root
/// parameter 6): (buffer, its mapping, GPU address). The `io` block cannot hold it: the GPU writes
/// the video-memory copy of unified memory, not the CPU's view (run 198: every record empty).
static TRACE_BUF: std::sync::OnceLock<Option<(usize, usize, u64)>> = std::sync::OnceLock::new();
/// (4,096: a recording of many traced draws reuses a slice only after this many; no fence check)
const TRACE_SLICES: usize = 4096;
/// the draws in the trace's slices: (slice, pipeline), the newest last
static ONE_PASS_TRACES: Mutex<std::collections::VecDeque<(usize, usize)>> = Mutex::new(std::collections::VecDeque::new());

unsafe fn trace_buffer(dev: &ID3D12Device) -> Option<(usize, usize, u64)> {
    *TRACE_BUF.get_or_init(|| {
        let bytes = (TRACE_SLICES * crate::gpu::recomp::emit::MESH_TRACE_BYTES) as u64;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
        let mut res: Option<ID3D12Resource> = None;
        if let Err(e) = dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut res) {
            crate::xlog_error!("one-pass trace: no system-memory buffer ({e})");
            return None;
        }
        let res = res?;
        let mut p = std::ptr::null_mut();
        res.Map(0, None, Some(&mut p)).ok()?;
        std::ptr::write_bytes(p as *mut u8, 0, bytes as usize);
        let va = res.GetGPUVirtualAddress();
        crate::xlog_info!("one-pass trace: {} KB of system memory at GPU {va:#x}", bytes >> 10);
        Some((res.into_raw() as usize, p as usize, va))
    })
}

/// The next draw's slice, zeroed and noted with its pipeline: its GPU address.
unsafe fn next_trace_slice(dev: &ID3D12Device, pso: usize) -> Option<u64> {
    let (_, map, va) = trace_buffer(dev)?;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let slice = (NEXT.fetch_add(1, Ordering::Relaxed) as usize) % TRACE_SLICES;
    let bytes = crate::gpu::recomp::emit::MESH_TRACE_BYTES;
    std::ptr::write_bytes((map + slice * bytes) as *mut u8, 0, bytes);
    let mut t = ONE_PASS_TRACES.lock().unwrap();
    if t.len() >= TRACE_SLICES {
        t.pop_front();
    }
    t.push_back((slice, pso));
    Some(va + (slice * bytes) as u64)
}

/// (trace) At a device loss: the records of the waves of the last one-pass draws that did not
/// finish (record word 0 not `EDEDEDED`), into the log.
pub fn dump_one_pass_traces() {
    use crate::gpu::recomp::emit::{MESH_TRACE_BYTES, MESH_TRACE_WORDS};
    let Some(Some((_, map, _))) = TRACE_BUF.get().copied() else {
        crate::xlog_error!("one-pass trace: no buffer");
        return;
    };
    let t = ONE_PASS_TRACES.lock().unwrap().clone();
    let (mut shown, mut done, mut empty) = (0usize, 0usize, 0usize);
    for (k, &(slice, pso)) in t.iter().enumerate().rev() {
        for slot in 0..MESH_TRACE_BYTES / (4 * MESH_TRACE_WORDS) {
            let at = (map + slice * MESH_TRACE_BYTES + slot * 4 * MESH_TRACE_WORDS) as *const u32;
            let w: Vec<u32> = (0..MESH_TRACE_WORDS).map(|i| unsafe { std::ptr::read_volatile(at.add(i)) }).collect();
            if w.iter().all(|&x| x == 0) {
                empty += 1;
            } else if w[0] == 0xEDED_EDED {
                done += 1;
            } else if shown < 64 {
                shown += 1;
                crate::xlog_error!("one-pass trace: draw -{} (pso {pso:#x}, MS {:016x}) slot {slot}: {:08x?}", t.len() - k, super::real::one_pass_ms_hash(pso), w);
            }
        }
    }
    crate::xlog_error!("one-pass trace: {} draws, {shown} unfinished wave records shown, {done} finished, {empty} empty", t.len());
}

/// (local) A one-pass mesh pipeline (`TwoPassPso::one_pass`) bound for a draw: its root
/// signature, the pixel shader's `io` (u0, the graphics user data) and the mesh shader's (u1, the
/// MS user data, where the two passes' records were), the memory windows (u2..), the pipeline.
unsafe fn bind_one_pass_stage(this: *mut c_void, list: &ID3D12GraphicsCommandList, real_mesh_pso: usize, tp: &super::real::TwoPassPso) -> Result<(), String> {
    super::list_x::ensure_view_heap(this);
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let first = first_window(super::real::uma_window_va(0), u.shadowed())?;
    let (layout, args, heaps) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone(), s.heaps));
    want_root_views(layout.as_deref(), &args, super::view_want::Path::MeshStage);
    let copy = |bytes: &[u8]| -> Result<usize, String> {
        let p = with(this, |s| s.arena.take(u, bytes.len().max(16), 256, &mut s.uma)).ok_or("unified memory is full")?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
        Ok(p as usize)
    };
    let trace = crate::gpu::recomp::emit::mesh_trace();
    let io_of = |ms: bool| -> Result<usize, String> {
        let inline = layout.as_ref().map_or(0, |l| if ms { l.inline_ms } else { l.inline_gfx } as usize).min(args.len());
        let table_at = copy(&args[inline..])?;
        let io = two_pass_io(layout.as_deref(), &args, table_at as u64, ms, u.base(), heaps);
        copy(&io.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>())
    };
    let (io_ps, io_ms) = (io_of(false)?, io_of(true)?);
    let va = |p: usize| super::real::uma_va_of(p).unwrap_or(first);
    let rs_raw = tp.mesh_rs as *mut c_void;
    let rs = ID3D12RootSignature::from_raw_borrowed(&rs_raw).ok_or("one-pass root signature")?;
    list.SetGraphicsRootSignature(rs);
    let views = [va(io_ps), va(io_ms)].into_iter().chain((0..4).map(|k| super::real::uma_window_va(k).unwrap_or(first)));
    for (k, v) in views.enumerate() {
        list.SetGraphicsRootUnorderedAccessView(k as u32, v);
    }
    if trace {
        let dev = super::real::list_device(this).ok_or("no device")?;
        let slice = next_trace_slice(&dev, real_mesh_pso).ok_or("no trace buffer")?;
        list.SetGraphicsRootUnorderedAccessView(6, slice);
    }
    let pso_raw = real_mesh_pso as *mut c_void;
    let pc = ID3D12PipelineState::from_raw_borrowed(&pso_raw).ok_or("one-pass mesh pipeline")?;
    list.SetPipelineState(pc);
    Ok(())
}

/// The pass that rewrites a two-pass ExecuteIndirect's arguments, on XCL's ABI: u0 = `count`,
/// `stride`, `offset` (of the DispatchMesh argument), `split`, `cap` (groups with a record), and
/// where the commands are (u2) and go (u3), in bytes. Each command goes out twice, for the compute
/// pass and then (at `count · stride`) for the mesh pass, with at most `cap` groups, all in X:
/// both passes find a group's record by `gid.x` alone. With a count buffer (u0 dword 7 set: the
/// count at dword 8's byte offset in u4), the commands past its count get no groups: they are run
/// one by one (`two_pass_steps`), so without one.
pub(super) const SCALE_ARGS_HLSL: &str = r#"RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer src : register(u2);
RWByteAddressBuffer dst : register(u3);
RWByteAddressBuffer cnt : register(u4);
[numthreads(64, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
    uint count = io.Load(0), stride = io.Load(4), offset = io.Load(8), split = io.Load(12), cap = io.Load(16);
    if (tid >= count) return;
    uint live = io.Load(28) != 0u ? min(count, cnt.Load(io.Load(32))) : count;
    uint s = io.Load(20) + tid * stride, d = io.Load(24) + tid * stride, m = d + count * stride;
    for (uint i = 0u; i < stride; i += 4u) {
        uint w = src.Load(s + i);
        dst.Store(d + i, w);
        dst.Store(m + i, w);
    }
    uint3 g = src.Load3(s + offset);
    uint n = 0u;
    if (tid < live && g.x != 0u && g.y != 0u && g.z != 0u) {
        n = min(g.x, cap);
        n = g.y > cap / n ? cap : n * g.y;
        n = g.z > cap / n ? cap : n * g.z;
    }
    dst.Store3(d + offset, uint3(n, 1u, 1u));
    dst.Store3(m + offset, uint3(n * split, 1u, 1u));
}
"#;

unsafe fn scale_indirect_pipeline(dev: &ID3D12Device) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
    // (local) one pipeline for each thread: under Wine an object's own vtable is put back while
    // it is handed to vkd3d-proton (`UnadoptGuard`, upstream PR #867), and a release of the one
    // pipeline all recording threads shared found it so (`pso_release`, a fault three runs ended in)
    // PORT-CHECK: a deferred draw's `Rewrite` names the pair by address; it lives as long as the thread
    thread_local! {
        static CACHE: std::cell::RefCell<HashMap<usize, (ID3D12RootSignature, ID3D12PipelineState)>> = std::cell::RefCell::new(HashMap::new());
    }
    let key = dev.as_raw() as usize;
    if let Some(pair) = CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return Ok(pair);
    }
    let dxil = crate::gpu::recomp::dxc::compile_cached(SCALE_ARGS_HLSL, "main", "cs_6_6")?.0;
    let rs = crate::gpu::recomp::pso::abi_root_signature(dev, false)?;
    let mut cdesc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const c_void, BytecodeLength: dxil.len() },
        ..Default::default()
    };
    let res_pso: Result<ID3D12PipelineState, _> = dev.CreateComputePipelineState(&cdesc);
    std::mem::ManuallyDrop::drop(&mut cdesc.pRootSignature);
    let pso = res_pso.map_err(|e| format!("scale pso create: {e}"))?;
    CACHE.with(|c| c.borrow_mut().insert(key, (rs.clone(), pso.clone())));
    Ok((rs, pso))
}

/// `bytes` of this recording's scratch blocks (COMMON between uses): the block and the offset.
/// Past `SCRATCH_CAP` a recording gets no more.
unsafe fn scratch_args(this: *mut c_void, dev: &ID3D12Device, bytes: u64) -> Result<(ID3D12Resource, u64), String> {
    let bytes = bytes.next_multiple_of(256);
    if let Some(r) = with(this, |s| {
        let (b, cap) = s.scratch.as_ref()?;
        (s.scratch_used + bytes <= *cap).then(|| {
            s.scratch_used += bytes;
            (b.clone(), s.scratch_used - bytes)
        })
    }) {
        return Ok(r);
    }
    let size = bytes.max(SCRATCH_BLOCK);
    let remaining = SCRATCH_CAP.saturating_sub(with(this, |s| s.scratch_taken));
    if size > remaining {
        return Err(format!("its rewritten arguments are past {} MiB this recording", SCRATCH_CAP >> 20));
    }
    let dev_key = dev.as_raw() as usize;
    let picked = take_free(dev_key, size, remaining);
    let block = match picked {
        Some(e) => e,
        None => TwoPassBuf::make(dev, size, "two-pass argument scratch")?,
    };
    let b = block.res.clone();
    with(this, |s| {
        s.scratch_taken += block.bytes;
        s.scratch_used = bytes;
        s.scratch = Some((block.res.clone(), block.bytes));
        s.scratch_in_args = false;
        s.res_retired.push(block);
    });
    Ok((b, 0))
}

/// The root UAVs of the pass that rewrites a two-pass ExecuteIndirect's arguments
/// (`SCALE_ARGS_HLSL`), as (root slot, GPU address): every view its program declares gets one.
///
/// u4 holds the count and the program reads it only with a count buffer – but the root signature
/// was just set, so a slot left out holds nothing, and a driver may load from it ahead of the
/// test: a load at GPU address 0, which an NVIDIA GPU answers with a fault and a device removal
/// (0x887A0006, the first operation not done in the list: this pass's Dispatch – a retail
/// title's world, three times, each at the first such draw of a list). Without a count buffer
/// u4 names the constants, which are always there.
fn scale_views(consts: u64, src: u64, dst: u64, count: Option<u64>) -> [(u32, u64); 4] {
    [(0, consts), (2, src), (3, dst), (4, count.unwrap_or(consts))]
}

/// The arguments of `max` commands at UMA address `src`, rewritten for the two passes
/// (`SCALE_ARGS_HLSL`) into a scratch block: the block (INDIRECT_ARGUMENT until the caller puts
/// it back to COMMON; with `batch` it stays there until the next rewrite into it,
/// `State::scratch_in_args`) and where the compute pass's commands are; the mesh pass's follow them.
///
/// They are read through the memory window over them, as translated shaders read memory, not
/// through the title's buffer: its state on the PC is the title's and not known here. So is the
/// count (`count`: its UMA address, when the title gave a count buffer).
#[allow(clippy::too_many_arguments)]
unsafe fn make_scaled_args_buffer(this: *mut c_void, list: &ID3D12GraphicsCommandList, src: usize, count: Option<usize>, max: u32, stride: u32, mesh_offset: u32, split: u32, cap: u32, batch: bool) -> Result<(ID3D12Resource, u64), String> {
    let (rewrite, dst, dst_off) = prepare_scaled_args(this, src, count, max, stride, mesh_offset, split, cap)?;
    // the title's writes of the arguments before this (`fresh_indirect_arguments`)
    // ((local) a one-pass ExecuteIndirect or an argument batch leaves the block in
    // INDIRECT_ARGUMENT also without `batch`)
    let before = scratch_state(with(this, |s| s.scratch_in_args));
    if batch {
        emit_uav_and_transition(list, None, &dst, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    } else {
        emit_uav_barrier(list, None);
        emit_transition(list, &dst, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    let (rs_raw, pso_raw) = (rewrite.rs as *mut c_void, rewrite.pso as *mut c_void);
    list.SetComputeRootSignature(ID3D12RootSignature::from_raw_borrowed(&rs_raw));
    list.SetPipelineState(ID3D12PipelineState::from_raw_borrowed(&pso_raw));
    for (slot, va) in rewrite.views {
        list.SetComputeRootUnorderedAccessView(slot, va);
    }
    list.Dispatch(rewrite.groups, 1, 1);
    emit_transition(list, &dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    // (without `batch` the caller puts it back to COMMON)
    with(this, |s| s.scratch_in_args = batch);
    Ok((dst, dst_off))
}

/// The pass that rewrites an ExecuteIndirect's arguments as bound (`SCALE_ARGS_HLSL`): its root
/// signature and pipeline (made once a device, kept), root views and groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rewrite {
    rs: usize,
    pso: usize,
    views: [(u32, u64); 4],
    groups: u32,
}

/// `make_scaled_args_buffer` up to what it records: the rewrite's constants in unified memory,
/// its place in a scratch block, the block and the offset.
#[allow(clippy::too_many_arguments)]
unsafe fn prepare_scaled_args(this: *mut c_void, src: usize, count: Option<usize>, max: u32, stride: u32, mesh_offset: u32, split: u32, cap: u32) -> Result<(Rewrite, ID3D12Resource, u64), String> {
    let dev = super::real::list_device(this).ok_or("no device")?;
    let (rs, pso) = scale_indirect_pipeline(&dev)?;
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let commands = (max as u64).saturating_mul(stride as u64);
    // only the GPU reads them: in a GPU view too, once its tiles are asked for
    want_view(super::view_want::Path::MemoryOp, src as u64, commands);
    super::real::uma_window_buffer_gpu(src, commands as usize).ok_or("the indirect arguments are not in one memory window")?;
    let src_va = super::real::uma_va_of(src).ok_or("unified memory is not open on the GPU")?;
    let count_va = match count {
        Some(p) => {
            want_view(super::view_want::Path::MemoryOp, p as u64, 4);
            super::real::uma_window_buffer_gpu(p, 4).ok_or("the indirect count is not in a memory window")?;
            Some(super::real::uma_va_of(p).ok_or("unified memory is not open on the GPU")?)
        }
        None => None,
    };
    let (dst, dst_off) = scratch_args(this, &dev, commands.saturating_mul(2))?;
    let consts: Vec<u8> = [max, stride, mesh_offset, split, cap, 0, 0, count_va.is_some() as u32, 0].iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = with(this, |s| s.arena.take(u, consts.len(), 256, &mut s.uma)).ok_or("unified memory is full")?;
    std::ptr::copy_nonoverlapping(consts.as_ptr(), p, consts.len());
    let consts_va = super::real::uma_va_of(p as usize).ok_or("unified memory is not open on the GPU")?;
    let views = scale_views(consts_va, src_va, dst.GetGPUVirtualAddress() + dst_off, count_va);
    // (the pair is the device's for good: `scale_indirect_pipeline` keeps it)
    Ok((Rewrite { rs: rs.as_raw() as usize, pso: pso.as_raw() as usize, views, groups: max.div_ceil(64) }, dst, dst_off))
}

/// The state the scratch block of a recording is in before a rewrite: INDIRECT_ARGUMENT when the
/// last two-pass ExecuteIndirect left it there (`two_pass_batch`), else COMMON. A block left in
/// INDIRECT_ARGUMENT when the recording moves on to another, or closes, is a buffer: it decays to
/// COMMON when its ExecuteCommandLists is done, before another recording can take it (`FREE_RES`
/// takes back a recording's blocks at its allocator's reset).
fn scratch_state(in_args: bool) -> D3D12_RESOURCE_STATES {
    if in_args {
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
    } else {
        D3D12_RESOURCE_STATE_COMMON
    }
}

/// (local, after cool_fox by codec_d, https://github.com/DoubleGhosty/cool_fox f4b48ad, `crates/victory-gpu/src/indirect/batch.rs`; GPL-3.0-or-later)
/// One pass rewrites the arguments of the one-pass ExecuteIndirects between two points where they
/// may change (a title barrier, flush or copy: `close_arg_batch`), recorded at the first of them.
/// It reads what to rewrite from a table in unified memory that the CPU goes on filling while the
/// list is recorded (threads past its count do nothing) and writes into a scratch block left
/// INDIRECT_ARGUMENT, so the draws after it take no pass or barrier of their own
/// (`make_scaled_args_buffer`: a pass, a UAV barrier and two transitions each). A job (8 words):
/// source window | count window << 4 | has count << 8, source offset, commands, stride, the
/// DispatchMesh argument's offset, count offset, destination offset, most groups (bit 9 of the
/// first word: two copies, the mesh pass's with `split` (bits 16-31) times the groups). Each command
/// is copied with its groups in X (`SCALE_ARGS_HLSL`); past the count it gets none.
/// `XCL_ARG_BATCH=0`: a pass for each.
pub(super) const BATCH_ARGS_HLSL: &str = r#"#define RS "UAV(u0), UAV(u1), UAV(u2), UAV(u3), UAV(u4), UAV(u5), UAV(u6), UAV(u7), UAV(u8), UAV(u9)"
RWByteAddressBuffer jobs : register(u0);
RWByteAddressBuffer dst : register(u1);
RWByteAddressBuffer w0 : register(u2);
RWByteAddressBuffer w1 : register(u3);
RWByteAddressBuffer w2 : register(u4);
RWByteAddressBuffer w3 : register(u5);
RWByteAddressBuffer w4 : register(u6);
RWByteAddressBuffer w5 : register(u7);
RWByteAddressBuffer w6 : register(u8);
RWByteAddressBuffer w7 : register(u9);
uint ld(uint w, uint a) {
    [branch] switch (w) {
    case 0u: return w0.Load(a);
    case 1u: return w1.Load(a);
    case 2u: return w2.Load(a);
    case 3u: return w3.Load(a);
    case 4u: return w4.Load(a);
    case 5u: return w5.Load(a);
    case 6u: return w6.Load(a);
    default: return w7.Load(a);
    }
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
    if (tid >= jobs.Load(0)) return;
    uint4 a = jobs.Load4(16u + tid * 32u), b = jobs.Load4(32u + tid * 32u);
    uint sw = a.x & 15u, cw = (a.x >> 4) & 15u, two = (a.x >> 9) & 1u, split = a.x >> 16;
    uint src = a.y, count = a.z, stride = a.w, off = b.x, d = b.z, cap = b.w;
    uint live = ((a.x >> 8) & 1u) != 0u ? min(count, ld(cw, b.y)) : count;
    for (uint c = 0u; c < count; c++) {
        uint s = src + c * stride, o = d + c * stride, m = o + count * stride;
        for (uint i = 0u; i < stride; i += 4u) {
            uint w = ld(sw, s + i);
            dst.Store(o + i, w);
            if (two != 0u) dst.Store(m + i, w);
        }
        uint3 g = uint3(ld(sw, s + off), ld(sw, s + off + 4u), ld(sw, s + off + 8u));
        uint n = 0u;
        if (c < live && g.x != 0u && g.y != 0u && g.z != 0u) {
            n = min(g.x, cap);
            n = g.y > cap / n ? cap : n * g.y;
            n = g.z > cap / n ? cap : n * g.z;
        }
        dst.Store3(o + off, uint3(n, 1u, 1u));
        if (two != 0u) dst.Store3(m + off, uint3(n * split, 1u, 1u));
    }
}
"#;
const BATCH_JOBS: u32 = 256;
const BATCH_TABLE: usize = 16 + BATCH_JOBS as usize * 32;
/// bytes of a batch's block of rewritten arguments
const BATCH_DST: u64 = 16 << 10;
/// most commands of one ExecuteIndirect a batch takes (more: a pass of its own)
const BATCH_MAX_COMMANDS: u32 = 64;

fn arg_batch_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_ARG_BATCH").map_or(true, |v| v.trim() != "0"))
}

/// (local) The title may have changed what ExecuteIndirects read: the next one opens a batch.
pub(super) fn close_arg_batch(list: *mut c_void) {
    with(list, |s| s.arg_batch = None);
}

unsafe fn batch_args_pipeline(dev: &ID3D12Device) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
    // (one for each thread, as `scale_indirect_pipeline`)
    thread_local! {
        static CACHE: std::cell::RefCell<HashMap<usize, (ID3D12RootSignature, ID3D12PipelineState)>> = std::cell::RefCell::new(HashMap::new());
    }
    let key = dev.as_raw() as usize;
    if let Some(pair) = CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return Ok(pair);
    }
    let dxil = crate::gpu::recomp::dxc::compile_cached(BATCH_ARGS_HLSL, "main", "cs_6_6")?.0;
    let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).map_err(|e| format!("batch root signature: {e}"))?;
    let mut cdesc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const c_void, BytecodeLength: dxil.len() },
        ..Default::default()
    };
    let res_pso: Result<ID3D12PipelineState, _> = dev.CreateComputePipelineState(&cdesc);
    std::mem::ManuallyDrop::drop(&mut cdesc.pRootSignature);
    let pso = res_pso.map_err(|e| format!("batch pso create: {e}"))?;
    CACHE.with(|c| c.borrow_mut().insert(key, (rs.clone(), pso.clone())));
    Ok((rs, pso))
}

/// Records a batch's pass (`BATCH_ARGS_HLSL`) on `list`, its table empty.
unsafe fn open_arg_batch(this: *mut c_void, list: &ID3D12GraphicsCommandList) -> Result<ArgBatch, String> {
    let dev = super::real::list_device(this).ok_or("no device")?;
    let (rs, pso) = batch_args_pipeline(&dev)?;
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let p = with(this, |s| s.arena.take(u, BATCH_TABLE, 256, &mut s.uma)).ok_or("unified memory is full")?;
    (p as *mut u32).write_volatile(0);
    let table_va = super::real::uma_va_of(p as usize).ok_or("unified memory is not open on the GPU")?;
    let first = (0..8).find_map(|k| super::real::uma_window_va(k).filter(|&v| v != 0)).ok_or("no memory window")?;
    let (dst, base) = scratch_args(this, &dev, BATCH_DST)?;
    // the title's writes of the arguments before this, as `make_scaled_args_buffer`
    let before = scratch_state(with(this, |s| s.scratch_in_args));
    emit_uav_and_transition(list, None, &dst, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list.SetComputeRootSignature(&rs);
    list.SetPipelineState(&pso);
    list.SetComputeRootUnorderedAccessView(0, table_va);
    list.SetComputeRootUnorderedAccessView(1, dst.GetGPUVirtualAddress());
    for k in 0..8u32 {
        list.SetComputeRootUnorderedAccessView(2 + k, super::real::uma_window_va(k as usize).filter(|&v| v != 0).unwrap_or(first));
    }
    list.Dispatch(BATCH_JOBS / 64, 1, 1);
    emit_transition(list, &dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    with(this, |s| s.scratch_in_args = true);
    Ok(ArgBatch { table: p as usize, jobs: 0, dst, base, used: 0 })
}

/// (local) `make_scaled_args_buffer` through the list's open batch: the block and where the
/// rewritten commands are (`split`: two copies, as two passes read them; None: one, for one pass).
/// None: not one a batch takes (it then takes a pass of its own).
#[allow(clippy::too_many_arguments)]
unsafe fn batched_args(this: *mut c_void, list: &ID3D12GraphicsCommandList, src: usize, count: Option<usize>, max: u32, stride: u32, mesh_offset: u32, split: Option<u32>, cap: u32) -> Option<(ID3D12Resource, u64)> {
    if !arg_batch_on() || max == 0 || max > BATCH_MAX_COMMANDS || stride % 4 != 0 || mesh_offset % 4 != 0 || split.is_some_and(|n| n == 0 || n > 0xFFFF) {
        return None;
    }
    let u = crate::gpu::uma::get()?;
    // (the window and the offset in it of `n` bytes at `p`)
    let at = |p: usize, n: usize| -> Option<(u32, u32)> {
        let k = u.window_of(p)?;
        let (base, size) = u.window(k);
        (k < 8 && p - base + n <= size && (p - base) % 4 == 0).then_some(())?;
        Some((k as u32, u32::try_from(p - base).ok()?))
    };
    let commands = max as u64 * stride as u64;
    // (only the GPU reads them: in a GPU view too, once its tiles are asked for – as `prepare_scaled_args`)
    want_view(super::view_want::Path::MemoryOp, src as u64, commands);
    let (sw, so) = at(src, commands as usize)?;
    let (cw, co, has) = match count {
        Some(p) => {
            want_view(super::view_want::Path::MemoryOp, p as u64, 4);
            let (w, o) = at(p, 4)?;
            (w, o, 1u32)
        }
        None => (0, 0, 0),
    };
    let bytes = (commands * if split.is_some() { 2 } else { 1 }).next_multiple_of(16);
    if bytes > BATCH_DST {
        return None;
    }
    if !with(this, |s| s.arg_batch.as_ref().is_some_and(|b| b.jobs < BATCH_JOBS && b.used + bytes <= BATCH_DST)) {
        match open_arg_batch(this, list) {
            Ok(b) => with(this, |s| s.arg_batch = Some(b)),
            Err(e) => {
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 4 {
                    crate::xlog_warn!("D3D12.X: one-pass ExecuteIndirect arguments not batched: {e}");
                }
                return None;
            }
        }
    }
    with(this, |s| {
        let b = s.arg_batch.as_mut()?;
        let at = b.base + b.used;
        let two = split.map_or(0, |n| 1 << 9 | n << 16);
        let job = [sw | cw << 4 | has << 8 | two, so, max, stride, mesh_offset, co, at as u32, cap];
        let t = b.table as *mut u32;
        for (i, w) in job.iter().enumerate() {
            t.add(4 + 8 * b.jobs as usize + i).write_volatile(*w);
        }
        b.jobs += 1;
        // (the count last: the job is whole before the pass can see it)
        t.write_volatile(b.jobs);
        b.used += bytes;
        Some((b.dst.clone(), at))
    })
}

/// What a DispatchMesh or ExecuteIndirect draws with, decided once from `resolved`, what
/// `pso_async::resolve_wait` gave for the bound `raw` pipeline.
enum DrawPipeline {
    /// a pipeline XCL built from console MS / GS code: its two passes, with `resolved` as the
    /// mesh pass's pipeline
    TwoPass(Arc<super::real::TwoPassPso>),
    /// the stand-in of a pipeline being built (or built since the resolve: the draw still has
    /// the stand-in, which was made with the title's root signature, not the two passes')
    StandIn,
    /// anything else: a PC pipeline, a translated one, a failed build's stand-in
    Other,
}

fn draw_pipeline(raw: usize, resolved: usize) -> DrawPipeline {
    if let Some(tp) = super::real::two_pass_entry(resolved) {
        return DrawPipeline::TwoPass(tp);
    }
    if super::pso_async::gave_standin(raw, resolved) {
        return DrawPipeline::StandIn;
    }
    DrawPipeline::Other
}

/// Before a draw with a PC pipeline (also a stand-in or fallback, made with the title's root
/// signature): the title's graphics root signature, which nothing else binds on the PC list – a
/// draw without one is invalid (the debug layer reports it on every GPU; WARP faults on it).
/// False: the list has none (the draw is skipped – it would run under whatever root signature a
/// translated draw left, XCL's ABI or a two-pass one), except on a bundle, which takes its
/// caller's.
unsafe fn bind_title_root_signature(this: *mut c_void, list: &ID3D12GraphicsCommandList) -> bool {
    let rs = with(this, |s| s.rs_gfx) as *mut c_void;
    if let Some(r) = ID3D12RootSignature::from_raw_borrowed(&rs) {
        list.SetGraphicsRootSignature(r);
        return true;
    }
    super::list_x::tracked_list_type(this) == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE)
}

/// A draw skipped by `bind_title_root_signature` (logged the first 4 times).
fn note_no_root_signature() {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 4 {
        crate::xlog_warn!("D3D12.X: a draw with a PC pipeline on a list with no graphics root signature set skipped");
    }
}

pub(super) unsafe extern "system" fn dispatch_mesh(this: *mut c_void, x: u32, y: u32, z: u32) {
    let _c = capture_work(this, || super::capture::WorkKind::DispatchMesh([x, y, z]), true);
    type PcDm = unsafe extern "system" fn(*mut c_void, u32, u32, u32);
    let pc_dm: PcDm = std::mem::transmute(super::real::orig_fn(this, 79));
    let raw_pso = with(this, |s| s.pso);
    let pso = super::pso_async::resolve_wait(raw_pso);
    let tp = match draw_pipeline(raw_pso, pso) {
        DrawPipeline::TwoPass(tp) => Some(tp),
        DrawPipeline::StandIn => {
            mesh_work(this, super::targets::STANDIN);
            count(DRAWS_SKIPPED);
            return;
        }
        DrawPipeline::Other => None,
    };
    // (XCL_TWO_PASS_BATCH=2: deferred, or the deferred ones recorded before this one)
    if let Some(tp) = &tp {
        // (local) `XCL_TWO_PASS=0`
        if two_pass_switch() == 0 {
            mesh_work(this, super::targets::SKIPPED);
            count(DRAWS_SKIPPED);
            return;
        }
        // ((local) a one-pass pipeline has no compute pass to defer)
        if two_pass_defer() && !tp.one_pass && defer_dispatch_mesh(this, tp, pso, raw_pso, [x, y, z]) {
            return;
        }
    }
    flush_two_pass(this, RunEnd::Draw);
    if let Some(tp) = tp {
        let groups = x.saturating_mul(y).saturating_mul(z);
        if groups == 0 {
            return;
        }
        let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
        if tp.one_pass {
            // (local) one pass: the console program is the mesh shader; it reads its group from X
            // alone, as the compute pass did (at most 65 535 groups, D3D12's limit a dimension)
            let n = groups.min(65_535);
            super::gpu_times::before(this, pso, super::gpu_times::Kind::Mesh(tp.kind), [x, y, z], false);
            let t = super::gpu_profile::begin(this, raw_pso, super::gpu_profile::Kind::OnePass, [n, 0, 0]);
            let r = bind_one_pass_stage(this, list, pso, &tp);
            if r.is_ok() {
                pc_dm(this, n, 1, 1);
            }
            super::gpu_profile::end(this, t);
            super::gpu_times::after(this);
            match r {
                Ok(()) => {
                    mesh_work(this, super::targets::RAN);
                    super::watch::note(raw_pso, super::watch::MESH, [groups, 0, 0]);
                    count(DRAWS);
                }
                Err(e) => {
                    mesh_work(this, super::targets::SKIPPED);
                    count(DRAWS_SKIPPED);
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, Ordering::Relaxed) < 8 {
                        crate::xlog_warn!("D3D12.X: one-pass DispatchMesh skipped: {e}");
                    }
                }
            }
            return;
        }
        let record_bytes = tp.record.record_dwords() as u64 * 4;
        let need = (groups as u64).saturating_mul(record_bytes);
        super::gpu_times::before(this, pso, super::gpu_times::Kind::Mesh(tp.kind), [x, y, z], false);
        let run = (|| -> Result<(), String> {
            let (res_buf, res_va, cap, records) = records_for(this, need, two_pass_batch())?;
            let n = groups.min(record_groups(cap, record_bytes, tp.mesh_groups_per_record));
            if n == 0 {
                return Err(format!("a record of {record_bytes} B does not fit the record buffer"));
            }
            if n < groups {
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::xlog_warn!("D3D12.X: two-pass DispatchMesh of {groups} groups: {} past the record buffer dropped", groups - n);
                }
            }
            let plan = two_pass_plan(1, records, with(this, |s| s.record_open));
            // (local) timed as one op: XCL_GPU_PROFILE
            let t = super::gpu_profile::begin(this, raw_pso, super::gpu_profile::Kind::TwoPass, [n, 0, 0]);
            let r = run_two_pass_plan(this, list, &plan, &res_buf, res_va, cap, false, |_, compute, va| {
                bind_two_pass_stage(this, list, pso, &tp, va, compute)?;
                if compute {
                    list.Dispatch(n, 1, 1);
                } else {
                    pc_dm(this, n * tp.mesh_groups_per_record, 1, 1);
                }
                Ok(())
            });
            super::gpu_profile::end(this, t);
            r
        })();
        super::gpu_times::after(this);
        match run {
            Ok(()) => {
                mesh_work(this, super::targets::RAN);
                super::watch::note(raw_pso, super::watch::MESH, [groups, 0, 0]);
                count(DRAWS);
                static LOGGED: AtomicU64 = AtomicU64::new(0);
                let n = LOGGED.fetch_add(1, Ordering::Relaxed) + 1;
                if n <= 8 || n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: two-pass DispatchMesh #{n} ({groups} groups, split {}, kind {})", tp.mesh_groups_per_record, tp.kind);
                }
            }
            Err(e) => {
                mesh_work(this, super::targets::SKIPPED);
                count(DRAWS_SKIPPED);
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::xlog_warn!("D3D12.X: two-pass DispatchMesh skipped: {e}");
                }
            }
        }
        return;
    }
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    if !bind_title_root_signature(this, list) {
        note_no_root_signature();
        mesh_work(this, super::targets::SKIPPED);
        count(DRAWS_SKIPPED);
        return;
    }
    mesh_work(this, super::targets::PC);
    count(DRAWS_PC);
    super::watch::note(raw_pso, super::watch::PC_MESH, [x, y, z]);
    pc_dm(this, x, y, z);
}

/// `XCL_DUMP_TARGETS`: a DispatchMesh (the two passes of a translated mesh pipeline count as the
/// one draw they stand for) with `outcome`
unsafe fn mesh_work(this: *mut c_void, outcome: usize) {
    if super::targets::enabled() {
        super::targets::note_draw(this, outcome);
    }
}

/// #59 ExecuteIndirect(signature, max count, arguments, offset, count buffer, offset): with a
/// translated pipeline, on XCL's ABI with the kept root arguments (signatures that change root
/// arguments themselves cannot be – skipped, logged)
pub(super) unsafe extern "system" fn execute_indirect(this: *mut c_void, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize, count_off: u64) {
    type F = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
    let pc: F = std::mem::transmute(super::real::orig_fn(this, 59));
    let dispatches = super::capture::on() && signature_dispatches(sig);
    let _c = capture_work(this, || super::capture::indirect(sig, dispatches, max, args, args_off, count_buf, count_off), !dispatches);
    let raw_pso = with(this, |s| s.pso);
    // (a dispatching one whose pipeline would be waited for here: at the submit, `list_x::defer`;
    // recorded there through this call, with everything below)
    let unbuilt_compute = signature_dispatches(sig) && super::pso_async::waits_for(raw_pso) && super::real::is_console_compute(raw_pso);
    if unbuilt_compute {
        flush_two_pass(this, RunEnd::Draw);
    }
    if unbuilt_compute && super::list_x::defer(this, super::list_x::Deferred::Indirect(sig, max, args, args_off, count_buf, count_off)) {
        super::capture::outcome(super::capture::Outcome::Deferred);
        return super::pso_async::need_by_submit(raw_pso);
    }
    let pso = super::pso_async::resolve_wait(raw_pso);
    // where the commands are in unified memory (a console buffer's offsets are the title's)
    let commands_at = super::real::console_buffer(args).map(|(p, n)| (p + args_off as usize, n.saturating_sub(args_off as usize)));
    let count_at = super::real::console_buffer(count_buf).filter(|&(_, n)| count_off as usize + 4 <= n).map(|(p, _)| p + count_off as usize);
    if super::targets::enabled() && signature_dispatches(sig) {
        super::targets::note_indirect_compute(raw_pso, commands_at, count_at, count_buf != 0, max, super::indirect_guard::plain_dispatch(sig));
    }
    let (args_off, count_off) = (args_off + super::real::buffer_delta(args), count_off + super::real::buffer_delta(count_buf));
    let tp = match draw_pipeline(raw_pso, pso) {
        DrawPipeline::TwoPass(tp) => Some(tp),
        // A pipeline still being built (or whose build finished after the resolve): its
        // stand-in draws nothing, and the list may hold XCL's ABI root signature from a
        // translated draw before it – the PC call with that pair is invalid (debug layer 953,
        // "the currently set Root Signature doesn't match the currently set Pipeline State
        // Object"; a retail title's world then hung the GPU, 0x887a0006). Skipped, like a draw or
        // dispatch of such a pipeline.
        DrawPipeline::StandIn => {
            ei_work(this, sig, super::targets::STANDIN);
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect with a pipeline that is still being built skipped");
            }
            return;
        }
        DrawPipeline::Other => None,
    };
    if let Some(tp) = tp {
        // Only a noted DispatchMesh signature runs the two passes. Any other (one that changes
        // root arguments, or whose companion Dispatch signature could not be made) would reach
        // the PC call under the title's root signature with the pass-through mesh pipeline on
        // the list, whose shaders read their memory through root views the title's root
        // signature does not have.
        let Some(msig) = dispatch_mesh_sig_of(sig) else {
            ei_work(this, sig, super::targets::SKIPPED);
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect with a two-pass pipeline and a signature that is not a plain DispatchMesh one (root arguments, or no companion Dispatch) skipped");
            }
            return;
        };
        // (local) `XCL_TWO_PASS=0` / `direct`
        if two_pass_switch() < 2 {
            ei_work(this, sig, super::targets::SKIPPED);
            count(DRAWS_SKIPPED);
            return;
        }
        // (XCL_TWO_PASS_BATCH=2: one command whose arguments and count are in unified memory
        // deferred; anything else as without, after the deferred draws before it; (local) a
        // one-pass pipeline has no compute pass to defer)
        if two_pass_defer() && !tp.one_pass && max == 1 && args != 0 {
            let src = commands_at.filter(|&(_, n)| msig.byte_stride as usize <= n).map(|(p, _)| p);
            let count = match (count_buf, count_at) {
                (0, _) => Some(None),
                (_, Some(p)) => Some(Some(p)),
                _ => None,
            };
            if let (Some(src), Some(count)) = (src, count) {
                if defer_execute_indirect(this, &tp, pso, raw_pso, sig, msig, max, src, count) {
                    return;
                }
            }
        }
        flush_two_pass(this, RunEnd::Draw);
        {
            let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
            super::gpu_times::before(this, pso, super::gpu_times::Kind::Mesh(tp.kind), [max, 0, 0], true);
            let run = (|| -> Result<(), String> {
                if max == 0 || args == 0 {
                    return Ok(());
                }
                let commands = max as usize * msig.byte_stride as usize;
                let src = match commands_at {
                    Some((p, n)) if commands <= n => p,
                    _ => return Err("its arguments are not in a console buffer in unified memory".into()),
                };
                if tp.one_pass {
                    // (local) one pass: the arguments rewritten as for two passes (groups in X, at
                    // most 65 535 a command, none past the count; the mesh copies at `count · stride`),
                    // then the draws alone – no compute pass, no record buffer, no barriers between
                    let count = match (count_buf, count_at) {
                        (0, _) => None,
                        (_, Some(p)) => Some(p),
                        _ => return Err("its count is not in a console buffer in unified memory".into()),
                    };
                    // (the batch's commands are in one copy; a pass of its own makes two; the block
                    // stays INDIRECT_ARGUMENT either way: the next rewrite puts it back)
                    let (scaled, first) = match batched_args(this, list, src, count, max, msig.byte_stride, msig.mesh_offset, None, 65_535) {
                        Some(r) => r,
                        None => {
                            let (scaled, at) = make_scaled_args_buffer(this, list, src, count, max, msig.byte_stride, msig.mesh_offset, 1, 65_535, true)?;
                            (scaled, at + commands as u64)
                        }
                    };
                    let t = super::gpu_profile::begin(this, raw_pso, super::gpu_profile::Kind::IndirectOnePass, [max, 0, 0]);
                    let r = (|| -> Result<(), String> {
                        bind_one_pass_stage(this, list, pso, &tp)?;
                        match plain_two_pass_signatures(sig) {
                            Some((_, mesh)) => {
                                for k in 0..max as u64 {
                                    // (trace: each command its own slice – its groups start at 0 again)
                                    if k > 0 && crate::gpu::recomp::emit::mesh_trace() {
                                        let dev = super::real::list_device(this).ok_or("no device")?;
                                        list.SetGraphicsRootUnorderedAccessView(6, next_trace_slice(&dev, pso).ok_or("no trace buffer")?);
                                    }
                                    pc(this, mesh, 1, scaled.as_raw() as usize, first + k * msig.byte_stride as u64 + msig.mesh_offset as u64, 0, 0);
                                }
                            }
                            None => pc(this, sig, max, scaled.as_raw() as usize, first, 0, 0),
                        }
                        Ok(())
                    })();
                    super::gpu_profile::end(this, t);
                    return r;
                }
                let record_bytes = tp.record.record_dwords() as u64 * 4;
                let batch = two_pass_batch();
                let (res_buf, res_va, region, records) = records_for(this, indirect_record_need(record_bytes, tp.mesh_groups_per_record), batch)?;
                let cap = record_groups(region, record_bytes, tp.mesh_groups_per_record);
                if cap == 0 {
                    return Err(format!("a record of {record_bytes} B does not fit the record buffer"));
                }
                let count = match (count_buf, count_at) {
                    (0, _) => None,
                    (_, Some(p)) => Some(p),
                    _ => return Err("its count is not in a console buffer in unified memory".into()),
                };
                // the groups past the record buffer, and the commands past the count, are dropped
                // on the GPU (`SCALE_ARGS_HLSL`)
                // ((local) through the list's open batch of rewrites when it takes them, `batched_args`)
                let (scaled, at, batched) = match batched_args(this, list, src, count, max, msig.byte_stride, msig.mesh_offset, Some(tp.mesh_groups_per_record), cap) {
                    Some((scaled, at)) => (scaled, at, true),
                    None => {
                        let (scaled, at) = make_scaled_args_buffer(this, list, src, count, max, msig.byte_stride, msig.mesh_offset, tp.mesh_groups_per_record, cap, batch)?;
                        (scaled, at, false)
                    }
                };
                let plan = two_pass_plan(max, records, with(this, |s| s.record_open));
                let plain = plain_two_pass_signatures(sig);
                // (local) timed as one op: XCL_GPU_PROFILE
                let t = super::gpu_profile::begin(this, raw_pso, super::gpu_profile::Kind::IndirectTwoPass, [max, 0, 0]);
                let passes = run_two_pass_plan(this, list, &plan, &res_buf, res_va, region, plain.is_none(), |k, compute, va| {
                    bind_two_pass_stage(this, list, pso, &tp, va, compute)?;
                    let (s, first) = if compute { (msig.companion_dispatch, at) } else { (sig, at + commands as u64) };
                    let command = first + k as u64 * msig.byte_stride as u64;
                    match plain {
                        // (local) the command's arguments alone, where they are in the command
                        Some((dispatch, mesh)) => pc(this, if compute { dispatch } else { mesh }, 1, scaled.as_raw() as usize, command + msig.mesh_offset as u64, 0, 0),
                        None => pc(this, s, 1, scaled.as_raw() as usize, command, 0, 0),
                    }
                    Ok(())
                });
                super::gpu_profile::end(this, t);
                // (with `batch` the block stays in INDIRECT_ARGUMENT for the next rewrite; (local) a
                // batch's block too)
                if !batch && !batched {
                    emit_transition(list, &scaled, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
                    // (local) the open batch's block may be this one, now COMMON: the next takes its own
                    close_arg_batch(this);
                }
                passes
            })();
            super::gpu_times::after(this);
            match run {
                Ok(()) => {
                    mesh_work(this, super::targets::RAN);
                    super::watch::note(raw_pso, super::watch::INDIRECT_MESH, [max, 0, 0]);
                    count(DRAWS);
                    static LOGGED: AtomicU64 = AtomicU64::new(0);
                    let n = LOGGED.fetch_add(1, Ordering::Relaxed) + 1;
                    if n <= 8 || n.is_power_of_two() {
                        crate::xlog_info!("D3D12.X: two-pass ExecuteIndirect(DispatchMesh) #{n} (max {max}, split {}, kind {})", tp.mesh_groups_per_record, tp.kind);
                    }
                }
                Err(e) => {
                    mesh_work(this, super::targets::SKIPPED);
                    count(DRAWS_SKIPPED);
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, Ordering::Relaxed) < 8 {
                        crate::xlog_warn!("D3D12.X: two-pass ExecuteIndirect skipped: {e}");
                    }
                }
            }
            return;
        }
    }
    flush_two_pass(this, RunEnd::Draw);
    let (gfx, compute) = (super::real::is_translated_graphics(pso), super::real::is_translated_compute(pso));
    if !gfx && !compute {
        // (a pipeline still being built: skipped above, `DrawPipeline::StandIn`)
        // A finished two-pass MS/GS pipeline is a PC mesh shader pipeline (MS+PS) registered in
        // TWO_PASS rather than TRANSLATED_GRAPHICS. Falling through to raw PC ExecuteIndirect
        // with the title's graphics root signature (e.g. a non-DispatchMesh indirect draw or a
        // root-changing signature) invalidates the command list (Close → E_INVALIDARG).
        // (`DrawPipeline::TwoPass` above takes these; this catches one `draw_pipeline` did not
        // see as two-pass, e.g. a build that finished after the resolve.)
        if let Some(tp) = super::real::two_pass_of(pso) {
            ei_work(this, sig, super::targets::SKIPPED);
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!(
                    "D3D12.X: ExecuteIndirect with two-pass {} pipeline ({pso:#x}) skipped – PC pipeline is a mesh shader",
                    tp.kind
                );
            }
            return;
        }
        // a console compute program without a translation (its build failed): nothing to run,
        // as for Dispatch. The PC call would run it with the list's compute root signature, which
        // can still be XCL's ABI one from a translated dispatch before it (debug layer 953).
        // A native PC compute pipeline is not one of these and still goes to the PC call.
        // (`CONSOLE_COMPUTE` holds the title's object, which for a shared stand-in differs from
        // the driver pipeline `resolve_wait` returned.)
        if signature_dispatches(sig) && (super::real::is_console_compute(raw_pso) || super::real::is_console_compute(pso)) {
            super::targets::note_dispatch();
            count(DISPATCHES_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect dispatch with a pipeline that has no translated console shader ({raw_pso:#x}) skipped");
            }
            return;
        }
        // a PC pipeline: the title's root signature, as for a draw (`graphics_bound`)
        // (a dispatch does not use it: it goes on without one)
        let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
        if !bind_title_root_signature(this, list) && !signature_dispatches(sig) {
            note_no_root_signature();
            ei_work(this, sig, super::targets::SKIPPED);
            count(DRAWS_SKIPPED);
            return;
        }
        ei_work(this, sig, super::targets::PC);
        let t = super::gpu_profile::begin(this, pso, super::gpu_profile::Kind::PcIndirect, [max, 0, 0]);
        fresh_indirect_arguments(this, || pc(this, sig, max, args, args_off, count_buf, count_off));
        super::gpu_profile::end(this, t);
        return;
    }
    if ROOT_SIGNATURES.lock().unwrap().as_ref().is_some_and(|s| s.contains(&sig)) {
        ei_work(this, sig, super::targets::SKIPPED);
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect that changes root arguments with a translated pipeline skipped");
        }
        return;
    }
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    if !gfx && super::real::skip_named_cs(pso, true) {
        return;
    }
    // an indirect draw gets its frame trace line as a direct one (`graphics_bound`), marked
    if gfx && super::present::trace_frame() {
        trace_ui_draw(this, raw_pso, true);
    }
    // (binds a compute root signature and a pipeline: before the dispatch's or draw's own)
    let out_ia = with(this, |s| s.guard_out_ia);
    let checked = super::indirect_guard::check_in(this, pso, sig, max, args, args_off, count_buf, out_ia);
    if checked.is_some() {
        // (the check wrote it: UNORDERED_ACCESS now)
        with(this, |s| s.guard_out_ia = false);
    }
    match bind_abi(this, list, pso, gfx, None, 0) {
        Ok(b) => {
            let whole = if compute {
                match super::real::dispatch_chain::whole_ready(pso) {
                    Ok(super::real::dispatch_chain::Whole::Plain) => None,
                    Ok(super::real::dispatch_chain::Whole::Ready(w)) => Some(w),
                    Ok(super::real::dispatch_chain::Whole::Late) => {
                        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
                        if n <= 4 || n.is_power_of_two() {
                            crate::xlog_warn!("D3D12.X: ExecuteIndirect of a split compute program whose whole form is still being built skipped ({n} so far)");
                        }
                        ei_work(this, sig, super::targets::SKIPPED);
                        count(DISPATCHES_SKIPPED);
                        return;
                    }
                    Err(e) => {
                        crate::xlog_warn!("D3D12.X: ExecuteIndirect whole compute PSO: {e}");
                        ei_work(this, sig, super::targets::SKIPPED);
                        count(DISPATCHES_SKIPPED);
                        return;
                    }
                }
            } else {
                None
            };
            // one dispatch: its group counts (as the guard let them through) folded or clamped on
            // the GPU (`clamp`, which writes a folded count into this dispatch's io), then the
            // binding set again. Only on a direct or compute list XCL keeps state for, as
            // `indirect_guard`: the pass records barriers, which a bundle cannot hold
            let mut clamped = None;
            let barriers = matches!(super::list_x::tracked_list_type(this), Some(t) if t == D3D12_COMMAND_LIST_TYPE_DIRECT || t == D3D12_COMMAND_LIST_TYPE_COMPUTE);
            if compute && max == 1 && signature_dispatches(sig) && barriers {
                let (from, at) = checked.as_ref().map_or((args, args_off), |c| c.arguments());
                clamped = super::clamp::before(list, from, at, b.io + 4 * crate::gpu::recomp::IO_GROUP_LIMIT as u64);
                if clamped.is_some() {
                    rebind(this, list, &b, false);
                }
            }
            if let Some(whole) = &whole {
                // (past the hook, as `bind_abi` binds the pipeline)
                super::list_x::set_compute_pipeline(this, whole.as_raw());
            }
            ei_work(this, sig, super::targets::RAN);
            count(if gfx { DRAWS } else { DISPATCHES });
            let t = super::gpu_profile::begin(this, pso, if gfx { super::gpu_profile::Kind::IndirectDraw } else { super::gpu_profile::Kind::IndirectDispatch }, [max, 0, 0]);
            match (&checked, clamped) {
                (_, Some((from, at))) => {
                    if let Some(c) = &checked {
                        super::gpu_profile::note_counts(&t, c.slot);
                        // (`clamp` read the checked counts there)
                        c.read(this);
                    }
                    fresh_indirect_arguments(this, || pc(this, sig, max, from, at, count_buf, count_off));
                    super::clamp::done(list);
                }
                (Some(c), None) => {
                    super::gpu_profile::note_counts(&t, c.slot);
                    c.run_keeping(this, |buffer, off| pc(this, sig, 1, buffer, off, 0, 0));
                    with(this, |s| s.guard_out_ia = true);
                }
                (None, None) => fresh_indirect_arguments(this, || pc(this, sig, max, args, args_off, count_buf, count_off)),
            }
            super::gpu_profile::end(this, t);
        }
        Err(e) => {
            ei_work(this, sig, super::targets::SKIPPED);
            count(if gfx { DRAWS_SKIPPED } else { DISPATCHES_SKIPPED });
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect with a translated pipeline skipped: {e}");
            }
        }
    }
}

/// `XCL_DUMP_TARGETS`: an ExecuteIndirect with `outcome` (a dispatch signature's: a dispatch)
unsafe fn ei_work(this: *mut c_void, sig: usize, outcome: usize) {
    if !super::targets::enabled() {
        return;
    }
    if signature_dispatches(sig) {
        super::targets::note_dispatch();
    } else {
        super::targets::note_draw(this, outcome);
    }
}

/// XCL's own buffer for `fresh_indirect_arguments` (device, buffer): COMMON between its uses
static ARGUMENT_ORDER: Mutex<Option<(usize, ID3D12Resource)>> = Mutex::new(None);

/// `op` (an ExecuteIndirect on `list`) reading the arguments and count the list wrote before it.
///
/// The console orders a dispatch that writes indirect arguments before the ExecuteIndirect that
/// reads them with a barrier that reaches the PC as a UAV barrier (`list_x::resource_barrier_x`).
/// NVIDIA's driver (RTX 5090) fetches an ExecuteIndirect's arguments without waiting for that
/// barrier: it ran the count that was there before the list in all but a few of 100 tries, also
/// when the dispatch wrote them through another resource over the same memory, as translated
/// shaders do through the memory window buffers (`execute_indirect_reads_the_arguments_written_before_it`).
/// A retail title's ExecuteIndirect after the dispatch that writes its group count then runs
/// whatever count the memory held before. A transition to INDIRECT_ARGUMENT makes the driver
/// wait for the writes – of any buffer, as measured there. The title's buffer is in the title's
/// state (its transitions reach the PC list), so a buffer of XCL's own takes it, COMMON ->
/// INDIRECT_ARGUMENT before and back after. Only on a list XCL made (`list_x::on_create`) and
/// not a bundle, which cannot hold barriers; its type comes from XCL's state, not a call to it.
unsafe fn fresh_indirect_arguments(list: *mut c_void, op: impl FnOnce()) {
    let tracked = super::list_x::tracked_list_type(list).is_some_and(|t| t != D3D12_COMMAND_LIST_TYPE_BUNDLE);
    let buffer = if tracked { ID3D12GraphicsCommandList::from_raw_borrowed(&list).and_then(|l| argument_order_buffer(l)) } else { None };
    let Some(buffer) = buffer else { return op() };
    let b = buffer.as_raw() as usize;
    super::list_x::transition(list, b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    op();
    super::list_x::transition(list, b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
}

/// The device's buffer for `fresh_indirect_arguments` (made on first use)
unsafe fn argument_order_buffer(list: &ID3D12GraphicsCommandList) -> Option<ID3D12Resource> {
    let mut dev: Option<ID3D12Device> = None;
    list.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let mut g = ARGUMENT_ORDER.lock().unwrap();
    if let Some((_, b)) = g.as_ref().filter(|(d, _)| *d == dev.as_raw() as usize) {
        return Some(b.clone());
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut b: Option<ID3D12Resource> = None;
    if let Err(e) = dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut b) {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect: no buffer to order its arguments after the writes before it ({e}) – it may read old ones");
        }
        return None;
    }
    let b = b?;
    *g = Some((dev.as_raw() as usize, b.clone()));
    Some(b)
}

/// A DrawInstanced of the console's rect list (three corners of each rectangle, the fourth
/// implied) as two triangles a rectangle; false: not a rect list, or one this cannot expand
/// (the caller draws the three vertices as a triangle – half of each rectangle).
///
/// The PC has no rect lists. The fourth corner is made here, on the CPU, from the title's
/// vertex data: v3 = v0 + v2 − v1 (seen in a retail title's full-screen passes: (−1,1) (−1,−1)
/// (1,−1) with texture coordinates (0,0) (0,1) (1,1) – the missing corner is (1,1), (1,0)).
/// The vertex layout is not looked at: each 4-byte word that is the same in all three vertices
/// is kept (a packed colour), any other is taken as a float. Vertices a shader makes from the
/// vertex index alone (no vertex buffer) are not expanded.
unsafe fn draw_rects(this: *mut c_void, list: &ID3D12GraphicsCommandList, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) -> bool {
    if !cpu_rect_expansion(with(this, |s| s.pso)) {
        return false;
    }
    let Some(vbs) = super::list_x::rect_list_state(this) else { return false };
    let rects = (vertices / 3) as usize;
    if vbs.is_empty() || rects == 0 || rects > 4096 {
        return false;
    }
    let mut views = Vec::new();
    for &(slot, (va, bytes, stride), _) in &vbs {
        let (s, start) = (stride as usize, va as usize + first_vertex as usize * stride as usize);
        let len = rects * 3 * s;
        // (a buffer in a GPU view is read where the CPU has it, its CPU view: the range of the
        // vertices, which must be one piece there)
        let at = if in_views(va) {
            let Some(c) = super::view_want::cpu_run(start as u64, len as u64) else { return false };
            c as usize
        } else {
            start
        };
        if va == 0 || s == 0 || s % 4 != 0 || first_vertex as usize * s + len > bytes as usize || !crate::gpu::shader::readable(at, len) {
            return false;
        }
        let p = allocate_upload_memory(this, rects * 6 * s, 16);
        if p.is_null() {
            return false;
        }
        let word = |v: usize, w: usize| (at as *const u8).add(v * s + w * 4).cast::<u32>().read_unaligned();
        for r in 0..rects {
            let out = p.add(r * 6 * s);
            let src = (at as *const u8).add(r * 3 * s);
            // v0 v1 v2, then v0 v2 v3
            std::ptr::copy_nonoverlapping(src, out, 3 * s);
            std::ptr::copy_nonoverlapping(src, out.add(3 * s), s);
            std::ptr::copy_nonoverlapping(src.add(2 * s), out.add(4 * s), s);
            for w in 0..s / 4 {
                let (a, b, c) = (word(r * 3, w), word(r * 3 + 1, w), word(r * 3 + 2, w));
                let d = if a == b && b == c { a } else { (f32::from_bits(a) + f32::from_bits(c) - f32::from_bits(b)).to_bits() };
                out.add(5 * s + w * 4).cast::<u32>().write_unaligned(d);
            }
        }
        views.push((slot, D3D12_VERTEX_BUFFER_VIEW { BufferLocation: super::list_x::pc_va_of(p as u64), SizeInBytes: (rects * 6 * s) as u32, StrideInBytes: stride }));
    }
    for (slot, v) in &views {
        list.IASetVertexBuffers(*slot, Some(&[*v]));
    }
    list.DrawInstanced(rects as u32 * 6, instances, 0, first_instance);
    for &(slot, _, pc) in &vbs {
        if let Some(v) = pc {
            list.IASetVertexBuffers(slot, Some(&[v]));
        }
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: rect lists are drawn as two triangles a rectangle, the fourth corner made on the CPU");
    }
    true
}

/// Whether a rect list drawn with the bound pipeline `pso` is expanded on the CPU (`draw_rects`):
/// not with the rect list GS (`real::has_rect_list_gs`), which completes each rect itself – both
/// would draw every rect twice (each CPU half is three corners of the rect).
fn cpu_rect_expansion(pso: usize) -> bool {
    !super::real::has_rect_list_gs(super::pso_async::resolve(pso))
}

/// #12 DrawInstanced
pub(super) unsafe extern "system" fn draw_instanced(this: *mut c_void, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
    let _c = capture_work(this, || super::capture::WorkKind::Draw { vertices, instances, first_vertex, first_instance }, true);
    DRAW_ARGS.with(|d| d.set((false, vertices, instances, first_vertex, 0)));
    if let Some(list) = graphics_bound(this) {
        let t = super::gpu_profile::begin(this, with(this, |s| s.pso), super::gpu_profile::Kind::Draw, [vertices, instances, 0]);
        if !draw_rects(this, &list, vertices, instances, first_vertex, first_instance) {
            super::list_x::sync_title_buffers(this, None, first_vertex, vertices, if first_instance != 0 { instances.max(2) } else { instances });
            list.DrawInstanced(vertices, instances, first_vertex, first_instance);
        }
        super::gpu_profile::end(this, t);
    }
}

thread_local! {
    /// the draw being bound: (indexed, count, instances, first index / vertex, base vertex) – for the frame trace
    static DRAW_ARGS: std::cell::Cell<(bool, u32, u32, u32, i32)> = const { std::cell::Cell::new((false, 0, 0, 0, 0)) };
}

/// #13 DrawIndexedInstanced
pub(super) unsafe extern "system" fn draw_indexed_instanced(this: *mut c_void, indices: u32, instances: u32, first_index: u32, base_vertex: i32, first_instance: u32) {
    let _c = capture_work(this, || super::capture::WorkKind::DrawIndexed { indices, instances, first_index, base_vertex, first_instance }, true);
    if let Some(list) = prepare_indexed_draw(this, indices, instances, first_index, base_vertex) {
        super::list_x::sync_title_buffers(this, Some((first_index, indices, base_vertex)), 0, 0, if first_instance != 0 { instances.max(2) } else { instances });
        emit_prepared_indexed_draw(&list, indices, instances, first_index, base_vertex, first_instance);
    }
}

/// Prepare an indexed draw once, before callers make temporary native state changes.
pub(super) unsafe fn prepare_indexed_draw(
    this: *mut c_void,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
) -> Option<ID3D12GraphicsCommandList> {
    DRAW_ARGS.with(|d| d.set((true, indices, instances, first_index, base_vertex)));
    graphics_bound(this)
}

/// Submit an indexed draw whose translated graphics state has already been prepared.
pub(super) unsafe fn emit_prepared_indexed_draw(
    list: &ID3D12GraphicsCommandList,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    let this = list.as_raw();
    let t = super::gpu_profile::begin(this, with(this, |s| s.pso), super::gpu_profile::Kind::Draw, [indices, instances, 0]);
    list.DrawIndexedInstanced(indices, instances, first_index, base_vertex, first_instance);
    super::gpu_profile::end(this, t);
}

/// Before a draw: with a translated pipeline its ABI root signature and the kept graphics
/// arguments are bound (None: the draw is skipped – a console pipeline without a
/// translation, or no unified memory).
unsafe fn graphics_bound(this: *mut c_void) -> Option<ID3D12GraphicsCommandList> {
    let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this)?.clone();
    let pso = with(this, |s| s.pso);
    // a pipeline still being built in the background is waited for (its stand-in draws nothing)
    let real = super::pso_async::resolve_wait(pso);
    if !super::real::is_translated_graphics(real) {
        // A finished two-pass MS/GS pipeline is a PC mesh shader pipeline (MS+PS) registered in
        // TWO_PASS rather than TRANSLATED_GRAPHICS. Binding the title's graphics RS and issuing
        // DrawInstanced / DrawIndexedInstanced against a mesh PSO invalidates the command list
        // (Close → E_INVALIDARG). It draws only through DispatchMesh.
        if let Some(tp) = super::real::two_pass_of(real) {
            super::targets::note_draw(this, super::targets::SKIPPED);
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!(
                    "D3D12.X: draw with two-pass {} pipeline ({real:#x}) skipped – PC pipeline is a mesh shader",
                    tp.kind
                );
            }
            return None;
        }
        if super::targets::enabled() {
            super::targets::note_draw(this, if super::pso_async::is_pending(pso) { super::targets::STANDIN } else { super::targets::PC });
        }
        count(DRAWS_PC);
        if super::present::trace_frame() {
            // which pipelines draw without a translation (first 12 distinct)
            static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
            let mut g = SEEN.lock().unwrap();
            if !g.contains(&pso) && g.len() < 12 {
                g.push(pso);
                crate::xlog_info!(
                    "frame trace: draw with an untranslated pipeline {pso:#x} (resolves to {real:#x}; pending {}, two-pass {}, RT0 {:?})",
                    super::pso_async::is_pending(pso),
                    super::real::two_pass_of(real).is_some(),
                    super::present::rt0_resource(this as usize).map(|r| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0) }))
                );
            }
        }
        // a pending stand-in draws nothing: an occlusion query around it resolves as visible
        if super::pso_async::is_pending(pso) {
            super::list_x::note_dropped_draw(this);
        }
        // a PC pipeline (also a stand-in or fallback): the title's root signature
        if !bind_title_root_signature(this, &list) {
            note_no_root_signature();
            return None;
        }
        return Some(list);
    }
    if super::present::trace_frame() {
        trace_ui_draw(this, pso, false);
    }
    match bind_abi(this, &list, pso, true, None, 0) {
        Ok(_) => {
            super::targets::note_draw(this, super::targets::RAN);
            count(DRAWS);
            Some(list)
        }
        Err(e) => {
            super::targets::note_draw(this, super::targets::SKIPPED);
            super::list_x::note_dropped_draw(this);
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: draw with a translated pipeline skipped: {e}");
            }
            None
        }
    }
}

/// `XCL_TRACE_FRAME`: draws into a presented buffer (the UI) once the title has presented for a
/// while – their pipeline, root arguments and the resources their descriptor tables start with
/// (format, size, kept in step with title memory or not). The first 24.
/// `XCL_TRACE_FRAME`, after 300 presents: one line per distinct (VS, PS, render target) drawn –
/// the shaders' dump hashes, the target and the textures of the first descriptor table.
/// ExecuteIndirect draws (`indirect`) are kinds of their own, marked.
unsafe fn trace_draw_kinds(this: *mut c_void, pso: usize, indirect: bool) {
    // keyed by the target's shape, not the object: a UI library's transient targets (Renoir's
    // layers) are new objects every frame and would fill the list at once
    static SEEN: Mutex<Vec<(u64, u64, Option<(u64, u32, i32, u32)>, bool)>> = Mutex::new(Vec::new());
    if super::present::activity().0 < 300 {
        return;
    }
    let Some((vs, ps)) = super::real::pso_shaders(super::pso_async::resolve(pso)) else { return };
    let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
    let desc = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.SampleDesc.Count) });
    // `XCL_TRACE_FORMATS=26,29` (DXGI format numbers): only draws into targets of those formats –
    // the first 160 kinds of a busy frame are its first passes, and the ones looked for (a
    // lighting target's) never got a line
    let formats = trace_formats();
    // (`XCL_TRACE_DSV_FORMATS`: or draws whose depth buffer, at least 2560 wide, has one of those
    // PC formats – the passes that write or test a depth buffer and draw into no target of the
    // formats above; known while `XCL_DUMP_TARGETS` keeps what a DSV names)
    let depth_wanted = || {
        let dsv = trace_dsv_formats();
        !dsv.is_empty()
            && super::list_x::bound_dsv(this).and_then(super::present::rtv_resource).and_then(desc).is_some_and(|d| d.0 >= 2560 && dsv.contains(&d.2))
    };
    if !formats.is_empty() && !desc(rt).is_some_and(|d| formats.contains(&d.2)) && !depth_wanted() {
        return;
    }
    // `XCL_TRACE_AFTER_S=110`: no kind is listed before that many seconds of the run (a title's
    // menus fill the list long before its world draws), and each kind listed then also gets its
    // root arguments: constants, the first words of root buffers, the first slots of its tables
    // with what the GPU holds of the title textures there (`root_args_summary`)
    static AFTER: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    let after = *AFTER.get_or_init(|| std::env::var("XCL_TRACE_AFTER_S").ok().and_then(|v| v.trim().parse().ok()));
    if after.is_some_and(|s| crate::log::uptime() < s) {
        return;
    }
    {
        let mut g = SEEN.lock().unwrap();
        let key = (vs, ps, desc(rt), indirect);
        if g.len() >= 160 || g.contains(&key) {
            return;
        }
        g.push(key);
    }
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut tex = Vec::new();
    if let (Some(layout), Some(u)) = (layout, crate::gpu::uma::get()) {
        for p in layout.params.iter() {
            if let Param::Table { at } = *p {
                let t = word(at as usize) as usize;
                for k in 0..4usize {
                    if let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) {
                        tex.push(format!("{:?}{}", desc(r), super::title_textures::address(r).map_or(String::new(), |a| format!(" at {a:#x}"))));
                    }
                }
            }
            if tex.len() >= 4 {
                break;
            }
        }
    }
    crate::xlog_info!(
        "frame trace: draws vs {vs:016x} ps {ps:016x} into {:?}{}{} – textures {}; {}",
        desc(rt),
        if super::present::is_presented(rt) { " (presented)" } else { "" },
        if indirect { " (ExecuteIndirect)" } else { "" },
        tex.join(", "),
        super::list_x::state_summary(this)
    );
    if after.is_some() {
        crate::xlog_info!("frame trace: … its root arguments: {}", root_args_summary(this, 8));
        // and, once, what the view heaps hold at that point of the run
        static CENSUS: std::sync::Once = std::sync::Once::new();
        CENSUS.call_once(super::console_heap::census);
    }
}

/// `XCL_TRACE_FORMATS=26,29` (DXGI format numbers): the render target formats the frame trace
/// keeps to – the kinds of draws it lists, and the pipelines whose state it logs. Empty: all.
pub(super) fn trace_formats() -> &'static [i32] {
    static FORMATS: std::sync::OnceLock<Vec<i32>> = std::sync::OnceLock::new();
    FORMATS.get_or_init(|| std::env::var("XCL_TRACE_FORMATS").unwrap_or_default().split(',').filter_map(|f| f.trim().parse().ok()).collect())
}

/// `XCL_TRACE_DSV_FORMATS=53,55` (DXGI format numbers): depth buffer formats the frame trace
/// also keeps to – draws whose bound depth buffer has one of them as its PC format (53: a D16
/// buffer), and pipelines whose DSV format is one of them (55). Empty: none.
pub(super) fn trace_dsv_formats() -> &'static [i32] {
    static FORMATS: std::sync::OnceLock<Vec<i32>> = std::sync::OnceLock::new();
    FORMATS.get_or_init(|| std::env::var("XCL_TRACE_DSV_FORMATS").unwrap_or_default().split(',').filter_map(|f| f.trim().parse().ok()).collect())
}

/// `XCL_TRACE_SHADERS=hash,…` (dump hashes, hex): the draws traced with their root arguments
/// are those whose vertex or pixel shader is listed – into any target, at any time (the first
/// 24); without it, draws into a presented target (or any – `XCL_TRACE_UI_ANY`) after 300
/// presents.
fn traced_shaders() -> &'static [u64] {
    static S: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    S.get_or_init(|| std::env::var("XCL_TRACE_SHADERS").unwrap_or_default().split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).collect())
}

/// `XCL_TRACE_FRAME` with a watched target (`present::watch_rt`): draws whose descriptor
/// tables name it – that object, or another at its title address (the first 16). An
/// ExecuteIndirect draw (`indirect`) has no arguments XCL reads: no line of them.
unsafe fn trace_reads_of_watched(this: *mut c_void, pso: usize, indirect: bool) {
    let Some(w) = super::present::watched_rt() else { return };
    let w_addr = super::real::title_address(w).or_else(|| super::title_textures::address(w));
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..16usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let addr = super::real::title_address(r).or_else(|| super::title_textures::address(r));
            if super::present::is_watched(r) || (addr.is_some() && addr == w_addr) {
                let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
                // each (source, target, shaders) once
                static SEEN: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());
                {
                    let mut g = SEEN.lock().unwrap();
                    if g.len() >= 32 || g.contains(&(r, rt, pso)) {
                        return;
                    }
                    g.push((r, rt, pso));
                }
                if !super::present::is_presented(rt) {
                    super::present::follow(rt);
                }
                if let Some(dsv) = super::list_x::bound_dsv(this) {
                    super::list_x::watch_dsv(dsv);
                }
                let d = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.Flags.0) });
                crate::xlog_info!(
                    "frame trace: a draw (shaders {:x?}, {}) reads a watched target {r:#x} {:?} at {addr:x?} (slot {k}), drawing into {rt:#x} {:?}{} – {}",
                    super::real::pso_shaders(super::pso_async::resolve(pso)),
                    super::list_x::state_summary(this),
                    d(r),
                    d(rt),
                    if super::present::is_presented(rt) { " (presented)" } else { "" },
                    root_args_summary(this, 32)
                );
                if indirect {
                    return;
                }
                let args = DRAW_ARGS.with(|d| d.get());
                crate::xlog_info!(
                    "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
                    super::real::pso_layout(super::pso_async::resolve(pso)).unwrap_or_default(),
                    super::list_x::geometry_summary(this, args)
                );
                return;
            }
        }
    }
}

/// `XCL_TRACE_FRAME`: render targets a draw into the watched target samples (a UI library's
/// layers) are dumped with it (`present::follow`).
unsafe fn follow_layer_sources(this: *mut c_void) {
    let (Some(w), Some(rt)) = (super::present::watched_rt(), super::present::rt0_resource(this as usize)) else { return };
    if rt != w {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..4usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let layer = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).is_some_and(|x| {
                let d = x.GetDesc();
                d.Flags.0 & 1 != 0 && d.Width >= 64 && d.Height >= 32 && d.SampleDesc.Count == 1
            });
            if layer && r != w {
                super::present::follow(r);
            }
        }
    }
}

/// `indirect`: an ExecuteIndirect draw, whose arguments are in GPU memory (`DRAW_ARGS` holds the
/// last direct draw's).
unsafe fn trace_ui_draw(this: *mut c_void, pso: usize, indirect: bool) {
    trace_draw_kinds(this, pso, indirect);
    trace_reads_of_watched(this, pso, indirect);
    follow_layer_sources(this);
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    // `XCL_TRACE_MANY=n`: n draws instead of 24 (a whole frame of a UI)
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let cap = *CAP.get_or_init(|| std::env::var("XCL_TRACE_MANY").ok().and_then(|v| v.parse().ok()).unwrap_or(24));
    if N.load(std::sync::atomic::Ordering::Relaxed) >= cap {
        return;
    }
    let Some(rt) = super::present::rt0_resource(this as usize) else { return };
    let wanted = traced_shaders();
    if wanted.is_empty() {
        // draws into a presented target, or – `XCL_TRACE_UI_ANY` – into any target
        if super::present::activity().0 < 300 || (!super::present::is_presented(rt) && std::env::var_os("XCL_TRACE_UI_ANY").is_none()) {
            return;
        }
    } else if !super::real::pso_shaders(super::pso_async::resolve(pso)).is_some_and(|(v, p)| wanted.contains(&v) || wanted.contains(&p)) {
        return;
    } else {
        super::present::watch_rt(rt);
    }
    N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let sh = super::real::pso_shaders(super::pso_async::resolve(pso)).map(|(v, p)| format!("vs {v:016x} ps {p:016x}"));
    crate::xlog_info!("frame trace: UI draw (pipeline {pso:#x}, {}{}): {}", sh.unwrap_or_default(), if indirect { ", ExecuteIndirect" } else { "" }, root_args_summary(this, 16));
    if !wanted.is_empty() && !indirect {
        // the picked shaders' geometry too (vertices dumped at the next present)
        let args = DRAW_ARGS.with(|d| d.get());
        crate::xlog_info!(
            "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
            super::real::pso_layout(super::pso_async::resolve(pso)).unwrap_or_default(),
            super::list_x::geometry_summary(this, args)
        );
    }
}

/// A draw's root arguments for the frame trace: constants, root buffers (their first words, as
/// the CPU sees them), the first slots of descriptor tables (the PC resources they name).
unsafe fn root_args_summary(this: *mut c_void, buffer_words: usize) -> String {
    let Some(u) = crate::gpu::uma::get() else { return String::new() };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return String::new() };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    // `XCL_TRACE_SLOTS=n`: the first n slots of each table instead of 4, with all eight words
    // of their image descriptors (a view's levels and its smallest level of detail are in the
    // later ones)
    static SLOTS: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let more = *SLOTS.get_or_init(|| std::env::var("XCL_TRACE_SLOTS").ok().and_then(|v| v.trim().parse().ok()));
    let (slots, words) = (more.unwrap_or(4).clamp(1, 64), if more.is_some() { 8 } else { 4 });
    let mut what = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => what.push(format!("p{i} constants {:x?}", (0..count.min(8) as usize).map(|k| word(at as usize + 4 * k)).collect::<Vec<_>>())),
            Param::Descriptor { at } => {
                let va = (word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32) as usize;
                let w: Vec<u32> = if va != 0 && crate::gpu::shader::readable(va, 4 * buffer_words) { (0..buffer_words).map(|k| ((va + 4 * k) as *const u32).read_unaligned()).collect() } else { Vec::new() };
                what.push(format!("p{i} buffer {va:#x} {w:x?}"));
            }
            Param::Table { at } => {
                let t = word(at as usize) as usize;
                let slots: Vec<String> = (0..slots)
                    .map(|k| {
                        let at = u.base() + t + 32 * k;
                        let w = super::console_heap::words_at((t + 32 * k) as u32);
                        // with `XCL_TRACE_SLOTS`: the image the T# describes (format, width,
                        // height, first and last level of the view, levels of the texture, smallest
                        // level of detail), the slot's own index in the shared heap, and the
                        // resource the T#'s index names there – a T# the title copied as memory
                        // carries its source slot's index, and samples whatever that slot holds now
                        let image = w.filter(|_| more.is_some()).and_then(|w| crate::gpu::desc::image(&w)).map_or(String::new(), |t| {
                            let own = super::console_heap::slot(at as u64).map(|s| s.index);
                            let named = super::console_heap::named_by_index(w.map_or(0, |w| w[0]));
                            let nd = (named != 0 && named != super::console_heap::GONE).then(|| ID3D12Resource::from_raw_borrowed(&(named as *mut c_void)).map(|x| x.GetDesc())).flatten();
                            if nd.is_some() && super::title_textures::address(named).is_some() {
                                super::title_textures::probe_gpu_content(named);
                            }
                            format!(
                                " image {:?}, own index {own:?}, its index names {named:#x} {:?} title {:?}",
                                (t.format, t.width, t.height, t.base_level, t.last_level, t.max_mip, t.min_lod),
                                nd.map(|d| (d.Width, d.Height, d.Format.0, d.MipLevels)),
                                super::title_textures::address(named).map(|a| format!("{a:#x}"))
                            )
                        });
                        match super::console_heap::resource_at(at) {
                            Some(r) => {
                                if super::title_textures::address(r).is_some() {
                                    super::title_textures::probe_gpu_content(r);
                                }
                                let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
                                format!("[{k}] {r:#x} {:?} title {:?} T# {:x?}{image}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), super::title_textures::address(r).map(|a| format!("{a:#x}")), w.map(|w| w[..words].to_vec()))
                            }
                            None => format!("[{k}] - T# {:x?}{image}", w.map(|w| w[..words].to_vec())),
                        }
                    })
                    .collect();
                what.push(format!("p{i} table {t:#x}: {}", slots.join(", ")));
            }
        }
    }
    what.join("; ")
}

/// #33 SetComputeRootSignature (PC #29)
pub(super) unsafe extern "system" fn set_compute_root_signature(this: *mut c_void, rs: *mut c_void) {
    let _c = super::capture::op(this, || Some(super::capture::Op::RootSig { gfx: false, rs: rs as usize }));
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        s.args = vec![0u8; layout.as_ref().map_or(0, |l| l.size as usize)];
        s.layout = layout;
    });
}

/// #35 SetComputeRootDescriptorTable (PC #31): the table's console address, 32 bits
pub(super) unsafe extern "system" fn set_compute_root_table(this: *mut c_void, i: u32, handle: u64) {
    put(this, i, 0, &(handle as u32).to_le_bytes());
}

/// #37 SetComputeRoot32BitConstant (PC #33)
pub(super) unsafe extern "system" fn set_compute_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    put(this, i, offset, &value.to_le_bytes());
}

/// #39 SetComputeRoot32BitConstants (PC #35)
pub(super) unsafe extern "system" fn set_compute_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    if !data.is_null() {
        let bytes: Vec<u8> = std::slice::from_raw_parts(data, n as usize).iter().flat_map(|w| w.to_le_bytes()).collect();
        put(this, i, offset, &bytes);
    }
}

/// #41 / #43 / #45 SetComputeRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #37 /
/// #39 / #41): the 64-bit GPU address
pub(super) unsafe extern "system" fn set_compute_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    note_address(va);
    put(this, i, 0, &va.to_le_bytes());
}

/// #25 SetPipelineState: noted, then bound as on the PC (`XCL_TWO_PASS_BATCH=2`: with deferred
/// draws on the list, a two-pass pipeline is bound in the run, between the mesh passes where it
/// came, `on_pipeline`; another records the run first)
pub(super) unsafe extern "system" fn set_pipeline_state(this: *mut c_void, pso: *mut c_void) {
    let deferring = DEFERRING_LISTS.load(Ordering::Relaxed) != 0 && DEFERRING.contains(this as usize);
    match on_pipeline(deferring, deferring && super::real::two_pass_of(pso as usize).is_some()) {
        OnPipeline::Queue => {
            let held = held_pso(pso as usize);
            with(this, |s| {
                s.pso = pso as usize;
                s.deferred.push(RunItem::Bind { pso: pso as usize, _held: held });
            });
            return;
        }
        OnPipeline::FlushThenBind => flush_two_pass(this, RunEnd::Pipeline),
        OnPipeline::Bind => {}
    }
    let _c = super::capture::op(this, || Some(super::capture::Op::Pipeline(pso as usize)));
    with(this, |s| s.pso = pso as usize);
    super::real::bind_pipeline(this, pso);
}

/// #14 Dispatch: with a translated pipeline, on XCL's ABI with the kept root arguments
pub(super) unsafe extern "system" fn dispatch(this: *mut c_void, x: u32, y: u32, z: u32) {
    dispatch_flags(this, x, y, z, 0);
}

/// `DispatchX` flag: x, y, z are thread counts, not group counts; the console launches whole
/// groups and leaves the lanes past the count out of EXEC (0x8000, wave32, needs nothing here)
const DISPATCH_THREAD_DIMENSIONS: u32 = 0x20;

/// What a dispatch of `dims` with `flags` launches on pipeline `pso`: its groups and the thread
/// limits for `IO_THREAD_LIMIT` (0: that axis ends on a whole group). None: it cannot be run
/// as the console would (thread dimensions with a group shape not known).
fn dispatch_groups(pso: usize, dims: [u32; 3], flags: u32) -> Option<([u32; 3], [u32; 3])> {
    if flags & DISPATCH_THREAD_DIMENSIONS == 0 {
        return Some((dims, [0; 3]));
    }
    let threads = super::real::pso_threads(pso)?.map(|t| t.max(1));
    let groups = std::array::from_fn(|k| dims[k].div_ceil(threads[k]));
    let limits = std::array::from_fn(|k| if dims[k] % threads[k] != 0 { dims[k] } else { 0 });
    Some((groups, limits))
}

/// What the frame capture records of a dispatch of `dims` with `flags` on `pso` (the pipeline
/// the title bound): its groups, or a `DispatchX` with thread dimensions' thread counts with the
/// groups they launch, when the group shape is known now (a pipeline still being built has none
/// yet; looked up with nothing sent to the workers, `pso_async::peek`).
fn captured_dispatch(pso: usize, dims: [u32; 3], flags: u32) -> super::capture::WorkKind {
    if flags & DISPATCH_THREAD_DIMENSIONS == 0 {
        return super::capture::WorkKind::Dispatch(dims);
    }
    let groups = if super::pso_async::has_real(pso) { dispatch_groups(super::pso_async::peek(pso), dims, flags).map(|(g, _)| g) } else { None };
    super::capture::WorkKind::DispatchThreads { threads: dims, flags, groups }
}

/// `DispatchX` with thread dimensions launched: how many, and at the first and every power of
/// two the last one's thread counts, its groups and the group shape, logged (whether a title
/// uses the flag, and with what).
fn note_thread_dimensions(threads: [u32; 3], groups: [u32; 3], shape: [u32; 3]) {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        let [t, g, a] = [threads, groups, shape].map(|v| format!("{}x{}x{}", v[0], v[1], v[2]));
        crate::xlog_info!("D3D12.X: DispatchX with thread dimensions: {n} so far, last {t} threads as {g} groups of {a}");
    }
}

/// `Dispatch` / `DispatchX(x, y, z, flags)`.
pub(super) unsafe fn dispatch_flags(this: *mut c_void, x: u32, y: u32, z: u32, flags: u32) {
    let pso = with(this, |s| s.pso);
    let _c = capture_work(this, || captured_dispatch(pso, [x, y, z], flags), false);
    // (recorded at the submit through this call, with everything below: counted there)
    if super::pso_async::waits_for(pso) && super::list_x::defer(this, super::list_x::Deferred::Dispatch(x, y, z, flags)) {
        super::capture::outcome(super::capture::Outcome::Deferred);
        return super::pso_async::need_by_submit(pso);
    }
    super::targets::note_dispatch();
    let bound = pso;
    let pso = super::pso_async::resolve_wait(pso);
    // what it launches: its groups (a `DispatchX` with thread dimensions: whole groups of the
    // pipeline's console group shape), and the thread limits; None: thread dimensions on a
    // pipeline whose group shape is not known, nothing launched (noted with no groups)
    let launch = dispatch_groups(pso, [x, y, z], flags);
    super::targets::note_compute(bound, super::targets::Dispatched::Direct(launch.map_or([0; 3], |(g, _)| g)));
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    if !super::real::is_translated_compute(pso) {
        // a console pipeline without a translation (including a background build that
        // failed): nothing to run (its root arguments were not given to the PC list either)
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: Dispatch with a pipeline that has no translated console shader ({pso:#x}) skipped");
        }
        count(DISPATCHES_SKIPPED);
        return;
    }
    let Some((groups, limits)) = launch else {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: DispatchX of {x}x{y}x{z} threads (flags {flags:#x}) with a pipeline whose group shape is not known ({pso:#x}) skipped");
        }
        count(DISPATCHES_SKIPPED);
        return;
    };
    if flags != 0 {
        // each kind of flags once: what titles pass, and what it launches
        static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        let first = { let mut g = SEEN.lock().unwrap(); !g.contains(&flags) && { g.push(flags); true } };
        if first {
            crate::xlog_info!("D3D12.X: DispatchX with flags {flags:#x}: {x}x{y}x{z} launched as {}x{}x{} groups (thread limits {limits:?})", groups[0], groups[1], groups[2]);
        }
    }
    if flags & DISPATCH_THREAD_DIMENSIONS != 0 {
        note_thread_dimensions([x, y, z], groups, super::real::pso_threads(pso).unwrap_or_default());
    }
    with(this, |s| s.limits = limits);
    let done = translated_dispatch(this, list, pso, groups[0], groups[1], groups[2]);
    with(this, |s| s.limits = [0; 3]);
    match done {
        Ok(()) => count(DISPATCHES),
        Err(e) => {
            count(DISPATCHES_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: Dispatch with a translated pipeline skipped: {e}");
            }
        }
    }
}

/// `XCL_TRACE_FRAME`: a full-screen dispatch (enough groups to cover a 4K target in tiles of up
/// to 128 pixels) – what its root arguments say. A bindless title passes the indices of its
/// descriptors in root constants and constant buffers: every small value there is looked up
/// in the dispatch's descriptor tables, and the resources found (the pass's inputs and
/// outputs) are logged.
unsafe fn trace_fullscreen(this: *mut c_void, pso: usize, groups: [u32; 3]) {
    if groups[0] < 30 || groups[1] < 17 {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout.clone(), s.args.clone()));
    let Some(layout) = layout else { return };
    static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    {
        let mut g = SEEN.lock().unwrap();
        if g.contains(&pso) || g.len() >= 12 {
            return;
        }
        g.push(pso);
    }
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let mut tables: Vec<u32> = Vec::new();
    let mut values: Vec<u32> = Vec::new();
    let mut what: Vec<String> = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => {
                let w: Vec<u32> = (0..count as usize).filter_map(|k| word(at as usize + 4 * k)).collect();
                what.push(format!("p{i} constants {w:x?}"));
                values.extend(&w);
            }
            Param::Descriptor { at } => {
                let lo = word(at as usize).unwrap_or(0) as u64;
                let hi = word(at as usize + 4).unwrap_or(0) as u64;
                let va = (lo | hi << 32) as usize;
                let n = 64usize;
                if va != 0 && crate::gpu::shader::readable(va, 4 * n) {
                    let w: Vec<u32> = (0..n).map(|k| ((va + 4 * k) as *const u32).read_unaligned()).collect();
                    what.push(format!("p{i} buffer {va:#x}: {w:x?}"));
                    values.extend(&w);
                } else {
                    what.push(format!("p{i} buffer {va:#x} (not readable)"));
                }
            }
            Param::Table { at } => {
                let v = word(at as usize).unwrap_or(0);
                what.push(format!("p{i} table {v:#x}"));
                tables.push(v);
            }
        }
    }
    let mut found: Vec<(u32, usize)> = Vec::new();
    for &v in &values {
        if v >= 1 << 20 {
            continue;
        }
        for &t in &tables {
            let slot = u.base() + t as usize + 32 * v as usize;
            if let Some(r) = super::console_heap::resource_at(slot) {
                if !found.iter().any(|x| x.1 == r) {
                    found.push((v, r));
                }
            }
        }
    }
    let res: Vec<String> = found
        .iter()
        .map(|&(v, r)| {
            let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
            format!("[{v}] {r:#x} {:?}{}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), if super::present::is_presented(r) { " PRESENTED" } else { "" })
        })
        .collect();
    crate::xlog_info!("frame trace: full-screen dispatch {groups:?}, pipeline {pso:#x}: {}", what.join("; "));
    crate::xlog_info!("frame trace: … its indices name: {}", if res.is_empty() { "nothing".to_string() } else { res.join("; ") });
}

/// #148 AllocateUploadMemory(size, alignment) → CPU/GPU pointer in unified memory (used by
/// `XboxBvhBuilder.dll` and other console middleware on default title command lists).
/// Freed automatically when the list's command allocator is reset.
pub(super) unsafe extern "system" fn allocate_upload_memory(this: *mut c_void, size: usize, align: usize) -> *mut u8 {
    let Some(u) = crate::gpu::uma::get() else {
        return std::ptr::null_mut();
    };
    let size = size.max(16);
    let align = align.max(16);
    if size <= crate::gpu::uma::ARENA_BLOCK / 2 {
        if let Some(p) = with(this, |s| s.arena.take(u, size, align, &mut s.uma)) {
            return p;
        }
    }
    if u.has_sparse() {
        if let Some((p, _)) = super::sparse::alloc(size, align) {
            with(this, |s| s.uma.push(p));
            return p as *mut u8;
        }
    }
    match u.alloc(size, align) {
        Some(p) => {
            with(this, |s| s.uma.push(p as usize));
            p
        }
        None => std::ptr::null_mut(),
    }
}

/// Diagnostic: what the committed window looks like when a draw's arguments find no room in it
/// (the first four times, then every power of two).
fn note_arena_full(u: &crate::gpu::uma::Uma, asked: usize) {
    static N: AtomicU64 = AtomicU64::new(0);
    let k = N.fetch_add(1, Ordering::Relaxed) + 1;
    if k > 4 && !k.is_power_of_two() {
        return;
    }
    let (free, largest, used, pool) = u.committed_state();
    let (allocators, blocks) = console_list::pending_blocks();
    crate::xlog_warn!(
        "D3D12.X: no unified memory for draw data ({k} so far): {asked} bytes asked; coherent window {} MB in use, {} MB free (largest run {} KB), {pool} of {} pool blocks left; {blocks} blocks held for {allocators} allocators",
        used >> 20,
        free >> 20,
        largest >> 10,
        crate::gpu::uma::POOL / crate::gpu::uma::ARENA_BLOCK
    );
}

unsafe fn translated_dispatch(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, x: u32, y: u32, z: u32) -> Result<(), String> {
    if super::present::trace_frame() {
        trace_fullscreen(this, pso, [x, y, z]);
    }
    if super::real::skip_named_cs(pso, false) {
        return Ok(());
    }
    // more groups in X than D3D12 takes: folded (`recomp::IO_GROUP_LIMIT`). The loop budget is
    // the console's groups' share: those are the groups that run
    let (limit, folded) = fold_counts(x, y, z);
    // (the PC takes 65 535 groups a dimension at most, the debug layer's id 1296: what it does
    // with more is not defined. One `fold` cannot fold is clamped: said here, with the program,
    // in place of `fold`'s line)
    if limit == 0 && folded != [x, y, z] {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 8 {
            let cs = super::real::pso_shaders(pso).map_or(0, |(cs, _)| cs);
            if left_out([x, y, z], folded) {
                crate::xlog_warn!("D3D12.X: Dispatch of {x} x {y} x {z} groups with cs {cs:016x}: more groups than a dispatch may have (XCL_INDIRECT_MAX_GROUPS) – left out");
            } else {
                crate::xlog_warn!("D3D12.X: Dispatch of {x} x {y} x {z} groups with cs {cs:016x}: past the PC's 65 535 groups a dimension and not foldable – clamped to 65535 a dimension");
            }
        }
    }
    bind_abi(this, list, pso, false, Some([x, y, z]), limit)?;
    // (one span for a split program's whole chain of dispatches, cut into slabs from the folded
    // groups; the span names the console's)
    let t = super::gpu_profile::begin(this, pso, super::gpu_profile::Kind::Dispatch, [x, y, z]);
    let ran = super::real::dispatch_chain::dispatch(list, pso, with(this, |s| s.allocator), folded);
    super::gpu_profile::end(this, t);
    ran
}

/// Window 0's GPU address (`window0`: `real::uma_window_va(0)`), which a bind on XCL's ABI
/// reads descriptor tables from and binds for a window without a buffer of its own. Shadowed
/// unified memory whose sparse windows could not be made (no reserved resources either) has
/// none there, and a null root descriptor is not bound.
fn first_window(window0: Option<u64>, shadowed: bool) -> Result<u64, String> {
    let first = window0.ok_or("unified memory is not open on the GPU")?;
    if first == 0 && shadowed {
        return Err("window 0 of unified memory has no buffer on the GPU".into());
    }
    Ok(first)
}

/// Whether dispatches pass `recomp::dispatch_loop_budget` (`XCL_DISPATCH_LOOP_BUDGET=0`, read
/// once: they leave `IO_LOOP_BUDGET` at 0 and every loop has the lane's 65 536 passes in both
/// translators, as before, to compare). Here, not in `recomp`: that module's source is part of
/// every translation's cache identity, and the switch changes no translated code.
pub(super) fn dispatch_loop_budget_wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_DISPATCH_LOOP_BUDGET").map_or(true, |v| v.trim() != "0"))
}

/// A dispatch of `x` × `y` × `z` groups as D3D12 takes it: (the count for
/// `recomp::IO_GROUP_LIMIT` – 0: not folded –, the groups to dispatch). More than 65 535 in X
/// with 1 in Y is folded into Y; any other count past 65 535 is clamped (logged once).
pub(super) fn fold(x: u32, y: u32, z: u32) -> (u32, [u32; 3]) {
    let (limit, groups) = fold_counts(x, y, z);
    if limit == 0 && groups != [x, y, z] {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            if left_out([x, y, z], groups) {
                crate::xlog_warn!("D3D12.X: a dispatch of {x}x{y}x{z} groups is left out: more groups than a dispatch may have (XCL_INDIRECT_MAX_GROUPS)");
            } else {
                crate::xlog_warn!("D3D12.X: a dispatch of {x}x{y}x{z} groups cannot be folded – clamped to 65535 a dimension");
            }
        }
    }
    (limit, groups)
}

/// Whether `fold_counts` left a dispatch of `asked` groups out (it gave no groups for counts
/// that had some).
fn left_out(asked: [u32; 3], groups: [u32; 3]) -> bool {
    groups == [0, 0, 0] && asked.iter().all(|&n| n != 0)
}

/// `fold` without its log line (for a caller that says it with the program)
fn fold_counts(x: u32, y: u32, z: u32) -> (u32, [u32; 3]) {
    fold_counts_within(x, y, z, super::indirect_guard::max_groups())
}

/// `fold_counts` with `most`, the groups in all a dispatch that needs folding or clamping may
/// still run (`indirect_guard`'s bound). Counts D3D12 takes pass as they are, whatever their
/// product. Past them, more groups than that are counts no title asks for – where the counts
/// should be lies something else: 1 065 353 216, the bits of 1.0f, is 16 257 rows of 65 535
/// groups once folded, and 70 000 x 70 000 x 16 clamped is still 68 thousand million – and
/// nothing runs: such a dispatch kept an NVIDIA GPU busy until the driver reset the device.
fn fold_counts_within(x: u32, y: u32, z: u32, most: u32) -> (u32, [u32; 3]) {
    use crate::gpu::recomp::FOLD_X;
    if x <= FOLD_X && y <= 65_535 && z <= 65_535 {
        return (0, [x, y, z]);
    }
    let (limit, groups) =
        if y == 1 && z <= 65_535 && x.div_ceil(FOLD_X) <= 65_535 { (x, [FOLD_X, x.div_ceil(FOLD_X), z]) } else { (0, [x.min(65_535), y.min(65_535), z.min(65_535)]) };
    // (a folded dispatch has up to one row of groups more than the console's count; the bound in
    // 32 bits, as `clamp`'s pass takes it)
    if groups.iter().map(|&n| n as u64).product::<u64>() > most.saturating_add(FOLD_X) as u64 {
        return (0, [0, 0, 0]);
    }
    (limit, groups)
}

/// What `bind_abi` set on a list, for setting it again after other work in between (`rebind`)
/// and the GPU address of its `io`.
pub(super) struct Bound {
    rs: ID3D12RootSignature,
    views: [u64; 6],
    pso: ID3D12PipelineState,
    pub io: u64,
}

/// The binding `bind_abi` made, set on `list` (`this`) again.
pub(super) unsafe fn rebind(this: *mut c_void, list: &ID3D12GraphicsCommandList, b: &Bound, gfx: bool) {
    if gfx {
        list.SetGraphicsRootSignature(&b.rs);
        for (k, v) in b.views.iter().enumerate() {
            list.SetGraphicsRootUnorderedAccessView(k as u32, *v);
        }
        list.SetPipelineState(&b.pso);
    } else {
        list.SetComputeRootSignature(&b.rs);
        for (k, v) in b.views.iter().enumerate() {
            list.SetComputeRootUnorderedAccessView(k as u32, *v);
        }
        // (resolved here, and a compute pipeline has none of the state the hook of a title's call keeps)
        super::list_x::set_compute_pipeline(this, b.pso.as_raw());
    }
}

/// XCL's ABI for a translated pipeline: its root signature, the user SGPRs (`io`, the kept
/// arguments the console's way – compute or graphics) and the argument table in unified
/// memory, u1 res, u2… the memory windows; then the pipeline. `groups`: the thread groups of a
/// dispatch whose counts are known here (`IO_LOOP_BUDGET`: the console's, also when folded).
/// `group_limit`: a folded dispatch's group count in X (`recomp::IO_GROUP_LIMIT`), 0 for any other.
unsafe fn bind_abi(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, gfx: bool, groups: Option<[u32; 3]>, group_limit: u32) -> Result<Bound, String> {
    if !with(this, |s| std::mem::replace(&mut s.heap_bound, true)) {
        super::list_x::ensure_view_heap(this);
    }
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let first = first_window(super::real::uma_window_va(0), u.shadowed())?;
    let (layout, args, heaps) = with(this, |s| {
        let (layout, args) = if gfx { (&s.layout_gfx, &s.args_gfx) } else { (&s.layout, &s.args) };
        (layout.clone(), args.clone(), s.heaps)
    });
    want_root_views(layout.as_deref(), &args, super::view_want::Path::RootView);
    // 32-bit console addresses: window 0; pieces of the list's arena (not a 64 KiB grain each)
    let copy = |bytes: &[u8]| -> Result<usize, String> {
        let p = with(this, |s| s.arena.take(u, bytes.len().max(16), 256, &mut s.uma)).ok_or_else(|| {
            note_arena_full(u, bytes.len());
            "unified memory is full"
        })?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
        Ok(p as usize)
    };
    // a pipeline still being built in the background is waited for
    let real = super::pso_async::resolve_wait(pso);
    // a compute shader takes no more arguments from registers than it has (`inline_for`)
    let user_sgprs = if gfx { None } else { super::real::pso_user_sgprs(real) };
    let inline = layout.as_ref().map_or(0, |l| if gfx { l.inline_gfx } else { console_list::inline_for(l, user_sgprs) } as usize).min(args.len());
    let table_at = copy(&args[inline..])?;
    let mut io = match layout.as_ref() {
        Some(l) if gfx => console_list::with_heaps(l, None, console_list::user_data_gfx(l, &args, table_at as u64), heaps),
        Some(l) => console_list::with_heaps(l, user_sgprs, console_list::user_data_for(l, user_sgprs, &args, table_at as u64), heaps),
        None => vec![table_at as u32, (table_at >> 32) as u32],
    };
    io.resize(crate::gpu::recomp::KERNEL_SGPRS, 0);
    io[crate::gpu::recomp::IO_UMA_BASE_HI] = (u.base() >> 32) as u32;
    if let Some(groups) = groups.filter(|_| dispatch_loop_budget_wanted()) {
        io[crate::gpu::recomp::IO_LOOP_BUDGET] = crate::gpu::recomp::dispatch_loop_budget(groups);
    }
    io[crate::gpu::recomp::IO_GROUP_LIMIT] = group_limit;
    if !gfx {
        for (k, l) in with(this, |s| std::mem::take(&mut s.limits)).into_iter().enumerate() {
            io[crate::gpu::recomp::IO_THREAD_LIMIT[k]] = l;
        }
    }
    let io_at = copy(&io.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>())?;
    let res_at = copy(&[0u8; 256])?;
    let va = |p: usize| super::real::uma_va_of(p).unwrap_or(first);
    let rs = super::real::pso_root_signature(real).or_else(|| super::real::abi_rs_cached().map(|p| p as usize)).ok_or("no XCL ABI root signature")?;
    let rs = rs as *mut c_void;
    let rs = ID3D12RootSignature::from_raw_borrowed(&rs).ok_or("root signature")?.clone();
    let mut views = [first; 6];
    views[0] = va(io_at);
    views[1] = va(res_at);
    // (every window slot gets a buffer: direct DXIL compute reads all four, `with_mem_raw_loads`)
    for k in 0..4 {
        views[2 + k] = super::real::uma_window_va(k).filter(|&v| v != 0).unwrap_or(first);
    }
    // ((local) `XCL_DXIL_AB`: a compute program's variant on odd frames, `real::ab::pick`)
    let real = if gfx { real } else { super::real::ab::pick(real) } as *mut c_void;
    let pc = ID3D12PipelineState::from_raw_borrowed(&real).ok_or("pipeline")?.clone();
    let b = Bound { rs, views, pso: pc, io: views[0] };
    rebind(this, list, &b, gfx);
    Ok(b)
}

#[cfg(test)]
pub(super) fn last_draw_args() -> (bool, u32, u32, u32, i32) {
    DRAW_ARGS.with(|d| d.get())
}

#[cfg(test)]
pub(super) fn counts_snapshot() -> [u64; 5] {
    std::array::from_fn(|k| COUNTS[k].load(Ordering::Relaxed))
}

#[cfg(test)]
pub(super) fn tracked_pso(list: *mut c_void) -> usize {
    if STATES.contains(list as usize) { with(list, |s| s.pso) } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::super::{device_x, list_x, pso_async, real, title_list};
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Counts D3D12 takes are dispatched as they are; more than 65 535 in X with 1 in Y are
    /// folded into Y with the console's count kept (a retail game's 72 512 × 1 × 1); others past
    /// the limit are clamped.
    #[test]
    fn dispatches_past_the_d3d12_limit_are_folded_or_clamped() {
        assert_eq!(fold(64, 2, 1), (0, [64, 2, 1]));
        assert_eq!(fold(65_535, 1, 1), (0, [65_535, 1, 1]));
        assert_eq!(fold(72_512, 1, 1), (72_512, [65_535, 2, 1]));
        assert_eq!(fold(131_071, 1, 3), (131_071, [65_535, 3, 3]));
        assert_eq!(fold(70_000, 2, 1), (0, [65_535, 2, 1]), "two dimensions in use: clamped");
        // every console group is in the folded grid exactly once, and no more than 65 535 more
        for n in [65_536u32, 72_512, 75_072, 200_000] {
            let (limit, [x, y, _]) = fold(n, 1, 1);
            assert_eq!(limit, n);
            assert!(x * y >= n && x * y - n < x, "{n}: {x}x{y}");
        }
    }

    /// A dispatch that needs folding or clamping runs at most the groups a dispatch may have: the
    /// bits of 1.0f where a count should be are 16 257 rows of 65 535 groups once folded, and
    /// nothing of that runs. Counts D3D12 takes are never looked at.
    #[test]
    fn a_dispatch_past_the_limit_with_more_groups_than_any_title_asks_for_is_left_out() {
        let most = 1 << 22;
        assert_eq!(fold_counts_within(most, 1, 1, most), (most, [65_535, 65, 1]), "the most that is folded");
        assert_eq!(fold_counts_within(most + 65_536, 1, 1, most), (0, [0, 0, 0]));
        assert_eq!(fold_counts_within(0x3F80_0000, 1, 1, most), (0, [0, 0, 0]), "1.0f in X");
        assert_eq!(fold_counts_within(70_000, 70_000, 16, most), (0, [0, 0, 0]), "clamped it is 65 535 x 65 535 x 16");
        assert_eq!(fold_counts_within(u32::MAX, 1, 1, most), (0, [65_535, 1, 1]), "not foldable: one row, as before");
        assert_eq!(fold_counts_within(70_000, 2, 1, most), (0, [65_535, 2, 1]));
        assert_eq!(fold_counts_within(4_096, 4_096, 1, most), (0, [4_096, 4_096, 1]), "within D3D12's limits: as it is, whatever the product");
        assert_eq!(fold_counts_within(60_000, 60_000, 2, most), (0, [60_000, 60_000, 2]));
        assert!(left_out([0x3F80_0000, 1, 1], [0, 0, 0]) && !left_out([0, 5, 1], [0, 0, 0]) && !left_out([70_000, 2, 1], [65_535, 2, 1]));
    }

    /// A pipeline with the rect list GS draws its rect lists whole: draw_rects leaves them alone.
    #[test]
    fn rect_lists_with_the_rect_list_gs_are_not_expanded_on_the_cpu() {
        let (with_gs, without) = (0x7E57_4000usize, 0x7E57_5000usize);
        real::register_rect_list_gs_for_test(with_gs);
        assert!(!cpu_rect_expansion(with_gs));
        assert!(cpu_rect_expansion(without) && cpu_rect_expansion(0));
        real::forget_pso_for_test(with_gs);
        assert!(cpu_rect_expansion(with_gs));
    }

    /// An ExecuteIndirect whose bound pipeline is still being built is skipped (its stand-in
    /// with XCL's ABI root signature on the list is the debug layer's 953 and a hung GPU);
    /// a built or failed one goes to the PC call. The gate `execute_indirect` uses.
    #[test]
    fn execute_indirect_with_a_pipeline_still_building_is_skipped() {
        let (building, built, failed) = (0x7E57_6001usize, 0x7E57_6002usize, 0x7E57_6003usize);
        // lazy submits: no worker starts them, so the states stay under the test's control
        pso_async::submit_lazy(building, false, Box::new(|| 0x1234), 0);
        pso_async::submit_lazy(built, false, Box::new(|| 0x1235), 0);
        pso_async::submit_lazy(failed, false, Box::new(|| 0), 0);
        assert!(pso_async::is_pending(building));
        assert_eq!(pso_async::resolve_wait(built), 0x1235);
        assert_eq!(pso_async::resolve_wait(failed), failed);
        assert!(!pso_async::is_pending(built) && !pso_async::is_pending(failed));
        // never resolved: still pending (the skip case), then forgotten
        assert!(pso_async::is_pending(building));
        pso_async::released(building);
        pso_async::released(built);
        pso_async::released(failed);
        assert!(!pso_async::is_pending(building));
    }

    /// (local) The batch pass of one-pass ExecuteIndirect arguments compiles with the root
    /// signature it is made with (`batch_args_pipeline`: from its DXIL).
    #[test]
    fn batch_args_program_compiles_with_its_root_signature() {
        if crate::gpu::recomp::dxc::location().is_err() {
            crate::gpu::recomp::runner::skip_test("no DXC");
            return;
        }
        let dxil = crate::gpu::recomp::dxc::compile(super::BATCH_ARGS_HLSL, "main", "cs_6_6", &[]).unwrap();
        assert!(dxil.windows(4).any(|w| w == b"RTS0"), "no root signature part");
    }

    /// A dispatch writes an ExecuteIndirect's group count through another buffer over the same
    /// memory (as translated shaders write through the memory windows), a UAV barrier (what a
    /// console barrier becomes), then the ExecuteIndirect: it runs the count just written, not
    /// the one the memory held before the list (`fresh_indirect_arguments`; without it an RTX
    /// 5090 ran the old count in nearly every try).
    #[test]
    fn execute_indirect_reads_the_arguments_written_before_it() {
        use crate::gpu::{fence, recomp::runner::skip_test};
        use std::mem::{transmute_copy, ManuallyDrop};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory2, IDXGIAdapter1, IDXGIFactory6, DXGI_CREATE_FACTORY_FLAGS};

        const TRIES: u32 = 40;
        // mode 0: write (x, 1, 1) at a[0..12]; mode 1: count the groups
        let src = r#"
#define RS "RootConstants(num32BitConstants=2, b0), UAV(u0), UAV(u1)"
cbuffer C : register(b0) { uint mode; uint x; };
RWByteAddressBuffer a : register(u0);
RWByteAddressBuffer c : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint t : SV_GroupIndex) {
    if (t != 0u) return;
    if (mode == 0u) a.Store3(0, uint3(x, 1u, 1u)); else c.InterlockedAdd(0, 1u);
}
"#;
        if crate::gpu::recomp::dxc::location().is_err() {
            skip_test("no DXC");
            return;
        }
        unsafe {
            let setup = || -> Result<(ID3D12Device, String), String> {
                let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| format!("DXGI factory unavailable: {e}"))?;
                let sel = crate::gpu::adapter::Selector::from_env_or("auto");
                let (adapter, name): (IDXGIAdapter1, String) = if sel == crate::gpu::adapter::Selector::Warp {
                    (factory.EnumWarpAdapter().map_err(|e| format!("WARP unavailable: {e}"))?, "WARP (software renderer)".into())
                } else {
                    let (adapter, desc) = crate::gpu::adapter::candidates(&factory, &sel).into_iter().next()
                        .ok_or_else(|| crate::gpu::adapter::no_match(&factory, &sel))?;
                    (adapter, crate::gpu::adapter::name(&desc))
                };
                let dev = crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_11_0).map_err(|e| format!("{name}: D3D12 device unavailable ({e})"))?;
                let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: D3D_SHADER_MODEL_6_6 };
                dev.CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &mut sm as *mut _ as _, std::mem::size_of_val(&sm) as u32)
                    .map_err(|e| format!("{name}: cannot query SM 6.6 ({e})"))?;
                if sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
                    return Err(crate::gpu::agility::format_no_sm66_message(&name, crate::gpu::agility::state()));
                }
                Ok((dev, name))
            };
            let (dev, name) = match setup() {
                Ok(selected) => selected,
                Err(reason) => { skip_test(&reason); return; }
            };
            println!("ExecuteIndirect adapter: {name}");
            let dxil = crate::gpu::recomp::dxc::compile(src, "main", "cs_6_6", &[]).unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).unwrap();
            let cs = D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&rs), CS: cs, ..Default::default() }).unwrap();
            let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() };
            let mut sig: Option<ID3D12CommandSignature> = None;
            dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 12, NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 }, None, &mut sig).unwrap();
            let sig = sig.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let desc = |width| D3D12_RESOURCE_DESC {
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
            // the arguments and the buffer the dispatch writes them through: one memory
            let mut heap: Option<ID3D12Heap> = None;
            let heap_desc = D3D12_HEAP_DESC { SizeInBytes: 1 << 16, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: 0, Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS };
            dev.CreateHeap(&heap_desc, &mut heap).unwrap();
            let heap = heap.unwrap();
            let placed = || {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreatePlacedResource(&heap, 0, &desc(1 << 16), D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                r.unwrap()
            };
            let (args, written) = (placed(), placed());
            // the group count, read on the CPU
            let cpu_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
            let mut groups: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&cpu_heap, D3D12_HEAP_FLAG_NONE, &desc(256), D3D12_RESOURCE_STATE_COMMON, None, &mut groups).unwrap();
            let groups = groups.unwrap();
            let mut count = std::ptr::null_mut();
            groups.Map(0, None, Some(&mut count)).unwrap();
            let count = count as *mut u32;
            let uav_barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }) },
            };
            let run = |list: &ID3D12GraphicsCommandList, close: &dyn Fn()| {
                close();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                let v = done.GetCompletedValue() + 1;
                queue.Signal(&done, v).unwrap();
                fence::wait(&done, v, &dev, None, 10_000).unwrap();
            };
            let (mut old, mut wrong) = (0u32, Vec::new());
            for k in 0..TRIES {
                // each try writes a count the memory does not hold yet
                let new = if k % 2 == 0 { 4096 } else { 0 };
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
                let record = |x: u32| -> ID3D12GraphicsCommandList {
                    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, &pso).unwrap();
                    list.SetComputeRootSignature(&rs);
                    list.SetComputeRootUnorderedAccessView(1, written.GetGPUVirtualAddress());
                    list.SetComputeRootUnorderedAccessView(2, groups.GetGPUVirtualAddress());
                    list.SetComputeRoot32BitConstants(0, 2, [0u32, x].as_ptr() as _, 0);
                    list.Dispatch(1, 1, 1);
                    list
                };
                let before = record(old);
                run(&before, &|| before.Close().unwrap());
                alloc.Reset().unwrap();
                *count = 0;
                let list = record(new);
                list.ResourceBarrier(&[uav_barrier.clone()]);
                list.SetComputeRoot32BitConstants(0, 2, [1u32, 0].as_ptr() as _, 0);
                // from here as XCL records a title's ExecuteIndirect: on the adopted list
                let raw = list.as_raw();
                real::adopt(raw, real::Kind::CommandList);
                list_x::on_create(raw, alloc.as_raw() as usize, 0, ty);
                type Ei = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
                type Close = unsafe extern "system" fn(*mut c_void) -> windows::core::HRESULT;
                let ei: Ei = std::mem::transmute(real::orig_fn(raw, 59));
                fresh_indirect_arguments(raw, || ei(raw, sig.as_raw() as usize, 1, args.as_raw() as usize, 0, 0, 0));
                let close: Close = std::mem::transmute(real::orig_fn(raw, 9));
                run(&list, &|| close(raw).unwrap());
                let ran = std::ptr::read_volatile(count);
                if ran != new {
                    wrong.push((k, ran));
                }
                old = new;
            }
            groups.Unmap(0, None);
            assert!(wrong.is_empty(), "{} of {TRIES} ExecuteIndirects ran another group count than the one written before them (try, groups): {wrong:?}", wrong.len());
        }
    }

    /// Minimal RTS0 v2 root signature blob with one 4-constant parameter.
    fn test_rts0_blob() -> Vec<u8> {
        let words: [u32; 12] = [
            2, 1, 24, 0, 0, 0, // version=2, num_params=1, params_offset=24
            1, 0, 36,          // param 0: type=Constants(1), visibility=0, payload_offset=36
            0, 0, 4,           // shader_reg=0, reg_space=0, num_32bit_values=4
        ];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn reset_and_clear_state_track_initial_pso_and_clear_root_args() {
        let list = 0x7100_0001usize as *mut c_void;
        let rs_compute = 0x7100_0010usize;
        let rs_gfx = 0x7100_0020usize;
        let alloc1 = 0x7100_0100usize;
        let alloc2 = 0x7100_0200usize;
        let pso_init = 0x7100_1000usize;
        let pso_mid = 0x7100_2000usize;
        let pso_cleared = 0x7100_3000usize;
        let pso_reset = 0x7100_4000usize;

        let blob = test_rts0_blob();
        console_list::note_root_signature(rs_compute, &blob);
        console_list::note_root_signature(rs_gfx, &blob);

        // Initial CreateCommandList / reset seeds PSO and allocator with clean root state.
        reset(list, alloc1, pso_init);
        with(list, |s| {
            assert_eq!(s.pso, pso_init);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Populate compute + graphics root signatures, constants, and a different PSO.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0xDEAD_BEEF, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0xCAFE_BABE, 0);
        }
        with(list, |s| {
            s.pso = pso_mid;
            assert!(s.layout.is_some() && !s.args.is_empty());
            assert!(s.layout_gfx.is_some() && !s.args_gfx.is_empty());
        });

        // ClearState(pso_cleared) updates PSO and clears root signatures/args while keeping allocator.
        clear_state(list, pso_cleared);
        with(list, |s| {
            assert_eq!(s.pso, pso_cleared);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // ClearState(null) clears PSO to 0.
        clear_state(list, 0);
        assert_eq!(tracked_pso(list), 0);

        // Re-populate root args, then Reset(alloc2, pso_reset) switches allocator/PSO and clears root args.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0x1234_5678, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0x8765_4321, 0);
        }
        reset(list, alloc2, pso_reset);
        with(list, |s| {
            assert_eq!(s.pso, pso_reset);
            assert_eq!(s.allocator, alloc2);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Reset with null initial PSO clears stale PSO back to 0.
        reset(list, alloc2, 0);
        assert_eq!(tracked_pso(list), 0);

        forget(list);
    }

    #[repr(C)]
    struct FakeAdoptedList {
        vt: *const usize,
        refs: AtomicUsize,
        dispatch_calls: AtomicUsize,
        set_compute_rs_calls: AtomicUsize,
        set_compute_uav_calls: AtomicUsize,
        set_pso_calls: AtomicUsize,
        execute_indirect_calls: AtomicUsize,
    }

    unsafe extern "system" fn fake_add_ref(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
    }

    unsafe extern "system" fn fake_release(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
    }

    unsafe extern "system" fn fake_dispatch(this: *mut c_void, _x: u32, _y: u32, _z: u32) {
        (*(this as *const FakeAdoptedList)).dispatch_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_pso(this: *mut c_void, _pso: *mut c_void) {
        (*(this as *const FakeAdoptedList)).set_pso_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_compute_rs(this: *mut c_void, _rs: *mut c_void) {
        (*(this as *const FakeAdoptedList)).set_compute_rs_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_compute_uav(this: *mut c_void, _slot: u32, _va: u64) {
        (*(this as *const FakeAdoptedList)).set_compute_uav_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_execute_indirect(
        this: *mut c_void,
        _sig: usize,
        _max: u32,
        _args: usize,
        _args_off: u64,
        _count_buf: usize,
        _count_off: u64,
    ) {
        (*(this as *const FakeAdoptedList)).execute_indirect_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// `DispatchX` with thread dimensions (flag 0x20, as the BVH builder's rehydration passes
    /// use it): the console launches whole groups of the pipeline's `numthreads` and leaves
    /// the lanes past the count out; without the flag the counts are groups. With the flag
    /// and a group shape not known, nothing is launched (the counts as groups was the bug).
    #[test]
    fn dispatch_x_with_thread_dimensions_launches_whole_groups() {
        let pso = 0x7300_0001usize;
        real::register_threads_for_test(pso, [64, 1, 1]);
        // 640 threads in groups of 64: 10 groups, the last one whole
        assert_eq!(dispatch_groups(pso, [640, 1, 1], 0x20), Some(([10, 1, 1], [0, 0, 0])));
        // 100 threads: 2 groups, and the lanes past 100 to leave out
        assert_eq!(dispatch_groups(pso, [100, 1, 1], 0x20), Some(([2, 1, 1], [100, 0, 0])));
        // the wave32 bit (0x8000) changes nothing here
        assert_eq!(dispatch_groups(pso, [100, 1, 1], 0x8020), Some(([2, 1, 1], [100, 0, 0])));
        // a 2-D group shape
        let pso2 = 0x7300_0002usize;
        real::register_threads_for_test(pso2, [8, 8, 1]);
        assert_eq!(dispatch_groups(pso2, [20, 16, 3], 0x20), Some(([3, 2, 3], [20, 0, 0])));
        // group counts (Dispatch, DispatchX without the flag) are launched as they are
        assert_eq!(dispatch_groups(pso, [7, 2, 1], 0), Some(([7, 2, 1], [0, 0, 0])));
        assert_eq!(dispatch_groups(pso, [7, 2, 1], 0x8000), Some(([7, 2, 1], [0, 0, 0])));
        // no threads: no groups
        assert_eq!(dispatch_groups(pso, [0, 1, 1], 0x20), Some(([0, 1, 1], [0, 0, 0])));
        // thread dimensions on a pipeline whose group shape is not known: not launched
        assert_eq!(dispatch_groups(0x7300_00FF, [100, 1, 1], 0x20), None);
        assert_eq!(dispatch_groups(0x7300_00FF, [100, 1, 1], 0), Some(([100, 1, 1], [0, 0, 0])));
    }

    /// A `DispatchX` with thread dimensions of more than 65 535 groups once converted is folded
    /// as a dispatch of that many groups (`translated_dispatch`: conversion first, then `fold`):
    /// the console's group count goes to `IO_GROUP_LIMIT`, the thread count to
    /// `IO_THREAD_LIMIT`. Taken as groups, the same counts were 64x the groups and left out.
    #[test]
    fn dispatch_x_with_thread_dimensions_past_the_d3d12_limit_is_folded_in_groups() {
        let most = 1 << 22;
        let pso = 0x7300_0011usize;
        real::register_threads_for_test(pso, [64, 1, 1]);
        // 10 000 001 threads: 156 251 groups of 64, the last with one thread
        let (groups, limits) = dispatch_groups(pso, [10_000_001, 1, 1], 0x20).unwrap();
        assert_eq!((groups, limits), ([156_251, 1, 1], [10_000_001, 0, 0]));
        let (limit, folded) = fold_counts_within(groups[0], groups[1], groups[2], most);
        assert_eq!((limit, folded), (156_251, [65_535, 3, 1]));
        assert!(folded[0] * folded[1] >= groups[0] && folded[0] * folded[1] - groups[0] < folded[0]);
        // the counts as groups: past what a dispatch may run, nothing at all
        assert_eq!(fold_counts_within(10_000_001, 1, 1, most), (0, [0, 0, 0]));
        // 8 x 8 groups, 600 000 x 8 threads: one row of 75 000 groups, foldable (as groups the
        // counts were two dimensions past the limit, clamped)
        let pso2 = 0x7300_0012usize;
        real::register_threads_for_test(pso2, [8, 8, 1]);
        let (groups, limits) = dispatch_groups(pso2, [600_000, 8, 1], 0x8020).unwrap();
        assert_eq!((groups, limits), ([75_000, 1, 1], [0, 0, 0]));
        assert_eq!(fold_counts_within(groups[0], groups[1], groups[2], most), (75_000, [65_535, 2, 1]));
    }

    /// `IO_THREAD_LIMIT` takes three io slots no other value has: in the trap-temp range
    /// (ttmp0..15, SGPRs 108..=123, which titles' programs do not use), apart from each other
    /// and from the GS, UMA base, loop budget and group limit slots.
    #[test]
    fn dispatch_thread_limit_slots_collide_with_no_other_io_slot() {
        use crate::gpu::recomp::{IO_GROUP_LIMIT, IO_GS_PRIMS, IO_GS_VERTICES, IO_LOOP_BUDGET, IO_THREAD_LIMIT, IO_UMA_BASE_HI, KERNEL_SCC_SLOT, KERNEL_SGPRS};
        let mut all = vec![IO_GS_PRIMS, IO_GS_VERTICES, IO_UMA_BASE_HI, IO_LOOP_BUDGET, IO_GROUP_LIMIT];
        all.extend(IO_THREAD_LIMIT);
        let mut unique = all.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), all.len(), "{all:?}");
        for s in all {
            assert!((108..=123).contains(&s) && s < KERNEL_SGPRS && s != KERNEL_SCC_SLOT, "slot {s}");
        }
    }

    /// The frame capture records a `DispatchX` with thread dimensions with its thread counts,
    /// flags and the groups they launch (none known for a pipeline whose group shape is not);
    /// other dispatches with their groups.
    #[test]
    fn a_dispatch_x_with_thread_dimensions_is_captured_with_its_groups() {
        use super::super::capture::WorkKind;
        let pso = 0x7300_0021usize;
        real::register_threads_for_test(pso, [64, 1, 1]);
        assert_eq!(captured_dispatch(pso, [100, 1, 1], 0x20), WorkKind::DispatchThreads { threads: [100, 1, 1], flags: 0x20, groups: Some([2, 1, 1]) });
        assert_eq!(captured_dispatch(0x7300_00FE, [100, 1, 1], 0x8020), WorkKind::DispatchThreads { threads: [100, 1, 1], flags: 0x8020, groups: None });
        assert_eq!(captured_dispatch(pso, [7, 2, 1], 0x8000), WorkKind::Dispatch([7, 2, 1]));
        assert_eq!(captured_dispatch(pso, [7, 2, 1], 0), WorkKind::Dispatch([7, 2, 1]));
    }

    #[test]
    fn async_compute_pso_success_and_failure_in_dispatch_and_execute_indirect() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[29] = fake_set_compute_rs as usize;
        orig_vt[30] = fake_noop_ptr as usize; // SetGraphicsRootSignature
        orig_vt[41] = fake_set_compute_uav as usize;
        orig_vt[59] = fake_execute_indirect as usize;
        // (a draw signature with a PC pipeline needs the title's graphics root signature)
        let title_rs = 0x7200_0200usize;

        // Extended vtable layout expected by `real::orig_fn`: entry -1 holds `orig_vt` pointer,
        // and entries 0..64 hold the COM method pointers used via `ID3D12GraphicsCommandList`.
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeAdoptedList {
            // SAFETY: `ext_vt` has 65 elements so `.add(1)` points to slot 0 with `orig_vt` at `-1`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeAdoptedList as *mut c_void;

        // 1. Background compute build SUCCEEDS: real PSO R is registered in TRANSLATED_COMPUTE,
        // stand-in S is NOT. ExecuteIndirect resolves S -> R and treats it as translated.
        let standin_ok = 0x7200_0001usize;
        let real_ok = 0x7200_0002usize;
        let sig_root_changing = 0x7200_00F0usize;
        note_root_changing_signature(sig_root_changing);

        pso_async::submit(
            standin_ok,
            true,
            Box::new(move || {
                real::register_translated_compute_for_test(real_ok);
                real_ok
            }),
        );
        assert_eq!(pso_async::resolve_wait(standin_ok), real_ok);
        assert!(!real::is_translated_compute(standin_ok));
        assert!(real::is_translated_compute(real_ok));

        reset(list_ptr, 0x7200_0100, standin_ok);
        unsafe {
            // With `sig_root_changing`, a translated compute PSO is caught by the root-changing
            // guard instead of falling through to `pc` (which happened when `R` wasn't registered).
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            0,
            "ExecuteIndirect must recognize resolved async compute PSO R as translated"
        );

        // 2. Background compute build FAILS (returns 0): stand-in S_fail is NOT in
        // TRANSLATED_COMPUTE. Dispatch must skip cleanly without binding XCL's ABI RS/UAVs
        // onto S_fail, and ExecuteIndirect must not enter bind_abi either.
        let standin_fail = 0x7200_0003usize;
        pso_async::submit(standin_fail, true, Box::new(|| 0));
        assert_eq!(pso_async::resolve_wait(standin_fail), standin_fail);
        assert!(!real::is_translated_compute(standin_fail));

        reset(list_ptr, 0x7200_0100, standin_fail);
        unsafe { set_graphics_root_signature(list_ptr, title_rs as *mut c_void) };
        let before = counts_snapshot();
        unsafe {
            dispatch(list_ptr, 1, 1, 1);
        }
        let after = counts_snapshot();
        assert_eq!(after[DISPATCHES], before[DISPATCHES]);
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 0);

        unsafe {
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            1,
            "failed async compute stand-in must not enter translated ExecuteIndirect path"
        );

        // 3. The same failed stand-in (made from a console shader) with a Dispatch signature:
        // nothing to run, as for Dispatch – the PC call would pair the stand-in with the list's
        // compute root signature (XCL's ABI one after a translated dispatch: debug layer 953)
        let sig_dispatch = 0x7200_00F8usize;
        real::register_console_compute_for_test(standin_fail);
        note_signature_dispatches(sig_dispatch, true);
        let before = counts_snapshot();
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        let after = counts_snapshot();
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 1, "a dispatch of an untranslated console compute pipeline must not reach the PC ExecuteIndirect");
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);

        // 3b. The same with a shared stand-in (`standin::shell`): the failed build resolves to the
        // driver pipeline, which was never registered – the title's object is
        let shell_fail = 0x7200_0005usize;
        let driver = 0x7200_0006usize;
        pso_async::submit(shell_fail, true, Box::new(|| 0));
        pso_async::shared(shell_fail, driver);
        assert_eq!(pso_async::resolve_wait(shell_fail), driver);
        real::register_console_compute_for_test(shell_fail);
        reset(list_ptr, 0x7200_0100, shell_fail);
        let before = counts_snapshot();
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        let after = counts_snapshot();
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 1, "a failed shared stand-in's dispatch must not reach the PC ExecuteIndirect");
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);
        let _ = pso_async::released(shell_fail);
        real::forget_pso_for_test(shell_fail);

        // 4. A native PC compute pipeline (not made from a console shader) with the same Dispatch
        // signature still goes to the PC call, as on main
        let native = 0x7200_0004usize;
        reset(list_ptr, 0x7200_0100, native);
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 2, "a native PC compute pipeline must still reach the PC ExecuteIndirect");

        // a signature made later at the same address without a Dispatch argument goes to the PC
        reset(list_ptr, 0x7200_0100, standin_fail);
        note_signature_dispatches(sig_dispatch, false);
        unsafe {
            set_graphics_root_signature(list_ptr, title_rs as *mut c_void);
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 3);

        forget(list_ptr);
        let _ = pso_async::released(standin_ok);
        let _ = pso_async::released(standin_fail);
        real::forget_pso_for_test(real_ok);
        real::forget_pso_for_test(standin_fail);
        assert!(!real::is_console_compute(standin_fail), "a released pipeline's address does not stay a console one");
    }

    /// Two-pass buffers of a recording with no known allocator are kept, not dropped at once (the
    /// GPU may not have run the list), and go once they are 10 s old; past the cap the oldest go.
    #[test]
    fn buffers_with_no_allocator_are_kept_until_old_enough() {
        use std::time::{Duration, Instant};
        let t = Instant::now();
        let mut a: Aging<u32> = Aging::new(100);
        assert_eq!(a.hold(t, vec![(10, 1), (10, 2)]), 0);
        assert_eq!(a.hold(t + Duration::from_secs(4), vec![(10, 3)]), 0);
        assert!(a.take_older(t + Duration::from_secs(9), UNKNOWN_RES_AGE).is_empty(), "the list may still run");
        assert_eq!(a.take_older(t + Duration::from_secs(10), UNKNOWN_RES_AGE), vec![1, 2]);
        assert_eq!(a.take_older(t + Duration::from_secs(14), UNKNOWN_RES_AGE), vec![3]);
        assert_eq!(a.bytes, 0);
        // past the cap: the oldest go, never the one just handed over
        assert_eq!(a.hold(t, vec![(60, 4)]), 0);
        assert_eq!(a.hold(t, vec![(60, 5)]), 1);
        assert_eq!(a.hold(t, vec![(200, 6)]), 1);
        assert_eq!(a.take_older(t + UNKNOWN_RES_AGE, UNKNOWN_RES_AGE), vec![6]);
        // through `hold_res_until_reset` with allocator 0: kept, not dropped
        unsafe {
            use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
            use windows::Win32::Graphics::Dxgi::*;
            // (WARP only: no hardware device)
            let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else { return };
            let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else { return };
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                return;
            }
            let buf = TwoPassBuf::make(&dev.unwrap(), 1 << 16, "test").unwrap();
            let raw = buf.res.as_raw() as usize;
            let kept = |r: usize| UNKNOWN_RES.lock().unwrap().held.iter().any(|h| h.2.res.as_raw() as usize == r);
            hold_res_until_reset(0, vec![buf]);
            assert!(kept(raw), "a buffer of a recording with no allocator is kept");
        }
    }

    /// A draw decides once what its pipeline is, from what `resolve_wait` gave: the stand-in of a
    /// two-pass pipeline whose build finished after that is still the stand-in (skipped), never
    /// the two passes with the stand-in as their mesh pipeline; the built one runs the two passes.
    #[test]
    fn a_draw_with_the_stand_in_of_a_two_pass_pipeline_is_not_recorded_as_its_passes() {
        let (standin, real_mesh, failed, native) = (0x7E57_7001usize, 0x7E57_7002usize, 0x7E57_7003usize, 0x7E57_7004usize);
        real::register_two_pass_for_test(real_mesh, "MS");
        // lazy submits: no worker starts them, so the states stay under the test's control
        pso_async::submit_lazy(standin, false, Box::new(move || real_mesh), 0);
        pso_async::submit_lazy(failed, false, Box::new(|| 0), 0);
        assert!(matches!(draw_pipeline(standin, standin), DrawPipeline::StandIn), "being built");
        // built now: a bind that resolved before has the stand-in (the second resolve that
        // `two_pass_of` makes finds the two passes – what recorded them with the stand-in)
        assert_eq!(pso_async::resolve_wait(standin), real_mesh);
        assert!(real::two_pass_of(standin).is_some());
        assert!(matches!(draw_pipeline(standin, standin), DrawPipeline::StandIn), "resolved before the build finished");
        assert!(matches!(draw_pipeline(standin, real_mesh), DrawPipeline::TwoPass(_)));
        assert_eq!(pso_async::resolve_wait(failed), failed);
        assert!(matches!(draw_pipeline(failed, failed), DrawPipeline::Other), "a failed build's stand-in draws as before");
        assert!(matches!(draw_pipeline(native, native), DrawPipeline::Other));
        // a shared stand-in (`standin::shell`): `resolve_wait` gives the driver pipeline behind it
        let (shell, driver) = (0x7E57_7005usize, 0x7E57_7006usize);
        pso_async::submit_lazy(shell, false, Box::new(|| 0), 0);
        pso_async::shared(shell, driver);
        assert!(matches!(draw_pipeline(shell, driver), DrawPipeline::StandIn), "a shared stand-in being built");
        pso_async::released(shell);
        pso_async::released(standin);
        pso_async::released(failed);
        real::forget_pso_for_test(real_mesh);
        assert!(matches!(draw_pipeline(real_mesh, real_mesh), DrawPipeline::Other));
    }

    static FAKE_DRAWS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "system" fn fake_draw_instanced(_this: *mut c_void, _v: u32, _i: u32, _sv: u32, _si: u32) {
        FAKE_DRAWS.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_noop_ptr(_this: *mut c_void, _p: *mut c_void) {}

    /// A two-pass pipeline never reaches a PC call under the title's root signature: not an
    /// ExecuteIndirect whose signature is not a noted DispatchMesh one, not a Draw. A draw with a
    /// PC pipeline and no graphics root signature on the list is skipped too.
    #[test]
    fn a_two_pass_pipeline_or_a_list_without_a_root_signature_does_not_reach_the_pc_call() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[12] = fake_draw_instanced as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[30] = fake_noop_ptr as usize; // SetGraphicsRootSignature
        orig_vt[59] = fake_execute_indirect as usize;
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);
        let fake = Box::new(FakeAdoptedList {
            // SAFETY: `ext_vt` has 65 elements so `.add(1)` points to slot 0 with `orig_vt` at `-1`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list = &*fake as *const FakeAdoptedList as *mut c_void;
        let (mesh, native, rs, sig) = (0x7E57_8001usize, 0x7E57_8002usize, 0x7E57_8003usize, 0x7E57_80F0usize);
        real::register_two_pass_for_test(mesh, "MS");
        reset(list, 0x7E57_8100, mesh);
        unsafe {
            set_graphics_root_signature(list, rs as *mut c_void);
            let before = counts_snapshot();
            execute_indirect(list, sig, 1, 0, 0, 0, 0);
            assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 0, "a signature that is not a noted DispatchMesh one");
            draw_instanced(list, 3, 1, 0, 0);
            assert_eq!(FAKE_DRAWS.load(Ordering::Relaxed), 0, "a Draw with a mesh pipeline");
            assert!(counts_snapshot()[DRAWS_SKIPPED] >= before[DRAWS_SKIPPED] + 2);
            // a PC pipeline draws, with the title's root signature
            reset(list, 0x7E57_8100, native);
            set_graphics_root_signature(list, rs as *mut c_void);
            draw_instanced(list, 3, 1, 0, 0);
            execute_indirect(list, sig, 1, 0, 0, 0, 0);
            assert_eq!((FAKE_DRAWS.load(Ordering::Relaxed), fake.execute_indirect_calls.load(Ordering::Relaxed)), (1, 1));
            // … and without one (a reset clears it), not: it would run under what the list holds
            reset(list, 0x7E57_8100, native);
            draw_instanced(list, 3, 1, 0, 0);
            execute_indirect(list, sig, 1, 0, 0, 0, 0);
            assert_eq!((FAKE_DRAWS.load(Ordering::Relaxed), fake.execute_indirect_calls.load(Ordering::Relaxed)), (1, 1));
            // a dispatch signature does not need it
            note_signature_dispatches(sig, true);
            execute_indirect(list, sig, 1, 0, 0, 0, 0);
            assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 2);
            note_signature_dispatches(sig, false);
        }
        forget(list);
        real::forget_pso_for_test(mesh);
    }

    #[test]
    fn execute_indirect_skips_failed_shared_console_compute_standin() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[29] = fake_set_compute_rs as usize;
        orig_vt[41] = fake_set_compute_uav as usize;
        orig_vt[59] = fake_execute_indirect as usize;

        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeAdoptedList {
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeAdoptedList as *mut c_void;

        // With shared stand-ins (`standin::shell`), `dev_create_compute_pso` registers the
        // title-facing shell in `CONSOLE_COMPUTE`, while `pso_async::resolve_wait` returns the
        // shared driver pipeline when the background build fails.
        let shell_fail = 0x7210_0001usize;
        let shell_ok = 0x7210_0002usize;
        let shared_driver = 0x7210_0003usize;
        let real_ok = 0x7210_0004usize;
        let sig_dispatch = 0x7210_00F8usize;
        let sig_root_changing = 0x7210_00F0usize;
        note_signature_dispatches(sig_dispatch, true);
        note_root_changing_signature(sig_root_changing);

        pso_async::submit_lazy(shell_fail, true, Box::new(|| 0), 0);
        pso_async::shared(shell_fail, shared_driver);
        real::register_console_compute_for_test(shell_fail);

        pso_async::submit_lazy(
            shell_ok,
            true,
            Box::new(move || {
                real::register_translated_compute_for_test(real_ok);
                real_ok
            }),
            0,
        );
        pso_async::shared(shell_ok, shared_driver);
        real::register_console_compute_for_test(shell_ok);

        assert_eq!(pso_async::resolve_wait(shell_fail), shared_driver);
        assert!(!pso_async::is_pending(shell_fail));
        assert!(real::is_console_compute(shell_fail));
        assert!(!real::is_console_compute(shared_driver));

        // 1. Failed shared console compute stand-in with a Dispatch signature must be skipped,
        // never forwarded to PC ExecuteIndirect with the shared driver pipeline.
        reset(list_ptr, 0x7210_0100, shell_fail);
        let before = counts_snapshot();
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        let after = counts_snapshot();
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            0,
            "failed shared console compute stand-in must not reach PC ExecuteIndirect"
        );
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);

        // 2. Another shell sharing the same driver pipeline whose build succeeds still takes the
        // translated path (caught here by the root-changing guard rather than falling through to PC).
        reset(list_ptr, 0x7210_0100, shell_ok);
        unsafe {
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 0);

        // 3. Releasing `shell_fail` forgets its `CONSOLE_COMPUTE` entry so a native PC compute
        // pipeline later allocated at the same address still reaches PC ExecuteIndirect.
        let _ = pso_async::released(shell_fail);
        real::forget_pso_for_test(shell_fail);
        assert!(!real::is_console_compute(shell_fail));
        reset(list_ptr, 0x7210_0100, shell_fail);
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            1,
            "native PC pipeline at a released shell's address must still reach PC ExecuteIndirect"
        );

        forget(list_ptr);
        note_signature_dispatches(sig_dispatch, false);
        let _ = pso_async::released(shell_ok);
        real::forget_pso_for_test(shell_ok);
        real::forget_pso_for_test(real_ok);
    }

    #[test]
    fn two_pass_pso_skips_direct_draw_and_unmatched_execute_indirect() {
        #[repr(C)]
        struct FakeTwoPassList {
            vt: *const usize,
            refs: AtomicUsize,
            set_graphics_rs_calls: AtomicUsize,
            execute_indirect_calls: AtomicUsize,
        }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            (*(this as *const FakeTwoPassList)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            (*(this as *const FakeTwoPassList)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
        }
        unsafe extern "system" fn set_graphics_rs(this: *mut c_void, _rs: *mut c_void) {
            (*(this as *const FakeTwoPassList)).set_graphics_rs_calls.fetch_add(1, Ordering::Relaxed);
        }
        unsafe extern "system" fn exec_indirect(
            this: *mut c_void,
            _sig: usize,
            _max: u32,
            _args: usize,
            _args_off: u64,
            _count_buf: usize,
            _count_off: u64,
        ) {
            (*(this as *const FakeTwoPassList)).execute_indirect_calls.fetch_add(1, Ordering::Relaxed);
        }

        let mut orig_vt = [0usize; 64];
        orig_vt[1] = add_ref as usize;
        orig_vt[2] = release as usize;
        orig_vt[30] = set_graphics_rs as usize;
        orig_vt[59] = exec_indirect as usize;

        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeTwoPassList {
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            set_graphics_rs_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeTwoPassList as *mut c_void;

        // Register a finished two-pass GS/MS PSO behind an async stand-in (registered in TWO_PASS,
        // not TRANSLATED_GRAPHICS).
        let standin_tp = 0x7250_0001usize;
        let real_tp = 0x7250_0002usize;
        pso_async::submit(
            standin_tp,
            false,
            Box::new(move || {
                real::register_two_pass_for_test(real_tp, "GS");
                real_tp
            }),
        );
        assert_eq!(pso_async::resolve_wait(standin_tp), real_tp);
        assert!(real::two_pass_of(real_tp).is_some());
        assert!(!real::is_translated_graphics(real_tp));

        reset(list_ptr, 0x7250_0100, standin_tp);
        unsafe {
            // Seed a non-null title graphics RS so a fallthrough would visibly call SetGraphicsRootSignature.
            set_graphics_root_signature(list_ptr, list_ptr);
        }

        // 1. Direct draw (`graphics_bound`) must return None without binding the title's RS.
        let before = counts_snapshot();
        let bound = unsafe { graphics_bound(list_ptr) };
        let after = counts_snapshot();
        assert!(bound.is_none(), "two-pass MS/GS PSO must not fall through to PC DrawInstanced");
        assert!(after[DRAWS_SKIPPED] > before[DRAWS_SKIPPED]);
        assert_eq!(after[DRAWS_PC], before[DRAWS_PC]);
        assert_eq!(fake.set_graphics_rs_calls.load(Ordering::Relaxed), 0);

        // 2. Unmatched `ExecuteIndirect` must skip without binding the title's RS or calling PC ExecuteIndirect.
        let before = counts_snapshot();
        unsafe {
            execute_indirect(list_ptr, 0x7250_00F0, 1, 0, 0, 0, 0);
        }
        let after = counts_snapshot();
        assert!(after[DRAWS_SKIPPED] > before[DRAWS_SKIPPED]);
        assert_eq!(fake.set_graphics_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 0);

        forget(list_ptr);
        let _ = pso_async::released(standin_tp);
        real::forget_pso_for_test(real_tp);
        assert!(real::two_pass_of(real_tp).is_none());
    }

    #[test]
    fn create_list_reset_and_clear_state_track_initial_pso_on_d3d12() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        if console_list::enabled() {
            return;
        }
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            real::adopt(dev.as_raw(), real::Kind::Device);

            // Build two real PC compute PSOs from the inbox dummy CS bytecode.
            let mut ser = None;
            D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).expect("serialize RS");
            let ser = ser.unwrap();
            let rs: ID3D12RootSignature = dev
                .CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize()))
                .expect("CreateRootSignature");
            let cs = real::dummy_cs_bytecode();
            let make_pso = || -> ID3D12PipelineState {
                let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as *const c_void, BytecodeLength: cs.len() },
                    ..Default::default()
                };
                let pso = dev.CreateComputePipelineState(&desc).expect("CreateComputePipelineState");
                std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
                pso
            };
            let pso1 = make_pso();
            let pso2 = make_pso();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");

            // Stand-in synthetic key backed by real `pso1` in pso_async: CreateCommandList must
            // resolve it via resolve_for_bind for PC D3D12 while tracking `standin` in `bind`.
            let standin = 0x7300_0001usize;
            let pso1_raw = pso1.as_raw() as usize;
            pso_async::submit(standin, true, Box::new(move || pso1_raw));
            assert_eq!(pso_async::resolve_wait(standin), pso1_raw);

            let mut raw_list: *mut c_void = std::ptr::null_mut();
            let iid = windows_sys::core::GUID::from_u128(0x5b160d0f_ac1b_4185_8ba8_b3ae42a5a455);
            let hr = device_x::create_command_list_pc(
                dev.as_raw(),
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT.0 as u32,
                alloc.as_raw() as usize,
                standin,
                &iid,
                &mut raw_list,
            );
            assert!(hr >= 0 && !raw_list.is_null(), "create_command_list_pc failed: {hr:#x}");
            let backing = title_list::backing(raw_list).expect("wrapped in title_list");
            assert_eq!(tracked_pso(backing), standin, "CreateCommandList must seed bind::State.pso");

            // ClearState(pso2) updates bind::State.pso to pso2 and clears root args.
            let rs_key = 0x7300_0010usize;
            console_list::note_root_signature(rs_key, &test_rts0_blob());
            set_compute_root_signature(backing, rs_key as *mut c_void);
            set_compute_root_constant(backing, 0, 0xABCD, 0);
            let cl = ID3D12GraphicsCommandList::from_raw_borrowed(&backing).unwrap();
            cl.ClearState(&pso2);
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "ClearState must update bind::State.pso");
            with(backing, |s| assert!(s.layout.is_none() && s.args.is_empty()));

            // ClearState(None) clears bind::State.pso to 0.
            cl.ClearState(None);
            assert_eq!(tracked_pso(backing), 0, "ClearState(None) must clear bind::State.pso");

            // Close + reset_x(alloc, pso2) updates bind::State.pso to pso2 and clears root args.
            set_graphics_root_signature(backing, rs_key as *mut c_void);
            set_graphics_root_constant(backing, 0, 0x1111, 0);
            cl.Close().expect("Close");
            let hr = list_x::reset_x(backing, alloc.as_raw() as usize, pso2.as_raw() as usize);
            assert!(hr >= 0, "reset_x failed: {hr:#x}");
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "Reset must update bind::State.pso");
            with(backing, |s| assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty()));

            // Release the wrapped title list cleanly (calls bind::forget + list_x::on_release).
            let release_fn: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(raw_list as *const *const usize)).add(2));
            assert_eq!(release_fn(raw_list), 0);
            let _ = pso_async::released(standin);
        }
    }

    #[test]
    fn dispatch_mesh_signature_tracking_returns_companion_and_layout() {
        let sig = 0x7400_0001usize;
        let companion = 0x7400_0002usize;
        assert_eq!(dispatch_mesh_sig_of(sig), None);
        note_dispatch_mesh_signature(sig, companion, 28, 16);
        assert_eq!(
            dispatch_mesh_sig_of(sig),
            Some(DispatchMeshSig { companion_dispatch: companion, byte_stride: 28, mesh_offset: 16 })
        );
        DISPATCH_MESH_SIGS.lock().unwrap().as_mut().unwrap().remove(&sig);
        assert_eq!(dispatch_mesh_sig_of(sig), None);
    }

    /// A signature made at a freed mesh dispatch signature's address (`create_command_signature_x`
    /// forgets every new one first) is not taken for a mesh dispatch with the old companion.
    #[test]
    fn a_signature_at_a_freed_mesh_signatures_address_has_no_companion() {
        let sig = 0x7400_0011usize;
        note_dispatch_mesh_signature(sig, 0x7400_0012, 16, 0);
        assert!(dispatch_mesh_sig_of(sig).is_some());
        forget_dispatch_mesh_signature(sig);
        assert_eq!(dispatch_mesh_sig_of(sig), None);
        // nothing noted: nothing to forget
        forget_dispatch_mesh_signature(0x7400_0013);
    }

    #[test]
    fn two_pass_io_carries_the_unified_memory_base() {
        use crate::gpu::recomp::{IO_UMA_BASE_HI, KERNEL_SGPRS};
        let base = 0x4_0000_0000usize;
        for compute_pass in [true, false] {
            let io = two_pass_io(None, &[], 0x4_1234_5600, compute_pass, base, [0; 2]);
            assert_eq!(io.len(), KERNEL_SGPRS);
            assert_eq!(io[IO_UMA_BASE_HI], 4, "compute pass {compute_pass}");
        }
        let io = two_pass_io(None, &[], 0x4_1234_5600, true, base, [0; 2]);
        assert_eq!((io[8], io[9]), (0x1234_5600, 4));
    }

    /// The root views of a bind that lie in a GPU view are asked for with `ROOT_VIEW_BYTES` each
    /// (their size is not known); constants that look like such an address, tables, null and
    /// addresses elsewhere are not, and nothing is without view windows.
    #[test]
    fn root_views_in_a_gpu_view_are_wanted_with_their_bytes() {
        use crate::gpu::uma::{Uma, MAX_HEAP_WINDOWS, MAX_WINDOWS, VIEW_WINDOWS, WINDOW};
        let u = Uma::for_runtime_windows(16 << 20, 16 << 20, MAX_HEAP_WINDOWS).expect("16-window aperture");
        let span = view_span_of(&u);
        assert_eq!(span.1, (VIEW_WINDOWS * WINDOW) as u64);
        let g = u.view_reserve(1 << 20).expect("a GPU view") as u64;
        assert!(in_span(span, g) && in_span(span, span.0 + span.1 - 1));
        assert!(!in_span(span, span.0 - 1) && !in_span(span, span.0 + span.1) && !in_span(span, 0));
        let layout = Layout {
            params: vec![Param::Constants { at: 0, count: 2 }, Param::Descriptor { at: 8 }, Param::Table { at: 16 }, Param::Descriptor { at: 20 }, Param::Descriptor { at: 28 }, Param::Descriptor { at: 36 }],
            size: 44,
            ..Default::default()
        };
        let mut args = Vec::new();
        for q in [g + 0x200, g + 0x100] {
            args.extend_from_slice(&q.to_le_bytes());
        }
        args.extend_from_slice(&(g as u32).to_le_bytes());
        for q in [0x7_1234_0000u64, 0, g + 0x3_0000] {
            args.extend_from_slice(&q.to_le_bytes());
        }
        let wanted: Vec<(u64, u64)> = root_views_wanted(&layout, &args, span).collect();
        assert_eq!(wanted, vec![(g + 0x100, ROOT_VIEW_BYTES), (g + 0x3_0000, ROOT_VIEW_BYTES)]);
        assert_eq!(ROOT_VIEW_BYTES, 64 << 10);
        assert_eq!(root_views_wanted(&layout, &args, (0, 0)).count(), 0, "no view windows");
        let four = Uma::for_runtime_windows(16 << 20, 16 << 20, MAX_WINDOWS).expect("4-window aperture");
        assert_eq!(view_span_of(&four), (0, 0));
        assert_eq!(u.view_release(g as usize), Some(1 << 20));
    }

    /// A list a deferred split records into next (`inherit_for_split`) leaves the two-pass record
    /// buffers and argument scratch of its previous recording to that recording's allocator, as
    /// `reset` does: the GPU may still read them. (Buffers of a software device: no GPU work.)
    #[test]
    fn a_split_holds_the_previous_record_buffers_for_their_allocator() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        let dev = unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            dev.unwrap()
        };
        let (from, to) = (0x7600_0001usize as *mut c_void, 0x7600_0002usize as *mut c_void);
        let (old, new) = (0x7600_0010usize, 0x7600_0020usize);
        let record = unsafe { TwoPassBuf::make(&dev, 1 << 16, "record buffer") }.unwrap();
        let block = unsafe { TwoPassBuf::make(&dev, 1 << 16, "argument scratch") }.unwrap();
        let (buf, scratch) = (record.res.clone(), block.res.clone());
        with(to, |s| {
            s.allocator = old;
            s.res_buf = Some(record);
            s.scratch = Some((scratch.clone(), 1 << 16));
            s.res_retired.push(block);
            (s.scratch_used, s.scratch_taken) = (256, 1 << 16);
        });
        inherit_for_split(from, to, new);
        with(to, |s| {
            assert_eq!(s.allocator, new);
            assert!(s.res_buf.is_none() && s.res_retired.is_empty() && s.scratch.is_none(), "fresh buffers for the new recording");
            assert_eq!((s.scratch_used, s.scratch_taken), (0, 0));
        });
        let held: Vec<usize> = ALLOC_RES.lock().unwrap().as_mut().and_then(|m| m.remove(&old)).unwrap_or_default().iter().map(|b| b.res.as_raw() as usize).collect();
        assert_eq!(held.len(), 2, "the record buffer and the scratch block wait for the old allocator");
        assert!(held.contains(&(buf.as_raw() as usize)) && held.contains(&(scratch.as_raw() as usize)));
        forget(from);
        forget(to);
    }

    /// The title's last Release of an allocator gives back the two-pass buffers ended recordings
    /// on it left (`alloc_pool::forget`), when nothing records on it and the GPU is done with it:
    /// no Reset of it comes any more, and they stayed counted against `two_pass_cap`. Those of an
    /// allocator a list still records on stay. (Buffers of a software device: no GPU work.)
    #[test]
    fn a_released_allocator_gives_back_its_two_pass_buffers() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        let dev = unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            dev.unwrap()
        };
        let key = dev.as_raw() as usize;
        let ids = Box::new([0u8; 4]);
        let (done, busy) = (ids.as_ptr() as usize, ids.as_ptr() as usize + 1);
        let (ended, recording) = (done + 2, done + 3);
        let ty = D3D12_COMMAND_LIST_TYPE_DIRECT.0;
        // a list recorded on `done` and is gone (never submitted); one still records on `busy`
        super::super::alloc_pool::created_on(ended, done, ty);
        super::super::alloc_pool::forget_list(ended);
        super::super::alloc_pool::created_on(recording, busy, ty);
        let a = unsafe { TwoPassBuf::make(&dev, 1 << 16, "record buffer") }.unwrap();
        let b = unsafe { TwoPassBuf::make(&dev, 1 << 16, "record buffer") }.unwrap();
        let (a_raw, b_raw) = (a.res.as_raw() as usize, b.res.as_raw() as usize);
        hold_res_until_reset(done, vec![a]);
        hold_res_until_reset(busy, vec![b]);
        unsafe {
            super::super::alloc_pool::forget(done);
            super::super::alloc_pool::forget(busy);
        }
        let held = |alloc: usize| -> Vec<usize> {
            ALLOC_RES.lock().unwrap().as_ref().and_then(|m| m.get(&alloc)).map_or(Vec::new(), |v| v.iter().map(|b| b.res.as_raw() as usize).collect())
        };
        let (held_done, held_busy) = (held(done), held(busy));
        let mut freed = Vec::new();
        while let Some(f) = FREE_RES.lock().unwrap().take(key, 0, u64::MAX) {
            freed.push(f.res.as_raw() as usize);
        }
        // (before the assertions: nothing of this test stays)
        let left = ALLOC_RES.lock().unwrap().as_mut().and_then(|m| m.remove(&busy));
        drop(left);
        assert!(held_done.is_empty(), "the released allocator's buffers do not wait for a Reset that never comes");
        assert_eq!(freed, vec![a_raw], "they are kept for reuse");
        assert_eq!(held_busy, vec![b_raw], "a list still records on the other: its buffers stay");
    }

    /// Binds on XCL's ABI (`bind_abi`, both two-pass stages) take window 0 the same way: not
    /// open on the GPU is refused, and so is shadowed memory without a buffer there (a null root
    /// descriptor); unshadowed memory with none keeps going as before.
    #[test]
    fn binds_refuse_shadowed_memory_without_window_0() {
        assert!(first_window(None, false).is_err());
        assert!(first_window(Some(0), true).is_err());
        assert_eq!(first_window(Some(0), false), Ok(0));
        assert_eq!(first_window(Some(0x7_0000_0000), true), Ok(0x7_0000_0000));
    }

    /// The two passes get the addresses of the heaps set after their arguments, the view heap's
    /// first, as `bind_abi`'s stages do (`console_list::with_heaps`): for the console's most
    /// common mesh layout (three tables, a root view, two constants) the compute pass's s23 /
    /// s24, the pixel shader's right after its arguments from s8.
    #[test]
    fn two_pass_heap_addresses_follow_the_arguments() {
        use crate::gpu::recomp::IO_UMA_BASE_HI;
        // RTS0 v2: three tables, a root SRV, two constants; the root signature's flags at 20
        let layout = |flags: u32| {
            let params = [(0u32, 0u32), (0, 0), (0, 0), (3, 0), (1, 2)];
            let n = params.len() as u32;
            let mut r = vec![2u32, n, 24, 0, 0, flags];
            for (k, &(ty, _)) in params.iter().enumerate() {
                r.extend([ty, 0, 24 + 12 * n + 12 * k as u32]);
            }
            for &(_, count) in &params {
                r.extend([0, 0, count]);
            }
            let blob: Vec<u8> = r.iter().flat_map(|w| w.to_le_bytes()).collect();
            console_list::layout_from_blob(&blob).unwrap()
        };
        let words = [0x11u32, 0x22, 0x33, 0x5000_0000, 0x7, 5, 6];
        let args: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let heaps = [0x6000_0000, 0x6381_0000];
        let io = |flags: u32, compute_pass: bool| two_pass_io(Some(&layout(flags)), &args, 0x4_0000_0200, compute_pass, 0x4_0000_0000, heaps);
        // the compute pass: table at s[8:9], arguments from s16, the heaps at s23 / s24
        let cs = io(0xC00, true);
        assert_eq!((cs[8], cs[9]), (0x200, 4));
        assert_eq!(&cs[16..23], &words);
        assert_eq!(&cs[23..26], &[0x6000_0000, 0x6381_0000, 0]);
        assert_eq!(cs[IO_UMA_BASE_HI], 4);
        assert_eq!(&io(0x800, true)[23..25], &[0x6381_0000, 0], "the sampler heap's alone");
        assert_eq!(&io(0x400, true)[23..25], &[0x6000_0000, 0], "the view heap's alone");
        assert_eq!(io(0, true)[23], 0, "no flags: no addresses");
        // the pixel shader: table at s[0:1], arguments from s8, the heaps after them
        let ps = io(0xC00, false);
        assert_eq!((ps[0], ps[1]), (0x200, 4));
        assert_eq!(&ps[8..15], &words);
        assert_eq!(&ps[15..18], &[0x6000_0000, 0x6381_0000, 0]);
        assert_eq!(io(0, false)[15], 0, "no flags: no addresses");
    }

    #[test]
    fn two_pass_groups_stay_inside_the_record_buffer() {
        // #304's transform record: 11 908 dwords
        let record = 11_908 * 4;
        assert_eq!(record_groups(32 << 20, record, 1), 704);
        assert_eq!(record_groups(128 << 20, record, 1), 2817);
        // the mesh pass's groups, `split` a record, all in X
        assert_eq!(record_groups(1 << 30, 64, 7), 65_535 / 7);
        assert_eq!(record_groups(1 << 20, 2 << 20, 1), 0);
    }

    /// Every root view the argument pass's program declares (u0, u2, u3, u4) gets an address,
    /// also without a count buffer: u4 then names the constants. (A slot left out after the root
    /// signature is set holds nothing, and a load from it is a load at GPU address 0.)
    #[test]
    fn the_argument_pass_names_a_buffer_in_every_view_it_declares() {
        for register in ["u0", "u2", "u3", "u4"] {
            assert!(SCALE_ARGS_HLSL.contains(&format!("register({register})")), "{register}");
        }
        assert!(!SCALE_ARGS_HLSL.contains("register(u1)") && !SCALE_ARGS_HLSL.contains("register(u5)"));
        let with_count = scale_views(0x1000, 0x2000, 0x3000, Some(0x4000));
        assert_eq!(with_count, [(0, 0x1000), (2, 0x2000), (3, 0x3000), (4, 0x4000)]);
        let without = scale_views(0x1000, 0x2000, 0x3000, None);
        assert_eq!(without, [(0, 0x1000), (2, 0x2000), (3, 0x3000), (4, 0x1000)]);
        assert!(without.iter().all(|&(_, va)| va != 0));
    }

    /// `SCALE_ARGS_HLSL` on `commands` (dwords each, the DispatchMesh argument at dword 1):
    /// the compute pass's and the mesh pass's commands.
    fn scale_args(commands: &[[u32; 4]], split: u32, cap: u32) -> Option<(Vec<[u32; 4]>, Vec<[u32; 4]>)> {
        scale_args_counted(commands, split, cap, None)
    }

    /// `scale_args` with a count buffer holding `count`
    fn scale_args_counted(commands: &[[u32; 4]], split: u32, cap: u32, count: Option<u32>) -> Option<(Vec<[u32; 4]>, Vec<[u32; 4]>)> {
        use crate::gpu::recomp::runner;
        if runner::adapter().is_err() {
            runner::skip_test("no GPU");
            return None;
        }
        let dxil = crate::gpu::recomp::dxc::compile(SCALE_ARGS_HLSL, "main", "cs_6_6", &[]).unwrap();
        let n = commands.len();
        let (src, dst) = (0u32, 256u32);
        let mut mem: Vec<u32> = commands.iter().flatten().copied().collect();
        mem.resize(64 + 2 * 4 * n, 0xDEAD);
        // the count at dword 60 (u4 is the same memory in the runner)
        assert!(4 * n <= 60);
        mem[60] = count.unwrap_or(0);
        let io = [n as u32, 16, 4, split, cap, src, dst, count.is_some() as u32, 240];
        let out = runner::dispatch(&dxil, [(n as u32).div_ceil(64), 1, 1], &io, &mem).unwrap();
        let cmd = |k: usize| [out[64 + 4 * k], out[65 + 4 * k], out[66 + 4 * k], out[67 + 4 * k]];
        Some(((0..n).map(cmd).collect(), (n..2 * n).map(cmd).collect()))
    }

    #[test]
    fn gpu_indirect_two_pass_groups_stay_inside_the_record_buffer() {
        let Some((compute, mesh)) = scale_args(&[[7, 100, 1, 1], [8, 3, 1, 1], [9, 0, 1, 1]], 2, 10) else { return };
        assert_eq!(compute, [[7, 10, 1, 1], [8, 3, 1, 1], [9, 0, 1, 1]]);
        assert_eq!(mesh, [[7, 20, 1, 1], [8, 6, 1, 1], [9, 0, 1, 1]]);
    }

    #[test]
    fn gpu_indirect_two_pass_groups_go_in_x() {
        let commands = [[1, 1, 2, 1], [2, 2, 3, 4], [3, 300, 300, 1], [4, 5, 0, 2], [5, 70_000, 70_000, 70_000], [6, 1000, 1, 1], [7, 5000, 1, 1], [8, 5000, 1, 0]];
        let Some((compute, mesh)) = scale_args(&commands, 3, 1000) else { return };
        assert_eq!(compute, [[1, 2, 1, 1], [2, 24, 1, 1], [3, 1000, 1, 1], [4, 0, 1, 1], [5, 1000, 1, 1], [6, 1000, 1, 1], [7, 1000, 1, 1], [8, 0, 1, 1]]);
        assert_eq!(mesh, [[1, 6, 1, 1], [2, 72, 1, 1], [3, 3000, 1, 1], [4, 0, 1, 1], [5, 3000, 1, 1], [6, 3000, 1, 1], [7, 3000, 1, 1], [8, 0, 1, 1]]);
    }

    /// Freed two-pass buffers are kept, many of one size too (a frame of recordings hands back
    /// tens of 32 MiB ones, and the next takes them again), up to the cap in bytes, oldest let go
    /// first; one no recording took for `FREE_RES_IDLE` goes.
    #[test]
    fn freed_two_pass_buffers_kept_for_reuse_are_capped() {
        use std::time::{Duration, Instant};
        let t = Instant::now();
        let mib = 1u64 << 20;
        let mut p: FreePool<u32> = FreePool::new(1 << 30);
        for k in 0..32 {
            assert!(p.put(t, 32 * mib, 7, k).is_empty(), "32 x 32 MiB fit 1 GiB");
        }
        assert_eq!(p.bytes, 1 << 30);
        assert_eq!(p.put(t, mib, 7, 99), vec![0], "past the cap the oldest goes");
        assert_eq!(p.bytes, (1 << 30) - 31 * mib);
        // nothing is idle yet; then all of them are
        assert!(p.take_idle(t + Duration::from_millis(1999), FREE_RES_IDLE).is_empty());
        assert_eq!(p.take_idle(t + FREE_RES_IDLE, FREE_RES_IDLE).len(), 32);
        assert_eq!((p.bytes, p.free.len()), (0, 0));
    }

    /// Two-pass buffers alive stay within the cap: one that would pass it is not counted (and not
    /// made), and freed ones go first to make room for it, oldest first, as many as it takes –
    /// none when all of them cannot make it (the draw is skipped anyway, and the next recordings
    /// can still take them).
    #[test]
    fn two_pass_buffers_alive_stay_within_the_cap() {
        use std::time::Instant;
        let mib = 1u64 << 20;
        let alive = AtomicU64::new(0);
        assert!(reserve(&alive, 32 * mib, 64 * mib));
        assert!(reserve(&alive, 32 * mib, 64 * mib), "up to the cap itself");
        assert!(!reserve(&alive, mib, 64 * mib));
        assert_eq!(alive.load(Ordering::Relaxed), 64 * mib, "a refused one is not counted");
        assert!(!reserve(&AtomicU64::new(u64::MAX), 1, u64::MAX), "no wrap");
        // room from the free ones
        let t = Instant::now();
        let mut p: FreePool<&str> = FreePool::new(1 << 30);
        p.put(t, 4 * mib, 1, "4");
        p.put(t, 32 * mib, 2, "32");
        p.put(t, mib, 1, "1");
        assert_eq!(p.take_room(0), Some(vec![]));
        assert_eq!(p.take_room(5 * mib), Some(vec!["4", "32"]));
        assert_eq!(p.bytes, mib);
        assert_eq!(p.take_room(64 * mib), None, "they cannot make the room: none go");
        assert_eq!((p.bytes, p.free.len()), (mib, 1));
        assert_eq!(p.take_room(mib), Some(vec!["1"]), "all there are, when they make it");
        assert_eq!((p.bytes, p.free.len()), (0, 0));
        assert!(two_pass_cap() > 0);
    }

    /// A recording takes the smallest freed buffer that is large enough, of its device and within
    /// its bound, the one handed back last of those: a large one stays for a large need, and the
    /// spare ones of a size grow old.
    #[test]
    fn a_recording_takes_the_smallest_freed_buffer_that_fits() {
        use std::time::{Duration, Instant};
        let t = Instant::now();
        let mib = 1u64 << 20;
        let mut p: FreePool<&str> = FreePool::new(1 << 30);
        p.put(t, 32 * mib, 1, "32 a");
        p.put(t, 4 * mib, 1, "4 a");
        p.put(t + Duration::from_millis(10), 4 * mib, 1, "4 b");
        p.put(t, mib, 2, "1 other device");
        assert_eq!(p.take(1, mib, u64::MAX), Some("4 b"));
        assert_eq!(p.take(1, 8 * mib, 16 * mib), None, "the 32 MiB one is past the bound");
        assert_eq!(p.take(1, 8 * mib, u64::MAX), Some("32 a"));
        assert_eq!(p.take(1, mib, u64::MAX), Some("4 a"));
        assert_eq!(p.take(1, 0, u64::MAX), None);
        assert_eq!(p.take(2, 0, u64::MAX), Some("1 other device"));
        assert_eq!(p.bytes, 0);
    }

    /// Record buffers come in powers of two from 1 MiB, and a draw has as many groups with a record
    /// as with the sizes before (its need, at least 4 MiB, in MiB): none drops groups it drew
    /// before. An ExecuteIndirect keeps the groups a command had with 32 MiB.
    #[test]
    fn two_pass_record_buffers_fit_what_their_draws_need() {
        let mib = 1u64 << 20;
        assert_eq!(record_buffer_bytes(0), mib);
        assert_eq!(record_buffer_bytes(2 * 47_632), mib, "a DispatchMesh of two groups of #304's record");
        assert_eq!(record_buffer_bytes(mib + 1), 2 * mib);
        assert_eq!(record_buffer_bytes(130 * 47_632), 8 * mib);
        assert_eq!(record_buffer_bytes(32 * mib), 32 * mib);
        assert_eq!(record_buffer_bytes(1 << 40), 128 * mib);
        let before = |need: u64| need.max(4 << 20).next_multiple_of(1 << 20).min(128 << 20);
        for record in [64u64, 500, 4_096, 47_632, 300_000, 3 << 20, 200 << 20] {
            for split in [1u32, 2, 4, 7] {
                for groups in [1u32, 2, 8, 52, 92, 130, 704, 2_817, 9_000, 65_535] {
                    let need = groups as u64 * record;
                    let old = groups.min(record_groups(before(need), record, split));
                    let new = groups.min(record_groups(record_buffer_bytes(need), record, split));
                    assert!(new >= old, "{groups} groups of {record} B, split {split}: {new} < {old}");
                }
                let ei = record_buffer_bytes(indirect_record_need(record, split));
                assert!(ei <= 32 * mib);
                assert_eq!(record_groups(ei, record, split), record_groups(32 * mib, record, split), "{record} B, split {split}");
            }
        }
        assert_eq!(indirect_record_need(64, 1), 65_535 * 64, "small records: their groups in X bound it");
        assert_eq!(record_buffer_bytes(indirect_record_need(64, 1)), 4 * mib);
        assert_eq!(indirect_record_need(47_632, 1), 32 * mib);
    }

    /// Without `XCL_TWO_PASS_BATCH` a two-pass draw records what it did before: each command's
    /// compute pass and mesh pass (`two_pass_steps`), a UAV barrier on the records after each.
    #[test]
    fn whole_two_pass_records_keep_a_barrier_after_every_pass() {
        for commands in 1..5u32 {
            let before: Vec<TwoPassStep> = two_pass_steps(commands).flat_map(|(k, compute)| [TwoPassStep::Pass { k, compute, region: WHOLE }, TwoPassStep::Barrier]).collect();
            assert_eq!(two_pass_plan(commands, Records::Whole, None), before, "{commands} commands");
            // (after a batched draw left a mesh pass reading: a barrier first)
            assert_eq!(two_pass_plan(commands, Records::Whole, Some(HALF_1))[1..], before[..]);
            assert_eq!(two_pass_plan(commands, Records::Whole, Some(HALF_1))[0], TwoPassStep::Barrier);
        }
        // a whole draw leaves nothing open
        let open = two_pass_plan(3, Records::Whole, None).into_iter().fold(None, open_after);
        assert_eq!(open, None);
    }

    /// Consecutive draws with records in halves: every compute pass and the barrier after it,
    /// then its mesh pass on the same region, and no barrier after a mesh pass – half the
    /// barriers of whole records, for DispatchMesh (one command) and ExecuteIndirect alike.
    #[test]
    fn two_pass_records_in_halves_record_half_the_barriers() {
        let barriers = |plans: &[Vec<TwoPassStep>]| plans.iter().flatten().filter(|s| **s == TwoPassStep::Barrier).count();
        for (draws, commands) in [(8usize, 1u32), (5, 3), (1, 4)] {
            let mut open = None;
            let mut plans = Vec::new();
            for _ in 0..draws {
                let plan = two_pass_plan(commands, Records::Halves, open);
                open = plan.iter().copied().fold(open, open_after);
                plans.push(plan);
            }
            assert_eq!(barriers(&plans), draws * commands as usize, "{draws} draws of {commands}");
            let whole: Vec<_> = (0..draws).map(|_| two_pass_plan(commands, Records::Whole, None)).collect();
            assert_eq!(barriers(&whole), 2 * draws * commands as usize);
            // each command: its compute pass, a barrier, its mesh pass on the same region
            for plan in &plans {
                for k in 0..commands {
                    let at = plan.iter().position(|s| matches!(s, TwoPassStep::Pass { k: c, compute: true, .. } if *c == k)).unwrap();
                    let TwoPassStep::Pass { region, .. } = plan[at] else { unreachable!() };
                    assert_eq!(plan[at + 1], TwoPassStep::Barrier);
                    assert_eq!(plan[at + 2], TwoPassStep::Pass { k, compute: false, region });
                    assert!(region == HALF_0 || region == HALF_1);
                }
            }
        }
    }

    /// Whatever the order of draws with records in halves or whole, of one or more commands, no
    /// compute pass writes a region a mesh pass may still be reading (no barrier between), and
    /// every mesh pass reads records its compute pass wrote before a barrier.
    #[test]
    fn two_pass_draws_never_write_records_a_mesh_pass_still_reads() {
        // (records, commands) of each draw: every sequence of three of these
        let kinds = [(Records::Halves, 1u32), (Records::Halves, 2), (Records::Halves, 3), (Records::Whole, 1), (Records::Whole, 2)];
        for a in kinds {
            for b in kinds {
                for c in kinds {
                    let (mut reading, mut writing, mut open) = (0u8, 0u8, None);
                    for (records, commands) in [a, b, c, a] {
                        let mut wrote = HashMap::new();
                        for step in two_pass_plan(commands, records, open) {
                            match step {
                                TwoPassStep::Barrier => (reading, writing) = (0, 0),
                                TwoPassStep::Pass { k, compute: true, region } => {
                                    assert_eq!(region & reading, 0, "{a:?} {b:?} {c:?}: a compute pass writes records a mesh pass reads");
                                    writing |= region;
                                    wrote.insert(k, region);
                                }
                                TwoPassStep::Pass { k, compute: false, region } => {
                                    assert_eq!(region & writing, 0, "{a:?} {b:?} {c:?}: a mesh pass reads records still being written");
                                    assert_eq!(wrote.get(&k), Some(&region), "the mesh pass reads its own compute pass's records");
                                    reading |= region;
                                }
                            }
                            open = open_after(open, step);
                            assert_eq!(open.unwrap_or(0), reading, "what is kept for the next draw");
                        }
                    }
                }
            }
        }
    }

    /// A record buffer in halves is twice the one the draw would have had whole, so each half
    /// has as many groups with a record; past `RECORD_MAX`, or without `XCL_TWO_PASS_BATCH`,
    /// the draw takes the whole buffer as before.
    #[test]
    fn two_pass_record_halves_hold_what_the_whole_buffer_held() {
        let mib = 1u64 << 20;
        assert_eq!(halves_bytes(0, true), Some(2 * mib));
        assert_eq!(halves_bytes(32 * mib, true), Some(64 * mib));
        assert_eq!(halves_bytes(64 * mib, true), Some(128 * mib));
        assert_eq!(halves_bytes(64 * mib + 1, true), None);
        assert_eq!(halves_bytes(32 * mib, false), None);
        for record in [64u64, 500, 4_096, 47_632, 300_000] {
            for split in [1u32, 2, 7] {
                let need = indirect_record_need(record, split);
                let half = halves_bytes(need, true).unwrap() / 2;
                assert_eq!(record_groups(half, record, split), record_groups(record_buffer_bytes(need), record, split), "{record} B, split {split}");
                for groups in [1u64, 92, 2_817, 9_000] {
                    let need = groups * record;
                    if let Some(both) = halves_bytes(need, true) {
                        assert!(record_groups(both / 2, record, split) >= record_groups(record_buffer_bytes(need), record, split));
                    }
                }
            }
        }
    }

    /// An ExecuteIndirect's argument block is in COMMON before its first rewrite; with
    /// `XCL_TWO_PASS_BATCH` the next rewrite into it takes it from INDIRECT_ARGUMENT.
    #[test]
    fn the_argument_block_is_taken_from_the_state_it_was_left_in() {
        assert_eq!(scratch_state(false), D3D12_RESOURCE_STATE_COMMON);
        assert_eq!(scratch_state(true), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        let list = 0x7500_0011usize as *mut c_void;
        assert!(!with(list, |s| s.scratch_in_args), "a new recording's block has not been used");
        with(list, |s| (s.scratch_in_args, s.record_open) = (true, Some(HALF_0)));
        reset(list, 0, 0);
        assert_eq!(with(list, |s| (s.scratch_in_args, s.record_open)), (false, None), "the next recording takes fresh buffers");
        forget(list);
    }

    /// A deferred DispatchMesh or ExecuteIndirect as `prepare_two_pass_stage` would snapshot it,
    /// draw `k` of a run with one pipeline pair: the title's root arguments of each draw give it
    /// io blocks of its own; its records are region `k` of the batch buffer at `0x9000_0000`.
    fn snap(k: u64, indirect: Option<usize>) -> DrawSnap {
        let records = 0x9000_0000 + k * 0x1_0000;
        let w = [0x5000_0000, 0x5100_0000, 0x5200_0000, 0x5300_0000];
        let compute = StageBind { rs: 0xC0, pso: 0xC1, views: [0x4000_0000 + k * 0x200, records, w[0], w[1], w[2], w[3]] };
        let mesh = StageBind { rs: 0xA0, pso: 0xA1, views: [0x4000_0100 + k * 0x200, records, w[0], w[1], w[2], w[3]] };
        let work = match indirect {
            None => DeferredWork::Mesh { groups: 3 + k as u32, mesh_groups: 2 * (3 + k as u32) },
            Some(block) => DeferredWork::Indirect {
                rewrite: Rewrite { rs: 0xE0, pso: 0xE1, views: [(0, 0x4100_0000 + k * 0x100), (2, 0x4200_0000 + k * 16), (3, 0x7000 + k * 32), (4, 0x4100_0000 + k * 0x100)], groups: 1 },
                block,
                before: D3D12_RESOURCE_STATE_COMMON,
                companion: 0xD0,
                sig: 0xD1,
                compute_at: k * 32,
                mesh_at: k * 32 + 16,
            },
        };
        DrawSnap { compute, mesh, work }
    }

    /// What the unbatched path records for `d`, as `make_scaled_args_buffer`,
    /// `emit_two_pass_stage` and the PC calls after it record it (barriers left out).
    fn unbatched_commands(d: &DrawSnap) -> Vec<Cmd> {
        let mut v = Vec::new();
        if let DeferredWork::Indirect { rewrite, .. } = d.work {
            v.extend([Cmd::ComputeRs(rewrite.rs), Cmd::Pso(rewrite.pso)]);
            v.extend(rewrite.views.iter().map(|&(s, va)| Cmd::ComputeView(s, va)));
            v.push(Cmd::Dispatch(rewrite.groups));
        }
        v.push(Cmd::ComputeRs(d.compute.rs));
        v.extend(d.compute.views.iter().enumerate().map(|(k, &va)| Cmd::ComputeView(k as u32, va)));
        v.push(Cmd::Pso(d.compute.pso));
        v.push(match d.work {
            DeferredWork::Mesh { groups, .. } => Cmd::Dispatch(groups),
            DeferredWork::Indirect { block, companion, compute_at, .. } => Cmd::Indirect(companion, block, compute_at),
        });
        v.push(Cmd::GraphicsRs(d.mesh.rs));
        v.extend(d.mesh.views.iter().enumerate().map(|(k, &va)| Cmd::GraphicsView(k as u32, va)));
        v.push(Cmd::Pso(d.mesh.pso));
        v.push(match d.work {
            DeferredWork::Mesh { mesh_groups, .. } => Cmd::DispatchMesh(mesh_groups),
            DeferredWork::Indirect { block, sig, mesh_at, .. } => Cmd::Indirect(sig, block, mesh_at),
        });
        v
    }

    /// A GPU call with the state it runs with: the call, the pipeline, and the root signature and
    /// root views of its side (compute for a Dispatch or an ExecuteIndirect of a companion
    /// Dispatch signature – `0xD0` / `0xE0`.. in `snap` – graphics otherwise).
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Call {
        call: Cmd,
        pso: usize,
        rs: usize,
        views: std::collections::BTreeMap<u32, u64>,
    }

    /// The calls of `cmds` with the state each runs with (a root signature that changes drops
    /// the root views, as D3D12 does).
    fn calls(cmds: &[Cmd]) -> Vec<Call> {
        use std::collections::BTreeMap;
        let (mut crs, mut grs, mut pso) = (0, 0, 0);
        let (mut cv, mut gv) = (BTreeMap::new(), BTreeMap::new());
        let mut out = Vec::new();
        for c in cmds {
            match *c {
                Cmd::ComputeRs(r) => {
                    if r != crs {
                        cv.clear();
                    }
                    crs = r;
                }
                Cmd::GraphicsRs(r) => {
                    if r != grs {
                        gv.clear();
                    }
                    grs = r;
                }
                Cmd::Pso(p) => pso = p,
                Cmd::ComputeView(s, va) => {
                    cv.insert(s, va);
                }
                Cmd::GraphicsView(s, va) => {
                    gv.insert(s, va);
                }
                Cmd::Dispatch(_) | Cmd::Indirect(0xD0, ..) => out.push(Call { call: c.clone(), pso, rs: crs, views: cv.clone() }),
                Cmd::DispatchMesh(_) | Cmd::Indirect(..) => out.push(Call { call: c.clone(), pso, rs: grs, views: gv.clone() }),
                // (the title's pipeline set, with what it sets: as a call of its own, in order)
                Cmd::Bind(p) => {
                    pso = p;
                    out.push(Call { call: c.clone(), pso, rs: 0, views: BTreeMap::new() });
                }
                Cmd::Barriers { .. } | Cmd::TimesBegin(_) | Cmd::TimesEnd => {}
            }
        }
        out
    }

    fn steps(draws: &[DrawSnap]) -> Vec<Step> {
        draws.iter().map(|&d| Step::Draw(d)).collect()
    }

    /// `snap` with the pipeline pair `pipe` (its compute pass's and mesh pass's PC pipelines).
    fn snap_on(k: u64, pipe: usize, indirect: Option<usize>) -> DrawSnap {
        let mut d = snap(k, indirect);
        d.compute.pso += pipe * 0x10;
        d.mesh.pso += pipe * 0x10;
        d
    }

    /// A run across the title's SetPipelineState calls (DispatchMesh and ExecuteIndirect draws
    /// on two pipelines, root arguments set between them): the compute passes each with their
    /// draw's pipeline, then the mesh passes with each pipeline set (`real::bind_pipeline`, with
    /// the depth bias and the rest it records) between them where the title made it, and each
    /// mesh pass with its own pipeline after it – what the unbatched path gives every call, in
    /// the order it gives the mesh passes and the sets.
    #[test]
    fn a_run_goes_on_across_pipeline_sets_in_the_titles_order() {
        let (a, b) = (0x7A00usize, 0x7B00usize);
        let run = vec![
            Step::Draw(snap_on(0, 0, None)),
            Step::Draw(snap_on(1, 0, Some(0xB1))),
            Step::Bind(b),
            Step::Draw(snap_on(2, 1, None)),
            Step::Draw(snap_on(3, 1, Some(0xB1))),
            Step::Bind(a),
            Step::Draw(snap_on(4, 0, None)),
        ];
        let cmds = batch_commands(&run, 0x900, false);
        let batched = calls(&cmds);
        let mut alone = Vec::new();
        for s in &run {
            match s {
                Step::Bind(p) => alone.push(calls(&[Cmd::Bind(*p)])[0].clone()),
                Step::Draw(d) => alone.extend(calls(&unbatched_commands(d))),
            }
        }
        // the mesh passes and the pipeline sets: the same calls with the same state, in order
        let mesh_side = |c: &[Call]| -> Vec<Call> { c.iter().filter(|x| matches!(x.call, Cmd::Bind(_) | Cmd::DispatchMesh(_) | Cmd::Indirect(0xD1, ..))).cloned().collect() };
        assert_eq!(mesh_side(&batched), mesh_side(&alone));
        assert_eq!(mesh_side(&batched).iter().map(|x| matches!(x.call, Cmd::Bind(_))).collect::<Vec<_>>(), [false, false, true, false, false, true, false]);
        // the compute passes and rewrites: each with its draw's pipeline, root signature, views
        let compute_side = |c: &[Call]| -> Vec<Call> { c.iter().filter(|x| matches!(x.call, Cmd::Dispatch(_) | Cmd::Indirect(0xD0, ..))).cloned().collect() };
        let (mut got, mut want) = (compute_side(&batched), compute_side(&alone));
        got.sort_by_key(|x| (x.pso, format!("{:?}", x.call)));
        want.sort_by_key(|x| (x.pso, format!("{:?}", x.call)));
        assert_eq!(got, want);
        // no pipeline set among the compute passes; a mesh pass sets its pipeline after a set
        let first_mesh = cmds.iter().position(|c| matches!(c, Cmd::TimesBegin(Some(_)))).unwrap();
        assert!(!cmds[..first_mesh].iter().any(|c| matches!(c, Cmd::Bind(_))));
        for (i, c) in cmds.iter().enumerate() {
            if matches!(c, Cmd::Bind(_)) {
                let next_pso = cmds[i..].iter().position(|c| matches!(c, Cmd::Pso(_))).unwrap();
                let next_call = cmds[i..].iter().position(|c| matches!(c, Cmd::DispatchMesh(_) | Cmd::Indirect(..))).unwrap();
                assert!(next_pso < next_call, "the mesh pass after a set binds its own pipeline");
            }
        }
        assert_eq!(cmds.iter().filter(|c| matches!(c, Cmd::Barriers { .. })).count(), 3, "still three ResourceBarrier calls");
    }

    /// SetPipelineState with no deferred draws binds as it always did; with them a two-pass
    /// pipeline goes into the run and any other pipeline records the run first. Into the run, it
    /// becomes the pipeline the next draws resolve, and nothing is recorded yet.
    #[test]
    fn set_pipeline_state_goes_into_a_run_only_with_a_two_pass_pipeline() {
        assert_eq!(on_pipeline(false, true), OnPipeline::Bind);
        assert_eq!(on_pipeline(false, false), OnPipeline::Bind);
        assert_eq!(on_pipeline(true, true), OnPipeline::Queue);
        assert_eq!(on_pipeline(true, false), OnPipeline::FlushThenBind);
        let list = 0x7500_0041usize as *mut c_void;
        let pso = 0x7C00_0041usize;
        real::register_two_pass_for_test(pso, "MS");
        with(list, |s| s.deferred.push(RunItem::Draw(dummy_draw())));
        mark_deferring(list);
        unsafe { set_pipeline_state(list, pso as *mut c_void) };
        assert_eq!(with(list, |s| s.pso), pso);
        assert_eq!(with(list, |s| s.deferred.iter().map(RunItem::step).collect::<Vec<_>>()), [Step::Draw(snap(0, None)), Step::Bind(pso)]);
        assert!(DEFERRING.contains(list as usize), "still deferring");
        forget(list);
    }

    /// SetDescriptorHeaps between deferred draws records the run first (the heap a pass reads its
    /// views and samplers from is the one bound when it was deferred), as do render target,
    /// barrier and other calls; the log names the call that ended each run.
    #[test]
    fn set_descriptor_heaps_between_deferred_draws_records_the_run_first() {
        for newer in [false, true] {
            assert!(flush_wanted(32, newer, true, true));
        }
        assert_eq!(super::super::real::list_slot_name(32), "SetDescriptorHeaps");
        assert_eq!(run_end_name(run_end_index(RunEnd::Call(32)), false), "SetDescriptorHeaps");
        assert_eq!(run_end_name(run_end_index(RunEnd::Call(26)), false), "ResourceBarrier");
        assert_eq!(run_end_name(run_end_index(RunEnd::Call(50)), false), "OMSetRenderTargets");
        assert_eq!(run_end_name(run_end_index(RunEnd::Call(12)), false), "DrawInstanced");
        assert_eq!(run_end_name(run_end_index(RunEnd::Call(29)), false), "EndQuery");
        assert_eq!(run_end_name(run_end_index(RunEnd::Call(93)), false), "console list method #93");
        assert_eq!(run_end_name(run_end_index(RunEnd::Call(93)), true), "console list slot #93 (newer layout)");
        assert_eq!(run_end_name(run_end_index(RunEnd::Pipeline), false), "SetPipelineState to a pipeline that is not two-pass");
        assert_eq!(run_end_name(run_end_index(RunEnd::Close), false), "Close");
    }

    fn dummy_draw() -> DeferredDraw {
        DeferredDraw {
            snap: snap(0, None),
            _tp: Arc::new(super::super::real::TwoPassPso { compute: 0, compute_rs: 0, mesh_rs: 0, record: Default::default(), mesh_groups_per_record: 1, kind: "MS", one_pass: false }),
            _mesh_pso: None,
            _block: None,
            raw_pso: 0,
            kind: "MS",
            size: [1, 1, 1],
        }
    }

    /// A run of N DispatchMesh draws, the title's root arguments set between them (an io block
    /// each): every compute pass, then one ResourceBarrier call (a UAV barrier on the records),
    /// then every mesh pass in the title's order; the root signatures and pipelines set once,
    /// each pass with all its root views. After a run whose mesh passes may still read the
    /// records, one barrier on them first.
    #[test]
    fn a_run_of_deferred_draws_records_its_compute_passes_then_one_barrier_then_its_mesh_passes() {
        let n = 5;
        let draws: Vec<DrawSnap> = (0..n).map(|k| snap(k, None)).collect();
        let cmds = batch_commands(&steps(&draws), 0x900, false);
        let mut want = vec![Cmd::TimesBegin(None), Cmd::ComputeRs(0xC0), Cmd::Pso(0xC1)];
        for d in &draws {
            want.extend(d.compute.views.iter().enumerate().map(|(k, &va)| Cmd::ComputeView(k as u32, va)));
            let DeferredWork::Mesh { groups, .. } = d.work else { unreachable!() };
            want.push(Cmd::Dispatch(groups));
        }
        want.extend([Cmd::TimesEnd, Cmd::Barriers { all: false, uav: Some(0x900), transitions: vec![] }]);
        for (i, d) in draws.iter().enumerate() {
            want.push(Cmd::TimesBegin(Some(i)));
            if i == 0 {
                want.extend([Cmd::GraphicsRs(0xA0), Cmd::Pso(0xA1)]);
            }
            want.extend(d.mesh.views.iter().enumerate().map(|(k, &va)| Cmd::GraphicsView(k as u32, va)));
            let DeferredWork::Mesh { mesh_groups, .. } = d.work else { unreachable!() };
            want.extend([Cmd::DispatchMesh(mesh_groups), Cmd::TimesEnd]);
        }
        assert_eq!(cmds, want);
        let barriers = |c: &[Cmd]| c.iter().filter(|c| matches!(c, Cmd::Barriers { .. })).count();
        assert_eq!(barriers(&cmds), 1, "one ResourceBarrier call for the run");
        let after_open = batch_commands(&steps(&draws), 0x900, true);
        assert_eq!(after_open[0], Cmd::Barriers { all: false, uav: Some(0x900), transitions: vec![] }, "the last run's mesh passes first");
        assert_eq!(after_open[1..], cmds[..]);
    }

    /// ExecuteIndirect draws in a run: one ResourceBarrier call before the argument rewrites (a
    /// UAV barrier on all resources and each argument block, once, from the state it was in, to
    /// UNORDERED_ACCESS), the rewrites, one call taking the blocks to INDIRECT_ARGUMENT, the
    /// compute passes, one barrier on the records, the mesh passes.
    #[test]
    fn a_run_of_deferred_execute_indirects_rewrites_every_argument_first() {
        use D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT as ARGS;
        use D3D12_RESOURCE_STATE_UNORDERED_ACCESS as UAV;
        let mut draws: Vec<DrawSnap> = vec![snap(0, Some(0xB1)), snap(1, None), snap(2, Some(0xB1)), snap(3, Some(0xB2))];
        if let DeferredWork::Indirect { before, .. } = &mut draws[2].work {
            // (a later draw in the block found it left in INDIRECT_ARGUMENT by the first)
            *before = ARGS;
        }
        let cmds = batch_commands(&steps(&draws), 0x900, true);
        let barriers: Vec<&Cmd> = cmds.iter().filter(|c| matches!(c, Cmd::Barriers { .. })).collect();
        assert_eq!(
            barriers,
            [
                &Cmd::Barriers { all: true, uav: None, transitions: vec![(0xB1, D3D12_RESOURCE_STATE_COMMON, UAV), (0xB2, D3D12_RESOURCE_STATE_COMMON, UAV)] },
                &Cmd::Barriers { all: false, uav: None, transitions: vec![(0xB1, UAV, ARGS), (0xB2, UAV, ARGS)] },
                &Cmd::Barriers { all: false, uav: Some(0x900), transitions: vec![] },
            ]
        );
        let at = |c: &Cmd| cmds.iter().position(|x| x == c).unwrap();
        let c = calls(&cmds);
        // three rewrites, four compute passes, four mesh passes, in that order
        let kinds: Vec<u8> = c.iter().map(|x| if x.pso == 0xE1 { 0 } else if x.pso == 0xC1 { 1 } else { 2 }).collect();
        assert_eq!(kinds, [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
        assert!(at(barriers[1]) > at(&Cmd::Dispatch(1)) && at(barriers[1]) < at(&Cmd::Indirect(0xD0, 0xB1, 0)));
    }

    /// Each draw of a run runs every call – the rewrite, the compute pass, the mesh pass – with the
    /// pipeline, root signature and root views the unbatched path gives it, and its mesh passes
    /// keep the title's order.
    #[test]
    fn deferred_draws_run_with_what_the_unbatched_path_gives_them() {
        let draws: Vec<DrawSnap> = (0..6).map(|k| snap(k, (k % 2 == 1).then_some(0xB0 + k as usize / 4))).collect();
        let batched = calls(&batch_commands(&steps(&draws), 0x900, false));
        let alone: Vec<Vec<Call>> = draws.iter().map(|d| calls(&unbatched_commands(d))).collect();
        let indirect: Vec<usize> = (0..draws.len()).filter(|&k| matches!(draws[k].work, DeferredWork::Indirect { .. })).collect();
        let rewrites: Vec<Call> = indirect.iter().map(|&k| alone[k][0].clone()).collect();
        let compute: Vec<Call> = alone.iter().map(|c| c[c.len() - 2].clone()).collect();
        let mesh: Vec<Call> = alone.iter().map(|c| c[c.len() - 1].clone()).collect();
        let want: Vec<Call> = rewrites.into_iter().chain(compute).chain(mesh).collect();
        assert_eq!(batched, want);
    }

    /// A run goes on only across the title's root signature and root argument sets, the calls
    /// that record nothing, and DispatchMesh / ExecuteIndirect (at their slots of the title's
    /// layout): every other title call records it first. Calls from XCL or the runtime, and calls
    /// on a list with nothing deferred, do not.
    #[test]
    fn a_title_call_records_the_deferred_draws_first_unless_it_cannot_change_them() {
        for newer in [false, true] {
            let kept: Vec<usize> = (0..600).filter(|&s| keeps_two_pass_batch(s, newer)).collect();
            let (ei, dm) = if newer { (61, 129) } else { (59, 127) };
            let want: Vec<usize> = (0..=8).chain([25]).chain(33..=46).chain([ei, dm]).collect();
            assert_eq!(kept, want, "newer layout: {newer}");
            // pipelines, barriers, draws, dispatches, copies, clears, IA / RS / OM state, heaps,
            // queries, predication, bundles, markers, Close, Reset, the console's own methods
            for slot in [26, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 9, 10, 11, 60, 67, 78, 81, 85, 93, 104, 105, 106, 124, 125, 139, 146, 148] {
                assert!(flush_wanted(slot, newer, true, true), "slot {slot}, newer layout: {newer}");
            }
            // the other layout's DispatchMesh / ExecuteIndirect slots are other methods there
            let (other_ei, other_dm) = if newer { (59, 127) } else { (61, 129) };
            assert!(flush_wanted(other_ei, newer, true, true) && flush_wanted(other_dm, newer, true, true));
            assert!(!flush_wanted(ei, newer, true, true) && !flush_wanted(36, newer, true, true));
            assert!(!flush_wanted(25, newer, false, true), "XCL's own calls");
            assert!(!flush_wanted(25, newer, true, false), "nothing deferred");
        }
    }

    /// A two-pass draw is deferred only on a direct list, without a capture, as one command, with
    /// records that fit a batch record buffer; otherwise the run is recorded and it goes at once.
    #[test]
    fn only_single_two_pass_draws_on_direct_lists_are_deferred() {
        let direct = Some(D3D12_COMMAND_LIST_TYPE_DIRECT);
        let mib = 1u64 << 20;
        assert_eq!(defer_check(true, direct, false, 1, 32 * mib, 128 * mib), Ok(()));
        assert_eq!(defer_check(true, direct, false, 1, 128 * mib, 128 * mib), Ok(()));
        assert_eq!(defer_check(false, direct, false, 1, mib, 128 * mib), Err(NoDefer::Off));
        for ty in [Some(D3D12_COMMAND_LIST_TYPE_BUNDLE), Some(D3D12_COMMAND_LIST_TYPE_COMPUTE), None] {
            assert_eq!(defer_check(true, ty, false, 1, mib, 128 * mib), Err(NoDefer::NotDirect));
        }
        assert_eq!(defer_check(true, direct, true, 1, mib, 128 * mib), Err(NoDefer::Capture));
        for commands in [0, 2, 64] {
            assert_eq!(defer_check(true, direct, false, commands, mib, 128 * mib), Err(NoDefer::Commands));
        }
        assert_eq!(defer_check(true, direct, false, 1, 128 * mib + 256, 128 * mib), Err(NoDefer::TooBig));
    }

    /// Each deferred draw has a region of its own in the batch record buffer, one after the other
    /// (256-aligned, as much as the draw alone would have); one that does not fit after them has
    /// the run recorded first and starts again at the start. An ExecuteIndirect keeps the groups
    /// it had alone.
    #[test]
    fn deferred_draws_take_regions_of_their_own_until_the_buffer_is_full() {
        let mib = 1u64 << 20;
        assert_eq!(batch_region_bytes(2 * 47_632), 95_488);
        assert_eq!(batch_region_bytes(0), 256);
        let (mut used, mut at) = (0, Vec::new());
        for need in [95_264u64, 32 * mib, 32 * mib, 32 * mib, 32 * mib] {
            let region = batch_region_bytes(need);
            let place = batch_place(used, region, 128 * mib);
            at.push(place);
            used = place.unwrap_or(0) + region;
        }
        assert_eq!(at, [Some(0), Some(95_488), Some(95_488 + 32 * mib), Some(95_488 + 64 * mib), None], "the fifth after a recorded run");
        assert_eq!(batch_place(96 * mib, 32 * mib, 128 * mib), Some(96 * mib), "up to the end");
        assert_eq!(batch_place(u64::MAX - 8, 256, 128 * mib), None);
        for record in [64u64, 4_096, 47_632, 300_000] {
            for split in [1u32, 3] {
                let need = indirect_record_need(record, split);
                assert_eq!(record_groups(batch_region_bytes(need), record, split), record_groups(record_buffer_bytes(need), record, split), "{record} B, split {split}");
            }
        }
    }

    /// A recording that ends with deferred draws (a Reset with no Close, a list let go) drops them,
    /// the list is no longer counted as deferring, and its batch record buffer goes with its others.
    #[test]
    fn a_recording_that_ends_drops_its_deferred_draws() {
        let list = 0x7500_0031usize as *mut c_void;
        let draw = || DeferredDraw {
            snap: snap(0, None),
            _tp: Arc::new(super::super::real::TwoPassPso { compute: 0, compute_rs: 0, mesh_rs: 0, record: Default::default(), mesh_groups_per_record: 1, kind: "MS", one_pass: false }),
            _mesh_pso: None,
            _block: None,
            raw_pso: 0,
            kind: "MS",
            size: [1, 1, 1],
        };
        let before = DEFERRING_LISTS.load(Ordering::Relaxed);
        with(list, |s| s.deferred.extend([RunItem::Draw(draw()), RunItem::Draw(draw())]));
        mark_deferring(list);
        mark_deferring(list);
        assert!(DEFERRING.contains(list as usize));
        reset(list, 0, 0);
        assert!(!DEFERRING.contains(list as usize));
        assert!(with(list, |s| s.deferred.is_empty() && s.batch_buf.is_none() && !s.batch_open));
        with(list, |s| s.deferred.push(RunItem::Draw(draw())));
        mark_deferring(list);
        forget(list);
        assert!(!DEFERRING.contains(list as usize));
        // (other tests' lists may come and go meanwhile; this one is counted once and gone)
        assert!(DEFERRING_LISTS.load(Ordering::Relaxed) <= before + 1);
    }

    #[test]
    fn gpu_two_pass_argument_scratch_is_shared_and_capped() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        let mut dev: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev) }.is_err() {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        let dev = dev.unwrap();
        let list = 0x7500_0001usize as *mut c_void;
        unsafe {
            // small rewrites share one block
            let (a, at_a) = scratch_args(list, &dev, 100).unwrap();
            // (a rewrite left it in INDIRECT_ARGUMENT, `two_pass_batch`)
            with(list, |s| s.scratch_in_args = true);
            let (b, at_b) = scratch_args(list, &dev, 300).unwrap();
            assert_eq!((a.as_raw(), at_a), (b.as_raw(), 0));
            assert_eq!(at_b, 256);
            assert!(with(list, |s| s.scratch_in_args), "the same block: in the state it was left in");
            // a recording takes at most SCRATCH_CAP of blocks
            let mut taken = 1;
            while scratch_args(list, &dev, SCRATCH_BLOCK).is_ok() {
                taken += 1;
                assert!(taken <= SCRATCH_CAP / SCRATCH_BLOCK, "past the cap");
                assert!(!with(list, |s| s.scratch_in_args), "a new block is in COMMON");
            }
            assert!(with(list, |s| s.scratch_taken) <= SCRATCH_CAP);
            // the next recording starts again
            reset(list, 0, 0);
            assert!(scratch_args(list, &dev, SCRATCH_BLOCK).is_ok());
        }
        forget(list);
    }

    #[test]
    fn gpu_indirect_two_pass_commands_past_the_count_get_no_groups() {
        let commands = [[1, 3, 1, 1], [2, 2, 1, 1], [3, 4, 1, 1]];
        let Some((compute, mesh)) = scale_args_counted(&commands, 2, 1000, Some(2)) else { return };
        assert_eq!(compute, [[1, 3, 1, 1], [2, 2, 1, 1], [3, 0, 1, 1]]);
        assert_eq!(mesh, [[1, 6, 1, 1], [2, 4, 1, 1], [3, 0, 1, 1]]);
        let Some((compute, _)) = scale_args_counted(&commands, 2, 1000, Some(7)) else { return };
        assert_eq!(compute, [[1, 3, 1, 1], [2, 2, 1, 1], [3, 4, 1, 1]]);
    }

    /// Commands of two-pass draws run in `two_pass_plan` order on the GPU (`run_two_pass_plan`),
    /// with records whole and in halves (`XCL_TWO_PASS_BATCH`), with stand-ins for the two passes:
    /// the "compute pass" writes each group's record (from the start of its region) with its
    /// command's constant, the "mesh pass" copies the records it reads out by command. One draw of
    /// three commands, then three draws of one, each mesh pass with no barrier after it in halves.
    #[test]
    fn gpu_two_pass_indirect_commands_keep_their_own_records() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        let mut dev: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev) }.is_err() {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        let dev = dev.unwrap();
        gpu_two_pass_records(&dev, Records::Whole);
        gpu_two_pass_records(&dev, Records::Halves);
    }

    fn gpu_two_pass_records(dev: &ID3D12Device, records: Records) {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        let src = r#"
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0), UAV(u1)"
cbuffer C : register(b0) { uint v; };
RWByteAddressBuffer rec : register(u0);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void cs(uint3 g : SV_GroupID) { rec.Store(g.x * 4u, v); }
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void ms(uint3 g : SV_GroupID) { o.Store((v * 16u + g.x) * 4u, rec.Load(g.x * 4u)); }
"#;
        // (constant, groups) of each command
        let commands = [(1u32, 3u32), (2, 2), (3, 4), (4, 1), (5, 4), (6, 2)];
        unsafe {
            let cs = crate::gpu::recomp::dxc::compile(src, "cs", "cs_6_6", &[]).unwrap();
            let ms = crate::gpu::recomp::dxc::compile(src, "ms", "cs_6_6", &[]).unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &cs).unwrap();
            let pso = |dxil: &[u8]| -> ID3D12PipelineState {
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&rs),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
                    ..Default::default()
                };
                dev.CreateComputePipelineState(&desc).unwrap()
            };
            let (pass1, pass2) = (pso(&cs), pso(&ms));
            let args = [
                D3D12_INDIRECT_ARGUMENT_DESC {
                    Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
                    Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 { Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 { RootParameterIndex: 0, DestOffsetIn32BitValues: 0, Num32BitValuesToSet: 1 } },
                },
                D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() },
            ];
            let mut sig: Option<ID3D12CommandSignature> = None;
            dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 16, NumArgumentDescs: 2, pArgumentDescs: args.as_ptr(), NodeMask: 0 }, &rs, &mut sig).unwrap();
            let sig = sig.unwrap();
            let buffer = |bytes: u64, cpu: bool| -> ID3D12Resource {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: bytes,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    ..Default::default()
                };
                let heap = if cpu {
                    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() }
                } else {
                    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }
                };
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                r.unwrap()
            };
            let (indirect, rec, out) = (buffer(256, true), buffer(256, false), buffer(1024, true));
            let map = |b: &ID3D12Resource| -> *mut u32 {
                let mut p = std::ptr::null_mut();
                b.Map(0, None, Some(&mut p)).unwrap();
                p as *mut u32
            };
            let (a, o) = (map(&indirect), map(&out));
            for (k, &(v, x)) in commands.iter().enumerate() {
                std::ptr::copy_nonoverlapping([v, x, 1, 1].as_ptr(), a.add(4 * k), 4);
            }
            std::ptr::write_bytes(o, 0, 256);
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, &pass1).unwrap();
            emit_transition(&list, &indirect, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
            list.SetComputeRootSignature(&rs);
            list.SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
            // (the recording's state: a key of its own)
            let this = 0x7500_0021usize as *mut c_void;
            let draws: [&[u64]; 4] = [&[0, 1, 2], &[3], &[4], &[5]];
            for draw in draws {
                let plan = two_pass_plan(draw.len() as u32, records, with(this, |s| s.record_open));
                // (the 256 B of records in halves of 128 B: 4 groups at most)
                run_two_pass_plan(this, &list, &plan, &rec, rec.GetGPUVirtualAddress(), 128, true, |k, compute, va| {
                    list.SetComputeRootUnorderedAccessView(1, va);
                    list.SetPipelineState(if compute { &pass1 } else { &pass2 });
                    list.ExecuteIndirect(&sig, 1, &indirect, draw[k as usize] * 16, None::<&ID3D12Resource>, 0);
                    Ok(())
                })
                .unwrap();
            }
            assert_eq!(with(this, |s| s.record_open).is_some(), records == Records::Halves, "a mesh pass left with no barrier after it");
            forget(this);
            emit_transition(&list, &indirect, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            queue.Signal(&done, 1).unwrap();
            crate::gpu::fence::wait(&done, 1, dev, None, 10_000).unwrap();
            for &(v, x) in &commands {
                let got: Vec<u32> = (0..x).map(|g| *o.add((v * 16 + g) as usize)).collect();
                assert_eq!(got, vec![v; x as usize], "the records the mesh pass of command {v} read ({records:?})");
            }
            indirect.Unmap(0, None);
            out.Unmap(0, None);
        }
    }

    /// (local probe) The same on a direct or a compute queue, with the title-like signature
    /// (constant + dispatch: vkd3d-proton's state template) or a plain one (dispatch only, the
    /// constant set by the CPU): what each command's mesh pass read, three times each.
    #[test]
    fn gpu_two_pass_indirect_probe() {
        for (name, ty, plain) in [
            ("compute queue, constant in the signature", D3D12_COMMAND_LIST_TYPE_COMPUTE, false),
            ("direct queue, constant in the signature", D3D12_COMMAND_LIST_TYPE_DIRECT, false),
            ("compute queue, plain signature", D3D12_COMMAND_LIST_TYPE_COMPUTE, true),
            ("direct queue, plain signature", D3D12_COMMAND_LIST_TYPE_DIRECT, true),
        ] {
            for _ in 0..3 {
                println!("PROBE {name}: {:?}", two_pass_indirect_probe(ty, plain));
            }
        }
    }

    fn two_pass_indirect_probe(ty: D3D12_COMMAND_LIST_TYPE, plain: bool) -> Vec<Vec<u32>> {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        let src = r#"
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0), UAV(u1)"
cbuffer C : register(b0) { uint v; };
RWByteAddressBuffer rec : register(u0);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void cs(uint3 g : SV_GroupID) { rec.Store(g.x * 4u, v); }
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void ms(uint3 g : SV_GroupID) { o.Store((v * 16u + g.x) * 4u, rec.Load(g.x * 4u)); }
"#;
        // (constant, groups) of each command
        let commands = [(1u32, 3u32), (2, 2), (3, 4)];
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                return Vec::new();
            }
            let dev = dev.unwrap();
            let cs = crate::gpu::recomp::dxc::compile(src, "cs", "cs_6_6", &[]).unwrap();
            let ms = crate::gpu::recomp::dxc::compile(src, "ms", "cs_6_6", &[]).unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &cs).unwrap();
            let pso = |dxil: &[u8]| -> ID3D12PipelineState {
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&rs),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
                    ..Default::default()
                };
                dev.CreateComputePipelineState(&desc).unwrap()
            };
            let (pass1, pass2) = (pso(&cs), pso(&ms));
            let args = [
                D3D12_INDIRECT_ARGUMENT_DESC {
                    Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
                    Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 { Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 { RootParameterIndex: 0, DestOffsetIn32BitValues: 0, Num32BitValuesToSet: 1 } },
                },
                D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() },
            ];
            let mut sig: Option<ID3D12CommandSignature> = None;
            if plain {
                dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 16, NumArgumentDescs: 1, pArgumentDescs: args[1..].as_ptr(), NodeMask: 0 }, None::<&ID3D12RootSignature>, &mut sig).unwrap();
            } else {
                dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 16, NumArgumentDescs: 2, pArgumentDescs: args.as_ptr(), NodeMask: 0 }, &rs, &mut sig).unwrap();
            }
            let sig = sig.unwrap();
            let buffer = |bytes: u64, cpu: bool| -> ID3D12Resource {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: bytes,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    ..Default::default()
                };
                let heap = if cpu {
                    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() }
                } else {
                    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }
                };
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                r.unwrap()
            };
            let (indirect, rec, out) = (buffer(256, true), buffer(256, false), buffer(1024, true));
            let map = |b: &ID3D12Resource| -> *mut u32 {
                let mut p = std::ptr::null_mut();
                b.Map(0, None, Some(&mut p)).unwrap();
                p as *mut u32
            };
            let (a, o) = (map(&indirect), map(&out));
            for (k, &(v, x)) in commands.iter().enumerate() {
                std::ptr::copy_nonoverlapping([v, x, 1, 1].as_ptr(), a.add(4 * k), 4);
            }
            std::ptr::write_bytes(o, 0, 256);
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, &pass1).unwrap();
            emit_transition(&list, &indirect, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
            list.SetComputeRootSignature(&rs);
            list.SetComputeRootUnorderedAccessView(1, rec.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
            for (k, compute) in two_pass_steps(commands.len() as u32) {
                list.SetPipelineState(if compute { &pass1 } else { &pass2 });
                if plain {
                    list.SetComputeRoot32BitConstant(0, commands[k as usize].0, 0);
                    list.ExecuteIndirect(&sig, 1, &indirect, k as u64 * 16 + 4, None::<&ID3D12Resource>, 0);
                } else {
                    list.ExecuteIndirect(&sig, 1, &indirect, k as u64 * 16, None::<&ID3D12Resource>, 0);
                }
                emit_uav_barrier(&list, Some(&rec));
            }
            emit_transition(&list, &indirect, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            queue.Signal(&done, 1).unwrap();
            crate::gpu::fence::wait(&done, 1, &dev, None, 10_000).unwrap();
            let got: Vec<Vec<u32>> = commands.iter().map(|&(v, x)| (0..x).map(|g| *o.add((v * 16 + g) as usize)).collect()).collect();
            indirect.Unmap(0, None);
            out.Unmap(0, None);
            got
        }
    }

    /// With heap windows the two-pass programs read the memory windows from the shared heap
    /// (`pso::mem_windows`): both passes' binds set it on a list that has none, as `bind_abi` does.
    /// (Heap windows are opt-in: run with `XCL_HEAP_WINDOWS=1`, else skipped.)
    #[test]
    fn two_pass_binds_set_the_shared_view_heap() {
        use crate::gpu::recomp::runner::skip_test;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        if !crate::gpu::recomp::pso::heap_windows() {
            skip_test("no heap windows (opt-in: XCL_HEAP_WINDOWS=1)");
            return;
        }
        #[repr(C)]
        struct FakeList {
            vt: *const usize,
            calls: Mutex<Vec<Vec<usize>>>,
        }
        unsafe extern "system" fn fake_set_heaps(this: *mut c_void, n: u32, heaps: *const usize) {
            let v = if heaps.is_null() { Vec::new() } else { std::slice::from_raw_parts(heaps, n as usize).to_vec() };
            (*(this as *const FakeList)).calls.lock().unwrap().push(v);
        }
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            // the shared heap with a window view, unless the backend made it already
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 1 << 16,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut window: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut window).unwrap();
            let window = window.unwrap();
            if super::super::console_heap::global_heap().is_none() && !super::super::console_heap::set_window_views(&dev, &[(window.as_raw() as usize, 1 << 16)]) {
                skip_test("no shared heap");
                return;
            }
            let g = super::super::console_heap::global_heap().unwrap().as_raw() as usize;
            let mut orig_vt = [0usize; 64];
            orig_vt[28] = fake_set_heaps as usize;
            let mut ext_vt = vec![0usize; 65];
            ext_vt[0] = orig_vt.as_ptr() as usize;
            ext_vt[1..].copy_from_slice(&orig_vt);
            let fake = Box::new(FakeList { vt: ext_vt.as_ptr().add(1), calls: Mutex::new(Vec::new()) });
            let this = &*fake as *const FakeList as *mut c_void;
            let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this).unwrap();
            // no root signatures: the bind stops before it records anything else
            let tp = super::super::real::TwoPassPso { compute: 0, compute_rs: 0, mesh_rs: 0, record: Default::default(), mesh_groups_per_record: 1, kind: "MS", one_pass: false };
            for compute_pass in [true, false] {
                super::super::list_x::set_descriptor_heaps_x(this, 0, std::ptr::null());
                assert!(super::super::list_x::bound_heaps(this, false).is_empty());
                assert!(bind_two_pass_stage(this, list, 0, &tp, 0, compute_pass).is_err());
                // (the shared sampler heap goes with it in the same call when there is one:
                // `list_x::ensure_view_heap`; the bind may have made it)
                let mut expected = vec![g];
                if let Some(s) = super::super::console_heap::global_sampler_heap() {
                    expected.push(s.as_raw() as usize);
                }
                assert_eq!(super::super::list_x::bound_heaps(this, false), expected, "compute pass {compute_pass}: the shared heap is bound");
                assert_eq!(fake.calls.lock().unwrap().last(), Some(&expected), "compute pass {compute_pass}");
            }
            super::super::list_x::bound_heaps(this, true);
            forget(this);
        }
    }


    /// A reusable block must fit the recording's remaining budget, not just its request.
    #[test]
    fn gpu_two_pass_pooled_scratch_respects_remaining_budget() {
        use windows::Win32::Graphics::{Direct3D::D3D_FEATURE_LEVEL_12_0, Dxgi::*};
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
            let wanted = std::env::var("XCL_ADAPTER").unwrap_or_default();
            let mut chosen = None;
            for i in 0.. {
                let Ok(adapter) = factory.EnumAdapters1(i) else { break };
                let desc = adapter.GetDesc1().unwrap();
                let end = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
                let name = String::from_utf16_lossy(&desc.Description[..end]);
                if !wanted.is_empty() && !name.contains(&wanted) { continue; }
                let mut dev: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut dev).is_ok() {
                    chosen = Some((name, dev.unwrap()));
                    break;
                }
            }
            let Some((name, dev)) = chosen else {
                crate::gpu::recomp::runner::skip_test("no selected D3D12 adapter");
                return;
            };
            println!("POOLED_SCRATCH_ADAPTER={name}");
            let key = dev.as_raw() as usize;
            // These are state keys only; scratch_args does not call through a list pointer.
            let (consumer, producer, empty) = (0x7500_1011usize as *mut c_void, 0x7500_1012usize as *mut c_void, 0x7500_1013usize as *mut c_void);
            let allocator = 0x7500_1014usize;
            let first = scratch_args(consumer, &dev, SCRATCH_BLOCK).unwrap();
            assert_eq!(with(consumer, |s| s.scratch_taken), SCRATCH_BLOCK);
            reset(producer, allocator, 0);
            let large = scratch_args(producer, &dev, SCRATCH_CAP).unwrap();
            assert_eq!(large.0.GetDesc().Width, SCRATCH_CAP);
            reset(producer, 0, 0);
            release_allocator_res(allocator);
            let second = scratch_args(consumer, &dev, SCRATCH_BLOCK).unwrap();
            let taken = with(consumer, |s| s.scratch_taken);
            println!("POOLED_SCRATCH taken={taken} second_bytes={} cap={SCRATCH_CAP}", second.0.GetDesc().Width);
            // If it was too large for consumer, the cached block remains reusable by an empty recording.
            let reused = if taken <= SCRATCH_CAP {
                let r = scratch_args(empty, &dev, SCRATCH_CAP).unwrap();
                assert_eq!(r.0.as_raw(), large.0.as_raw());
                assert_eq!(with(empty, |s| s.scratch_taken), SCRATCH_CAP);
                assert!(scratch_args(empty, &dev, 256).is_err());
                Some(r)
            } else { None };
            drop((first, second, large, reused));
            for list in [consumer, producer, empty] { forget(list); }
            while FREE_RES.lock().unwrap().take(key, 0, u64::MAX).is_some() {}
            assert!(taken <= SCRATCH_CAP, "pooled block exceeds recording budget: {taken} > {SCRATCH_CAP}");
        }
    }
}
