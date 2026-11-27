//! The title's own memory, seen by the GPU: sparse windows.
//!
//! A console title keeps GPU data in memory it reserves and commits itself (XDefiant: one
//! 8 GB reservation of 64 KiB pages it commits in steps and places buffers in) – on the
//! console all of it is GPU-visible at the same addresses. XCL gives such reservations
//! addresses in the *sparse windows* of the unified-memory block (`gpu::uma`: windows 0..2
//! in `for_runtime`), so translated shaders reach them through `mem`…`mem2` like any unified
//! memory. On the GPU each sparse window is a reserved (tiled) buffer of 4 GiB whose 64 KiB
//! tiles are mapped onto video-memory heaps as the title commits pages.
//! The CPU's view stays the title's memory: what the CPU wrote (write-watch) is copied into
//! the tiles before the title's next ExecuteCommandLists via an asynchronous ring-buffered
//! staging upload queue, and the title's queues GPU-wait on the upload fence (no CPU stall).
//! Only the bytes the CPU changed go (64-byte chunks against a hash of what was sent last,
//! `changed_chunks`): the rest of a written page may hold what the GPU wrote there (counters,
//! indirect arguments) and must not be overwritten with the CPU's stale view of it.
//! Buffers the title places there get reserved resources over the same tiles, so PC-side
//! copies, views and barriers see the same memory.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ffi::c_void;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_DECOMMIT, PAGE_GUARD,
    PAGE_NOACCESS, PAGE_READWRITE,
};

const TILE: usize = 0x1_0000;
/// tiles of a video-memory pool heap (64 MiB)
const POOL_TILES: u32 = 1024;
/// tiles decommitted while no title work was submitted that stay mapped (`State::parked`) before
/// they are unmapped together: 256 MB of video memory
const PARKED_MAX: usize = 4096;
/// tiles waiting for their mapping call (`State::pending`) before they are sent at once: 256 MB
const PENDING_MAX: usize = 4096;
/// a pool tile freed is not reused before this (`State::cooling`): work the title submitted
/// before its unmap may still read it on the title's queues, and the sparse queue that maps,
/// clears and fills it for its next address does not wait for those
const COOL: std::time::Duration = std::time::Duration::from_secs(2);
/// bytes of the ring-buffered upload staging buffer
const STAGING: usize = 64 << 20;
/// alignment of sub-allocations inside the upload staging ring (write-combined cacheline)
const RING_ALIGN: usize = 64;
/// number of in-flight upload command allocators / lists
const UPLOAD_SLOTS: usize = 16;
/// bytes of the dedicated pre-zeroed buffer for tile/hole clears (1 MiB)
const ZERO_BUF: usize = 1 << 20;
/// bytes of the GPU→CPU readback staging buffer
const READBACK_STAGING: usize = 16 << 20;

/// changed runs of fewer bytes go up together in a compute dispatch (`scatter`) when there are
/// this many of them, of this many chunks at most (4 MiB, so the ring fills in pieces)
const SCATTER_RUN: usize = 4 << 10;
const SCATTER_FROM: usize = 16;
const SCATTER_CHUNKS: usize = 1 << 16;

const PAGE: usize = 0x1000;
const PAGES_PER_TILE: usize = TILE / PAGE;
const CHUNK: usize = super::scatter::CHUNK;
const CHUNKS: usize = PAGE / CHUNK;

/// Fast non-cryptographic integer hasher for page/tile lookup maps on the hot submit path.
#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        }
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn finish(&self) -> u64 {
        // Tile addresses are 64 KiB aligned. Move mixed bits into the low bucket-index bits.
        self.0.rotate_right(23)
    }
}

type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
type FxHashSet<K> = std::collections::HashSet<K, BuildHasherDefault<FxHasher>>;

struct TileEntry {
    pool: (u32, u32),
    /// bit `i` set => 4 KiB page `i` of this 64 KiB tile is uncommitted (a hole)
    holes: u16,
    /// per-4 KiB-page hashes of the 64-byte chunks last sent to the GPU (`None` = all zero)
    sent: [Option<Box<[u64; CHUNKS]>>; PAGES_PER_TILE],
    /// 0: the bytes are at the tile's own address. Else the 64 KiB-aligned CPU address they are
    /// read from and land at (a tile of a GPU view, `State::mirrors`); `holes` are then the
    /// uncommitted pages there. The tile's own address is no CPU memory: the title reserved it
    /// with no access, as the console's GPU-only view of the same pages
    src: usize,
    /// what the CPU memory of a tile over a CPU view is: new for each such entry (`next_gen`), and
    /// again whenever a page of it becomes a hole. A copy back or an immediate recorded with
    /// another lands nowhere: the memory it was for is gone (decommitted and committed again, or
    /// another CPU view shown). 0: a tile of the window's own, whose memory is its address's
    gen: u32,
}

/// The mapped tiles of the windows' own (not GPU views), and how many of them hold video memory
/// no CPU write was ever sent to: every committed page still has no `sent` hashes. Such a tile
/// holds zeros unless the GPU wrote to it.
fn never_sent(mapped: &FxHashMap<usize, TileEntry>) -> (usize, usize) {
    let (mut own, mut unsent) = (0, 0);
    for e in mapped.values().filter(|e| e.src == 0 && e.pool != LAZY) {
        own += 1;
        if e.sent.iter().all(|p| p.is_none()) {
            unsent += 1;
        }
    }
    (own, unsent)
}

/// `TileEntry::pool` of a tile the title committed that has no video memory yet
/// (`XCL_SPARSE_LAZY=1`): it gets a pool tile once the CPU's first bytes go up to it or a
/// placed buffer or a copy back names it (`State::materialize`). Until then the window reads
/// zeros there, as the title's freshly committed pages do.
const LAZY: (u32, u32) = (u32::MAX, u32::MAX);

/// `XCL_SPARSE_LAZY=1`: pages a title commits in a sparse window get their video memory at
/// their first use, not at the commit. Off by default: a compute program that writes through a
/// window to memory no buffer was placed on would write nothing there.
fn lazy_commits() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::LAZY_COMMITS_FOR_TEST.with(|c| c.get()) {
        return on;
    }
    lazy_asked() && !LAZY_REFUSED.load(Ordering::Relaxed)
}

/// `XCL_SPARSE_LAZY=1` was given
fn lazy_asked() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SPARSE_LAZY").is_ok_and(|v| v.trim() == "1"))
}

/// `XCL_SPARSE_LAZY=1` on a device whose unmapped tiles do not read zeros (`null_tiles_read_zero`):
/// commits map at once, as without it
static LAZY_REFUSED: AtomicBool = AtomicBool::new(false);

/// Whether a tile of a reserved buffer with no video memory reads zeros and takes writes that go
/// nowhere on `dev` (tiled resources tier 2). On tier 1 a touch of one is undefined.
unsafe fn null_tiles_read_zero(dev: &ID3D12Device) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.TiledResourcesTier.0 >= D3D12_TILED_RESOURCES_TIER_2.0
}

/// The parts of `runs` (address, bytes) that lie in tiles `keep` holds for, cut where a tile
/// ends; neighbours left together are one run.
fn runs_where(runs: &[(usize, usize)], mut keep: impl FnMut(usize) -> bool) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for &(a, n) in runs {
        let end = a.saturating_add(n);
        let mut at = a;
        while at < end {
            let t = at & !(TILE - 1);
            let stop = t.saturating_add(TILE).min(end);
            if keep(t) {
                push_run(&mut out, at, stop - at);
            }
            at = stop;
        }
    }
    out
}

/// the last `TileEntry::gen` given out
static TILE_GEN: AtomicU32 = AtomicU32::new(0);

/// A `TileEntry::gen` for a tile over a CPU view (never 0)
fn next_gen() -> u32 {
    TILE_GEN.fetch_add(1, Ordering::Relaxed) % u32::MAX + 1
}

impl TileEntry {
    fn with_holes(pool: (u32, u32), holes: u16) -> Self {
        Self { pool, holes, sent: std::array::from_fn(|_| None), src: 0, gen: 0 }
    }

    /// The CPU address of the tile at `t` (this entry's tile)
    #[inline]
    fn cpu(&self, t: usize) -> usize {
        if self.src == 0 {
            t
        } else {
            self.src
        }
    }
}

/// The hole masks (`query_tile_holes`) of the 64 KiB tiles `tiles` of the title's CPU memory, in
/// their order. A page in the working set is committed: one query asks that of all their pages.
/// The others go to VirtualQuery, once for each region it reports: it walks a region to its end,
/// so a call a tile over a long committed range grew with the square of the range.
unsafe fn source_holes(tiles: &[usize]) -> Vec<u16> {
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    /// tiles a working-set query asks for
    const BATCH: usize = 256;
    let mut out = Vec::with_capacity(tiles.len());
    let mut ws: Vec<PSAPI_WORKING_SET_EX_INFORMATION> = Vec::with_capacity(BATCH.min(tiles.len()) * PAGES_PER_TILE);
    // the region the last VirtualQuery reported: [start, end), and whether its pages are accessible
    let mut region = (0usize, 0usize, false);
    for batch in tiles.chunks(BATCH) {
        ws.clear();
        for &t in batch {
            for i in 0..PAGES_PER_TILE {
                ws.push(PSAPI_WORKING_SET_EX_INFORMATION { VirtualAddress: (t + i * PAGE) as *mut c_void, ..std::mem::zeroed() });
            }
        }
        let asked = K32QueryWorkingSetEx(GetCurrentProcess(), ws.as_mut_ptr() as *mut c_void, std::mem::size_of_val(&ws[..]) as u32) != 0;
        for (j, &t) in batch.iter().enumerate() {
            let mut holes = 0xFFFFu16;
            for i in 0..PAGES_PER_TILE {
                let pg = t + i * PAGE;
                // PSAPI_WORKING_SET_EX_BLOCK: bit 0 Valid, bits 4-14 Win32Protection
                let flags = ws[j * PAGES_PER_TILE + i].VirtualAttributes.Flags;
                let protect = ((flags >> 4) & 0x7FF) as u32;
                let mut accessible = asked && flags & 1 != 0 && protect != 0 && protect & (PAGE_NOACCESS | PAGE_GUARD) == 0;
                if !accessible {
                    if !(region.0..region.1).contains(&pg) {
                        #[cfg(test)]
                        tests::VIRTUAL_QUERIES.with(|n| n.set(n.get() + 1));
                        let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                        region = if VirtualQuery(pg as *const c_void, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) == 0 || mbi.RegionSize == 0 {
                            (pg, pg + PAGE, false)
                        } else {
                            let ok = mbi.State == MEM_COMMIT && mbi.Protect != 0 && (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
                            (pg, (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize).max(pg + PAGE), ok)
                        };
                    }
                    accessible = region.2;
                }
                if accessible {
                    holes &= !(1u16 << i);
                }
            }
            out.push(holes);
        }
    }
    out
}

/// Query actual OS page commitment across the 64 KiB tile at `tile`.
/// Bit `i` in the returned mask is set whenever 4 KiB page `tile + i * PAGE` is not committed
/// and accessible (`MEM_RESERVE`, `MEM_FREE`, `PAGE_NOACCESS`, `PAGE_GUARD`, etc.).
unsafe fn query_tile_holes(tile: usize) -> u16 {
    let mut holes = 0xFFFFu16;
    let end = tile.saturating_add(TILE);
    let mut cur = tile;
    while cur < end {
        let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        // SAFETY: `VirtualQuery` inspects process virtual memory metadata without dereferencing `cur`.
        let n = VirtualQuery(cur as *const c_void, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>());
        if n == 0 || mbi.RegionSize == 0 {
            break;
        }
        let region_end = (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize).min(end);
        let accessible = mbi.State == MEM_COMMIT && mbi.Protect != 0 && (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
        if accessible {
            let mut pg = cur;
            while pg < region_end {
                holes &= !(1u16 << ((pg - tile) / PAGE));
                pg += PAGE;
            }
        }
        if region_end <= cur {
            break;
        }
        cur = region_end;
    }
    holes
}

/// Update `mapped`, `free`, and `to_zero` for a decommit of `[a, a + n)`.
/// Returns the base addresses of tiles that became completely uncommitted (`holes == 0xFFFF`)
/// and were removed from `mapped` (ready to be unmapped from their sparse window buffers).
fn decommit_tiles(
    mapped: &mut FxHashMap<usize, TileEntry>,
    pinned: &FxHashMap<(u32, u32), u32>,
    free: &mut Vec<(u32, u32)>,
    to_zero: &mut BTreeSet<(usize, usize)>,
    a: usize,
    n: usize,
) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let mut unmapped: Vec<usize> = Vec::new();
    let mut pg = a & !(PAGE - 1);
    let end = a.saturating_add(n);
    while pg < end {
        let t = pg & !(TILE - 1);
        let tile_end = t.saturating_add(TILE).min(end);
        if let Some(entry) = mapped.get_mut(&t) {
            let mut newly_holed = 0u16;
            let mut p = pg;
            while p < tile_end {
                let idx = (p - t) / PAGE;
                entry.sent[idx] = None;
                let bit = 1u16 << idx;
                if entry.holes & bit == 0 {
                    entry.holes |= bit;
                    newly_holed |= bit;
                }
                p += PAGE;
            }
            if newly_holed != 0 && entry.src != 0 {
                entry.gen = next_gen();
            }
            if entry.holes == 0xFFFF {
                unmapped.push(t);
            } else if newly_holed != 0 {
                for idx in 0..PAGES_PER_TILE {
                    if newly_holed & (1u16 << idx) != 0 {
                        to_zero.insert((t + idx * PAGE, PAGE));
                    }
                }
            }
        }
        if tile_end <= pg {
            break;
        }
        pg = tile_end;
    }
    for &t in &unmapped {
        if let Some(entry) = mapped.remove(&t) {
            if entry.pool != LAZY && !pinned.contains_key(&entry.pool) {
                free.push(entry.pool);
            }
        }
    }
    // (their pieces only: going over all of them for each page decommitted while a title loads,
    // with no submit to clear them, grew with the square of the pages)
    for &t in &unmapped {
        while let Some(&piece) = to_zero.range((t, 0)..(t + TILE, 0)).next() {
            to_zero.remove(&piece);
        }
    }
    unmapped
}

/// The pieces to clear (`State::to_zero`) as runs: neighbours and pieces inside another together.
fn zero_runs(pieces: BTreeSet<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (a, n) in pieces {
        match out.last_mut() {
            Some(l) if a <= l.0 + l.1 => l.1 = l.1.max(a + n - l.0),
            _ => out.push((a, n)),
        }
    }
    out
}

/// Whether the oldest tile of `cooling` was freed `COOL` ago or more at `now`
fn cooled(cooling: &VecDeque<((u32, u32), std::time::Instant)>, now: std::time::Instant) -> bool {
    cooling.front().is_some_and(|&(_, at)| now.duration_since(at) >= COOL)
}

/// The pool tile to use next (`State::take_tile`): the oldest of `cooling` if it cooled, else
/// one of `free`. None: neither, though tiles freed less than `COOL` ago may be cooling.
fn reuse_tile(free: &mut Vec<(u32, u32)>, cooling: &mut VecDeque<((u32, u32), std::time::Instant)>, now: std::time::Instant) -> Option<(u32, u32)> {
    if cooled(cooling, now) {
        return cooling.pop_front().map(|(t, _)| t);
    }
    free.pop()
}

/// The pool heaps (`pools` of them) all of whose tiles are in `free` or cooled in `cooling` at
/// `now` (`State::retire_free_pools`): a tile still cooling keeps its heap.
fn free_heaps(free: &[(u32, u32)], cooling: &VecDeque<((u32, u32), std::time::Instant)>, pools: usize, now: std::time::Instant) -> Vec<usize> {
    let ready = cooling.iter().take_while(|&&(_, at)| now.duration_since(at) >= COOL).map(|&(t, _)| t);
    if free.len() + ready.clone().count() < POOL_TILES as usize {
        return Vec::new();
    }
    let mut count = vec![0u32; pools];
    for (h, _) in free.iter().copied().chain(ready) {
        if let Some(c) = count.get_mut(h as usize) {
            *c += 1;
        }
    }
    (0..pools).filter(|&h| count[h] == POOL_TILES).collect()
}

/// Whether a submit that freed no tile tries `State::retire_free_pools` at `now`: when a tile
/// of `cooling` cooled or a heap waits in `retired` (`waiting`), and `COOL` after the last try
/// (`tried`) at the earliest (a try counts the cooled tiles).
fn retire_due(cooling: &VecDeque<((u32, u32), std::time::Instant)>, waiting: bool, tried: std::time::Instant, now: std::time::Instant) -> bool {
    (waiting || cooled(cooling, now)) && now.duration_since(tried) >= COOL
}

struct UploadSlot {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence_val: u64,
}

/// A copy back on `State::rbq`: its list, its readback buffer, the `rb_fence` value it waits
/// for, the `rbq` fence value that marks it done, and the copies into the title's memory
/// (window address, offset in `buf`, bytes, `TileEntry::gen` of the address's tile then:
/// `State::copy_back`).
struct RbJob {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    buf: ID3D12Resource,
    ptr: *mut u8,
    cap: usize,
    rb_v: u64,
    done_v: u64,
    copies: Vec<(usize, usize, usize, u32)>,
    /// the type of queue `list` is for (a copy back runs on the title's own queue)
    ty: D3D12_COMMAND_LIST_TYPE,
    /// the title queue's fences for it (`State::rb_queue_fences`): the first reaches `rb_v` when
    /// the lists it follows are done, the second `done_v` when the copy is. None: `rb_fence` and
    /// the readback queue's fence (the form before)
    fences: Option<(ID3D12Fence, ID3D12Fence)>,
    /// place among the submissions' jobs (`State::job_seq`): copies and immediates land in it
    seq: u64,
}

/// In an immediate's byte count: the cell is a counter that the title's queues write in turn,
/// with values that only rise as they are recorded. A list that runs late – a retail title's
/// compute jobs, tens of frames after their recording when its submits are slow – writes an
/// earlier value over a later one; the title's CPU waits for "at least" the later one, which
/// was reached. The CPU's view keeps the larger value.
pub(super) const IMM_KEEP_MAX: u8 = 0x80;

/// Immediates the title's lists wrote into a window (`list_x::imm_by_cpu`): stored in the
/// title's memory once `fence` reaches `value` (the lists are done on the GPU).
struct ImmJob {
    /// place among the copies back (`RbJob::seq`)
    seq: u64,
    fence: ID3D12Fence,
    value: u64,
    /// (address, value, bytes, `TileEntry::gen` of the address's tile then)
    writes: Vec<(usize, u64, u8, u32)>,
}

/// How far `drain_readbacks` goes
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Drain {
    /// only copies the GPU has finished
    Finished,
    /// also those whose title work is done (waiting for the copy itself): before a CPU read
    /// the title made sure of, e.g. after its wait for the value a list wrote
    TitleDone,
}

struct RetiredTiles {
    tiles: Vec<(u32, u32)>,
    /// Fence marks on `self.queue` and every known title queue (`None` = signal failed, keep held).
    marks: Option<Vec<(ID3D12Fence, u64)>>,
    /// when they were unmapped: they cool (`State::cooling`) from then on once the marks are reached
    at: std::time::Instant,
}

struct State {
    dev: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    /// reusable auto-reset Win32 event for fence waits (no per-wait CreateEventW/CloseHandle)
    event: HANDLE,
    /// fence signalled on title queues after their lists, so a copy back waits for them
    rb_fence: ID3D12Fence,
    rb_value: u64,
    /// title queue -> the two fences it signals around a copy back, and the last value. One set
    /// a queue: with one fence for all, a queue that got to its Signal first let the copies of a
    /// queue still waiting run early
    rb_queue_fences: FxHashMap<usize, (ID3D12Fence, ID3D12Fence, u64)>,
    /// copies back after the title's submissions (`readback_after_execute`): on a queue of their
    /// own (`rbq`, None: on `queue`, waited for on the CPU), landing in the title's memory once
    /// done (`drain_readbacks`). The title's queues can wait for work the title submits after
    /// this one, from the same thread: waiting here for its lists stopped that submission – and,
    /// on `queue`, every upload behind it (`before_execute` waiting for ring room).
    rbq: Option<(ID3D12CommandQueue, ID3D12Fence)>,
    rbq_value: u64,
    /// submitted copies back, oldest first, and finished ones to reuse
    rb_jobs: VecDeque<RbJob>,
    rb_pool: Vec<RbJob>,
    /// immediates of the title's submissions, oldest first, and the count that orders them
    /// with `rb_jobs`
    imm_jobs: VecDeque<ImmJob>,
    job_seq: u64,
    /// (window start, reserved buffer) of each sparse window
    windows: Vec<(usize, ID3D12Resource)>,
    /// video-memory pool heaps (None: given back, its index free for the next one)
    pools: Vec<Option<ID3D12Heap>>,
    /// pool tiles to use: those of a pool that were never used, and cooled ones (`cooling`)
    free: Vec<(u32, u32)>,
    /// pool tiles freed (`flush_parked`, once `quarantined` lets them go, and `release_if_unused`),
    /// oldest first, with when: used again `COOL` after that at the earliest (`take_tile`).
    /// Before, a tile unmapped went to `free` and was the next one taken, mapped, cleared and
    /// filled for another address on the sparse queue while title work submitted before the
    /// unmap could still read it.
    cooling: VecDeque<((u32, u32), std::time::Instant)>,
    /// pools all of whose tiles came free: released once the GPU is past `value` and a moment
    /// went by (titles' queues may still read what was mapped there)
    retired: Vec<(ID3D12Heap, u64, std::time::Instant)>,
    /// when `retire_free_pools` last ran (`retire_due`)
    retire_tried: std::time::Instant,
    /// tile address → pool tile + uncommitted page hole mask + per-page sent chunk hashes
    mapped: FxHashMap<usize, TileEntry>,
    /// pool tiles also used by placed buffers (not freed while they are)
    pinned: FxHashMap<(u32, u32), u32>,
    /// tile address → pool tile of tiles decommitted since the last title submit, still mapped
    /// in their window: unmapped together (`flush_parked`) before the next submit, when the free
    /// list runs out, or past `PARKED_MAX`; a commit there before that takes its tile back with
    /// no new mapping. (A retail title moving physical pages one at a time while it loads
    /// decommits and commits again thousands of tiles a second, and each `UpdateTileMappings`
    /// took ~0.15 ms.)
    parked: FxHashMap<usize, (u32, u32)>,
    /// tile address → pool tile of decommitted tiles a placed buffer uses: they stay mapped in
    /// their window until the buffer is gone. Memory committed there again is then the same for
    /// the window (what shaders write) and the buffer (what an ExecuteIndirect or a copy of the
    /// title reads); with a new tile in the window the buffer kept the old one, with old bytes
    kept: FxHashMap<usize, (u32, u32)>,
    /// tile addresses given a pool tile in `mapped` whose `UpdateTileMappings` has not been sent
    /// yet: mapped together (`flush_pending`, one call per window) before the next title submit,
    /// before XCL's own GPU work on the windows, or past `PENDING_MAX`. (A retail title starting
    /// on a Series S profile maps 4.3 GB of physical pages in small pieces: one mapping call
    /// each, ~0.4 ms, held its loading thread ~25 s and its stall watchdog ended it.)
    pending: Vec<usize>,
    /// placed buffers → the tiles they use (address, pool tile)
    buffers: FxHashMap<usize, Vec<(usize, (u32, u32))>>,
    /// unmapped parked pool tiles awaiting `self.queue` and in-flight title queues before they
    /// cool (`cooling`)
    quarantined: VecDeque<RetiredTiles>,
    /// per-title-queue fences used to mark in-flight GPU work (`note_queue`)
    queue_fences: FxHashMap<usize, (ID3D12CommandQueue, ID3D12Fence, u64)>,
    /// reused tiles / decommitted partial pages to clear: (address, bytes)
    to_zero: BTreeSet<(usize, usize)>,
    /// ring-buffered upload staging heap (`STAGING` bytes, persistently mapped)
    staging: Option<(ID3D12Resource, *mut u8)>,
    /// monotonic ring-buffer write/submitted/retired byte cursors
    ring_write: u64,
    ring_submitted: u64,
    ring_free: u64,
    /// submitted ring chunks awaiting GPU completion: `(fence_val, ring_end_pos)`
    inflight: VecDeque<(u64, u64)>,
    /// dedicated 1 MiB pre-zeroed buffer for tile/hole clears (zero staging bandwidth)
    zero_buf: Option<ID3D12Resource>,
    /// ring of async upload command allocators + lists
    slots: Vec<UploadSlot>,
    next_slot: usize,
    /// dedicated readback staging buffer + command allocator / list
    readback: Option<(ID3D12Resource, *mut u8)>,
    rb_alloc: ID3D12CommandAllocator,
    rb_list: ID3D12GraphicsCommandList,
    rb_last_fence: u64,
    /// the fence value each title queue last waited for
    waited: FxHashMap<usize, u64>,
    /// `mapped` as runs of neighbouring tiles (start -> end), kept with it: sorted out of all of
    /// them again (a retail title has 150,000) for each look behind a commit
    runs: BTreeMap<usize, usize>,
    /// reusable buffer for dirty runs in `before_execute` (avoids per-submit heap alloc)
    scratch_runs: Vec<(usize, usize)>,
    /// `XCL_LOOK_STATS=2`: the title's pages XCL wrote itself since the last look (copies back
    /// landed, immediates stored; `OWN_PAGES_MAX` at most), for `look_stats::own_writes`
    own_pages: Vec<usize>,
    /// CPU tile → the window tiles over it (tiles of a GPU view, `map_mirror`), with video memory
    /// or not: what the CPU writes there goes to each of them. Several can show one CPU tile
    mirrors: FxHashMap<usize, Vec<usize>>,
    /// window tile → the CPU tile under it, for each tile in `mirrors`
    mirror_src: FxHashMap<usize, usize>,
    /// the CPU tiles of `mirrors` as runs (start -> end): where a CPU write has to reach the GPU
    /// too (`mapped_ranges`). Tiles over a CPU view are never in `runs`: their own addresses are
    /// not the CPU's
    src_runs: BTreeMap<usize, usize>,
    /// `src_runs` for the threads that decommit the title's CPU memory, and their decommits under
    /// way (`SOURCES`; a test's state has one of its own)
    sources: &'static Mutex<Sources>,
    /// window tiles given video memory over their CPU view since the last submit: all of their
    /// committed pages go up with it (the CPU wrote them before the tile was there, and those
    /// writes were taken by a look before)
    fill: Vec<usize>,
    /// window tiles asked for (`map_mirror`) and not handled yet
    asks: Vec<usize>,
    /// window tiles whose GPU-view page changed (`remapped`), linked again at the next submit
    remaps: Vec<usize>,
    /// window tiles linked over their CPU tiles (`mirror_src`) that wait for their video memory
    /// and fill, oldest first (`fill_later`)
    later: VecDeque<(usize, usize)>,
    /// committed pages filled at a submit at most, for tiles given video memory there
    /// (`XCL_MIRROR_FILL_MB`): the rest wait for the next
    fill_pages: usize,
    /// tiles over a CPU view with video memory now, and the most there may be (`XCL_MIRROR_MB`)
    mirrored: usize,
    mirror_cap: usize,
    /// tiles over a CPU view left without video memory for `mirror_cap`, or as none was to be had:
    /// given it once there is room (`retry_refused`), or once their CPU view is written
    refused: FxHashSet<usize>,
    /// when a tile over a CPU view last found no video memory (`take_tile`): for a moment the
    /// others do not try to make a heap again (each tried, and logged it), and the refused ones
    /// wait
    no_memory: Option<std::time::Instant>,
    /// tiles listed with no video memory (`LAZY`) whose CPU bytes found none at an upload
    /// (`upload_dirty`): all that is committed there goes up at a submit once there is some
    starved: Vec<usize>,
    counts: MirrorCounts,
    diag: ViewDiag,
    /// `XCL_VIEW_POISON`: the tile the windows' tiles without video memory show (`poison_windows`)
    poison: Option<Poison>,
}

/// The one 64 KiB of video memory the tiles of the windows without video memory of their own are
/// mapped to (`D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE`), a buffer placed over it, and the
/// pattern's upload copy it is filled from again (`refresh_poison`: what a shader wrote there
/// lands in the one tile).
struct Poison {
    heap: ID3D12Heap,
    buf: ID3D12Resource,
    src: ID3D12Resource,
    at: std::time::Instant,
}

/// What the tiles over CPU views did, for the timing line
#[derive(Default, Debug)]
struct MirrorCounts {
    /// the most tiles with video memory at once
    peak: usize,
    /// tiles filled whole from their CPU view, and the bytes of their committed pages
    fills: u64,
    fill_bytes: u64,
    /// tiles refused for `mirror_cap` or for want of video memory (each once until it has some)
    refused: u64,
    /// tiles whose video memory went back (their CPU view decommitted, or the view gone)
    given_back: u64,
    /// pages that were not committed in a CPU view, written since and asked about again
    rechecks: u64,
}

/// Diagnostics of the GPU-view tiles, all off unless set: `XCL_VIEW_KEEP=1` (a tile over a CPU
/// view keeps its video memory once filled, within `mirror_cap`), `XCL_VIEW_POISON=1` (or a
/// 32-bit pattern, `0x…`: what a tile of the windows without video memory reads instead of zeros,
/// `State::poison_windows`) and `XCL_VIEW_STATS=1`. Any of them logs `view_stats_line` every 2 s.
struct ViewSwitches {
    keep: bool,
    poison: Option<u32>,
    stats: bool,
}

fn view_switches() -> &'static ViewSwitches {
    static ON: std::sync::OnceLock<ViewSwitches> = std::sync::OnceLock::new();
    ON.get_or_init(|| {
        let var = |k: &str| std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty() && v != "0");
        let (keep, poison) = (var("XCL_VIEW_KEEP").is_some(), poison_pattern(var("XCL_VIEW_POISON").as_deref()));
        ViewSwitches { keep, poison, stats: keep || poison.is_some() || var("XCL_VIEW_STATS").is_some() }
    })
}

/// The poison pattern of `XCL_VIEW_POISON` (set, not "0"): a NaN as a float, a wild index or count
/// as an integer, or the pattern given in hex.
fn poison_pattern(v: Option<&str>) -> Option<u32> {
    let v = v?;
    match v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
        Some(hex) => u32::from_str_radix(hex, 16).ok().filter(|&p| p != 0),
        None => Some(0x7FC0_DEAD),
    }
}

/// Whether a tile over a CPU view whose video memory would go back keeps it mapped where it is
/// (`XCL_VIEW_KEEP`): while the tiles with video memory and kept, this one among them, are within
/// the cap.
fn keep_view_tile(keep: bool, held: usize, cap: usize) -> bool {
    keep && held <= cap
}

/// The bucket of a tile that got video memory again `submits` after it went: 0-1, 2-8, 9-64, more.
fn back_bucket(submits: u64) -> usize {
    match submits {
        0..=1 => 0,
        2..=8 => 1,
        9..=64 => 2,
        _ => 3,
    }
}

/// What the GPU-view diagnostics count (`ViewSwitches`; nothing while they are off but submits).
#[derive(Default, Debug)]
struct ViewDiag {
    keep: bool,
    stats: bool,
    /// submits so far (`before_execute_on`)
    submits: u64,
    /// tiles kept mapped that would have gone back (`XCL_VIEW_KEEP`)
    kept: u64,
    /// tile given back or kept → the submit then, until it gets video memory again
    left_at: FxHashMap<usize, u64>,
    /// tiles that got video memory again, by submits since they went (`back_bucket`), and the
    /// bytes of their committed pages filled again
    back: [u64; 4],
    back_bytes: u64,
    /// submits that left tiles asked for waiting (`later`, `asks`) for a later one
    deferred: u64,
}

/// Tiles of GPU views asked for (`map_mirror` queued them), while `ViewSwitches::stats`.
static VIEW_ASKED: AtomicU64 = AtomicU64::new(0);

/// `view_stats`: [asked, filled, given back, kept, back 0-1, 2-8, 9-64, more, bytes back, submits,
/// deferred] so far, then [with video memory, kept mapped, waiting, refused] now.
const VS: usize = 15;
const VS_NOW: usize = 11;

/// tiles over CPU views asked for taken in at a time, and refused ones tried again at a submit at
/// most (`State::map_mirrors`); the rest wait for the next one
const MIRROR_ASKS_MAX: usize = 4096;

/// Committed pages of the CPU views filled into tiles given video memory at a submit at most
/// (`XCL_MIRROR_FILL_MB`, 16 MB by default; at least a page): the fills are hashed and go
/// through the staging ring on the title's submitting thread, with the state's lock held.
fn fill_pages() -> usize {
    static PAGES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PAGES.get_or_init(|| {
        let mb = std::env::var("XCL_MIRROR_FILL_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(16);
        (mb.saturating_mul(1 << 20) / PAGE).max(1)
    })
}

/// Video memory for tiles over CPU views at most, in tiles (`XCL_MIRROR_MB`, 4096 MB by default):
/// a GPU view covers the whole of a 24 GB CPU reservation, and the card is mostly in use already.
fn mirror_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let mb = std::env::var("XCL_MIRROR_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(4096);
        mb.saturating_mul(1 << 20) / TILE
    })
}

impl Drop for State {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }
}

const K0: u64 = 0x9E37_79B9_7F4A_7C15;
const K1: u64 = 0xBF58_476D_1CE4_E5B9;
const K2: u64 = 0x94D0_49BB_1331_11EB;
const K3: u64 = 0x243F_6A88_85A3_08D3;

/// 4-way parallel hash over 8 `u64` words (64 bytes): breaks serial multiply latency chain.
#[inline(always)]
const fn hash_words(w: [u64; 8]) -> u64 {
    let a = (w[0] ^ K3).wrapping_mul(K0) ^ w[4].wrapping_mul(K1);
    let b = (w[1] ^ K0).wrapping_mul(K1) ^ w[5].wrapping_mul(K2);
    let c = (w[2] ^ K1).wrapping_mul(K2) ^ w[6].wrapping_mul(K3);
    let d = (w[3] ^ K2).wrapping_mul(K3) ^ w[7].wrapping_mul(K0);
    let h = a.rotate_left(27).wrapping_add(b.rotate_left(13)) ^ c.rotate_left(41).wrapping_add(d.rotate_left(53));
    h.wrapping_mul(K0) ^ (h >> 32)
}

const ZERO_HASH: u64 = hash_words([0; 8]);
const ZERO_HASHES: [u64; CHUNKS] = [ZERO_HASH; CHUNKS];

#[inline(always)]
fn read_chunk_words(c: &[u8]) -> [u64; 8] {
    debug_assert_eq!(c.len(), CHUNK);
    let p = c.as_ptr() as *const [u8; 8];
    unsafe {
        [
            u64::from_le_bytes(p.add(0).read_unaligned()),
            u64::from_le_bytes(p.add(1).read_unaligned()),
            u64::from_le_bytes(p.add(2).read_unaligned()),
            u64::from_le_bytes(p.add(3).read_unaligned()),
            u64::from_le_bytes(p.add(4).read_unaligned()),
            u64::from_le_bytes(p.add(5).read_unaligned()),
            u64::from_le_bytes(p.add(6).read_unaligned()),
            u64::from_le_bytes(p.add(7).read_unaligned()),
        ]
    }
}

/// hash of 64 bytes
#[inline(always)]
fn chunk_hash(c: &[u8]) -> u64 {
    hash_words(read_chunk_words(c))
}

#[inline(always)]
fn push_run(out: &mut Vec<(usize, usize)>, addr: usize, len: usize) {
    match out.last_mut() {
        Some(last) if last.0 + last.1 == addr => last.1 += len,
        _ => out.push((addr, len)),
    }
}

/// Copy `len` bytes from `src` to offset `at` of the title's page `pg`, 64-byte chunk by chunk:
/// a chunk the CPU wrote since it was last sent keeps its (newer) bytes and goes up with the next
/// look; a chunk as it was sent gets what the GPU copied back, and its hash with it. Whether it
/// wrote any.
unsafe fn land_page(slot: &mut Option<Box<[u64; CHUNKS]>>, pg: usize, at: usize, src: *const u8, len: usize) -> bool {
    let sent = slot.get_or_insert_with(|| Box::new(ZERO_HASHES));
    let end = at + len;
    let mut o = at;
    let mut wrote = false;
    while o < end {
        let k = o / CHUNK;
        let chunk_end = ((k + 1) * CHUNK).min(end);
        let chunk = std::slice::from_raw_parts((pg + k * CHUNK) as *const u8, CHUNK);
        if chunk_hash(chunk) == sent[k] {
            std::ptr::copy_nonoverlapping(src.add(o - at), (pg + o) as *mut u8, chunk_end - o);
            sent[k] = chunk_hash(std::slice::from_raw_parts((pg + k * CHUNK) as *const u8, CHUNK));
            wrote = true;
        }
        o = chunk_end;
    }
    wrote
}

/// `State::own_pages` keeps this many pages at most (a look's worth)
const OWN_PAGES_MAX: usize = 1 << 16;

thread_local! {
    /// `XCL_LOOK_STATS=2`: the next `before_execute` on this thread follows a look (`note_look`)
    static LOOKED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The submit about to call `before_execute` on this thread looked for CPU writes
/// (`XCL_LOOK_STATS=2`): XCL's own writes since the look before are counted against it.
pub(super) fn note_look() {
    LOOKED.with(|l| l.set(true));
}

/// Scan a 4 KiB page at `pg`, appending any changed 64-byte chunk runs directly to `out`
/// (zero per-page heap allocation; skips allocating `slot` completely when a fresh zero page
/// is still all zeros).
#[inline]
fn scan_page(page: &[u8], pg: usize, slot: &mut Option<Box<[u64; CHUNKS]>>, out: &mut Vec<(usize, usize)>) {
    if let Some(sent) = slot.as_deref_mut() {
        for (k, c) in page.chunks_exact(CHUNK).enumerate() {
            let h = chunk_hash(c);
            if h == sent[k] {
                continue;
            }
            sent[k] = h;
            push_run(out, pg + k * CHUNK, CHUNK);
        }
        return;
    }
    // Page has never had non-zero data sent: skip zero chunks with a cheap OR-reduction and
    // allocate the hash array only if at least one chunk is non-zero.
    let mut chunks = page.chunks_exact(CHUNK).enumerate();
    while let Some((k, c)) = chunks.next() {
        let w = read_chunk_words(c);
        if (w[0] | w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) == 0 {
            continue;
        }
        let h = hash_words(w);
        if h == ZERO_HASH {
            continue;
        }
        let sent = slot.insert(Box::new(ZERO_HASHES));
        sent[k] = h;
        push_run(out, pg + k * CHUNK, CHUNK);
        for (k2, c2) in chunks {
            let h2 = chunk_hash(c2);
            if h2 == ZERO_HASH {
                continue;
            }
            sent[k2] = h2;
            push_run(out, pg + k2 * CHUNK, CHUNK);
        }
        break;
    }
}

/// The byte ranges (offset, length) of `page` whose 64-byte chunks differ from `sent`
/// (updated to the page's chunks).
pub fn changed_chunks(page: &[u8], sent: &mut [u64; CHUNKS]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (k, c) in page.chunks_exact(CHUNK).enumerate() {
        let h = chunk_hash(c);
        if h == sent[k] {
            continue;
        }
        sent[k] = h;
        push_run(&mut out, k * CHUNK, CHUNK);
    }
    out
}

/// Plan a contiguous sub-allocation `(aligned_pos, chunk_len)` inside a ring buffer of `cap`
/// bytes starting at monotonic cursor `ring_write`, keeping each chunk `<= cap / 2` and
/// avoiding mid-copy wrap-around without wasting large tail gaps.
#[inline]
fn plan_ring_chunk(ring_write: u64, want: usize, cap: usize) -> (u64, usize) {
    let mut chunk = want.min(cap / 2);
    let mut aligned = (ring_write + (RING_ALIGN as u64 - 1)) & !(RING_ALIGN as u64 - 1);
    let off = (aligned as usize) % cap;
    if off + chunk > cap {
        let tail = cap - off;
        if tail >= PAGE {
            chunk = tail;
        } else {
            aligned += tail as u64;
        }
    }
    (aligned, chunk)
}

// D3D12 objects are free-threaded; staging pointers and event handle are only used under STATE lock
unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// The state's queue again, outside `STATE` (`queue`). `STATE` is held around D3D12 calls (an
/// upload's allocator `Reset` in `begin_slot`), and `late_release` asks for the queue from a
/// resource's last `Release`, which D3D12 can make inside a call of its own on another thread
/// (a list's `Reset`, with `XCL_DRED=1` on every start): the two then waited for each other.
static QUEUE: Mutex<Option<ID3D12CommandQueue>> = Mutex::new(None);

/// (bytes of video memory in pool heaps, of those mapped) – `XCL_ACTIVITY`
pub fn pool_bytes() -> (usize, usize) {
    let g = lock_state();
    let _held = holding(Holder::Other);
    let Some(s) = g.as_ref() else { return (0, 0) };
    let pools = s.pools.iter().filter(|p| p.is_some()).count() * POOL_TILES as usize * TILE;
    let quarantined: usize = s.quarantined.iter().map(|b| b.tiles.len()).sum();
    (pools, pools.saturating_sub((s.free.len() + s.cooling.len() + quarantined) * TILE))
}
/// commits before the device existed: the pages committed, as disjoint page ranges (start →
/// end) merged with their neighbours, so a commit or decommit costs O(log n) and not a scan of
/// every page committed so far. Like `committed` / `decommitted` with a state, a range counts
/// every page it touches; `init` commits them once the windows exist.
static EARLY: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());
/// Ranges committed on the CPU in sparse windows whose VRAM tiles are mapped lazily on first
/// non-zero CPU write (`alloc_lazy`) or GPU write target registration, avoiding eager VRAM
/// allocation for untouched pages or GPU-owned render targets placed inside large title heaps.
static LAZY_RANGES: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());
static LAZY_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn in_lazy_range(lazy: &BTreeMap<usize, usize>, a: usize) -> bool {
    lazy.range(..=a).next_back().is_some_and(|(_, &e)| a < e)
}

fn lazy_overlaps_tile(lazy: &BTreeMap<usize, usize>, t: usize) -> bool {
    lazy.range(..t.saturating_add(TILE)).next_back().is_some_and(|(_, &e)| e > t)
}

fn lazy_tile_holes(lazy: &BTreeMap<usize, usize>, t: usize) -> u16 {
    let mut holes = 0u16;
    for k in 0..PAGES_PER_TILE {
        if !in_lazy_range(lazy, t + k * PAGE) {
            holes |= 1u16 << k;
        }
    }
    holes
}

fn page_has_nonzero(bytes: &[u8]) -> bool {
    let mut chunks = bytes.chunks_exact(8);
    if chunks.any(|c| u64::from_ne_bytes(c.try_into().unwrap()) != 0) {
        return true;
    }
    chunks.remainder().iter().any(|&b| b != 0)
}

/// The pages `[a, a + n)` touches, as `[start, end)`.
fn page_span(a: usize, n: usize) -> (usize, usize) {
    (a & !(PAGE - 1), a.saturating_add(n).saturating_add(PAGE - 1) & !(PAGE - 1))
}

/// Add the pages `[a, b)` to `early` (ranges as start -> end), merged with the ranges they overlap
/// or touch.
fn early_add(early: &mut BTreeMap<usize, usize>, mut a: usize, mut b: usize) {
    if a >= b {
        return;
    }
    if let Some((&s, &e)) = early.range(..=a).next_back() {
        if e >= a {
            a = s;
            b = b.max(e);
            early.remove(&s);
        }
    }
    loop {
        let Some((s, e)) = early.range(a..=b).next().map(|(&s, &e)| (s, e)) else { break };
        b = b.max(e);
        early.remove(&s);
    }
    early.insert(a, b);
}

/// Take the pages `[a, b)` out of `early`; a range that overlaps it keeps the rest.
fn early_remove(early: &mut BTreeMap<usize, usize>, a: usize, b: usize) {
    if a >= b {
        return;
    }
    if let Some((&s, &e)) = early.range(..a).next_back() {
        if e > a {
            early.insert(s, a);
            if e > b {
                early.insert(b, e);
                return;
            }
        }
    }
    loop {
        let Some((s, e)) = early.range(a..b).next().map(|(&s, &e)| (s, e)) else { break };
        early.remove(&s);
        if e > b {
            early.insert(b, e);
            break;
        }
    }
}
/// generation counter incremented whenever the set of mapped tiles changes (for watch_ranges cache)
static MAPPED_GEN: AtomicU64 = AtomicU64::new(0);

/// Generation counter of `mapped_ranges()` (changes only when tiles are mapped or unmapped).
pub fn mapped_gen() -> u64 {
    MAPPED_GEN.load(Ordering::Acquire)
}

fn buffer_desc(bytes: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    }
}

unsafe fn new_state(dev: &ID3D12Device, windows: Vec<(usize, ID3D12Resource)>) -> windows::core::Result<State> {
    if lazy_asked() && !null_tiles_read_zero(dev) && !LAZY_REFUSED.swap(true, Ordering::Relaxed) {
        crate::xlog_warn!("sparse windows: XCL_SPARSE_LAZY=1 is left out - on this device a tile with no video memory does not read zeros (tiled resources tier under 2)");
    }
    let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())?;
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
    let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
    let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None)?;
    let rbq = (|| -> windows::core::Result<(ID3D12CommandQueue, ID3D12Fence)> {
        let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())?;
        let _ = q.SetName(windows::core::w!("xcl sparse readback queue"));
        Ok((q, dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?))
    })()
    .ok();
    let mut slots = Vec::with_capacity(UPLOAD_SLOTS);
    for _ in 0..UPLOAD_SLOTS {
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
        list.Close()?;
        let _ = list.SetName(windows::core::w!("xcl sparse upload list"));
        let _ = alloc.SetName(windows::core::w!("xcl sparse upload allocator"));
        slots.push(UploadSlot { alloc, list, fence_val: 0 });
    }
    let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
    let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None)?;
    rb_list.Close()?;
    let _ = rb_list.SetName(windows::core::w!("xcl sparse readback list"));
    let _ = rb_alloc.SetName(windows::core::w!("xcl sparse readback allocator"));
    let _ = queue.SetName(windows::core::w!("xcl sparse queue"));
    Ok(State {
        dev: dev.clone(),
        queue,
        fence,
        value: 0,
        event,
        rb_fence,
        rb_value: 0,
        rb_queue_fences: FxHashMap::default(),
        rbq,
        rbq_value: 0,
        rb_jobs: VecDeque::new(),
        rb_pool: Vec::new(),
        imm_jobs: VecDeque::new(),
        job_seq: 0,
        windows,
        pools: Vec::new(),
        free: Vec::new(),
        cooling: VecDeque::new(),
        retired: Vec::new(),
        retire_tried: std::time::Instant::now(),
        mapped: FxHashMap::default(),
        pinned: FxHashMap::default(),
        parked: FxHashMap::default(),
        kept: FxHashMap::default(),
        pending: Vec::new(),
        buffers: FxHashMap::default(),
        quarantined: VecDeque::new(),
        queue_fences: FxHashMap::default(),
        to_zero: BTreeSet::new(),
        staging: None,
        ring_write: 0,
        ring_submitted: 0,
        ring_free: 0,
        inflight: VecDeque::new(),
        zero_buf: None,
        slots,
        next_slot: 0,
        readback: None,
        rb_alloc,
        rb_list,
        rb_last_fence: 0,
        waited: FxHashMap::default(),
        runs: BTreeMap::new(),
        scratch_runs: Vec::new(),
        own_pages: Vec::new(),
        mirrors: FxHashMap::default(),
        mirror_src: FxHashMap::default(),
        src_runs: BTreeMap::new(),
        sources: &SOURCES,
        fill: Vec::new(),
        asks: Vec::new(),
        remaps: Vec::new(),
        later: VecDeque::new(),
        fill_pages: fill_pages(),
        mirrored: 0,
        mirror_cap: mirror_cap(),
        refused: FxHashSet::default(),
        no_memory: None,
        starved: Vec::new(),
        counts: MirrorCounts::default(),
        diag: ViewDiag { keep: view_switches().keep, stats: view_switches().stats, ..Default::default() },
        poison: None,
    })
}

/// pool heap number of the tiles of a shadowed window: they are the window's own buffer, not
/// pool tiles (never mapped, freed or pinned)
const SHADOW_POOL: u32 = u32::MAX;

/// A committed window the GPU could not open as a heap (`real::open_uma`): `buf` in video
/// memory stands for `[base, base + size)` and is kept in step like the tiles of a sparse
/// window -- the CPU's writes (write-watch, changed 64-byte chunks) go to it before each
/// ExecuteCommandLists, GPU writes come back on Map and after the lists' memory operations.
/// All of it counts as mapped from the start: the window is committed. False: no queue for
/// the copies (the buffer then never sees the CPU's writes).
pub unsafe fn shadow_window(dev: &ID3D12Device, base: usize, size: usize, buf: &ID3D12Resource) -> bool {
    let mut g = STATE.lock().unwrap();
    if g.is_none() {
        // no sparse window was made (`init`): the copies still need the queue and the lists
        match new_state(dev, Vec::new()) {
            Ok(s) => *g = Some(s),
            Err(e) => {
                crate::xlog_warn!("shadowed window at {base:#x}: {e}");
                return false;
            }
        }
    }
    let Some(s) = g.as_mut() else { return false };
    s.windows.push((base, buf.clone()));
    for (i, t) in (base..base + size).step_by(TILE).enumerate() {
        s.mapped.insert(t, TileEntry::with_holes((SHADOW_POOL, i as u32), 0));
    }
    early_add(&mut s.runs, base, base + size.next_multiple_of(TILE));
    MAPPED_GEN.fetch_add(1, Ordering::Release);
    true
}

/// The sparse windows' buffers on `dev`: `(window_index, buffer_raw, gpu_va)` for each
/// sparse window in `gpu::uma` and each view window, made the same way (empty: no sparse
/// windows). Nothing of a view window is mapped here: its tiles get video memory when asked for.
pub unsafe fn init(dev: &ID3D12Device) -> Vec<(usize, usize, u64)> {
    let Some(u) = crate::gpu::uma::get() else { return Vec::new() };
    let mut out = Vec::new();
    let mut windows = Vec::new();
    for k in 0..u.shader_windows() {
        if !u.is_sparse_window(k) && !u.is_view_window(k) {
            continue;
        }
        let (base, size) = u.window(k);
        let mut r: Option<ID3D12Resource> = None;
        if let Err(e) = dev.CreateReservedResource(&buffer_desc(size as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut r) {
            crate::xlog_warn!("sparse window {k}: CreateReservedResource: {e}");
            return Vec::new();
        }
        let Some(r) = r else { return Vec::new() };
        // (a name for what a device removal reports about an address, `real::log_device_removed`)
        let _ = r.SetName(&windows::core::HSTRING::from(format!("xcl memory window {k}{}", if u.is_view_window(k) { " (GPU views)" } else { "" })));
        out.push((k, r.clone().into_raw() as usize, r.GetGPUVirtualAddress()));
        windows.push((base, r));
    }
    if out.is_empty() {
        return out;
    }
    let made = new_state(dev, windows);
    match made {
        Ok(mut s) => {
            let views = out.iter().filter(|w| u.is_view_window(w.0)).count();
            crate::xlog_info!("D3D12.X: {} sparse window(s) for the title's own memory and {views} for GPU views of it (tiled buffers on video memory)", out.len() - views);
            let sw = view_switches();
            if let Some(pattern) = sw.poison {
                s.poison_windows(pattern);
            }
            if sw.stats {
                crate::xlog_info!(
                    "D3D12.X: GPU-view diagnostics: keep {}, poison {}, a line every 2 s",
                    if sw.keep { format!("on ({} tiles at most, XCL_MIRROR_MB)", s.mirror_cap) } else { "off".into() },
                    match (sw.poison, s.poison.is_some()) {
                        (Some(p), true) => format!("{p:#010x}"),
                        (Some(_), false) => "failed (zeros)".into(),
                        (None, _) => "off".into(),
                    }
                );
                start_view_stats();
            }
            // a state this replaces (the windows set up again) goes after the lock: the release of
            // its last buffers reaches `forget_buffer`, which takes the lock (#344)
            // (the queue first: `queue()` never answers with one older than the state's)
            let old_queue = QUEUE.lock().unwrap().replace(s.queue.clone());
            let old = STATE.lock().unwrap().replace(s);
            drop(old);
            drop(old_queue);
            MAPPED_GEN.fetch_add(1, Ordering::Release);
            for (a, b) in std::mem::take(&mut *EARLY.lock().unwrap()) {
                committed(a, b - a);
            }
            out
        }
        Err(e) => {
            crate::xlog_warn!("sparse windows: {e}");
            Vec::new()
        }
    }
}

impl State {
    #[inline]
    fn window_index(&self, a: usize) -> Option<usize> {
        self.windows.iter().position(|(b, _)| a >= *b && a < *b + crate::gpu::uma::WINDOW)
    }

    /// A pool tile (true: used before, to be cleared). One freed less than `COOL` ago, or one
    /// still `quarantined`, is not taken: a retired pool is taken back or a heap made instead.
    /// (Parked tiles are not flushed for it: freed now, they would wait and cool all the same.)
    unsafe fn take_tile(&mut self) -> Option<((u32, u32), bool)> {
        #[cfg(test)]
        if tests::NO_TILES.with(|c| c.get()) {
            return None;
        }
        self.drain_quarantined();
        if let Some(t) = reuse_tile(&mut self.free, &mut self.cooling, std::time::Instant::now()) {
            return Some((t, true));
        }
        // a pool retired a moment ago (its last tile came free) is taken back before a new
        // 64 MB heap is made: freeing and allocating a tile in turn made a new pool each time,
        // up to 18 GB of retired heaps in 2 s (#913). Its tiles are reused as freed tiles are
        // (zeroed), and they all cooled before it was retired.
        if let Some((heap, _, _)) = self.retired.pop() {
            let h = self.add_pool(heap);
            return Some(((h, 0), true));
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_TILES as u64 * TILE as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
        };
        // (a heap takes its time, with the state locked: what waits for its mapping goes first, a
        // CPU signal that asks meanwhile gives up after 2 ms)
        self.flush_pending();
        let mut heap: Option<ID3D12Heap> = None;
        if let Err(e) = super::title_textures::own_work(|| self.dev.CreateHeap(&desc, &mut heap)) {
            crate::xlog_warn!("sparse windows: no video memory for more tiles ({e})");
            return None;
        }
        let heap = heap?;
        let _ = heap.SetName(windows::core::w!("xcl sparse tile pool"));
        let h = self.add_pool(heap);
        Some(((h, 0), false))
    }

    /// `heap` becomes a pool (in a given-back pool's index if there is one); its tiles but the
    /// first, which the caller takes, go to the free list.
    fn add_pool(&mut self, heap: ID3D12Heap) -> u32 {
        let h = match self.pools.iter().position(|p| p.is_none()) {
            Some(i) => {
                self.pools[i] = Some(heap);
                i as u32
            }
            None => {
                self.pools.push(Some(heap));
                self.pools.len() as u32 - 1
            }
        };
        self.free.extend((1..POOL_TILES).rev().map(|t| (h, t)));
        h
    }

    /// Pools whose every tile is free again (cooled, `COOL`) go (64 MB of video memory each –
    /// and as much of the system's commit: a retail game's homes of physical pages left 0.75 GB
    /// of pools for 0.29 GB of tiles in use). Call after their tiles are unmapped and signalled.
    fn retire_free_pools(&mut self) {
        let now = std::time::Instant::now();
        self.retire_tried = now;
        let done = unsafe { super::real::fence_completed(&self.fence) };
        self.retired.retain(|(_, v, at)| !(done >= *v && now.duration_since(*at).as_secs() >= 2));
        let gone: Vec<usize> = free_heaps(&self.free, &self.cooling, self.pools.len(), now).into_iter().filter(|&h| self.pools[h].is_some()).collect();
        if gone.is_empty() {
            return;
        }
        self.free.retain(|&(h, _)| !gone.contains(&(h as usize)));
        self.cooling.retain(|&((h, _), _)| !gone.contains(&(h as usize)));
        for h in gone {
            if let Some(heap) = self.pools[h].take() {
                self.retired.push((heap, self.value, now));
            }
        }
    }

    /// Map `tiles` (tile addresses) of `res` (starting at `res_base`) onto pool tiles,
    /// batched per pool heap with contiguous runs merged into single regions.
    unsafe fn map_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[(usize, (u32, u32))]) {
        if tiles.is_empty() {
            return;
        }
        let mut by_heap: FxHashMap<u32, Vec<(u32, u32)>> = FxHashMap::default();
        debug_assert!(tiles.iter().all(|t| t.1 != LAZY), "a tile with no video memory is mapped nowhere");
        for &(a, (h, t)) in tiles {
            by_heap.entry(h).or_default().push((((a - res_base) / TILE) as u32, t));
        }
        count_tile_calls(res_base, false, by_heap.len(), tiles.len());
        for (h, mut pairs) in by_heap {
            pairs.sort_unstable_by_key(|&(x, _)| x);
            let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::with_capacity(pairs.len());
            let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::with_capacity(pairs.len());
            let mut flags: Vec<D3D12_TILE_RANGE_FLAGS> = Vec::with_capacity(pairs.len());
            let mut offsets: Vec<u32> = Vec::with_capacity(pairs.len());
            let mut counts: Vec<u32> = Vec::with_capacity(pairs.len());
            let mut i = 0;
            while i < pairs.len() {
                let (x0, t0) = pairs[i];
                let mut run = 1u32;
                while i + (run as usize) < pairs.len() {
                    let (nx, nt) = pairs[i + run as usize];
                    if nx == x0 + run && nt == t0 + run {
                        run += 1;
                    } else {
                        break;
                    }
                }
                coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
                regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
                flags.push(D3D12_TILE_RANGE_FLAG_NONE);
                offsets.push(t0);
                counts.push(run);
                i += run as usize;
            }
            let _tm = crate::hle::api_cost::time(crate::hle::api_cost::Call::WindowMap);
            self.queue.UpdateTileMappings(
                res,
                coords.len() as u32,
                Some(coords.as_ptr()),
                Some(regions.as_ptr()),
                self.pools[h as usize].as_ref().expect("a pool tile of a live pool"),
                flags.len() as u32,
                Some(flags.as_ptr()),
                Some(offsets.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
    }

    unsafe fn unmap_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[usize]) {
        if tiles.is_empty() {
            return;
        }
        let mut xs: Vec<u32> = tiles.iter().map(|&a| ((a - res_base) / TILE) as u32).collect();
        xs.sort_unstable();
        let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::with_capacity(xs.len());
        let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::with_capacity(xs.len());
        let mut flags: Vec<D3D12_TILE_RANGE_FLAGS> = Vec::with_capacity(xs.len());
        let mut counts: Vec<u32> = Vec::with_capacity(xs.len());
        // (`XCL_VIEW_POISON`: onto the poison tile, every tile of a run on its one tile)
        let poison = self.poison.as_ref().map(|p| &p.heap);
        let mut i = 0;
        while i < xs.len() {
            let x0 = xs[i];
            let mut run = 1u32;
            while i + (run as usize) < xs.len() && xs[i + run as usize] == x0 + run {
                run += 1;
            }
            coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
            regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
            flags.push(if poison.is_some() { D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE } else { D3D12_TILE_RANGE_FLAG_NULL });
            counts.push(run);
            i += run as usize;
        }
        let offsets = vec![0u32; if poison.is_some() { counts.len() } else { 0 }];
        count_tile_calls(res_base, true, 1, tiles.len());
        let _tm = crate::hle::api_cost::time(crate::hle::api_cost::Call::WindowUnmap);
        self.queue.UpdateTileMappings(
            res,
            coords.len() as u32,
            Some(coords.as_ptr()),
            Some(regions.as_ptr()),
            poison,
            flags.len() as u32,
            Some(flags.as_ptr()),
            poison.map(|_| offsets.as_ptr()),
            Some(counts.as_ptr()),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
    }

    /// `XCL_VIEW_POISON`: every tile of `res` (`tiles` of them) onto the poison tile, before the
    /// tiles with video memory are mapped over it.
    unsafe fn poison_all(&self, res: &ID3D12Resource, tiles: u32) {
        let Some(p) = &self.poison else { return };
        if tiles == 0 {
            return;
        }
        let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
        let region = D3D12_TILE_REGION_SIZE { NumTiles: tiles, UseBox: false.into(), ..Default::default() };
        let (flag, off) = (D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE, 0u32);
        self.queue.UpdateTileMappings(res, 1, Some(&coord), Some(&region), &p.heap, 1, Some(&flag), Some(&off), Some(&tiles), D3D12_TILE_MAPPING_FLAG_NONE);
    }

    /// `XCL_VIEW_POISON`: a tile of video memory filled with `pattern`, and every tile of the
    /// windows mapped onto it (one call each, before any has video memory of its own); unmapped
    /// tiles go back to it from here on (`unmap_tiles`), and a buffer placed in a window starts
    /// on it (`place_buffer`). Many tiles of reserved buffers on one tile of a heap is what
    /// `REUSE_SINGLE_TILE` is for. Logged and left off when the tile cannot be made.
    unsafe fn poison_windows(&mut self, pattern: u32) {
        let made = (|| -> windows::core::Result<Poison> {
            let desc = D3D12_HEAP_DESC {
                SizeInBytes: TILE as u64,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
            };
            let mut heap: Option<ID3D12Heap> = None;
            self.dev.CreateHeap(&desc, &mut heap)?;
            let heap = heap.ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_FAIL))?;
            let _ = heap.SetName(windows::core::w!("xcl poison tile"));
            let mut d = buffer_desc(TILE as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            let mut buf: Option<ID3D12Resource> = None;
            self.dev.CreatePlacedResource(&heap, 0, &d, D3D12_RESOURCE_STATE_COMMON, None, &mut buf)?;
            let mut src: Option<ID3D12Resource> = None;
            self.dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut src)?;
            let (Some(buf), Some(src)) = (buf, src) else { return Err(windows::Win32::Foundation::E_FAIL.into()) };
            let mut p = std::ptr::null_mut();
            src.Map(0, None, Some(&mut p))?;
            std::slice::from_raw_parts_mut(p as *mut u32, TILE / 4).fill(pattern);
            src.Unmap(0, None);
            Ok(Poison { heap, buf, src, at: std::time::Instant::now() })
        })();
        match made {
            Ok(p) => self.poison = Some(p),
            Err(e) => {
                crate::xlog_warn!("XCL_VIEW_POISON: no poison tile ({e}) – tiles without video memory read zeros");
                return;
            }
        }
        let windows: Vec<ID3D12Resource> = self.windows.iter().map(|w| w.1.clone()).collect();
        let mut tiles = 0u64;
        for r in &windows {
            let n = (r.GetDesc().Width as usize).div_ceil(TILE) as u32;
            self.poison_all(r, n);
            tiles += n as u64;
        }
        self.refresh_poison();
        crate::xlog_info!("XCL_VIEW_POISON: {tiles} tiles of {} window(s) show {pattern:#010x} where they have no video memory", windows.len());
    }

    /// The poison tile filled with its pattern again (at most once every 16 ms: at a submit),
    /// on the sparse queue the title's queues wait for.
    unsafe fn refresh_poison(&mut self) {
        let Some((buf, src)) = self.poison.as_mut().map(|p| {
            p.at = std::time::Instant::now();
            (p.buf.clone(), p.src.clone())
        }) else {
            return;
        };
        let idx = self.begin_slot();
        self.slots[idx].list.CopyBufferRegion(&buf, 0, &src, 0, TILE as u64);
        self.submit_slot(idx);
    }

    /// `ensure_mapped` for a commit with `lazy_commits`: the tiles of `[a, a + n)` are listed
    /// (watched, scanned, their holes kept) with no video memory (`LAZY`). A tile still mapped
    /// from before (parked, or kept for a buffer) is taken back as it is.
    unsafe fn ensure_listed(&mut self, a: usize, n: usize) {
        let end = a.saturating_add(n);
        let mut t = a & !(TILE - 1);
        let mut new = false;
        while t < end {
            if !self.mapped.contains_key(&t) && !self.mirror_src.contains_key(&t) && self.window_index(t).is_some() {
                if self.parked.contains_key(&t) || self.kept.contains_key(&t) {
                    self.ensure_mapped(t, TILE, |_| 0xFFFF);
                } else {
                    self.mapped.insert(t, TileEntry::with_holes(LAZY, 0xFFFF));
                    early_add(&mut self.runs, t, t + TILE);
                    new = true;
                }
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        if new {
            MAPPED_GEN.fetch_add(1, Ordering::Release);
        }
    }

    /// The `LAZY` tiles of `[a, a + n)` get their video memory now (their mapping is pending,
    /// as `ensure_mapped` leaves it): (the tiles that did, those left without). A tile used
    /// before goes to `reused` as the run to clear, which the caller sees to before anything is
    /// written there. A moment after no video memory was found (`no_memory`) only tiles at hand
    /// are taken, as in `new_mirror`, but with `force` (a placed buffer maps its tiles once).
    unsafe fn materialize(&mut self, a: usize, n: usize, force: bool, reused: &mut Vec<(usize, usize)>) -> (usize, usize) {
        let end = a.saturating_add(n);
        let mut t = a & !(TILE - 1);
        let (mut got, mut left) = (0, 0);
        while t < end {
            if self.mapped.get(&t).is_some_and(|e| e.pool == LAZY) {
                let at_hand = !self.free.is_empty() || cooled(&self.cooling, std::time::Instant::now()) || !self.retired.is_empty();
                let wait = left > 0 || (!force && self.short_of_memory() && !at_hand);
                match if wait { None } else { self.take_tile() } {
                    Some((tile, used)) => {
                        if let Some(e) = self.mapped.get_mut(&t) {
                            e.pool = tile;
                        }
                        if used {
                            reused.push((t, TILE));
                        }
                        self.pending.push(t);
                        got += 1;
                    }
                    None => {
                        if !wait {
                            self.no_memory = Some(std::time::Instant::now());
                        }
                        left += 1;
                    }
                }
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        (got, left)
    }

    /// The title's lists about to be submitted write `ranges` on the GPU (`will_write`): their
    /// tiles listed with no video memory get it, cleared at this submit before the lists run.
    unsafe fn give_memory(&mut self, ranges: &[(usize, usize)]) {
        let mut reused = Vec::new();
        for &(a, n) in ranges {
            self.materialize(a, n, false, &mut reused);
        }
        self.to_zero.extend(reused);
    }

    /// tiles of `[a, a + n)` get video memory (mapped in their window); `initial_holes` supplies
    /// the initial 16-bit hole mask for each newly mapped 64 KiB tile. Tiles over a CPU view
    /// (`mirror_src`) are left to `new_mirror`.
    unsafe fn ensure_mapped(&mut self, a: usize, n: usize, mut initial_holes: impl FnMut(usize) -> u16) {
        if n == 0 {
            return;
        }
        let first = a & !(TILE - 1);
        let end = a.saturating_add(n);
        let mut new = false;
        let mut unparked = false;
        let mut t = first;
        while t < end {
            if !self.mapped.contains_key(&t) && !self.mirror_src.contains_key(&t) && self.window_index(t).is_some() {
                // still mapped there: no new mapping, what it held is cleared
                if let Some(tile) = self.parked.remove(&t).or_else(|| self.kept.remove(&t)) {
                    self.mapped.insert(t, TileEntry::with_holes(tile, initial_holes(t)));
                    early_add(&mut self.runs, t, t + TILE);
                    self.to_zero.insert((t, TILE));
                    unparked = true;
                } else {
                    let Some((tile, reused)) = self.take_tile() else { break };
                    self.mapped.insert(t, TileEntry::with_holes(tile, initial_holes(t)));
                    early_add(&mut self.runs, t, t + TILE);
                    if reused {
                        self.to_zero.insert((t, TILE));
                    }
                    self.pending.push(t);
                    new = true;
                }
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        if new || unparked {
            MAPPED_GEN.fetch_add(1, Ordering::Release);
        }
        if self.pending.len() >= PENDING_MAX {
            self.flush_pending();
        } else if !self.pending.is_empty() {
            pending_soon();
        }
    }

    /// The pending tiles (`pending`) mapped onto their pool tiles, one call per window, then the
    /// fence signalled (title queues wait for it in `before_execute`). An address decommitted
    /// since has no `mapped` entry and is skipped; one committed again maps its current tile. One
    /// decommitted and parked (or kept for a placed buffer) stays pending: its mapping was never
    /// sent, and a commit that takes it back from there sends none.
    unsafe fn flush_pending(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let mut addrs = std::mem::take(&mut self.pending);
        addrs.sort_unstable();
        addrs.dedup();
        let mut by_window: FxHashMap<usize, Vec<(usize, (u32, u32))>> = FxHashMap::default();
        for t in addrs {
            if let (Some(e), Some(wi)) = (self.mapped.get(&t).filter(|e| e.pool != LAZY), self.window_index(t)) {
                by_window.entry(wi).or_default().push((t, e.pool));
            } else if self.parked.contains_key(&t) || self.kept.contains_key(&t) {
                self.pending.push(t);
            }
        }
        if by_window.is_empty() {
            return;
        }
        for (wi, tiles) in by_window {
            let (b, r) = (self.windows[wi].0, self.windows[wi].1.clone());
            TILE_WORK[0].fetch_add(tiles.len() as u64, Ordering::Relaxed);
            TILE_WORK[1].fetch_add(1, Ordering::Relaxed);
            self.map_tiles(&r, b, &tiles);
        }
        self.signal();
    }

    #[inline]
    unsafe fn signal(&mut self) {
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }

    #[inline]
    unsafe fn wait_for(&self, target: u64) {
        if target == 0 || super::real::fence_completed(&self.fence) >= target {
            return;
        }
        // The event is one for all waits here, and a wait that gave up leaves its registration
        // behind: that one sets the event later, and the next wait would return at once with the
        // queue not there (10 such "after 5 s" lines within 100 ms in one run, each followed by a
        // Reset of an allocator the GPU was still in). So: until the fence says so, or 5 s.
        let t0 = std::time::Instant::now();
        loop {
            let _ = self.fence.SetEventOnCompletion(target, self.event);
            let left = 5000u64.saturating_sub(t0.elapsed().as_millis() as u64).max(1) as u32;
            windows::Win32::System::Threading::WaitForSingleObject(self.event, left);
            if super::real::fence_completed(&self.fence) >= target {
                return;
            }
            if t0.elapsed().as_millis() >= 5000 {
                break;
            }
        }
        crate::xlog_warn!("sparse: the queue has not reached fence value {target} after 5 s (at {})", super::real::fence_completed(&self.fence));
    }

    #[inline]
    unsafe fn reclaim_ring(&mut self) {
        if self.inflight.is_empty() {
            return;
        }
        let completed = super::real::fence_completed(&self.fence);
        while let Some(&(fv, end_pos)) = self.inflight.front() {
            if fv <= completed {
                self.ring_free = end_pos;
                self.inflight.pop_front();
            } else {
                break;
            }
        }
    }

    /// Acquire the next upload command slot, waiting only if that slot's previous submission is
    /// still executing on the GPU, then reset its allocator and list.
    unsafe fn begin_slot(&mut self) -> usize {
        let idx = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        let fv = self.slots[idx].fence_val;
        if fv > 0 && super::real::fence_completed(&self.fence) < fv {
            self.wait_for(fv);
            self.reclaim_ring();
        }
        let slot = &self.slots[idx];
        let _ = slot.alloc.Reset();
        let _ = slot.list.Reset(&slot.alloc, None);
        idx
    }

    /// Close and submit upload slot `idx` asynchronously (NO CPU wait — callers order GPU work
    /// via `queue.Wait(&self.fence, self.value)`).
    unsafe fn submit_slot(&mut self, idx: usize) {
        // the copies on `queue` see every tile mapped (`pending` goes first on the same queue)
        self.flush_pending();
        let closed = self.slots[idx].list.Close().is_ok();
        if closed {
            if let Ok(l) = self.slots[idx].list.cast::<ID3D12CommandList>() {
                super::title_textures::own_work(|| {
                    self.queue.ExecuteCommandLists(&[Some(l)]);
                });
            }
        }
        self.signal();
        self.slots[idx].fence_val = self.value;
        if self.ring_write > self.ring_submitted {
            self.inflight.push_back((self.value, self.ring_write));
            self.ring_submitted = self.ring_write;
        }
    }

    unsafe fn staging_ptr(&mut self) -> Option<*mut u8> {
        if self.staging.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(STAGING as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).ok()?;
            self.staging = Some((r, p as *mut u8));
        }
        self.staging.as_ref().map(|&(_, p)| p)
    }

    unsafe fn ensure_zero_buf(&mut self) -> bool {
        if self.zero_buf.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(ZERO_BUF as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            if self
                .dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut r,
                )
                .is_err()
            {
                return false;
            }
            let Some(r) = r else { return false };
            let mut p = std::ptr::null_mut();
            if r.Map(0, None, Some(&mut p)).is_err() {
                return false;
            }
            std::ptr::write_bytes(p as *mut u8, 0, ZERO_BUF);
            r.Unmap(0, None);
            self.zero_buf = Some(r);
        }
        true
    }

    unsafe fn readback_buf(&mut self) -> Option<(ID3D12Resource, *mut u8)> {
        if self.readback.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(READBACK_STAGING as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).ok()?;
            self.readback = Some((r, p as *mut u8));
        }
        self.readback.clone()
    }

    /// Clear `sent` hashes for every page overlapping `[a, a + n)`.
    fn clear_sent_range(&mut self, a: usize, n: usize) {
        let mut pg = a & !(PAGE - 1);
        let end = a + n;
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = (t + TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    entry.sent[(p - t) / PAGE] = None;
                    p += PAGE;
                }
            }
            pg = tile_end;
        }
    }

    /// Update `sent` hashes for all 64-byte chunks overlapping `[va, va + len)` so the CPU
    /// copy-back we just wrote to `va` is not re-uploaded by the next `before_execute`.
    unsafe fn sync_sent_range(&mut self, va: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut pg = va & !(PAGE - 1);
        let end = va + len;
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = (t + TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let cpu = entry.cpu(t);
                let mut p = pg;
                while p < tile_end {
                    let idx = (p - t) / PAGE;
                    if entry.holes & (1u16 << idx) == 0 {
                        let bytes = std::slice::from_raw_parts((cpu + (p - t)) as *const u8, PAGE);
                        let sent = entry.sent[idx].get_or_insert_with(|| Box::new(ZERO_HASHES));
                        let lo = va.saturating_sub(p) / CHUNK;
                        let hi = (end.min(p + PAGE) - p).div_ceil(CHUNK).min(CHUNKS);
                        for k in lo..hi {
                            sent[k] = chunk_hash(&bytes[k * CHUNK..(k + 1) * CHUNK]);
                        }
                    }
                    p += PAGE;
                }
            }
            pg = tile_end;
        }
    }

    /// Whether nobody wrote the 64-byte chunk holding `va` since it was last sent to the GPU (its
    /// page has video memory and its hash is the one known).
    unsafe fn chunk_clean(&self, va: usize) -> bool {
        let pg = va & !(PAGE - 1);
        let t = pg & !(TILE - 1);
        let idx = (pg - t) / PAGE;
        let Some(entry) = self.mapped.get(&t).filter(|e| e.pool != LAZY) else { return false };
        if entry.holes & (1u16 << idx) != 0 {
            return false;
        }
        let page = std::slice::from_raw_parts((entry.cpu(t) + (pg - t)) as *const u8, PAGE);
        let k = (va - pg) / CHUNK;
        let h = chunk_hash(&page[k * CHUNK..(k + 1) * CHUNK]);
        entry.sent[idx].as_deref().map_or(h == ZERO_HASH, |sent| h == sent[k])
    }

    /// `len` bytes from readback memory `p + off` into the title's `dst` (committed pages only, in
    /// one tile), a window address: they land at its tile's CPU address, if the tile's memory is
    /// what it was when the copy was recorded (`gen`, `TileEntry::gen`). One shown over another
    /// CPU view since, or with a page decommitted (and maybe committed again), is left out: what
    /// the GPU had there before is not that memory's. (A tile of the window's own lands as before.)
    unsafe fn copy_back(&mut self, p: *mut u8, dst: usize, off: usize, len: usize, gen: u32) {
        let mut at = dst;
        let mut src_off = off;
        let end = dst + len;
        while at < end {
            let pg = at & !(PAGE - 1);
            let t = pg & !(TILE - 1);
            let idx = (pg - t) / PAGE;
            let step = (pg + PAGE).min(end) - at;
            // (a tile with no video memory holds nothing the GPU wrote)
            if let Some(entry) = self.mapped.get_mut(&t).filter(|e| e.pool != LAZY) {
                let here = entry.cpu(t);
                if entry.holes & (1u16 << idx) == 0 && entry.gen == gen {
                    let wrote = land_page(&mut entry.sent[idx], here + (pg - t), at - pg, p.add(src_off), step);
                    if wrote && self.own_pages.len() < OWN_PAGES_MAX && super::look_stats::deep() {
                        self.own_pages.push(here + (pg - t));
                    }
                }
            }
            at += step;
            src_off += step;
        }
    }

    /// An immediate the GPU wrote at `va` (`bytes`: 4 or 8), into the title's memory there (at the
    /// tile's CPU address), if it is the memory it was when the immediate was recorded (`gen`, as
    /// in `copy_back`)
    unsafe fn store_imm(&mut self, va: usize, value: u64, bytes: u8, gen: u32) {
        let pg = va & !(PAGE - 1);
        let t = pg & !(TILE - 1);
        let Some(cpu) = self.mapped.get(&t).filter(|e| e.holes & (1u16 << ((pg - t) / PAGE)) == 0 && e.gen == gen).map(|e| e.cpu(t) + (va - t)) else { return };
        // (a chunk the title's CPU wrote since it was last sent keeps its state: it goes up with the
        // next look, with this value in it)
        let clean = self.chunk_clean(va);
        match bytes & 0x7f {
            4 => (*(cpu as *const std::sync::atomic::AtomicU32)).store(value as u32, Ordering::Release),
            8 if bytes & IMM_KEEP_MAX != 0 => {
                if (*(cpu as *const AtomicU64)).fetch_max(value, Ordering::AcqRel) > value {
                    static LATE: AtomicU64 = AtomicU64::new(0);
                    let k = LATE.fetch_add(1, Ordering::Relaxed) + 1;
                    if k.is_power_of_two() {
                        crate::xlog_info!("sparse: {k} immediates so far left out – their cell is a counter that held a later value ({value:#x} for {va:#x})");
                    }
                    return;
                }
            }
            8 => (*(cpu as *const AtomicU64)).store(value, Ordering::Release),
            _ => return,
        }
        if self.own_pages.len() < OWN_PAGES_MAX && super::look_stats::deep() {
            self.own_pages.push(cpu & !(PAGE - 1));
        }
        if clean {
            self.sync_sent_range(va, (bytes & 0x7f) as usize);
        }
    }

    /// A copy back's list and a readback buffer of at least `need` bytes (reused when one fits)
    unsafe fn rb_job(&mut self, need: usize, ty: D3D12_COMMAND_LIST_TYPE) -> Option<RbJob> {
        if let Some(i) = self.rb_pool.iter().enumerate().filter(|(_, j)| j.cap >= need && j.ty == ty).min_by_key(|(_, j)| j.cap).map(|(i, _)| i) {
            return Some(self.rb_pool.swap_remove(i));
        }
        let cap = need.max(64 << 10).next_power_of_two().min(READBACK_STAGING);
        let alloc: ID3D12CommandAllocator = self.dev.CreateCommandAllocator(ty).ok()?;
        let list: ID3D12GraphicsCommandList = self.dev.CreateCommandList(0, ty, &alloc, None).ok()?;
        list.Close().ok()?;
        let _ = list.SetName(windows::core::w!("xcl sparse readback list"));
        let mut d = buffer_desc(cap as u64);
        d.Flags = D3D12_RESOURCE_FLAG_NONE;
        let mut r: Option<ID3D12Resource> = None;
        self.dev
            .CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r)
            .ok()?;
        let buf = r?;
        let mut p: *mut c_void = std::ptr::null_mut();
        buf.Map(0, None, Some(&mut p)).ok()?;
        Some(RbJob { alloc, list, buf, ptr: p as *mut u8, cap, rb_v: 0, done_v: 0, copies: Vec::new(), ty, fences: None, seq: 0 })
    }

    /// Land submitted copies back in the title's memory as far as `how` says: every job that is
    /// done, a queue's own in their order (a job of a queue that waits does not hold back the
    /// other queues' jobs).
    unsafe fn drain_readbacks(&mut self, how: Drain) {
        self.drain_readbacks_spin(how, 2000);
    }

    /// `drain_readbacks`, waiting for a finished list's copy for `spin_us` at most.
    unsafe fn drain_readbacks_spin(&mut self, how: Drain, spin_us: u128) {
        let rbq_done = self.rbq.as_ref().map(|(_, f)| f.clone());
        let mut i = 0;
        let mut k = 0;
        loop {
            // the immediates of submissions before the next copy back first
            while k < self.imm_jobs.len() && self.rb_jobs.get(i).is_none_or(|j| self.imm_jobs[k].seq < j.seq) {
                if super::real::fence_completed(&self.imm_jobs[k].fence) < self.imm_jobs[k].value {
                    k += 1;
                    continue;
                }
                let Some(job) = self.imm_jobs.remove(k) else { break };
                for &(va, value, bytes, gen) in &job.writes {
                    self.store_imm(va, value, bytes, gen);
                }
                RB_PENDING.fetch_sub(1, Ordering::Release);
            }
            if i >= self.rb_jobs.len() {
                break;
            }
            let (title_f, done_f, rb_v, done_v) = {
                let j = &self.rb_jobs[i];
                match (&j.fences, &rbq_done) {
                    (Some((a, b)), _) => (a.clone(), b.clone(), j.rb_v, j.done_v),
                    (None, Some(d)) => (self.rb_fence.clone(), d.clone(), j.rb_v, j.done_v),
                    (None, None) => {
                        i += 1;
                        continue;
                    }
                }
            };
            if super::real::fence_completed(&done_f) < done_v {
                if how == Drain::Finished || super::real::fence_completed(&title_f) < rb_v {
                    i += 1;
                    continue;
                }
                // the title's lists are done: only the copy itself is left
                // (for 2 ms at most: the state's lock is held here, and every submit needs it; a copy
                // that stands behind something else on its queue lands when it is done)
                // (no event: `self.event` is the tile queue's, see `wait_for`)
                let t0 = std::time::Instant::now();
                while super::real::fence_completed(&done_f) < done_v && t0.elapsed().as_micros() < spin_us {
                    std::thread::yield_now();
                }
                if super::real::fence_completed(&done_f) < done_v {
                    static LATE: AtomicU64 = AtomicU64::new(0);
                    let k = LATE.fetch_add(1, Ordering::Relaxed) + 1;
                    if k.is_power_of_two() {
                        crate::xlog_warn!("sparse: {k} copies back so far not finished 2 ms after the title's work they follow – left for later");
                    }
                    i += 1;
                    continue;
                }
            }
            let Some(mut job) = self.rb_jobs.remove(i) else { break };
            let copies = std::mem::take(&mut job.copies);
            for &(dst, off, len, gen) in &copies {
                self.copy_back(job.ptr, dst, off, len, gen);
            }
            TILE_WORK[3].fetch_add(copies.iter().map(|c| c.2 as u64).sum(), Ordering::Relaxed);
            RB_PENDING.fetch_sub(1, Ordering::Release);
            job.copies = copies;
            job.copies.clear();
            // (a burst of copies back does not keep its buffers for good: each is up to 16 MiB of
            // host memory)
            if self.rb_pool.len() < 32 {
                self.rb_pool.push(job);
            }
        }
    }

    /// `readback_ranges` without a CPU wait: the copies are recorded on the title's own `queue`,
    /// right behind the lists that wrote `ranges` (so they wait for those lists and for nothing
    /// else), and land in the title's memory in `drain_readbacks` once `fences.1` reaches `v`.
    unsafe fn readback_async(&mut self, queue: &ID3D12CommandQueue, ranges: &[(usize, usize)], fences: &(ID3D12Fence, ID3D12Fence), v: u64) {
        let ty = queue.GetDesc().Type;
        // the mapped parts, in readback-buffer sized batches: (window, address, bytes,
        // `TileEntry::gen` of its tile)
        let mut batches: Vec<Vec<(usize, usize, usize, u32)>> = vec![Vec::new()];
        let mut used = 0usize;
        for &(a, n) in ranges {
            let mut at = 0usize;
            while at < n {
                let cur = a + at;
                let tile = cur & !(TILE - 1);
                let Some(wi) = self.window_index(cur) else { break };
                let to_tile_end = (tile + TILE) - cur;
                // (a tile with no video memory has nothing to copy back, like one not mapped)
                let Some(gen) = self.mapped.get(&tile).filter(|e| e.pool != LAZY).map(|e| e.gen) else {
                    at += (n - at).min(to_tile_end);
                    continue;
                };
                if used == READBACK_STAGING {
                    batches.push(Vec::new());
                    used = 0;
                }
                let to_win_end = (self.windows[wi].0 + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - at).min(READBACK_STAGING - used).min(to_tile_end).min(to_win_end);
                batches.last_mut().expect("batch").push((wi, cur, chunk, gen));
                used += chunk;
                at += chunk;
            }
        }
        for batch in batches.into_iter().filter(|b| !b.is_empty()) {
            let need = batch.iter().map(|c| c.2).sum::<usize>();
            let Some(mut job) = self.rb_job(need, ty) else {
                crate::xlog_warn!("sparse: no readback buffer for a copy back of {need} bytes – left out");
                // (the Signal after the loop still goes out: what was recorded for it stands)
                break;
            };
            let _ = job.alloc.Reset();
            let _ = job.list.Reset(&job.alloc, None);
            let mut off = 0usize;
            for (wi, cur, chunk, gen) in batch {
                let (base, res) = (self.windows[wi].0, self.windows[wi].1.clone());
                job.list.CopyBufferRegion(&job.buf, off as u64, &res, (cur - base) as u64, chunk as u64);
                job.copies.push((cur, off, chunk, gen));
                off += chunk;
            }
            if job.list.Close().is_ok() {
                pc_execute(queue, &job.list);
            }
            job.rb_v = v;
            job.done_v = v;
            job.fences = Some(fences.clone());
            self.job_seq += 1;
            job.seq = self.job_seq;
            self.rb_jobs.push_back(job);
            RB_PENDING.fetch_add(1, Ordering::Release);
            RB_ASYNC.store(true, Ordering::Relaxed);
        }
        let _ = queue.Signal(&fences.1, v);
    }

    /// Copy `ranges` (address, bytes) from sparse VRAM back into the CPU view and refresh
    /// `sent` hashes. Caller has already ordered `self.queue` after any GPU writes if needed.
    unsafe fn readback_ranges(&mut self, ranges: &[(usize, usize)]) {
        self.flush_pending();
        // copies back of earlier submissions first: their data is older than this one's
        self.drain_readbacks(Drain::TitleDone);
        let Some((rb, p)) = self.readback_buf() else { return };
        let mut pending: Vec<(usize, usize, usize, u32)> = Vec::new(); // (va, rb_off, chunk, gen of its tile)
        let mut used = 0usize;
        let mut open = false;
        let flush_and_copy = |s: &mut State, open: &mut bool, pending: &mut Vec<(usize, usize, usize, u32)>, used: &mut usize, p: *mut u8| {
            if *open {
                *open = false;
                if s.rb_list.Close().is_ok() {
                    if let Ok(l) = s.rb_list.cast::<ID3D12CommandList>() {
                        super::title_textures::own_work(|| {
                            s.queue.ExecuteCommandLists(&[Some(l)]);
                        });
                    }
                }
                s.signal();
                s.rb_last_fence = s.value;
                s.wait_for(s.value);
                s.reclaim_ring();
            }
            for &(dst, off, len, gen) in pending.iter() {
                s.copy_back(p, dst, off, len, gen);
            }
            pending.clear();
            *used = 0;
        };
        for &(a, n) in ranges {
            if n == 0 {
                continue;
            }
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let tile = cur & !(TILE - 1);
                let Some(wi) = self.window_index(cur) else { break };
                let b = self.windows[wi].0;
                let to_tile_end = (tile + TILE) - cur;
                // (a tile with no video memory has nothing to copy back: the GPU's writes there went
                // nowhere, and a tile it got now would hold what another address left in it)
                let Some(gen) = self.mapped.get(&tile).filter(|e| e.pool != LAZY).map(|e| e.gen) else {
                    done += (n - done).min(to_tile_end);
                    continue;
                };
                if used == READBACK_STAGING {
                    flush_and_copy(self, &mut open, &mut pending, &mut used, p);
                }
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - done).min(READBACK_STAGING - used).min(to_tile_end).min(to_win_end);
                if !open {
                    self.wait_for(self.rb_last_fence);
                    let _ = self.rb_alloc.Reset();
                    let _ = self.rb_list.Reset(&self.rb_alloc, None);
                    open = true;
                }
                self.rb_list.CopyBufferRegion(&rb, used as u64, &self.windows[wi].1, (cur - b) as u64, chunk as u64);
                pending.push((cur, used, chunk, gen));
                used += chunk;
                done += chunk;
            }
        }
        if open {
            flush_and_copy(self, &mut open, &mut pending, &mut used, p);
        }
    }

    /// Asynchronously zero `runs` (address, bytes) in sparse VRAM using the pre-zeroed `zero_buf`
    /// (zero staging ring bytes consumed, zero CPU stall).
    unsafe fn upload_zeros(&mut self, runs: &[(usize, usize)]) {
        // (a tile with no video memory reads zeros already; a run goes on into its neighbours)
        let kept: Vec<(usize, usize)>;
        let runs = if lazy_commits() {
            kept = runs_where(runs, |t| self.mapped.get(&t).is_none_or(|e| e.pool != LAZY));
            &kept[..]
        } else {
            runs
        };
        if runs.is_empty() || !self.ensure_zero_buf() {
            return;
        }
        let slot_idx = self.begin_slot();
        let mut last_wi: Option<usize> = None;
        let zero_buf = self.zero_buf.as_ref().unwrap();
        let list = &self.slots[slot_idx].list;
        for &(a, n) in runs {
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let wi = match last_wi {
                    Some(i) if cur >= self.windows[i].0 && cur < self.windows[i].0 + crate::gpu::uma::WINDOW => i,
                    _ => match self.window_index(cur) {
                        Some(i) => {
                            last_wi = Some(i);
                            i
                        }
                        None => break,
                    },
                };
                let b = self.windows[wi].0;
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - done).min(ZERO_BUF).min(to_win_end);
                list.CopyBufferRegion(&self.windows[wi].1, (cur - b) as u64, zero_buf, 0, chunk as u64);
                done += chunk;
            }
        }
        self.submit_slot(slot_idx);
    }

    /// Asynchronously copy `runs` (address, bytes) from the CPU view into sparse VRAM tiles via
    /// the 64 MiB ring staging buffer (zero CPU stall on the common path).
    unsafe fn upload_dirty(&mut self, runs: &[(usize, usize)]) {
        if runs.is_empty() {
            return;
        }
        // the CPU's first bytes for a tile listed with no video memory: it gets it now, and its
        // mapping is sent before the copies that fill it. One used before is cleared first, here
        // and not at the next submit (`to_zero`), which would clear these bytes with it
        let kept: Vec<(usize, usize)>;
        let mut runs = runs;
        if lazy_commits() {
            let mut reused = Vec::new();
            let (mut got, mut left) = (0, 0);
            for &(a, n) in runs {
                let (g, l) = self.materialize(a, n, false, &mut reused);
                got += g;
                left += l;
            }
            if got > 0 {
                self.flush_pending();
                self.upload_zeros(&reused);
            }
            // none to be had: the bytes are not sent, whatever the scan noted, and all of the tile
            // goes up once there is some (`starved`)
            if left > 0 {
                let mut starved: Vec<usize> = Vec::new();
                kept = runs_where(runs, |t| {
                    let has = self.mapped.get(&t).is_none_or(|e| e.pool != LAZY);
                    if !has && starved.last() != Some(&t) {
                        starved.push(t);
                    }
                    has
                });
                for &t in &starved {
                    self.clear_sent_range(t, TILE);
                }
                self.starved.extend(starved);
                runs = &kept[..];
                if runs.is_empty() {
                    return;
                }
            }
        }
        let Some(p) = self.staging_ptr() else { return };
        let mut open_slot: Option<usize> = None;
        let mut last_wi: Option<usize> = None;
        // the copies into the ring (source address, ring offset, bytes), done before their slot is
        // submitted, by the worker threads when there are many
        let mut jobs: Vec<(usize, usize, usize)> = Vec::new();
        // the many small ones first, together (the windows the open slot's dispatches write are
        // UAVs in it until their first copy)
        let mut uavs: Vec<usize> = Vec::new();
        let single = self.scatter_small(runs, p, &mut open_slot, &mut jobs, &mut uavs);
        for &(a, n) in single.as_deref().unwrap_or(runs) {
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let wi = match last_wi {
                    Some(i) if cur >= self.windows[i].0 && cur < self.windows[i].0 + crate::gpu::uma::WINDOW => i,
                    _ => match self.window_index(cur) {
                        Some(i) => {
                            last_wi = Some(i);
                            i
                        }
                        None => break,
                    },
                };
                let b = self.windows[wi].0;
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let (src, len) = self.cpu_span(cur, (n - done).min(to_win_end));
                let (aligned, chunk) = plan_ring_chunk(self.ring_write, len, STAGING);
                self.ring_room(aligned + chunk as u64, p, &mut open_slot, &mut jobs);
                let slot_idx = match open_slot {
                    Some(idx) => idx,
                    None => {
                        uavs.clear();
                        let idx = self.begin_slot();
                        open_slot = Some(idx);
                        idx
                    }
                };
                if let Some(k) = uavs.iter().position(|&w| w == wi) {
                    uavs.swap_remove(k);
                    self.slots[slot_idx].list.ResourceBarrier(&[uav_to_copy(&self.windows[wi].1)]);
                }
                let dst_off = (aligned as usize) % STAGING;
                jobs.push((src, dst_off, chunk));
                let staging_res = &self.staging.as_ref().unwrap().0;
                self.slots[slot_idx].list.CopyBufferRegion(&self.windows[wi].1, (cur - b) as u64, staging_res, dst_off as u64, chunk as u64);
                self.ring_write = aligned + chunk as u64;
                done += chunk;
            }
        }
        if let Some(slot_idx) = open_slot {
            run_copies(&mut jobs, p);
            self.submit_slot(slot_idx);
        }
    }

    /// Room in the upload ring up to its position `end`: what the GPU is done with is taken back;
    /// when that is not enough, the open slot is submitted and the GPU waited for.
    unsafe fn ring_room(&mut self, end: u64, p: *mut u8, open_slot: &mut Option<usize>, jobs: &mut Vec<(usize, usize, usize)>) {
        if end.saturating_sub(self.ring_free) <= STAGING as u64 {
            return;
        }
        self.reclaim_ring();
        if end.saturating_sub(self.ring_free) <= STAGING as u64 {
            return;
        }
        if let Some(slot_idx) = open_slot.take() {
            run_copies(jobs, p);
            self.submit_slot(slot_idx);
            self.reclaim_ring();
        }
        while end.saturating_sub(self.ring_free) > STAGING as u64 {
            let Some(&(fv, _)) = self.inflight.front() else { break };
            self.wait_for(fv);
            self.reclaim_ring();
        }
    }

    /// Where the CPU bytes of window addresses `[a, a + n)` are: (CPU address of `a`, bytes from
    /// `a` that lie in one piece there). The piece ends where a tile's CPU address does not follow
    /// the one before it: a GPU view shown over another part of its CPU view, or a tile of the
    /// window's own next to one over a CPU view (whose own address the CPU cannot read).
    fn cpu_span(&self, a: usize, n: usize) -> (usize, usize) {
        if self.mirrors.is_empty() {
            return (a, n);
        }
        let src = |t: usize| self.mapped.get(&t).map_or(t, |e| e.cpu(t));
        let t0 = a & !(TILE - 1);
        let s0 = src(t0);
        let mut end = (t0 + TILE).min(a + n);
        while end < a + n && src(end) == s0 + (end - t0) {
            end = (end + TILE).min(a + n);
        }
        (s0 + (a - t0), end - a)
    }

    /// Record the small ones of `runs` as dispatches (`scatter`) that put many of them in their
    /// places at once, in the open slot: the runs left for a copy each (None: all of them). `uavs`
    /// are the windows the open slot's dispatches write.
    unsafe fn scatter_small(&mut self, runs: &[(usize, usize)], p: *mut u8, open_slot: &mut Option<usize>, jobs: &mut Vec<(usize, usize, usize)>, uavs: &mut Vec<usize>) -> Option<Vec<(usize, usize)>> {
        if runs.len() < SCATTER_FROM || !super::scatter::on() {
            return None;
        }
        // (a run that ends in the next window is cut where the window ends, by the copies; one
        // whose CPU bytes are not in one piece is cut where they break, by the copies too)
        let small = |&(a, n): &(usize, usize)| n < SCATTER_RUN && a % CHUNK == 0 && n % CHUNK == 0 && self.window_index(a).is_some_and(|w| a + n <= self.windows[w].0 + crate::gpu::uma::WINDOW) && self.cpu_span(a, n).1 == n;
        let (small, single): (Vec<_>, Vec<_>) = runs.iter().copied().partition(small);
        if small.len() < SCATTER_FROM {
            return None;
        }
        let pass = super::scatter::of(&self.dev)?;
        let ring_va = self.staging.as_ref()?.0.GetGPUVirtualAddress();
        let pieces = super::scatter::pieces(&small, SCATTER_CHUNKS, |a| self.window_index(a).map(|w| (w, self.windows[w].0)));
        let mut bound = None;
        for piece in &pieces {
            // the table of the piece and its chunks in one part of the ring
            let bytes = super::scatter::Scatter::bytes(piece.chunks);
            let mut at = self.ring_write.next_multiple_of(RING_ALIGN as u64);
            let to_end = STAGING - at as usize % STAGING;
            if bytes > to_end {
                at += to_end as u64;
            }
            self.ring_room(at + bytes as u64, p, open_slot, jobs);
            let slot_idx = match *open_slot {
                Some(idx) => idx,
                None => {
                    uavs.clear();
                    let idx = self.begin_slot();
                    *open_slot = Some(idx);
                    idx
                }
            };
            let list = self.slots[slot_idx].list.clone();
            if bound != Some((slot_idx, self.slots[slot_idx].fence_val)) {
                bound = Some((slot_idx, self.slots[slot_idx].fence_val));
                pass.bind(&list);
            }
            let off = at as usize % STAGING;
            let mut place = p.add(off) as *mut u32;
            let mut data = off + super::scatter::Scatter::data_at(piece.chunks);
            for &(a, n) in &small[piece.runs.clone()] {
                for k in 0..n / CHUNK {
                    place.write((a - piece.base + k * CHUNK) as u32);
                    place = place.add(1);
                }
                jobs.push((self.cpu_span(a, n).0, data, n));
                data += n;
            }
            let (base, buf) = &self.windows[piece.window];
            pass.dispatch(&list, piece.chunks, ring_va + off as u64, buf.GetGPUVirtualAddress() + (piece.base - base) as u64);
            if !uavs.contains(&piece.window) {
                uavs.push(piece.window);
            }
            self.ring_write = at + bytes as u64;
            SCATTERED[0].fetch_add(1, Ordering::Relaxed);
        }
        SCATTERED[1].fetch_add(small.len() as u64, Ordering::Relaxed);
        Some(single)
    }
}

/// dispatches of `scatter_small` so far, and the runs in them
static SCATTERED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// `r`, written by a dispatch of the list, becomes the destination of a copy.
fn uav_to_copy(r: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: the list keeps no reference and the window outlives it
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                StateAfter: D3D12_RESOURCE_STATE_COPY_DEST,
            }),
        },
    }
}

/// Allocate a zeroed, write-watched block in the sparse VRAM windows (never straddling a
/// 4 GiB window boundary when `size <= 4 GiB`). Commits CPU pages immediately but defers
/// mapping 64 KiB VRAM tiles until the CPU writes non-zero data (or `place_buffer` pins them).
// PORT-CHECK: #877's lazy ranges kept beside HEAD's own lazy tiles (`XCL_SPARSE_LAZY`, `LAZY` entries); kernelx / real.rs / list_x still call alloc_lazy / ensure_lazy_mapped
pub unsafe fn alloc_lazy(size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc(size, align)?;
    if VirtualAlloc(addr as *const c_void, rounded, MEM_COMMIT, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    let (start, end) = page_span(addr, rounded);
    early_add(&mut LAZY_RANGES.lock().unwrap(), start, end);
    LAZY_ACTIVE.store(true, Ordering::Release);
    MAPPED_GEN.fetch_add(1, Ordering::Release);
    Some((addr, rounded))
}

/// Ensure any lazily-allocated sparse tiles overlapping `[a, a + n)` are backed by VRAM
/// (called when the GPU is about to write directly to a buffer UAV or memory-op target).
pub unsafe fn ensure_lazy_mapped(a: usize, n: usize) {
    if n == 0 || !LAZY_ACTIVE.load(Ordering::Acquire) {
        return;
    }
    let mut g = lock_state();
    let Some(s) = g.as_mut() else { return };
    if s.window_index(a).is_none() {
        return;
    }
    let lazy = LAZY_RANGES.lock().unwrap();
    if lazy.is_empty() {
        return;
    }
    let first = a & !(TILE - 1);
    let end = a.saturating_add(n);
    let mut t = first;
    while t < end {
        if !s.mapped.contains_key(&t) && lazy_overlaps_tile(&lazy, t) {
            let holes = lazy_tile_holes(&lazy, t);
            s.ensure_mapped(t, TILE, |_| holes);
        }
        match t.checked_add(TILE) {
            Some(next) => t = next,
            None => break,
        }
    }
}

/// Allocate a zeroed, write-watched block in the sparse VRAM windows (never straddling a
/// 4 GiB window boundary when `size <= 4 GiB`). Commits the CPU pages and maps VRAM tiles
/// (or queues them until the D3D12 device exists). Returns `(address, rounded_size)`.
pub unsafe fn alloc(size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc(size, align)?;
    if VirtualAlloc(addr as *const c_void, rounded, MEM_COMMIT, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    committed(addr, rounded);
    Some((addr, rounded))
}

/// Allocate a zeroed, write-watched block strictly inside sparse window `window` (e.g.
/// window 0 for 32-bit-addressed descriptor heaps). Returns `(address, rounded_size)`.
pub unsafe fn alloc_in(window: usize, size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc_in(window, size, align)?;
    if VirtualAlloc(addr as *const c_void, rounded, MEM_COMMIT, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    committed(addr, rounded);
    Some((addr, rounded))
}

/// Release a sparse allocation from `alloc` / `alloc_in` (false: not a sparse allocation).
pub unsafe fn free(p: usize) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    if !u.is_sparse(p) {
        return false;
    }
    // decommitted before the address goes back: once released, another thread may get it,
    // commit it and write there
    let Some((start, len)) = u.sparse_reservation(p).filter(|&(start, _)| start == p) else { return false };
    decommitted(start, len);
    VirtualFree(p as *mut c_void, len, MEM_DECOMMIT);
    u.sparse_release(p).is_some()
}

/// After `queue.ExecuteCommandLists`: copy GPU-written sparse ranges (e.g. `Write32/64`,
/// `FillMemoryWith32BitValueX`, `CopyMemoryToMemoryX`) back to their CPU view so CPU reads
/// and spin-waits see the GPU writes without clobbering `sent` hashes.
/// The queue tile mappings go out on, once there is one. Answered without `STATE` (`QUEUE`):
/// a caller may be inside a D3D12 call that a holder of `STATE` waits for.
pub fn queue() -> Option<ID3D12CommandQueue> {
    QUEUE.lock().unwrap().clone()
}

pub unsafe fn readback_after_execute(queue: &ID3D12CommandQueue, ranges: &[(usize, usize)], imms: &[(usize, u64, u8)]) {
    if ranges.is_empty() && imms.is_empty() {
        return;
    }
    let t0 = super::look_stats::deep().then(std::time::Instant::now);
    let mut g = lock_for_submit();
    let _held = holding(Holder::After);
    if let Some(t0) = t0 {
        HELD.add(HELD_AFTER_WAIT, t0.elapsed().as_micros() as u64);
    }
    let Some(s) = g.as_mut() else { return };
    s.note_queue(queue);
    s.after_execute(queue, ranges, imms);
}

/// The PC `ExecuteCommandLists` of `queue` for `list` (an adopted title queue's own method,
/// not XCL's hook: this is not a submit of the title's).
pub(super) unsafe fn pc_execute(queue: &ID3D12CommandQueue, list: &ID3D12GraphicsCommandList) {
    type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
    const SLOT: usize = 10;
    let q = queue.as_raw();
    let f: ExecFn = if super::real::is_adopted(q) { std::mem::transmute(super::real::orig_fn(q, SLOT)) } else { std::mem::transmute(*(*(q as *const *const usize)).add(SLOT)) };
    let one = [list.as_raw()];
    f(q, 1, one.as_ptr());
}

/// copies back submitted and not landed yet, and whether any ran on the readback queue
static RB_PENDING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static RB_ASYNC: AtomicBool = AtomicBool::new(false);

/// Whether copies back land asynchronously (any ran on the readback queue so far)
pub fn readbacks_async() -> bool {
    RB_ASYNC.load(Ordering::Relaxed)
}

/// Copies back that are done on the GPU, into the title's memory, and those whose title work
/// is done (waiting for the copy): before a CPU read of memory the title's lists wrote.
pub unsafe fn settle_readbacks() {
    let _cost = crate::hle::api_cost::time(crate::hle::api_cost::Call::SettleReadbacks);
    if RB_PENDING.load(Ordering::Acquire) == 0 {
        return;
    }
    let mut g = STATE.lock().unwrap();
    let _held = holding(Holder::Settle);
    if let Some(s) = g.as_mut() {
        s.drain_readbacks(Drain::TitleDone);
    }
}

/// `settle_readbacks` from a fence observation: the state's lock is only tried (for ~2 ms), as
/// XCL may be waiting on a fence of its own with the lock held.
pub unsafe fn settle_readbacks_try() {
    settle_readbacks_within(5);
}

/// `settle_readbacks` with the state's lock waited for `ms` ms at most (false: it was not had:
/// what is pending stays pending). A copy back whose title work is done is waited for 0.3 ms.
pub unsafe fn settle_readbacks_within(ms: u64) -> bool {
    if RB_PENDING.load(Ordering::Acquire) == 0 {
        return true;
    }
    // (`XCL_LOOK_STATS=2`: the time a title's thread spends here when it looks at a fence)
    let _cost = super::look_stats::deep().then(|| crate::hle::api_cost::time(crate::hle::api_cost::Call::FenceSettle));
    let Some(mut g) = try_lock_giving_way(&STATE, &SUBMITS_WAITING, std::time::Duration::from_millis(ms), submit_lock_first()) else { return false };
    let _held = holding(Holder::Settle);
    if let Some(s) = g.as_mut() {
        s.drain_readbacks_spin(Drain::TitleDone, 300);
    }
    true
}

/// The thread that moves finished copies back into the title's memory, so it sees what its
/// lists wrote without submitting again (started with the first copy back).
fn start_readback_thread() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-sparse-readback".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_micros(500));
            // (`XCL_SUBMIT_LOCK_FIRST=1`: nothing to land, or a submit waits for the lock: not now)
            if submit_lock_first() && (RB_PENDING.load(Ordering::Acquire) == 0 || SUBMITS_WAITING.load(Ordering::Acquire) > 0) {
                continue;
            }
            if let Ok(mut g) = STATE.try_lock() {
                let _held = holding(Holder::ReadbackThread);
                if let Some(s) = g.as_mut().filter(|s| !s.rb_jobs.is_empty() || !s.imm_jobs.is_empty()) {
                    unsafe { s.drain_readbacks(Drain::Finished) };
                }
            }
        });
    });
}

impl State {
    /// `readback_after_execute` on this state
    unsafe fn after_execute(&mut self, queue: &ID3D12CommandQueue, ranges: &[(usize, usize)], imms: &[(usize, u64, u8)]) {
        let s = self;
        let mut filtered: Vec<(usize, usize)> = ranges.iter().copied().filter(|&(a, n)| n > 0 && s.window_index(a).is_some()).collect();
        // immediates: stored by XCL when the lists are done. One that another write of these
        // lists overlaps, and all of them without a queue of the title's to order them on,
        // come back with a copy like the rest.
        let mut imm: Vec<(usize, u64, u8, u32)> = Vec::new();
        for &(a, v, n) in imms {
            if s.window_index(a).is_none() {
                continue;
            }
            let len = (n & 0x7f) as usize;
            if s.rbq.is_none() || ranges.iter().any(|&(ra, rn)| a < ra + rn && ra < a + len) {
                filtered.push((a, len));
            } else {
                // (with the memory it is for, `TileEntry::gen`: no tile there yet counts as the
                // window's own, as before)
                imm.push((a, v, n, s.mapped.get(&(a & !(TILE - 1))).map_or(0, |e| e.gen)));
            }
        }
        if filtered.is_empty() && imm.is_empty() {
            return;
        }
        filtered.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(filtered.len());
        for (a, n) in filtered {
            match merged.last_mut() {
                Some(last) if a <= last.0.saturating_add(last.1) => {
                    let end = (last.0 + last.1).max(a + n);
                    last.1 = end - last.0;
                }
                _ => merged.push((a, n)),
            }
        }
        if s.rbq.is_some() {
            // on this queue: [the title's lists] Signal(first, v) [the copies] Signal(second, v)
            let key = queue.as_raw() as usize;
            if !s.rb_queue_fences.contains_key(&key) {
                let (Ok(a), Ok(b)) = (s.dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE), s.dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE)) else { return };
                s.rb_queue_fences.insert(key, (a, b, 0));
            }
            let (fences, v) = {
                let e = s.rb_queue_fences.get_mut(&key).expect("fences");
                e.2 += 1;
                ((e.0.clone(), e.1.clone()), e.2)
            };
            let _ = queue.Signal(&fences.0, v);
            if !imm.is_empty() {
                s.job_seq += 1;
                s.imm_jobs.push_back(ImmJob { seq: s.job_seq, fence: fences.0.clone(), value: v, writes: imm });
                RB_PENDING.fetch_add(1, Ordering::Release);
                RB_ASYNC.store(true, Ordering::Relaxed);
            }
            if !merged.is_empty() {
                s.readback_async(queue, &merged, &fences, v);
            }
            start_readback_thread();
            return;
        }
        s.rb_value += 1;
        let v = s.rb_value;
        if queue.Signal(&s.rb_fence, v).is_ok() {
            let _ = s.queue.Wait(&s.rb_fence, v);
            super::list_x::note_internal_wait(s.queue.as_raw() as usize, "sparse readback behind a title queue", &s.rb_fence, v);
        }
        s.readback_ranges(&merged);
    }
}

/// Synchronize `[va, va + len)` in a sparse window for a CPU read (`ID3D12Resource::Map` with
/// a non-empty read range, or `Map(0, NULL)` on a sparse readback resource): flushes pending
/// tile clears and unsubmitted CPU-changed chunks first so CPU writes are preserved, then
/// copies the range back from VRAM into CPU memory.
pub unsafe fn readback_now(va: usize, len: usize) {
    if len == 0 {
        return;
    }
    let mut g = lock_state();
    let _held = holding(Holder::CpuRead);
    let Some(s) = g.as_mut() else { return };
    if s.window_index(va).is_none() {
        return;
    }
    let zeroed = zero_runs(std::mem::take(&mut s.to_zero));
    if !zeroed.is_empty() {
        for &(a, n) in &zeroed {
            s.clear_sent_range(a, n);
        }
        s.upload_zeros(&zeroed);
    }
    // Preserve any unsubmitted CPU writes on pages overlapping `[va, va + len)`
    let mut dirty_runs: Vec<(usize, usize)> = Vec::new();
    let mut pg = va & !(PAGE - 1);
    let end = va + len;
    while pg < end {
        let t = pg & !(TILE - 1);
        let tile_end = (t + TILE).min(end);
        if let Some(entry) = s.mapped.get_mut(&t) {
            let cpu = entry.cpu(t);
            let mut p = pg;
            while p < tile_end {
                let idx = (p - t) / PAGE;
                if entry.holes & (1u16 << idx) == 0 {
                    let bytes = std::slice::from_raw_parts((cpu + (p - t)) as *const u8, PAGE);
                    scan_page(bytes, p, &mut entry.sent[idx], &mut dirty_runs);
                }
                p += PAGE;
            }
        }
        pg = tile_end;
    }
    if !dirty_runs.is_empty() {
        s.upload_dirty(&dirty_runs);
    }
    s.readback_ranges(&[(va, len)]);
}

/// Submits waiting for the state's lock. A thread that takes it again and again – a title mapping
/// its memory a page at a time – can keep them out for seconds: the lock goes to the first to ask
/// once it is free, and the thread that just gave it up is back before a waiting one has woken (a
/// retail title's loading screen stood still for 8 s while another of its threads mapped 1.2 GB).
static SUBMITS_WAITING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// what a commit waits for them at most
const GIVE_WAY_US: u128 = 2000;

/// `m` for a submit: the others let it in first (`lock_in_turn`).
fn lock_first<'a, T>(m: &'a Mutex<T>, waiting: &std::sync::atomic::AtomicUsize) -> std::sync::MutexGuard<'a, T> {
    waiting.fetch_add(1, Ordering::AcqRel);
    let g = m.lock();
    waiting.fetch_sub(1, Ordering::AcqRel);
    g.unwrap()
}

/// Submits in `before_execute` waiting for the state's lock (they send the put-off mappings).
static FLUSHERS_WAITING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `XCL_SUBMIT_LOCK_FIRST=1` (off unless set): a submit's other uses of the state's lock (its
/// after stage's copies back, its ranges' mapped tiles) are let in first like its `before_execute`,
/// and the threads that only try the lock in a loop (copies back settled at a fence look, the
/// readback thread, put-off mappings sent) do not try it while a submit waits. A retail title's
/// world: each submit waited ~2 ms for the lock while title threads looked at fences ~170 times a
/// second, each trying the lock for up to 5 ms; a thread that tries the lock as it is let go takes
/// it before the waiting submit has woken.
fn submit_lock_first() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SUBMIT_LOCK_FIRST").is_ok_and(|v| v.trim() == "1"))
}

/// The state's lock for a submit's after stage and ranges: `lock_first` with
/// `XCL_SUBMIT_LOCK_FIRST=1`, else `lock_state` as before.
fn lock_for_submit() -> std::sync::MutexGuard<'static, Option<State>> {
    if !submit_lock_first() {
        return lock_state();
    }
    let mut g = lock_first(&STATE, &SUBMITS_WAITING);
    forget_released(&mut g);
    g
}

/// `m` tried until it is had or `budget` has gone by (None). With `give_way`, it is not tried
/// while a submit waits for it (`waiting`); that time does not count against `budget`, up to
/// `budget` more (a caller that may not wait for the lock still gives up in the end).
fn try_lock_giving_way<'a, T>(m: &'a Mutex<T>, waiting: &std::sync::atomic::AtomicUsize, budget: std::time::Duration, give_way: bool) -> Option<std::sync::MutexGuard<'a, T>> {
    let t0 = std::time::Instant::now();
    let mut gave = std::time::Duration::ZERO;
    loop {
        if give_way && gave < budget && waiting.load(Ordering::Acquire) > 0 {
            let t = std::time::Instant::now();
            std::thread::yield_now();
            gave += t.elapsed();
            continue;
        }
        if let Ok(g) = m.try_lock() {
            return Some(g);
        }
        if t0.elapsed().saturating_sub(gave) >= budget {
            return None;
        }
        std::thread::yield_now();
    }
}

/// `m` once no submit waits for it (or `GIVE_WAY_US` have gone by).
fn lock_in_turn<'a, T>(m: &'a Mutex<T>, waiting: &std::sync::atomic::AtomicUsize) -> std::sync::MutexGuard<'a, T> {
    if waiting.load(Ordering::Acquire) > 0 {
        let t0 = std::time::Instant::now();
        while waiting.load(Ordering::Acquire) > 0 && t0.elapsed().as_micros() < GIVE_WAY_US {
            std::thread::yield_now();
        }
    }
    m.lock().unwrap()
}

/// The title committed `[a, a + n)` in a sparse window. The pages of tiles mapped for it that
/// lie outside the range are not committed (holes); the range's pages are.
pub unsafe fn committed(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    super::title_textures::mark_hot(a, n);
    let mut g = lock_in_turn(&STATE, &SUBMITS_WAITING);
    let _held = holding(Holder::Commits);
    forget_released(&mut g);
    let Some(s) = g.as_mut() else {
        let (start, end) = page_span(a, n);
        early_add(&mut EARLY.lock().unwrap(), start, end);
        return;
    };
    s.commit_range(a, n);
}

/// The title decommitted `[a, a + n)`: tiles whose pages are all uncommitted (whether in one
/// full-tile decommit or incrementally via sub-tile decommits) give their video memory back
/// (unless a placed buffer still pins the pool tile).
pub unsafe fn decommitted(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    let (start, end) = page_span(a, n);
    let lazy_changed = {
        let mut lazy = LAZY_RANGES.lock().unwrap();
        let before = lazy.len();
        let had = lazy.range(..end).next_back().is_some_and(|(_, &e)| e > start);
        if had {
            early_remove(&mut lazy, start, end);
        }
        had || lazy.len() != before
    };
    let mut g = lock_in_turn(&STATE, &SUBMITS_WAITING);
    let _held = holding(Holder::Commits);
    forget_released(&mut g);
    let Some(s) = g.as_mut() else {
        early_remove(&mut EARLY.lock().unwrap(), start, end);
        if lazy_changed {
            MAPPED_GEN.fetch_add(1, Ordering::Release);
        }
        return;
    };
    s.decommit_range(a, n);
    if lazy_changed {
        MAPPED_GEN.fetch_add(1, Ordering::Release);
    }
}

/// `decommitted` around the caller's decommit `os` of `[a, a + n)` in a sparse window (its
/// VirtualFree, called with no lock of XCL's held; its result is returned): a CPU view there (a
/// title's reservation of 64 KiB pages that a GPU view shows) has its pages taken as holes by
/// submits until Windows is done, as `source_decommit` does outside the windows. For a release,
/// drop the pair and `forget_mirrors` inside `os`, after the VirtualFree.
pub unsafe fn decommit<R>(a: usize, n: usize, os: impl FnOnce() -> R) -> R {
    if n == 0 {
        return os();
    }
    let (span, _) = freeing_begins(&SOURCES, a, n);
    decommitted(a, n);
    let r = os();
    freeing_ends(&SOURCES, span);
    r
}

/// Tiles wait for their mapping call (`State::pending`; read without the lock).
static PENDING: AtomicBool = AtomicBool::new(false);
static PENDING_WAKE: (Mutex<()>, std::sync::Condvar) = (Mutex::new(()), std::sync::Condvar::new());

/// A tile's mapping was put off: sent by the thread started here in 2 ms, unless something sends
/// it before. Work already on the GPU can reach the memory without another submit (it waits for a
/// value the CPU writes).
fn pending_soon() {
    if PENDING.swap(true, Ordering::AcqRel) {
        return;
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-tile-maps".into()).spawn(|| loop {
            {
                let mut g = PENDING_WAKE.0.lock().unwrap_or_else(|e| e.into_inner());
                while !PENDING.load(Ordering::Acquire) {
                    g = PENDING_WAKE.1.wait(g).unwrap_or_else(|e| e.into_inner());
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
            if PENDING.load(Ordering::Acquire) {
                send_pending();
            }
        });
    });
    drop(PENDING_WAKE.0.lock().unwrap_or_else(|e| e.into_inner()));
    PENDING_WAKE.1.notify_one();
}

/// The mappings put off go to the GPU now: the CPU is about to signal a fence, and work submitted
/// before the title committed the memory may run because of it.
pub fn send_pending() {
    // (a signal under the state's lock must not wait for it; the thread sends them then)
    let t0 = std::time::Instant::now();
    loop {
        // (asked each time: another thread may be sending them, and has sent them when the flag goes)
        if !PENDING.load(Ordering::Acquire) {
            return;
        }
        // (`XCL_SUBMIT_LOCK_FIRST=1`: a submit that waits for the lock in `before_execute` sends
        // them itself, `flush_pending`)
        if submit_lock_first() && FLUSHERS_WAITING.load(Ordering::Acquire) > 0 {
            if t0.elapsed().as_millis() >= 2 {
                return;
            }
            std::thread::yield_now();
            continue;
        }
        match STATE.try_lock() {
            Ok(mut g) => {
                let _held = holding(Holder::TileMaps);
                if let Some(s) = g.as_mut() {
                    unsafe { s.flush_pending() };
                }
                PENDING.store(false, Ordering::Release);
                return;
            }
            Err(std::sync::TryLockError::Poisoned(_)) => return PENDING.store(false, Ordering::Release),
            Err(std::sync::TryLockError::WouldBlock) if t0.elapsed().as_millis() >= 2 => return,
            Err(std::sync::TryLockError::WouldBlock) => std::thread::yield_now(),
        }
    }
}

impl State {
    /// `committed` with the state locked
    unsafe fn commit_range(&mut self, a: usize, n: usize) {
        // a tile over a CPU view committed at its own address: its own memory from here on
        if !self.mirror_src.is_empty() {
            let mut t = a & !(TILE - 1);
            while t < a.saturating_add(n) {
                if self.unlink_mirror(t) {
                    if let Some(e) = self.mapped.get_mut(&t).filter(|e| e.src != 0) {
                        // (what it held was the CPU view's: cleared, and the pages committed here
                        // are no holes below)
                        *e = TileEntry::with_holes(e.pool, 0xFFFF);
                        self.to_zero.insert((t, TILE));
                        self.mirrored -= 1;
                        early_add(&mut self.runs, t, t + TILE);
                        MAPPED_GEN.fetch_add(1, Ordering::Release);
                    }
                }
                match t.checked_add(TILE) {
                    Some(next) => t = next,
                    None => break,
                }
            }
        }
        if lazy_commits() {
            self.ensure_listed(a, n);
        } else {
            self.ensure_mapped(a, n, |_| 0xFFFF);
        }
        let mut pg = a & !(PAGE - 1);
        let end = a.saturating_add(n);
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = t.saturating_add(TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    entry.holes &= !(1u16 << ((p - t) / PAGE));
                    p += PAGE;
                }
            }
            if tile_end <= pg {
                break;
            }
            pg = tile_end;
        }
    }

    /// `decommitted` with the state locked: the tiles that go are parked (`flush_parked` unmaps
    /// them and frees their pool tiles)
    unsafe fn decommit_range(&mut self, a: usize, n: usize) {
        let mut pools: Vec<(usize, (u32, u32))> = Vec::new();
        let end = a.saturating_add(n);
        // tiles over a CPU view go as a whole (their holes are the CPU view's pages, not these)
        if !self.mirror_src.is_empty() {
            let mut t = a & !(TILE - 1);
            while t < end {
                if self.mirror_src.contains_key(&t) {
                    self.drop_mirror(t);
                }
                match t.checked_add(TILE) {
                    Some(next) => t = next,
                    None => break,
                }
            }
            if self.parked.len() >= PARKED_MAX && self.flush_parked() {
                self.retire_free_pools();
            }
        }
        // a CPU view in a window (a title's reservation of 64 KiB pages a GPU view shows): the
        // tiles over it have holes there too (below, only the entries of these addresses get them)
        if !self.src_runs.is_empty() {
            self.source_decommitted_in(a, n);
        }
        let mut t = a & !(TILE - 1);
        while t < end {
            if let Some(e) = self.mapped.get(&t).filter(|e| e.pool != LAZY) {
                pools.push((t, e.pool));
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        let tiles = decommit_tiles(&mut self.mapped, &self.pinned, &mut Vec::new(), &mut self.to_zero, a, n);
        if tiles.is_empty() {
            return;
        }
        for &t in &tiles {
            early_remove(&mut self.runs, t, t + TILE);
            if let Ok(i) = pools.binary_search_by_key(&t, |p| p.0) {
                self.parked.insert(t, pools[i].1);
            }
        }
        MAPPED_GEN.fetch_add(1, Ordering::Release);
        if self.parked.len() >= PARKED_MAX && self.flush_parked() {
            self.drain_quarantined();
            self.retire_free_pools();
        }
    }

    /// A pool tile a buffer used at `t` is freed (`cooling`), unless its window maps it, it
    /// is parked there, it waits in `quarantined`, or another buffer uses it. Kept mapped for
    /// the buffer: parked from here on.
    fn release_if_unused(&mut self, t: usize, tile: (u32, u32)) {
        if self.pinned.contains_key(&tile) {
            return;
        }
        if self.kept.get(&t) == Some(&tile) {
            self.kept.remove(&t);
            self.parked.insert(t, tile);
            return;
        }
        if self.mapped.get(&t).is_none_or(|e| e.pool != tile)
            && self.parked.get(&t) != Some(&tile)
            && !self.quarantined.iter().any(|b| b.tiles.contains(&tile))
        {
            self.cooling.push_back((tile, std::time::Instant::now()));
        }
    }

    unsafe fn note_queue(&mut self, q: &ID3D12CommandQueue) -> bool {
        let key = q.as_raw() as usize;
        if self.queue_fences.contains_key(&key) {
            return true;
        }
        // Ignore queues belonging to a different device (e.g. isolated unit tests).
        let mut qdev: Option<ID3D12Device> = None;
        if q.GetDevice(&mut qdev).is_err() || qdev.as_ref().is_none_or(|d| d.as_raw() != self.dev.as_raw()) {
            return true;
        }
        match self.dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) {
            Ok(fence) => {
                self.queue_fences.insert(key, (q.clone(), fence, 0));
                true
            }
            Err(_) => false,
        }
    }

    unsafe fn sync_known_queues(&mut self) -> bool {
        let mut ok = true;
        for raw in super::present::known_queues() {
            if !self.queue_fences.contains_key(&raw) {
                let ptr = raw as *mut c_void;
                if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&ptr) {
                    ok &= self.note_queue(q);
                }
            }
        }
        ok
    }

    unsafe fn quarantine_tiles(&mut self, tiles: Vec<(u32, u32)>) {
        if tiles.is_empty() {
            return;
        }
        let mut ok = self.sync_known_queues();
        let mut marks = Vec::with_capacity(1 + self.queue_fences.len());
        marks.push((self.fence.clone(), self.value));
        if ok {
            for (q, fence, val) in self.queue_fences.values_mut() {
                let next = *val + 1;
                if q.Signal(&*fence, next).is_ok() {
                    *val = next;
                    marks.push((fence.clone(), next));
                } else {
                    ok = false;
                    break;
                }
            }
        }
        self.quarantined.push_back(RetiredTiles { tiles, marks: ok.then_some(marks), at: std::time::Instant::now() });
    }

    /// The `quarantined` tiles the GPU is past go to `cooling` (from when they were unmapped).
    /// (XCL's own query of the fences: the title's `GetCompletedValue` hook takes the state's
    /// lock, which the caller holds.)
    unsafe fn drain_quarantined(&mut self) -> bool {
        let mut freed = false;
        while let Some(front) = self.quarantined.front() {
            let ready = front.marks.as_ref().is_some_and(|marks| marks.iter().all(|(f, v)| super::real::fence_completed(f) >= *v));
            if !ready {
                break;
            }
            let batch = self.quarantined.pop_front().unwrap();
            let at = batch.at;
            for tile in batch.tiles {
                if !self.pinned.contains_key(&tile) {
                    self.cooling.push_back((tile, at));
                    freed = true;
                }
            }
        }
        freed
    }

    /// The parked tiles unmapped from their windows (one call per window) and their pool tiles
    /// (but those of placed buffers) quarantined behind `self.queue` and every known title
    /// queue's completion fence, then freed to `cooling` (`drain_quarantined`). Pools left all
    /// free are not retired here: callers do that (`retire_free_pools`).
    unsafe fn flush_parked(&mut self) -> bool {
        if self.parked.is_empty() {
            return false;
        }
        let mut by_window: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        let mut tiles: Vec<(u32, u32)> = Vec::new();
        for (t, tile) in std::mem::take(&mut self.parked) {
            // (a placed buffer uses it)
            if self.pinned.contains_key(&tile) {
                self.kept.insert(t, tile);
                continue;
            }
            if let Some(wi) = self.window_index(t) {
                by_window.entry(wi).or_default().push(t);
            }
            tiles.push(tile);
        }
        if by_window.is_empty() {
            // (nothing to unmap)
            self.quarantine_tiles(tiles);
            return false;
        }
        for (wi, ts) in by_window {
            let (b, r) = (self.windows[wi].0, self.windows[wi].1.clone());
            TILE_WORK[2].fetch_add(ts.len() as u64, Ordering::Relaxed);
            self.unmap_tiles(&r, b, &ts);
        }
        self.signal();
        self.quarantine_tiles(tiles);
        true
    }
}

/// Window tiles asked for by `map_mirror`: each is queued once, until it is forgotten
/// (`forget_mirrors`) or left out (`State::map_mirrors`).
static MIRROR_ASKED: super::sharded::Sharded<()> = super::sharded::Sharded::new();
/// How often a tile left `MIRROR_ASKED` (a view that went, a tile given back with its source):
/// whoever remembers having asked (`view_want`'s threads) asks again.
static MIRROR_FORGOTTEN: AtomicU64 = AtomicU64::new(0);

/// Counts up whenever a tile that was asked for may be asked for again (`map_mirror` would
/// queue it): a caller that keeps what it asked for forgets it when this changes.
pub fn mirror_asks_forgotten() -> u64 {
    MIRROR_FORGOTTEN.load(Ordering::Acquire)
}

/// Tile `t` is asked for no more.
fn forget_ask(t: usize) {
    if MIRROR_ASKED.remove(t).is_some() {
        MIRROR_FORGOTTEN.fetch_add(1, Ordering::Release);
        #[cfg(test)]
        tests::ASKS_FORGOTTEN.with(|n| n.set(n.get() + 1));
    }
}
/// The tiles asked for that no submit has taken yet. A lock of its own: `map_mirror` is called
/// while recording, on any thread, and never waits for `STATE`.
static MIRROR_ASKS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static MIRROR_ASKING: AtomicBool = AtomicBool::new(false);

/// What a thread that decommits the title's CPU memory needs to know of the tiles over CPU views,
/// and what a submit needs to know of those decommits. A lock of its own, which takes no other:
/// a decommit asks it with no lock of the state's (most of them are of memory no tile shows, and
/// a title's loading threads decommit all the time), a submit with `STATE` held.
#[derive(Default)]
struct Sources {
    /// `State::src_runs`, changed with it (`link_mirror`, `unlink_mirror`)
    runs: BTreeMap<usize, usize>,
    /// the pages `[start, end)` of decommits under way (`source_decommit`): told, Windows not
    /// done yet. A submit takes them as holes, whatever Windows says of them for now
    freeing: Vec<(usize, usize)>,
}

impl Sources {
    /// Whether tiles show any of `[a, end)`
    fn shows(&self, a: usize, end: usize) -> bool {
        self.runs.range(..end).next_back().is_some_and(|(_, &e)| e > a)
    }
}

static SOURCES: Mutex<Sources> = Mutex::new(Sources { runs: BTreeMap::new(), freeing: Vec::new() });

fn lock_sources(m: &Mutex<Sources>) -> std::sync::MutexGuard<'_, Sources> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A decommit of the pages of `[a, a + n)` begins: submits take them as holes until it ends
/// (`freeing_ends` with the span returned). True: tiles show some of them, and the state has to
/// be told before Windows frees them. Either a submit linked a tile there before this (and the
/// state is told), or it links it after and finds the pages here (in the one lock, in turn).
fn freeing_begins(m: &Mutex<Sources>, a: usize, n: usize) -> ((usize, usize), bool) {
    let span = page_span(a, n);
    let mut l = lock_sources(m);
    l.freeing.push(span);
    (span, l.shows(span.0, span.1))
}

/// The decommit of `span` (`freeing_begins`) is done.
fn freeing_ends(m: &Mutex<Sources>, span: (usize, usize)) {
    let mut l = lock_sources(m);
    if let Some(i) = l.freeing.iter().position(|&f| f == span) {
        l.freeing.swap_remove(i);
    }
}

/// What window tile `g` shows (`tile_shows`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Under {
    /// not in a GPU view
    NoView,
    /// nothing (zeros): unmapped by the queue's page mappings, no page behind it, or a page kept
    /// where kernelx does not see the memory go (no tile may show it there)
    Nothing,
    /// the 64 KiB-aligned CPU tile: the CPU view's at the same offset, or where the page mapped
    /// there is kept
    Cpu(usize),
}

/// What window tile `g` of a GPU view shows (`kernelx::mirrors::shows`; locks of its own, which
/// take no other: it may be asked with `STATE` held). While nothing is mapped, the CPU view's tile
/// as before.
fn tile_shows(g: usize) -> Under {
    use crate::hle::kernelx::mirrors::{shows, Shows};
    match shows(g) {
        Shows::NoView => Under::NoView,
        Shows::Same(c) => Under::Cpu(c & !(TILE - 1)),
        Shows::Page { at, tile: true } => Under::Cpu(at & !(TILE - 1)),
        Shows::Page { .. } | Shows::Nothing => Under::Nothing,
    }
}

/// The 64 KiB-aligned CPU tile behind window tile `g` of a GPU view (`tile_shows`).
fn mirror_source_tile(g: usize) -> Option<usize> {
    match tile_shows(g) {
        Under::Cpu(c) => Some(c),
        _ => None,
    }
}

/// 64 KiB of XCL's own, reserved and never committed: the source of a tile that shows nothing but
/// keeps its video memory, as a placed buffer maps its tiles once (`buffer_tiles`,
/// `remap_mirrors`). Its pages are always holes: nothing scans it, and no copy back lands there.
fn no_source() -> usize {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *AT.get_or_init(|| {
        use windows_sys::Win32::System::Memory::MEM_RESERVE;
        unsafe { VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE, PAGE_NOACCESS) as usize }
    })
}

/// Window tiles whose page the queue's page mappings changed (`remapped`), for the next submit to
/// link again (`State::remap_mirrors`). A lock of its own, as `MIRROR_ASKS`.
static REMAPS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static REMAPPING: AtomicBool = AtomicBool::new(false);

/// What the GPU-view tiles `tiles` show changed (`kernelx::mirrors`: a mapping, an unmap, a freed
/// page, a page that got or lost its place): those asked for (`map_mirror`) are linked again at
/// the next submit. Any thread; no state's lock.
pub fn remapped(tiles: &[usize]) {
    let asked: Vec<usize> = tiles.iter().copied().filter(|&t| MIRROR_ASKED.contains(t)).collect();
    if asked.is_empty() {
        return;
    }
    REMAPS.lock().unwrap().extend(asked);
    REMAPPING.store(true, Ordering::Release);
}

/// The tiles of `q` (`REMAPS`) join those waiting to be linked again (`State::remaps`). Past
/// `MIRROR_ASKS_MAX` the queue is sorted and each tile kept once: a page changed at every frame,
/// or by several calls, waits once, and the queue holds no more than the tiles asked for.
fn take_remaps(remaps: &mut Vec<usize>, q: &mut Vec<usize>) {
    remaps.append(q);
    if remaps.len() > MIRROR_ASKS_MAX {
        remaps.sort_unstable();
        remaps.dedup();
    }
}

/// The physical page the GPU-view tiles `tiles` show moved from `from` to `to` (both 64 KiB
/// aligned; `whole`: its bytes were copied there): the tiles linked over `from` follow it now,
/// with their video memory (`State::retarget`). Called on the moving thread (kernelx, AWAY held)
/// before the memory at `from` can go. The state's lock only when tiles show `from` (it gives way
/// to submits).
pub unsafe fn moved(tiles: &[usize], from: usize, to: usize, whole: bool) {
    if tiles.is_empty() || from == 0 || to == 0 || !lock_sources(&SOURCES).shows(from, from + TILE) {
        return;
    }
    let mut g = lock_in_turn(&STATE, &SUBMITS_WAITING);
    let _held = holding(Holder::Sources);
    forget_released(&mut g);
    let Some(s) = g.as_mut() else { return };
    for &w in tiles {
        s.retarget(w, from, to, whole);
    }
}

/// The GPU-view tiles linked again so far: [to show what the page mappings put there (or nothing),
/// that followed their page].
static VIEW_TILES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// `VIEW_TILES` so far.
pub fn page_tiles() -> [u64; 2] {
    std::array::from_fn(|k| VIEW_TILES[k].load(Ordering::Relaxed))
}

/// The CPU tile a placed buffer's tile is linked over for what it shows: a tile that shows nothing
/// keeps its video memory over `no_source` (the buffer maps its tiles once). None: no view.
fn buffer_source(under: Under) -> Option<usize> {
    match under {
        Under::Cpu(c) => Some(c),
        Under::Nothing => Some(no_source()).filter(|&n| n != 0),
        Under::NoView => None,
    }
}

/// The window tiles of `[va, va + len)` are to show the title's CPU view behind them (`va` in a
/// GPU view of its memory, `kernelx::mirrors`): video memory, filled from the CPU view and kept in
/// step with it, from the next submit on. Any thread; it waits for no submit (no `STATE`) and
/// asks Windows nothing. Tiles that are not in a sparse window or in a GPU view are left out
/// there. Returns how many of its tiles were queued now (the others were asked for before).
pub fn map_mirror(va: usize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let end = va.saturating_add(len);
    let mut new: Vec<usize> = Vec::new();
    let mut t = va & !(TILE - 1);
    while t < end {
        if MIRROR_ASKED.insert(t, ()).is_none() {
            new.push(t);
        }
        match t.checked_add(TILE) {
            Some(next) => t = next,
            None => break,
        }
    }
    if new.is_empty() {
        return 0;
    }
    let queued = new.len();
    if view_switches().stats {
        VIEW_ASKED.fetch_add(queued as u64, Ordering::Relaxed);
    }
    MIRROR_ASKS.lock().unwrap().extend(new);
    MIRROR_ASKING.store(true, Ordering::Release);
    queued
}

/// The title's CPU memory `[a, a + n)` is decommitted by `os` (the caller's VirtualFree, called
/// with no lock of XCL's held; its result is returned): the tiles shown over it have holes
/// there first (`source_decommitted`), and while `os` runs a submit takes those pages as holes,
/// whatever Windows says of them. Told only before the decommit, a submit in between asked Windows
/// about a page written before, found it still committed, read it as Windows freed it, and left it
/// committed in its tile for good (a later copy back or immediate wrote to it). The state's lock
/// is taken only when tiles show some of the range.
pub unsafe fn source_decommit<R>(a: usize, n: usize, os: impl FnOnce() -> R) -> R {
    if n == 0 {
        return os();
    }
    let (span, shown) = freeing_begins(&SOURCES, a, n);
    if shown {
        source_decommitted(a, n);
    }
    let r = os();
    freeing_ends(&SOURCES, span);
    r
}

/// `source_decommit` for a release of the title's CPU memory `[a, a + n)` by `os`: after it the
/// tiles over it are forgotten as well (`forget_mirrors`: the address may be given out anew),
/// before the pages stop being holes for submits. Drop the pair (`kernelx::mirrors::released`)
/// before `os` returns, so that no tile is linked over the address after the forget.
pub unsafe fn source_release<R>(a: usize, n: usize, os: impl FnOnce() -> R) -> R {
    if n == 0 {
        return os();
    }
    let (span, shown) = freeing_begins(&SOURCES, a, n);
    if shown {
        source_decommitted(a, n);
    }
    let r = os();
    forget_mirrors(a, n);
    freeing_ends(&SOURCES, span);
    r
}

/// The part of `source_decommit` that tells the state: the tiles shown over the title's CPU
/// memory `[a, a + n)` have holes there, cleared on the GPU (`sent = None` says the GPU holds
/// zeros), and a tile with no committed page left gives its video memory back. A submit's scan
/// reads those pages with the state's lock held: after this it does not – unless it asks Windows
/// before the decommit lands, which only `source_decommit` rules out (call that).
pub unsafe fn source_decommitted(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    let (start, end) = page_span(a, n);
    if !lock_sources(&SOURCES).shows(start, end) {
        return;
    }
    let mut g = lock_in_turn(&STATE, &SUBMITS_WAITING);
    let _held = holding(Holder::Sources);
    forget_released(&mut g);
    let Some(s) = g.as_mut() else { return };
    s.source_decommitted_in(a, n);
}

/// `[a, a + n)` is no GPU view or CPU view any more (released, or the pair is gone): tiles in it,
/// or over CPU memory in it, show nothing from here on (their video memory goes back), and may be
/// asked for again (the addresses may be given out anew). Call it once the pair is dropped
/// (`kernelx::mirrors::released`): with no tile linked anywhere it takes no state's lock, and a
/// tile linked after that looks its pair up again (`State::take_asks`), so one of the two sees
/// the other.
pub fn forget_mirrors(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    let end = a.saturating_add(n);
    // (out of the queue before out of the set: an ask in between finds the tile still asked
    // for, or stays queued, and the count after it has the thread ask again. The other way
    // round, an ask in between went with the queue and left the tile asked for, never mapped)
    MIRROR_ASKS.lock().unwrap().retain(|&t| !(t < end && t + TILE > a));
    #[cfg(test)]
    if let Some(f) = tests::BETWEEN_FORGETS.with(|f| f.take()) {
        f();
    }
    let mut asked: Vec<usize> = Vec::new();
    MIRROR_ASKED.for_each(|t, _| {
        if t < end && t + TILE > a {
            asked.push(t);
        }
    });
    for t in asked {
        forget_ask(t);
    }
    if lock_sources(&SOURCES).runs.is_empty() {
        return;
    }
    let mut g = lock_in_turn(&STATE, &SUBMITS_WAITING);
    let _held = holding(Holder::Sources);
    forget_released(&mut g);
    let Some(s) = g.as_mut() else { return };
    unsafe { s.forget_mirrors_in(a, end) };
}

/// The CPU address behind window address `va`, for what reads or writes the title's memory
/// there: `va` itself, unless it lies in a GPU view of the title's CPU memory, whose tiles show
/// their CPU view (`State::map_mirrors` takes them by the same rule), or where the page the
/// queue's page mappings put there is kept (wherever that is: the CPU reads and writes any). No
/// `STATE`.
pub fn cpu_of(va: usize) -> usize {
    use crate::hle::kernelx::mirrors::{shows, Shows};
    match shows(va) {
        Shows::Same(c) | Shows::Page { at: c, .. } => c,
        Shows::NoView | Shows::Nothing => va,
    }
}

impl State {
    /// Window tile `w` shows CPU tile `c` (with video memory or not yet).
    fn link_mirror(&mut self, w: usize, c: usize) {
        self.mirror_src.insert(w, c);
        let over = self.mirrors.entry(c).or_default();
        over.push(w);
        if over.len() == 1 {
            early_add(&mut self.src_runs, c, c + TILE);
            // (before the CPU tile's pages are asked about: a decommit from here on tells)
            early_add(&mut lock_sources(self.sources).runs, c, c + TILE);
            MAPPED_GEN.fetch_add(1, Ordering::Release);
        }
    }

    /// Window tile `w` shows no CPU tile any more (its entry in `mapped`, if any, is the caller's),
    /// and may be asked for again. False: it showed none.
    fn unlink_mirror(&mut self, w: usize) -> bool {
        let linked = self.unlink_kept(w);
        if linked {
            forget_ask(w);
        }
        linked
    }

    /// `unlink_mirror`, the ask kept: `w` is linked again (`remap_mirrors`), or waits asked for
    /// until its GPU-view page shows something. (Forgotten, every recording thread's memory of its
    /// asks would empty at each such change, `view_want`.)
    fn unlink_kept(&mut self, w: usize) -> bool {
        let Some(c) = self.mirror_src.remove(&w) else { return false };
        if let Some(over) = self.mirrors.get_mut(&c) {
            over.retain(|&x| x != w);
            if over.is_empty() {
                self.mirrors.remove(&c);
                early_remove(&mut self.src_runs, c, c + TILE);
                early_remove(&mut lock_sources(self.sources).runs, c, c + TILE);
                MAPPED_GEN.fetch_add(1, Ordering::Release);
            }
        }
        self.refused.remove(&w);
        true
    }

    /// Window tile `w` shows its CPU tile no more: its video memory is parked (unmapped before the
    /// next submit), and it may be asked for again.
    fn drop_mirror(&mut self, w: usize) {
        self.unlink_mirror(w);
        self.park(w);
    }

    /// The video memory of window tile `w` over a CPU view is parked (unmapped before the next
    /// submit, unless a placed buffer uses it, or it is taken back first: `new_mirror`).
    fn park(&mut self, w: usize) {
        if self.mapped.get(&w).is_some_and(|e| e.src != 0) {
            let e = self.mapped.remove(&w).expect("a tile over a CPU view");
            while let Some(&piece) = self.to_zero.range((w, 0)..(w + TILE, 0)).next() {
                self.to_zero.remove(&piece);
            }
            self.mirrored -= 1;
            self.give_back(w, e.pool);
        }
    }

    /// The video memory `pool` of window tile `w` over a CPU view goes back (no longer counted in
    /// `mirrored`): parked – or kept mapped there (`XCL_VIEW_KEEP`, `keep_view_tile`), to be taken
    /// back as a parked tile is (`new_mirror`).
    fn give_back(&mut self, w: usize, pool: (u32, u32)) {
        if keep_view_tile(self.diag.keep, self.mirrored + 1 + self.kept.len(), self.mirror_cap) {
            if self.diag.kept == 0 {
                crate::xlog_info!("XCL_VIEW_KEEP: tiles over CPU views keep their video memory once filled ({} tiles at most)", self.mirror_cap);
            }
            self.kept.insert(w, pool);
            self.diag.kept += 1;
        } else {
            self.parked.insert(w, pool);
            self.counts.given_back += 1;
        }
        if self.diag.stats {
            // (a tile that never comes back is not kept for ever)
            if self.diag.left_at.len() >= 1 << 20 {
                self.diag.left_at.clear();
            }
            self.diag.left_at.insert(w, self.diag.submits);
        }
    }

    /// Video memory for window tile `w` over CPU tile `src` (`holes`: its pages not committed
    /// there): taken back from `parked` / `kept` (cleared; it is held already, so past
    /// `mirror_cap` too: a kept tile refused stayed mapped there with old bytes), or a tile of a
    /// pool (cleared when it was used before), mapped with the next flush; all of it goes up at
    /// the next submit (`fill`). `force`: past `mirror_cap` too (it counts all the same), for a
    /// placed buffer's tile: the buffer maps its tiles once, and video memory the window tile got
    /// later would not be the buffer's. False: past `mirror_cap` (logged at powers of two) or no
    /// video memory (now or a moment ago, `no_memory`): `w` is in `refused` then (counted once).
    unsafe fn new_mirror(&mut self, w: usize, src: usize, holes: u16, force: bool) -> bool {
        if let Some(tile) = self.parked.remove(&w).or_else(|| self.kept.remove(&w)) {
            self.mapped.insert(w, TileEntry { src, gen: next_gen(), ..TileEntry::with_holes(tile, holes) });
            self.to_zero.insert((w, TILE));
        } else {
            // (`XCL_VIEW_KEEP`: the tiles kept hold video memory too)
            let held = self.mirrored + if self.diag.keep { self.kept.len() } else { 0 };
            let at_cap = !force && held >= self.mirror_cap;
            if at_cap && self.diag.keep && !self.kept.is_empty() {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| crate::xlog_warn!("XCL_VIEW_KEEP: {} tiles kept and {} with video memory reach the cap of {} (XCL_MIRROR_MB) – tiles are refused from here on, and read zeros", self.kept.len(), self.mirrored, self.mirror_cap));
            }
            // (a moment after a heap could not be made, only tiles at hand: parked ones and those
            // still quarantined or cooling are not)
            self.drain_quarantined();
            let at_hand = !self.free.is_empty() || cooled(&self.cooling, std::time::Instant::now()) || !self.retired.is_empty();
            let wait = at_cap || (!force && self.short_of_memory() && !at_hand);
            let got = if wait { None } else { self.take_tile() };
            let Some((tile, reused)) = got else {
                if !wait {
                    self.no_memory = Some(std::time::Instant::now());
                }
                if self.refused.insert(w) {
                    self.counts.refused += 1;
                    let n = self.counts.refused;
                    if at_cap && n.is_power_of_two() {
                        crate::xlog_warn!("sparse: {n} GPU-view tiles left without video memory (XCL_MIRROR_MB) – the GPU reads zeros there");
                    }
                }
                return false;
            };
            self.mapped.insert(w, TileEntry { src, gen: next_gen(), ..TileEntry::with_holes(tile, holes) });
            if reused {
                self.to_zero.insert((w, TILE));
            }
            self.pending.push(w);
        }
        self.refused.remove(&w);
        self.fill.push(w);
        self.mirrored += 1;
        self.counts.peak = self.counts.peak.max(self.mirrored);
        if self.diag.stats {
            if let Some(at) = self.diag.left_at.remove(&w) {
                self.diag.back[back_bucket(self.diag.submits - at)] += 1;
                self.diag.back_bytes += ((PAGES_PER_TILE - holes.count_ones() as usize) * PAGE) as u64;
            }
        }
        true
    }

    /// A tile over a CPU view found no video memory a moment ago (`no_memory`, 1 s)
    fn short_of_memory(&self) -> bool {
        self.no_memory.is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(1))
    }

    /// Tiles refused (`refused`) wait for video memory again (`later`) once there is room, as
    /// many as there is room for and `MIRROR_ASKS_MAX` at most: what their CPU view held goes up
    /// then (written once before, it may never be written again). One no longer linked or with
    /// video memory leaves `refused`.
    unsafe fn retry_refused(&mut self) {
        if self.refused.is_empty() || self.mirrored >= self.mirror_cap || self.short_of_memory() {
            return;
        }
        let room = (self.mirror_cap - self.mirrored).min(MIRROR_ASKS_MAX);
        let mut take: Vec<(usize, usize)> = Vec::new();
        let mut gone: Vec<usize> = Vec::new();
        for &w in &self.refused {
            if take.len() == room {
                break;
            }
            match self.mirror_src.get(&w) {
                Some(&c) if !self.mapped.contains_key(&w) => take.push((w, c)),
                _ => gone.push(w),
            }
        }
        for w in gone {
            self.refused.remove(&w);
        }
        // (in the order of their CPU tiles: a query for each region of them)
        take.sort_unstable_by_key(|t| t.1);
        self.later.extend(take);
    }

    /// At the top of a submit: tiles shown over their CPU views get video memory and are filled
    /// from them (`fill_later`), the committed pages of `fill_pages` at most (a burst of asks
    /// hashed up to 256 MiB and pushed it through the staging ring with the state's lock held,
    /// on the title's submitting thread). The tiles waiting from submits before go first, in
    /// their order; then the tiles refused before, when there is room again (`retry_refused`);
    /// then the tiles asked for since (`take_asks`).
    unsafe fn map_mirrors(&mut self) {
        let mut budget = self.fill_pages;
        if self.later.is_empty() {
            self.retry_refused();
        }
        loop {
            self.fill_later(&mut budget);
            if budget == 0 || !self.later.is_empty() || self.asks.is_empty() {
                return;
            }
            self.take_asks();
        }
    }

    /// The tiles waiting in `later` (linked, oldest first) get video memory and are filled while
    /// `budget` (committed pages) lasts; one whose CPU tile has no committed page leaves it (and
    /// `refused`: a write there gives it video memory, `mirror_writes`). One refused for want of
    /// room or video memory goes to `refused` (`new_mirror`).
    unsafe fn fill_later(&mut self, budget: &mut usize) {
        /// tiles asked about at a time
        const BATCH: usize = 256;
        while *budget > 0 && !self.later.is_empty() {
            let batch: Vec<(usize, usize)> = self.later.iter().take(BATCH).copied().collect();
            let holes = self.holes_now(&batch.iter().map(|b| b.1).collect::<Vec<_>>());
            let mut done = 0;
            for (&(w, c), h) in batch.iter().zip(holes) {
                if *budget == 0 {
                    break;
                }
                done += 1;
                if self.mirror_src.get(&w) != Some(&c) || self.mapped.contains_key(&w) {
                    continue;
                }
                if h == 0xFFFF {
                    self.refused.remove(&w);
                } else if self.new_mirror(w, c, h, false) {
                    *budget = budget.saturating_sub(PAGES_PER_TILE - h.count_ones() as usize);
                }
            }
            self.later.drain(..done);
            if done < batch.len() {
                return;
            }
        }
    }

    /// The tiles asked for (`asks`, `MIRROR_ASKS_MAX` at most at a time) linked over their CPU
    /// views, to wait for their video memory in `later`: a tile of a sparse window in a GPU view
    /// gets it unless its CPU tile has no committed page (it waits in `mirrors` for the CPU to
    /// write there). Others are left out, and may be asked for again.
    unsafe fn take_asks(&mut self) {
        if self.asks.is_empty() {
            return;
        }
        let n = self.asks.len().min(MIRROR_ASKS_MAX);
        let mut asks: Vec<usize> = self.asks.drain(..n).collect();
        asks.sort_unstable();
        asks.dedup();
        let mut want: Vec<(usize, usize)> = Vec::new();
        for g in asks {
            if self.mirror_src.contains_key(&g) {
                continue;
            }
            let under = if !self.mapped.contains_key(&g) && self.window_index(g).is_some() { tile_shows(g) } else { Under::NoView };
            match under {
                Under::Cpu(c) => want.push((g, c)),
                // (it stays asked for: a page mapped there later links it, `remapped`)
                Under::Nothing => {}
                // (not counted: an address left out at every submit would empty every thread's
                // memory of its asks at every submit; a pair made later changes the views)
                Under::NoView => {
                    MIRROR_ASKED.remove(g);
                }
            }
        }
        if want.is_empty() {
            return;
        }
        #[cfg(test)]
        if let Some(f) = tests::BEFORE_LINK.with(|f| f.take()) {
            f();
        }
        // linked before their CPU tiles' pages are asked about (`holes_now`)
        for &(g, c) in &want {
            self.link_mirror(g, c);
        }
        // (a pair dropped since it was looked up: its `forget_mirrors` may have found nothing
        // linked, and the CPU tile may be given out anew; a page mapped or unmapped there since:
        // linked again, the ask kept)
        want.retain_mut(|(g, c)| match tile_shows(*g) {
            Under::Cpu(now) if now == *c => true,
            Under::Cpu(now) => {
                self.unlink_kept(*g);
                self.link_mirror(*g, now);
                *c = now;
                true
            }
            Under::Nothing => {
                self.unlink_kept(*g);
                false
            }
            Under::NoView => !self.unlink_mirror(*g),
        });
        self.later.extend(want);
    }

    /// Window tile `w`, linked over `from`, follows its page to `to` (`moved`): the link moves. With
    /// video memory and `whole` (the page's 64 KiB were copied to `to`) its entry keeps its pool
    /// tile, gen and sent hashes – nothing is cleared or filled: the known write kernelx notes at
    /// `to` has the next scan send what differs, and a copy back recorded before lands at `to`.
    /// Not `whole` (a page never used, not committed at `to`): its video memory is parked and it
    /// waits to be filled from `to` (`later`), as one with no video memory does.
    fn retarget(&mut self, w: usize, from: usize, to: usize, whole: bool) {
        if self.mirror_src.get(&w) != Some(&from) || from == to {
            return;
        }
        self.unlink_kept(w);
        self.link_mirror(w, to);
        VIEW_TILES[1].fetch_add(1, Ordering::Relaxed);
        match self.mapped.get_mut(&w).filter(|e| e.src == from) {
            Some(e) if whole => {
                e.src = to;
                e.holes = 0;
            }
            Some(_) => {
                self.park(w);
                self.later.push_back((w, to));
            }
            None => self.later.push_back((w, to)),
        }
    }

    /// The tiles whose GPU-view page the queue's page mappings changed (`remaps`, from `remapped`;
    /// `MIRROR_ASKS_MAX` at most a submit) are linked to what they show now: video memory over a
    /// place is cleared and filled from its committed pages (`fill_later`, before the asks), one
    /// that shows nothing is given back – but a placed buffer's, which keeps its video memory
    /// cleared over `no_source` – and stays asked for. Called after the finished copies back land
    /// (a copy back recorded for the old link finds a new gen) and before `map_mirrors`.
    unsafe fn remap_mirrors(&mut self) {
        if self.remaps.is_empty() {
            return;
        }
        let n = self.remaps.len().min(MIRROR_ASKS_MAX);
        let mut tiles: Vec<usize> = self.remaps.drain(..n).collect();
        tiles.sort_unstable();
        tiles.dedup();
        for g in tiles {
            // (a tile of the window's own is not a view's)
            if self.mapped.get(&g).is_some_and(|e| e.src == 0) {
                continue;
            }
            let now = tile_shows(g);
            let linked = self.mirror_src.get(&g).copied();
            match now {
                Under::Cpu(c) if linked == Some(c) => continue,
                Under::Nothing if linked.is_none_or(|l| l == no_source()) => continue,
                _ => {}
            }
            let pinned = self.mapped.get(&g).is_some_and(|e| self.pinned.contains_key(&e.pool));
            self.unlink_kept(g);
            self.park(g);
            if pinned {
                // (a kept tile must not keep the bytes it showed)
                self.to_zero.insert((g, TILE));
            }
            match now {
                Under::Cpu(c) if self.window_index(g).is_some() => {
                    self.link_mirror(g, c);
                    // (it was shown: before the older asks)
                    self.later.push_front((g, c));
                    VIEW_TILES[0].fetch_add(1, Ordering::Relaxed);
                }
                Under::Nothing => {
                    if pinned && no_source() != 0 {
                        self.link_mirror(g, no_source());
                        self.new_mirror(g, no_source(), 0xFFFF, true);
                    }
                    VIEW_TILES[0].fetch_add(1, Ordering::Relaxed);
                }
                _ => forget_ask(g),
            }
        }
    }

    /// The hole masks of CPU tiles `tiles` (`source_holes`), with the pages of decommits under way
    /// as holes too (`Sources::freeing`): Windows has them still, and frees them while the submit
    /// reads them. Asked once the tiles are linked (`link_mirror`): a decommit that begins after
    /// this finds them and tells the state before Windows frees anything; one that began before
    /// is in the list read here.
    unsafe fn holes_now(&self, tiles: &[usize]) -> Vec<u16> {
        let freeing = lock_sources(self.sources).freeing.clone();
        let mut holes = source_holes(tiles);
        for &(a, end) in &freeing {
            for (h, &t) in holes.iter_mut().zip(tiles) {
                let (lo, hi) = (a.max(t), end.min(t + TILE));
                if lo < hi {
                    *h |= ((1u32 << ((hi - t) / PAGE)) - (1u32 << ((lo - t) / PAGE))) as u16;
                }
            }
        }
        holes
    }

    /// Before a submit, the CPU pages written (`dirty`, `page` bytes each) that tiles show: a tile
    /// over one with no video memory gets it now (its CPU tile had no committed page, gave its
    /// memory back, or was refused), and a written page that was a hole there is asked about
    /// again: committed since (a page of the CPU view committed lazily, on its first write), it is
    /// one no more. Every `mapped` insert of a submit is made here, before pointers into the
    /// entries are taken.
    unsafe fn mirror_writes(&mut self, dirty: &[usize], page: usize) {
        if self.mirrors.is_empty() {
            return;
        }
        // CPU tiles written that tiles show, and the pages written there
        let mut written: Vec<(usize, u16)> = Vec::new();
        for &a in dirty {
            let mut x = a & !(PAGE - 1);
            while x < a + page {
                let t = x & !(TILE - 1);
                if self.mirrors.contains_key(&t) {
                    let bit = 1u16 << ((x - t) / PAGE);
                    match written.last_mut() {
                        Some(l) if l.0 == t => l.1 |= bit,
                        _ => written.push((t, bit)),
                    }
                }
                x += PAGE;
            }
        }
        // (tile over it, CPU tile, holes written – None: the tile has no video memory)
        let mut ask: Vec<(usize, usize, Option<u16>)> = Vec::new();
        // (a tile refused is not asked about while there is no room or memory for it: it would be
        // refused again, and its CPU tile may be written at every submit)
        let waiting = self.mirrored >= self.mirror_cap || self.short_of_memory();
        for &(t, bits) in &written {
            for &w in &self.mirrors[&t] {
                match self.mapped.get(&w) {
                    Some(e) if e.src == t => {
                        if e.holes & bits != 0 {
                            ask.push((w, t, Some(e.holes & bits)));
                        }
                    }
                    Some(_) => {}
                    None if waiting && self.refused.contains(&w) => {}
                    None => ask.push((w, t, None)),
                }
            }
        }
        if ask.is_empty() {
            return;
        }
        let now = self.holes_now(&ask.iter().map(|a| a.1).collect::<Vec<_>>());
        for (&(w, t, holes), h) in ask.iter().zip(now) {
            match holes {
                Some(bits) => {
                    self.counts.rechecks += bits.count_ones() as u64;
                    let committed = bits & !h;
                    if let Some(e) = self.mapped.get_mut(&w) {
                        e.holes &= !committed;
                        for i in 0..PAGES_PER_TILE {
                            if committed & (1u16 << i) != 0 {
                                e.sent[i] = None;
                            }
                        }
                    }
                }
                None if h != 0xFFFF && !self.mapped.contains_key(&w) => {
                    self.new_mirror(w, t, h, false);
                }
                None => {}
            }
        }
    }

    /// The CPU tiles of `mirrors` that `[a, end)` touches (by `src_runs`).
    fn src_tiles_in(&self, a: usize, end: usize) -> Vec<usize> {
        let a0 = a & !(TILE - 1);
        let mut out = Vec::new();
        let before = self.src_runs.range(..a0).next_back().map(|(&s, &e)| (s, e));
        for (s, e) in before.into_iter().chain(self.src_runs.range(a0..end).map(|(&s, &e)| (s, e))) {
            let mut t = s.max(a0);
            while t < e.min(end) {
                out.push(t);
                t += TILE;
            }
        }
        out
    }

    /// `source_decommitted` with the state locked: the pieces of the tiles over `[a, a + n)` get
    /// holes (`decommit_tiles`); a tile all holes is parked, and stays in `mirrors` (a commit and
    /// write there gives it video memory again).
    unsafe fn source_decommitted_in(&mut self, a: usize, n: usize) {
        let end = a.saturating_add(n);
        for t in self.src_tiles_in(a, end) {
            let (lo, hi) = (a.max(t), end.min(t + TILE));
            let over = self.mirrors.get(&t).cloned().unwrap_or_default();
            for w in over {
                let Some(pool) = self.mapped.get(&w).filter(|e| e.src == t).map(|e| e.pool) else { continue };
                if !decommit_tiles(&mut self.mapped, &self.pinned, &mut Vec::new(), &mut self.to_zero, w + (lo - t), hi - lo).is_empty() {
                    self.mirrored -= 1;
                    self.give_back(w, pool);
                }
            }
        }
        if self.parked.len() >= PARKED_MAX && self.flush_parked() {
            self.retire_free_pools();
        }
    }

    /// `forget_mirrors` with the state locked (`[a, end)`)
    unsafe fn forget_mirrors_in(&mut self, a: usize, end: usize) {
        let touches = |x: usize| x < end && x + TILE > a;
        let gone: Vec<usize> = self.mirror_src.iter().filter(|&(&w, &c)| touches(w) || touches(c)).map(|(&w, _)| w).collect();
        for w in gone {
            self.drop_mirror(w);
        }
        self.asks.retain(|&w| {
            let keep = !touches(w) && !mirror_source_tile(w).is_some_and(touches);
            if !keep {
                forget_ask(w);
            }
            keep
        });
        if self.parked.len() >= PARKED_MAX && self.flush_parked() {
            self.retire_free_pools();
        }
    }

    /// The tiles of a buffer placed at `[start, start + len)` (`place_buffer`, `start` a tile's),
    /// with video memory where they can have it: a tile of a GPU view shows its CPU view
    /// (`new_mirror`, with no committed page and past `mirror_cap` too: the buffer needs the tile,
    /// and maps it once), the others are the
    /// window's own, with the pages committed there.
    unsafe fn buffer_tiles(&mut self, start: usize, len: usize) -> Vec<(usize, (u32, u32))> {
        let end = start + len;
        let mut over: Vec<(usize, usize)> = Vec::new();
        for t in (start..end).step_by(TILE) {
            if self.mapped.contains_key(&t) || self.window_index(t).is_none() {
                continue;
            }
            if let Some(c) = self.mirror_src.get(&t).copied().or_else(|| buffer_source(tile_shows(t))) {
                over.push((t, c));
            }
        }
        if !over.is_empty() {
            // linked before their CPU tiles' pages are asked about (`holes_now`), and given up
            // when their pair went meanwhile, linked again when their page changed (as in
            // `take_asks`)
            over.retain_mut(|(t, c)| {
                if self.mirror_src.contains_key(t) {
                    return true;
                }
                self.link_mirror(*t, *c);
                match buffer_source(tile_shows(*t)) {
                    Some(now) if now == *c => true,
                    Some(now) => {
                        self.unlink_kept(*t);
                        self.link_mirror(*t, now);
                        *c = now;
                        true
                    }
                    None => !self.unlink_mirror(*t),
                }
            });
            let holes = self.holes_now(&over.iter().map(|o| o.1).collect::<Vec<_>>());
            for (&(t, c), h) in over.iter().zip(holes) {
                self.new_mirror(t, c, h, true);
            }
        }
        // tiles listed with no video memory get it for the buffer (before `ensure_mapped`, which
        // sends what is pending or asks for it soon); one that finds none is left out
        let mut reused = Vec::new();
        self.materialize(start, len, true, &mut reused);
        self.to_zero.extend(reused);
        self.ensure_mapped(start, len, |t| query_tile_holes(t));
        (start..end).step_by(TILE).filter_map(|t| self.mapped.get(&t).filter(|e| e.pool != LAZY).map(|e| (t, e.pool))).collect()
    }

    /// `mapped_ranges` of this state
    fn watched_ranges(&self) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = self.runs.iter().chain(self.src_runs.iter()).map(|(&a, &b)| (a, b)).collect();
        if !self.src_runs.is_empty() {
            out.sort_unstable();
        }
        out
    }
}

/// The sparse windows' tiles listed (with video memory, or to get it at their first use,
/// `LAZY`) as sorted (start, end) runs, and the CPU memory
/// tiles of GPU views show (never the views' own addresses): where a CPU write has to reach the
/// GPU.
pub fn mapped_ranges() -> Vec<(usize, usize)> {
    let t0 = super::look_stats::deep().then(std::time::Instant::now);
    let g = lock_for_submit();
    let _held = holding(Holder::Ranges);
    if let Some(t0) = t0 {
        HELD.add(HELD_RANGES_WAIT, t0.elapsed().as_micros() as u64);
    }
    let Some(s) = g.as_ref() else { return Vec::new() };
    let runs = s.watched_ranges();
    // (local, #877) with the lazily allocated ranges (`alloc_lazy`): a write there is what maps
    // their tiles
    if LAZY_ACTIVE.load(Ordering::Acquire) {
        let lazy = LAZY_RANGES.lock().unwrap();
        if !lazy.is_empty() {
            let mut runs = runs;
            runs.extend(lazy.iter().map(|(&a, &b)| (a, b)));
            runs.sort_unstable();
            let mut out: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
            for (a, b) in runs {
                match out.last_mut() {
                    Some(last) if a <= last.1 => last.1 = last.1.max(b),
                    _ => out.push((a, b)),
                }
            }
            return out;
        }
    }
    runs
}

/// time spent and pages found by the write-watch scans (`refresh_once`), for the timing line
static WATCH: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// [lock wait, chunk scan, upload] in µs, runs uploaded
static PHASES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Microseconds so far in [looks for CPU writes, waits for the state lock, chunk scans, uploads].
pub fn timing_totals() -> [u64; 4] {
    [WATCH[0].load(Ordering::Relaxed), PHASES[0].load(Ordering::Relaxed), PHASES[1].load(Ordering::Relaxed), PHASES[2].load(Ordering::Relaxed)]
}

pub fn note_watch(t: std::time::Duration, pages: usize) {
    WATCH[0].fetch_add(t.as_micros() as u64, Ordering::Relaxed);
    WATCH[1].fetch_add(pages as u64, Ordering::Relaxed);
}

/// One page to scan: the CPU address of its bytes, its address in its window, and the slot of its
/// hashes (disjoint for every page of one submit: the window pages are distinct and their tiles'
/// entries are only touched through these).
struct PageSlot {
    cpu: usize,
    win: usize,
    slot: *mut Option<Box<[u64; CHUNKS]>>,
}

/// Of the pages `found` (CPU addresses, sorted), how many slots of `pages` the scan went over and
/// how many of those it found unchanged: no run of `runs` (window addresses, sorted, as the scan
/// made them) in the slot's window page.
fn own_scanned(found: &[usize], pages: &[PageSlot], runs: &[(usize, usize)]) -> (usize, usize) {
    if found.is_empty() {
        return (0, 0);
    }
    let (mut scanned, mut unchanged) = (0, 0);
    for p in pages.iter().filter(|p| found.binary_search(&p.cpu).is_ok()) {
        scanned += 1;
        let k = runs.partition_point(|r| r.0 + r.1 <= p.win);
        if !runs.get(k).is_some_and(|r| r.0 < p.win + PAGE) {
            unchanged += 1;
        }
    }
    (scanned, unchanged)
}

/// The pages of one submit, shared with the worker threads (`worker_pool`).
struct SharedPages<'a>(&'a [PageSlot]);
// SAFETY: the slots are disjoint, and the scan finishes before the state's lock is given up
unsafe impl Sync for SharedPages<'_> {}

/// Scan `pages` (`scan_page` on each, the bytes at their CPU address); the changed runs go to
/// `out` as window addresses, in their order. How the work was shared with the workers, if it was.
unsafe fn scan_pages(pages: &[PageSlot], out: &mut Vec<(usize, usize)>) -> Option<super::worker_pool::Shares> {
    /// pages a worker takes at a time, and the number from which the work is shared
    const PIECE: usize = 64;
    const SHARE_FROM: usize = 256;
    let scan = |p: &PageSlot, out: &mut Vec<(usize, usize)>| {
        let bytes = std::slice::from_raw_parts(p.cpu as *const u8, PAGE);
        scan_page(bytes, p.win, &mut *p.slot, out);
    };
    if pages.len() < SHARE_FROM {
        pages.iter().for_each(|p| scan(p, out));
        return None;
    }
    let pieces = pages.len().div_ceil(PIECE);
    let outs: Vec<Mutex<Vec<(usize, usize)>>> = (0..pieces).map(|_| Mutex::new(Vec::new())).collect();
    let shared = SharedPages(pages);
    let work = |k: usize| {
        // (the whole struct, not its field: that is what is Sync)
        let shared = &shared;
        let mut mine = Vec::new();
        for p in &shared.0[k * PIECE..((k + 1) * PIECE).min(shared.0.len())] {
            scan(p, &mut mine);
        }
        *outs[k].lock().unwrap_or_else(|e| e.into_inner()) = mine;
    };
    let shares = super::worker_pool::run(pieces, &work);
    for o in outs {
        for (addr, len) in o.into_inner().unwrap_or_else(|e| e.into_inner()) {
            push_run(out, addr, len);
        }
    }
    Some(shares)
}

/// Copies from the title's memory into the staging ring: shared with the worker threads when
/// there are many bytes.
unsafe fn run_copies(jobs: &mut Vec<(usize, usize, usize)>, staging: *mut u8) {
    const SHARE_FROM: usize = 512 << 10;
    let total: usize = jobs.iter().map(|j| j.2).sum();
    if total < SHARE_FROM || jobs.len() < 2 {
        for &(src, off, n) in jobs.iter() {
            std::ptr::copy_nonoverlapping(src as *const u8, staging.add(off), n);
        }
    } else {
        // pieces of about the same bytes, in the order of the jobs
        let target = (total / 16).max(64 << 10);
        let mut pieces: Vec<(usize, usize)> = Vec::new();
        let (mut from, mut bytes) = (0usize, 0usize);
        for (k, j) in jobs.iter().enumerate() {
            bytes += j.2;
            if bytes >= target {
                pieces.push((from, k + 1));
                from = k + 1;
                bytes = 0;
            }
        }
        if from < jobs.len() {
            pieces.push((from, jobs.len()));
        }
        struct Staging(*mut u8);
        // SAFETY: the copies go to disjoint pieces of the ring, all from the caller's job list
        unsafe impl Sync for Staging {}
        let st = Staging(staging);
        let list: &[(usize, usize, usize)] = jobs;
        super::worker_pool::run(pieces.len(), &|k| {
            let st = &st;
            for &(src, off, n) in &list[pieces[k].0..pieces[k].1] {
                unsafe { std::ptr::copy_nonoverlapping(src as *const u8, st.0.add(off), n) };
            }
        });
    }
    jobs.clear();
}

/// The tiles of lazily allocated ranges (`alloc_lazy`) that get video memory now: those without
/// it that have a non-zero byte in a page of `dirty` (sorted) inside such a range. Sorted.
unsafe fn map_lazy_written(s: &mut State, dirty: &[usize], page: usize) -> Vec<usize> {
    let mut fresh: Vec<usize> = Vec::new();
    if !LAZY_ACTIVE.load(Ordering::Acquire) {
        return fresh;
    }
    let lazy = LAZY_RANGES.lock().unwrap();
    if lazy.is_empty() {
        return fresh;
    }
    for &a in dirty {
        let mut pg = a;
        while pg < a + page {
            let t = pg & !(TILE - 1);
            if fresh.last() != Some(&t) && in_lazy_range(&lazy, pg) && !s.mapped.contains_key(&t) && page_has_nonzero(std::slice::from_raw_parts(pg as *const u8, PAGE)) {
                let holes = lazy_tile_holes(&lazy, t);
                s.ensure_mapped(t, TILE, |_| holes);
                if s.mapped.contains_key(&t) {
                    fresh.push(t);
                }
            }
            pg += PAGE;
        }
    }
    fresh
}

/// Before the title's ExecuteCommandLists on `queue`: the pages the CPU wrote (`dirty`, 4 KiB
/// pages from the write-watch, sorted) and the reused tiles are copied asynchronously into video
/// memory, and `queue` GPU-waits for that and for new tile mappings.
pub unsafe fn before_execute(queue: &ID3D12CommandQueue, dirty: &[usize], page: usize) {
    let t0 = std::time::Instant::now();
    // (who had the lock when this began to wait for it: `XCL_LOOK_STATS=2`)
    let behind = (HOLDER.load(Ordering::Relaxed) as usize).min(HOLDERS - 1);
    FLUSHERS_WAITING.fetch_add(1, Ordering::AcqRel);
    let mut g = lock_first(&STATE, &SUBMITS_WAITING);
    FLUSHERS_WAITING.fetch_sub(1, Ordering::AcqRel);
    let _held = holding(Holder::Submit);
    let t_got = super::look_stats::deep().then(std::time::Instant::now);
    forget_released(&mut g);
    let Some(s) = g.as_mut() else { return };
    if MIRROR_ASKING.swap(false, Ordering::AcqRel) {
        s.asks.append(&mut MIRROR_ASKS.lock().unwrap());
    }
    if REMAPPING.swap(false, Ordering::AcqRel) {
        take_remaps(&mut s.remaps, &mut REMAPS.lock().unwrap());
    }
    if WRITE_ASKING.swap(false, Ordering::AcqRel) {
        let asks = std::mem::take(&mut *WRITE_ASKS.lock().unwrap_or_else(|e| e.into_inner()));
        s.give_memory(&asks);
    }
    let waited = t0.elapsed().as_micros() as u64;
    PHASES[0].fetch_add(waited, Ordering::Relaxed);
    SUBMIT_US.fetch_add(waited, Ordering::Relaxed);
    if let Some(t_got) = t_got {
        LOCK_SPLIT.add(LS_WAIT, (t_got - t0).as_micros() as u64);
        LOCK_SPLIT.add(LS_ALL, waited);
        HELD.add(HOLDERS + behind, (t_got - t0).as_micros() as u64);
    }
    s.before_execute_on(queue, dirty, page);
}

/// What the title's lists being recorded will write on the GPU in the sparse windows (address,
/// bytes; `will_write`), taken by the next submit: with `lazy_commits` a tile there may have no
/// video memory yet, and the write would go nowhere.
static WRITE_ASKS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
static WRITE_ASKING: AtomicBool = AtomicBool::new(false);

/// A list being recorded writes `[va, va + len)` on the GPU with a memory operation (an
/// immediate, a fill, a copy, a timestamp): with `lazy_commits`, its tiles get their video memory
/// at the next submit, which comes before the list's (`State::give_memory`). Never waits for the
/// state.
pub(super) fn will_write(va: usize, len: usize) {
    if len == 0 || !lazy_commits() {
        return;
    }
    WRITE_ASKS.lock().unwrap_or_else(|e| e.into_inner()).push((va, len));
    WRITE_ASKING.store(true, Ordering::Release);
}

/// microseconds in `before_execute`, the wait for the state's lock with it
static SUBMIT_US: AtomicU64 = AtomicU64::new(0);

/// `XCL_LOOK_STATS=2`: the parts of what `submit_cost` calls "sparse lock" (`PHASES[0]`: the wait
/// for the state's lock and the work before the scan), in µs but where said: a long wait asks for
/// less work under the lock elsewhere, the parts for less of that work here. (The pages of the
/// tiles that came to show a CPU view are gathered after it: the scan split's gather.)
static LOCK_SPLIT: super::look_stats::Sums<11> = super::look_stats::Sums::new();
const LS_SUBMITS: usize = 0;
/// `PHASES[0]` of these submits
const LS_ALL: usize = 1;
const LS_WAIT: usize = 2;
/// landing finished copies back, and their bytes
const LS_BACK: usize = 3;
const LS_BACK_BYTES: usize = 4;
/// the tiles over CPU views (`map_mirrors`, `mirror_writes`)
const LS_VIEWS: usize = 5;
/// `flush_pending`, its tiles and calls
const LS_MAP: usize = 6;
const LS_MAP_TILES: usize = 7;
const LS_MAP_CALLS: usize = 8;
/// `flush_parked` (with the pools it left free), its tiles
const LS_UNMAP: usize = 9;
const LS_UNMAP_TILES: usize = 10;

/// so far: [tiles `flush_pending` mapped, its calls, tiles `flush_parked` unmapped, bytes of
/// copies back landed] (read under the state's lock, which all of them are done with)
static TILE_WORK: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// so far, always counted: the sparse queue's UpdateTileMappings calls and their tiles, [map
/// calls, tiles mapped, unmap calls, tiles unmapped] in the sparse windows, then the same in
/// the view windows (GPU views). A placed buffer's mapping counts with its window's.
static TILE_CALLS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// `calls` UpdateTileMappings of `tiles` tiles of the resource at `base` (`TILE_CALLS`)
fn count_tile_calls(base: usize, unmap: bool, calls: usize, tiles: usize) {
    let k = if super::view_want::is_view(base as u64) { 4 } else { 0 } + if unmap { 2 } else { 0 };
    TILE_CALLS[k].fetch_add(calls as u64, Ordering::Relaxed);
    TILE_CALLS[k + 1].fetch_add(tiles as u64, Ordering::Relaxed);
}

/// `TILE_CALLS` so far, for the periodic log line (`tile_calls_line`)
pub fn tile_calls() -> [u64; 8] {
    std::array::from_fn(|k| TILE_CALLS[k].load(Ordering::Relaxed))
}

/// The tile mapping calls between two `tile_calls`, None when there were none.
pub fn tile_calls_line(now: &[u64; 8], before: &[u64; 8]) -> Option<String> {
    let d: [u64; 8] = std::array::from_fn(|k| now[k].saturating_sub(before[k]));
    if d.iter().all(|&n| n == 0) {
        return None;
    }
    Some(format!(
        "sparse windows {} calls mapping {} tiles, {} calls unmapping {} tiles; GPU views {} calls mapping {} tiles, {} calls unmapping {} tiles",
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
    ))
}

/// The GPU-view diagnostics' counts (`VS`), all zero but the asks before there is a state.
fn view_stats() -> [u64; VS] {
    let mut x = [0u64; VS];
    x[0] = VIEW_ASKED.load(Ordering::Relaxed);
    let g = lock_state();
    let _held = holding(Holder::Other);
    let Some(s) = g.as_ref() else { return x };
    let d = &s.diag;
    let rest = [s.counts.fills, s.counts.given_back, d.kept, d.back[0], d.back[1], d.back[2], d.back[3], d.back_bytes, d.submits, d.deferred];
    x[1..VS_NOW].copy_from_slice(&rest);
    let now = [s.mirrored, s.kept.len(), s.later.len() + s.asks.len(), s.refused.len()];
    for (k, n) in now.into_iter().enumerate() {
        x[VS_NOW + k] = n as u64;
    }
    x
}

/// The GPU-view diagnostics' line between two `view_stats`, None when nothing was counted.
fn view_stats_line(now: &[u64; VS], before: &[u64; VS]) -> Option<String> {
    let d: [u64; VS_NOW] = std::array::from_fn(|k| now[k].saturating_sub(before[k]));
    if d.iter().all(|&n| n == 0) {
        return None;
    }
    Some(format!(
        "GPU-view tiles (2 s): {} asked for, {} filled, {} given back, {} kept; back after 0-1 / 2-8 / 9-64 / more submits: {} / {} / {} / {} ({:.1} MB filled again); {} of {} submits left asked tiles waiting; now {} with video memory, {} kept, {} waiting, {} refused",
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8] as f64 / (1 << 20) as f64, d[10], d[9],
        now[VS_NOW], now[VS_NOW + 1], now[VS_NOW + 2], now[VS_NOW + 3]
    ))
}

/// `view_stats_line` every 2 s on a thread of its own (`ViewSwitches::stats`; once).
fn start_view_stats() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-view-stats".into()).spawn(|| {
            let mut before = view_stats();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                let now = view_stats();
                if let Some(line) = view_stats_line(&now, &before) {
                    crate::xlog_info!("D3D12.X: {line}");
                }
                before = now;
            }
        });
    });
}

/// The sparse lock split line of `x` (`LOCK_SPLIT`'s sums), as averages a submit.
fn lock_split_line(x: &[u64; 11]) -> String {
    let n = x[LS_SUBMITS].max(1) as f64;
    let ms = |k: usize| x[k] as f64 / n / 1e3;
    let parts = [LS_WAIT, LS_BACK, LS_VIEWS, LS_MAP, LS_UNMAP].iter().map(|&k| x[k]).sum::<u64>();
    format!(
        "sparse lock split ({} submits): wait {:.3} ms, copies back {:.3} ms ({:.0} KB), GPU-view tiles {:.3} ms, mappings {:.3} ms ({:.1} tiles in {:.1} calls), unmappings {:.3} ms ({:.1} tiles), rest {:.3} ms a submit (of {:.3} ms)",
        x[LS_SUBMITS],
        ms(LS_WAIT),
        ms(LS_BACK),
        x[LS_BACK_BYTES] as f64 / n / 1024.0,
        ms(LS_VIEWS),
        ms(LS_MAP),
        x[LS_MAP_TILES] as f64 / n,
        x[LS_MAP_CALLS] as f64 / n,
        ms(LS_UNMAP),
        x[LS_UNMAP_TILES] as f64 / n,
        x[LS_ALL].saturating_sub(parts) as f64 / n / 1e3,
        ms(LS_ALL)
    )
}

impl State {
    /// `before_execute` with the state locked
    unsafe fn before_execute_on(&mut self, queue: &ID3D12CommandQueue, dirty: &[usize], page: usize) {
        let t0 = std::time::Instant::now();
        let s = self;
        // (`XCL_LOOK_STATS=2`: the time after each step, and the work counted so far)
        let split = super::look_stats::deep();
        let stamp = || split.then(std::time::Instant::now);
        let work0: [u64; 4] = std::array::from_fn(|k| if split { TILE_WORK[k].load(Ordering::Relaxed) } else { 0 });
        // XCL's own writes since the look before, and those of them this look found (before the
        // copies back below land: those the next look finds)
        let own = (split && LOOKED.with(|l| l.replace(false))).then(|| {
            let mut own = std::mem::take(&mut s.own_pages);
            own.sort_unstable();
            own.dedup();
            let found: Vec<usize> = own.iter().copied().filter(|p| dirty.binary_search(p).is_ok()).collect();
            (own.len(), found)
        });
        // (copies back done before the tiles change below: a page that stops being a hole there
        // would take what the GPU had in it)
        s.drain_readbacks(Drain::Finished);
        let t_back = stamp();
        // tiles whose page the queue's page mappings changed (after the copies back: one for the
        // old link lands nowhere), tiles asked to show CPU views, and the CPU writes that change
        // them: before the flushes below, so their mappings reach the title's work (a tile taken
        // back from `parked` needs none)
        s.remap_mirrors();
        s.map_mirrors();
        s.diag.submits += 1;
        if s.diag.stats && !(s.later.is_empty() && s.asks.is_empty()) {
            s.diag.deferred += 1;
        }
        s.mirror_writes(dirty, page);
        let t_views = stamp();
        // (local, #877) lazily allocated ranges (`alloc_lazy`): a tile gets video memory with the
        // first non-zero CPU write to it (before the mappings are sent and the reused tiles
        // cleared, below)
        let fresh = map_lazy_written(s, dirty, page);
        // the title's work sees what was committed mapped …
        s.flush_pending();
        PENDING.store(false, Ordering::Release);
        let t_map = stamp();
        // the title's work sees what was decommitted unmapped, as it was (pools whose last tiles
        // cooled since go with no tile freed now too: those a flush frees wait for the GPU in
        // `quarantined`, then still cool there: `retire_due` sees them). This queue's work so far
        // is waited for too.
        s.note_queue(queue);
        let flushed = s.flush_parked();
        s.drain_quarantined();
        if flushed || retire_due(&s.cooling, !s.retired.is_empty(), s.retire_tried, std::time::Instant::now()) {
            s.retire_free_pools();
        }
        let t_locked = std::time::Instant::now();
        PHASES[0].fetch_add((t_locked - t0).as_micros() as u64, Ordering::Relaxed);
        if let (Some(t_back), Some(t_views), Some(t_map)) = (t_back, t_views, t_map) {
            let us = |a: std::time::Instant, b: std::time::Instant| (b - a).as_micros() as u64;
            let work = |k: usize| TILE_WORK[k].load(Ordering::Relaxed) - work0[k];
            LOCK_SPLIT.add(LS_ALL, us(t0, t_locked));
            LOCK_SPLIT.add(LS_BACK, us(t0, t_back));
            LOCK_SPLIT.add(LS_BACK_BYTES, work(3));
            LOCK_SPLIT.add(LS_VIEWS, us(t_back, t_views));
            LOCK_SPLIT.add(LS_MAP, us(t_views, t_map));
            LOCK_SPLIT.add(LS_MAP_TILES, work(0));
            LOCK_SPLIT.add(LS_MAP_CALLS, work(1));
            LOCK_SPLIT.add(LS_UNMAP, us(t_map, t_locked));
            LOCK_SPLIT.add(LS_UNMAP_TILES, work(2));
            if LOCK_SPLIT.add(LS_SUBMITS, 1) >= 1024 {
                let x = LOCK_SPLIT.take();
                crate::xlog_info!("D3D12.X: {}; {}", lock_split_line(&x), holders_line(&HELD.take(), x[LS_SUBMITS]));
            }
        }
        let zero_runs = zero_runs(std::mem::take(&mut s.to_zero));
        for &(a, n) in &zero_runs {
            s.clear_sent_range(a, n);
        }
        let mut dirty_runs = std::mem::take(&mut s.scratch_runs);
        dirty_runs.clear();
        let mut pages: Vec<PageSlot> = Vec::with_capacity(dirty.len());
        // tiles that came to show a CPU view: all of it (what is committed there)
        let mut fill = std::mem::take(&mut s.fill);
        fill.sort_unstable();
        fill.dedup();
        for &w in &fill {
            let Some(entry) = s.mapped.get_mut(&w).filter(|e| e.src != 0) else { continue };
            s.counts.fills += 1;
            for idx in 0..PAGES_PER_TILE {
                if entry.holes & (1u16 << idx) == 0 {
                    pages.push(PageSlot { cpu: entry.src + idx * PAGE, win: w + idx * PAGE, slot: &mut entry.sent[idx] });
                    s.counts.fill_bytes += PAGE as u64;
                }
            }
        }
        // tiles whose bytes found no video memory at an upload: all that is committed there, once
        // a moment went by since (the upload gives them their tile, or leaves them here again)
        let mut again = false;
        if !s.starved.is_empty() && !s.short_of_memory() {
            let mut starved = std::mem::take(&mut s.starved);
            starved.sort_unstable();
            starved.dedup();
            for &t in &starved {
                let Some(entry) = s.mapped.get_mut(&t).filter(|e| e.src == 0) else { continue };
                again = true;
                for idx in 0..PAGES_PER_TILE {
                    if entry.holes & (1u16 << idx) == 0 {
                        pages.push(PageSlot { cpu: t + idx * PAGE, win: t + idx * PAGE, slot: &mut entry.sent[idx] });
                    }
                }
            }
        }
        // (the tile of `fresh` whose pages are taken already)
        let mut fresh_taken = usize::MAX;
        let mut i = 0;
        while i < dirty.len() {
            let a = dirty[i];
            let mut n = page;
            i += 1;
            while i < dirty.len() && dirty[i] == a + n {
                n += page;
                i += 1;
            }
            // only what has video memory (committed tiles of a sparse window, and the tiles that
            // show CPU memory written)
            let mut x = a;
            while x < a + n {
                let t = x & !(TILE - 1);
                let end = (t + TILE).min(a + n);
                if let Some(entry) = s.mapped.get_mut(&t).filter(|e| e.src == 0) {
                    if !fresh.is_empty() && fresh.binary_search(&t).is_ok() {
                        // mapped just now (`map_lazy_written`): all of it goes up, once
                        if fresh_taken != t {
                            fresh_taken = t;
                            for idx in 0..PAGES_PER_TILE {
                                if entry.holes & (1u16 << idx) == 0 {
                                    pages.push(PageSlot { cpu: t + idx * PAGE, win: t + idx * PAGE, slot: &mut entry.sent[idx] });
                                }
                            }
                        }
                    } else {
                        let mut pg = x & !(PAGE - 1);
                        while pg < end {
                            let idx = (pg - t) / PAGE;
                            if entry.holes & (1u16 << idx) == 0 {
                                pages.push(PageSlot { cpu: pg, win: pg, slot: &mut entry.sent[idx] });
                            }
                            pg += PAGE;
                        }
                    }
                }
                if let Some(over) = s.mirrors.get(&t) {
                    for &w in over {
                        let Some(entry) = s.mapped.get_mut(&w).filter(|e| e.src == t) else { continue };
                        let mut pg = x & !(PAGE - 1);
                        while pg < end {
                            let idx = (pg - t) / PAGE;
                            if entry.holes & (1u16 << idx) == 0 {
                                pages.push(PageSlot { cpu: pg, win: w + (pg - t), slot: &mut entry.sent[idx] });
                            }
                            pg += PAGE;
                        }
                    }
                }
                x = end;
            }
        }
        if !s.mirrors.is_empty() || again {
            // (in window order, for the runs; a page filled and written is one slot, scanned once)
            pages.sort_unstable_by_key(|p| p.win);
            pages.dedup_by_key(|p| p.win);
        }
        fill.clear();
        s.fill = fill;
        // (`XCL_LOOK_STATS=2`: the gather of the pages apart from their scan)
        let t_gathered = super::look_stats::deep().then(std::time::Instant::now);
        let shares = scan_pages(&pages, &mut dirty_runs);
        let t_scanned = std::time::Instant::now();
        super::look_stats::scan(pages.len(), &dirty_runs);
        if let Some(t_gathered) = t_gathered {
            super::look_stats::scan_split(t_gathered - t_locked, t_scanned - t_gathered, shares);
        }
        if let Some((written, found)) = own {
            let (scanned, unchanged) = own_scanned(&found, &pages, &dirty_runs);
            super::look_stats::own_writes(written, found.len(), scanned, unchanged);
        }
        PHASES[1].fetch_add((t_scanned - t_locked).as_micros() as u64, Ordering::Relaxed);
        PHASES[3].fetch_add((zero_runs.len() + dirty_runs.len()) as u64, Ordering::Relaxed);
        // Submit tile clears before CPU writes so overlapping CopyBufferRegion destinations have no WAW hazard.
        if !zero_runs.is_empty() {
            s.upload_zeros(&zero_runs);
        }
        if !dirty_runs.is_empty() {
            s.upload_dirty(&dirty_runs);
        }
        if s.poison.as_ref().is_some_and(|p| p.at.elapsed() >= std::time::Duration::from_millis(16)) {
            s.refresh_poison();
        }
        PHASES[2].fetch_add(t_scanned.elapsed().as_micros() as u64, Ordering::Relaxed);
        {
            static CALLS: AtomicU64 = AtomicU64::new(0);
            static BYTES: AtomicU64 = AtomicU64::new(0);
            let bytes: usize = zero_runs.iter().chain(dirty_runs.iter()).map(|r| r.1).sum();
            let n = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
            let b = BYTES.fetch_add(bytes as u64, Ordering::Relaxed) + bytes as u64;
            let us = SUBMIT_US.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed) + t0.elapsed().as_micros() as u64;
            if n.is_power_of_two() && n >= 16 {
                let listed = s.mapped.values().filter(|e| e.pool == LAZY).count();
                crate::xlog_info!("sparse windows: {n} submits, {} MB of CPU writes uploaded, {} ms spent, {} tiles mapped", b >> 20, us / 1000, s.mapped.len() - listed);
                let (own, unsent) = never_sent(&s.mapped);
                crate::xlog_info!("sparse windows: {unsent} of the windows' own {own} mapped tiles ({} of {} MB) never got a byte from the CPU (zeros, or what the GPU wrote); {listed} listed with no video memory (XCL_SPARSE_LAZY)", unsent * TILE >> 20, own * TILE >> 20);
                let ms = |x: &AtomicU64| x.load(Ordering::Relaxed) / 1000;
                let c = &s.counts;
                crate::xlog_info!(
                    "sparse windows timing: write-watch {} ms ({} dirty pages), lock wait {} ms, chunk scan {} ms, upload {} ms ({} runs, {} of them in {} dispatches); GPU views: {} tiles with video memory ({} at most, cap {}), {} filled ({} MB), {} refused, {} given back, {} holes asked again",
                    ms(&WATCH[0]),
                    WATCH[1].load(Ordering::Relaxed),
                    ms(&PHASES[0]),
                    ms(&PHASES[1]),
                    ms(&PHASES[2]),
                    PHASES[3].load(Ordering::Relaxed),
                    SCATTERED[1].load(Ordering::Relaxed),
                    SCATTERED[0].load(Ordering::Relaxed),
                    s.mirrored,
                    c.peak,
                    s.mirror_cap,
                    c.fills,
                    c.fill_bytes >> 20,
                    c.refused,
                    c.given_back,
                    c.rechecks
                );
            }
        }
        dirty_runs.clear();
        s.scratch_runs = dirty_runs;
        let q = queue.as_raw() as usize;
        if s.value > 0 && s.waited.get(&q).copied().unwrap_or(0) < s.value {
            let _ = queue.Wait(&s.fence, s.value);
            super::list_x::note_internal_wait(q, "sparse uploads and tile mappings", &s.fence, s.value);
            s.waited.insert(q, s.value);
        }
    }
}

/// A buffer the title places at `va` in a sparse window (`bytes` long, `desc` the PC
/// description): a reserved resource over the window's tiles there, starting at the 64 KiB
/// below `va` (returned: its distance from there).
pub unsafe fn place_buffer(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<(crate::util::HRESULT, usize)> {
    let mut g = lock_state();
    let _held = holding(Holder::Buffers);
    let s = g.as_mut()?;
    let start = va & !(TILE - 1);
    let delta = va - start;
    let bytes = desc.Width as usize;
    if bytes == 0 {
        return None;
    }
    s.window_index(start)?;
    s.window_index(va.checked_add(bytes)?.checked_sub(1)?)?;
    let mut d = *desc;
    d.Width = (delta + bytes) as u64;
    d.Alignment = 0;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // Reserved buffers only accept NONE or ALLOW_UNORDERED_ACCESS (strip console flags and ALLOW_CROSS_ADAPTER).
    d.Flags &= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if state & (D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32 | D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32) != 0 {
        d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    let hr = super::real::dev_create_reserved_resource(this, &d as *const _ as usize, state as usize, 0, riid, ppv);
    if hr < 0 || ppv.is_null() || (*ppv).is_null() {
        return None;
    }
    // (a name for what a device removal reports about an address, `real::log_device_removed`)
    if let Some(r) = ID3D12Resource::from_raw_borrowed(&*ppv) {
        let _ = r.SetName(&windows::core::HSTRING::from(format!("title buffer over tiles at {va:#x} ({bytes:#x} bytes)")));
    }
    // its tiles: the window's video memory at the same addresses; probe actual OS commitment
    // so uncommitted pages (e.g. [start, va) or reserved tails) stay marked as holes (of the CPU
    // view a tile of a GPU view shows)
    let tiles = s.buffer_tiles(start, delta + bytes);
    let raw = *ppv;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?.clone();
    // (`XCL_VIEW_POISON`: its tiles without video memory show the poison tile as the window's do)
    s.poison_all(&res, (delta + bytes).div_ceil(TILE) as u32);
    s.map_tiles(&res, start, &tiles);
    for &(_, tile) in &tiles {
        *s.pinned.entry(tile).or_default() += 1;
    }
    s.buffers.insert(raw as usize, tiles);
    s.signal();
    Some((hr, delta))
}

/// Placed buffers released while `STATE` was held, let go by the next `lock_state`. Held only
/// to push or take, never across a call out.
static FORGOTTEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// `STATE`, with the buffers released while it was held let go first.
fn lock_state() -> std::sync::MutexGuard<'static, Option<State>> {
    let mut g = STATE.lock().unwrap();
    forget_released(&mut g);
    g
}

/// Who holds `STATE` (`HOLDER`): the time a submit waits for it is put down to the holder it found
/// there when it began to wait, and each holder's own time under it is summed (`XCL_LOOK_STATS=2`,
/// with the sparse lock split line).
#[derive(Clone, Copy)]
#[repr(u8)]
enum Holder {
    Nobody = 0,
    Submit,
    After,
    Ranges,
    Settle,
    ReadbackThread,
    TileMaps,
    Commits,
    Sources,
    Buffers,
    CpuRead,
    Other,
}
const HOLDERS: usize = 12;
const HOLDER_NAMES: [&str; HOLDERS] = [
    "nobody",
    "submits",
    "after stages",
    "ranges",
    "copies back settled",
    "readback thread",
    "put-off mappings",
    "commits",
    "GPU-view sources",
    "placed buffers",
    "CPU reads",
    "other",
];
static HOLDER: AtomicU8 = AtomicU8::new(0);
/// [µs each holder held the lock], [µs submits waited behind each], then the submit's other
/// waits for it: in its after stage, and for its ranges
static HELD: super::look_stats::Sums<{ 2 * HOLDERS + 2 }> = super::look_stats::Sums::new();
const HELD_AFTER_WAIT: usize = 2 * HOLDERS;
const HELD_RANGES_WAIT: usize = 2 * HOLDERS + 1;

/// `STATE` is held by its holder until this is dropped (declared after the guard: it goes first).
struct Holding(Holder, Option<std::time::Instant>);

fn holding(who: Holder) -> Holding {
    HOLDER.store(who as u8, Ordering::Relaxed);
    Holding(who, super::look_stats::deep().then(std::time::Instant::now))
}

impl Drop for Holding {
    fn drop(&mut self) {
        HOLDER.store(Holder::Nobody as u8, Ordering::Relaxed);
        if let Some(t) = self.1 {
            HELD.add(self.0 as usize, t.elapsed().as_micros() as u64);
        }
    }
}

/// The holders part of the sparse lock split line (`HELD`'s sums over `submits`), in ms a submit:
/// who held the lock how long, and whom the submits' waits for it were behind.
fn holders_line(x: &[u64; 2 * HOLDERS + 2], submits: u64) -> String {
    let n = submits.max(1) as f64;
    let part = |from: usize| -> String {
        let items: Vec<String> = (0..HOLDERS).filter(|&k| x[from + k] > 0).map(|k| format!("{} {:.3}", HOLDER_NAMES[k], x[from + k] as f64 / n / 1e3)).collect();
        if items.is_empty() { "none".into() } else { items.join(", ") }
    };
    format!(
        "held by: {} ms a submit; waits behind: {} ms a submit; the submit's other waits for it: after stage {:.3}, ranges {:.3} ms a submit",
        part(0),
        part(HOLDERS),
        x[HELD_AFTER_WAIT] as f64 / n / 1e3,
        x[HELD_RANGES_WAIT] as f64 / n / 1e3
    )
}

fn forget_released(g: &mut Option<State>) {
    let gone = std::mem::take(&mut *FORGOTTEN.lock().unwrap());
    let Some(s) = g.as_mut() else { return };
    for res in gone {
        forget_in(s, res);
    }
}

/// A placed buffer is gone: its tiles are not pinned by it any more. This runs from the
/// resource's last Release, which D3D12 can make while it holds a lock of its own (in a title's
/// command list Reset), and `before_execute` calls D3D12 with `STATE` held: waiting for `STATE`
/// here deadlocked the two threads. So it does not wait: with `STATE` held, the holder's next
/// `lock_state` lets the buffer go.
pub fn forget_buffer(res: usize) {
    FORGOTTEN.lock().unwrap().push(res);
    match STATE.try_lock() {
        Ok(mut g) => {
            let _held = holding(Holder::Other);
            forget_released(&mut g)
        }
        Err(std::sync::TryLockError::WouldBlock) => {}
        Err(e) => panic!("sparse state: {e}"),
    }
}

fn forget_in(s: &mut State, res: usize) {
    let Some(tiles) = s.buffers.remove(&res) else { return };
    for (t, tile) in tiles {
        if let Some(c) = s.pinned.get_mut(&tile) {
            *c -= 1;
            if *c == 0 {
                s.pinned.remove(&tile);
                s.release_if_unused(t, tile);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    thread_local! {
        /// The VirtualQuery calls `source_holes` made on this thread.
        pub(super) static VIRTUAL_QUERIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// Run by `take_asks` on this thread between the look up of the asked tiles' pairs and
        /// their link (once)
        pub(super) static BEFORE_LINK: std::cell::Cell<Option<Box<dyn FnOnce()>>> = const { std::cell::Cell::new(None) };
        /// Run by `forget_mirrors` on this thread between dropping the queued asks and the asked
        /// set's tiles (once)
        pub(super) static BETWEEN_FORGETS: std::cell::Cell<Option<Box<dyn FnOnce()>>> = const { std::cell::Cell::new(None) };
        /// The asks `forget_ask` counted on this thread (`MIRROR_FORGOTTEN` is every test's)
        pub(super) static ASKS_FORGOTTEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        /// `lazy_commits` for the test on this thread (`LazyCommits`; None: as the environment
        /// says). A thread's own: the other tests' states run beside it in this process.
        pub(super) static LAZY_COMMITS_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
        /// `take_tile` finds no video memory on this thread
        pub(super) static NO_TILES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// The runs of the tiles mapped, made from all of them (but those over a CPU view: their
    /// own addresses are no CPU memory).
    fn runs_of(s: &State) -> BTreeMap<usize, usize> {
        let mut tiles: Vec<usize> = s.mapped.iter().filter(|(_, e)| e.src == 0).map(|(&t, _)| t).collect();
        tiles.sort_unstable();
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for t in tiles {
            match runs.last_mut() {
                Some(last) if last.1 == t => last.1 = t + TILE,
                _ => runs.push((t, t + TILE)),
            }
        }
        runs.into_iter().collect()
    }

    /// The tiles waiting to be linked again keep their order while few; past `MIRROR_ASKS_MAX` each
    /// waits once, however often its page changed.
    #[test]
    fn a_tile_remapped_again_and_again_waits_once() {
        let (mut remaps, mut q) = (Vec::new(), vec![3 * TILE, TILE, 3 * TILE]);
        take_remaps(&mut remaps, &mut q);
        assert!(q.is_empty());
        assert_eq!(remaps, vec![3 * TILE, TILE, 3 * TILE], "few: as they came");
        // the same 100 tiles changed at each of 100 submits, none taken meanwhile
        for _ in 0..100 {
            let mut q: Vec<usize> = (0..100).map(|k| k * TILE).collect();
            take_remaps(&mut remaps, &mut q);
            assert!(remaps.len() <= MIRROR_ASKS_MAX + 100);
        }
        remaps.sort_unstable();
        remaps.dedup();
        assert_eq!(remaps, (0..100).map(|k| k * TILE).collect::<Vec<_>>());
    }

    /// XCL's own writes in a look: a slot counts as scanned when its CPU page is one of them, and
    /// as unchanged when no changed run lies in its window page (a GPU view's slot has its own).
    #[test]
    fn own_writes_scanned_and_found_unchanged() {
        let slot = |cpu: usize, win: usize| PageSlot { cpu, win, slot: std::ptr::null_mut() };
        let (c, w) = (0x10_0000usize, 0x8000_0000usize);
        // CPU pages 0..4, and page 1 shown again by a GPU view's tile at `w`
        let pages = [slot(c, c), slot(c + PAGE, c + PAGE), slot(c + PAGE, w + PAGE), slot(c + 2 * PAGE, c + 2 * PAGE), slot(c + 3 * PAGE, c + 3 * PAGE)];
        // changed: a chunk of page 0, the end of page 1 (own address only), page 3 whole
        let runs = [(c + 0x40, 0x40), (c + 2 * PAGE - CHUNK, CHUNK), (c + 3 * PAGE, PAGE)];
        // XCL wrote pages 1, 2 and 3 (and 9, which was not scanned)
        let found = [c + PAGE, c + 2 * PAGE, c + 3 * PAGE, c + 9 * PAGE];
        assert_eq!(own_scanned(&found, &pages, &runs), (4, 2));
        assert_eq!(own_scanned(&[], &pages, &runs), (0, 0));
        assert_eq!(own_scanned(&found, &pages, &[]), (4, 4));
    }

    /// The sparse lock split: the parts and the rest are the whole, as averages a submit.
    #[test]
    fn the_sparse_lock_parts_and_rest_are_the_whole() {
        let mut x = [0u64; 11];
        x[LS_SUBMITS] = 4;
        x[LS_ALL] = 4_000;
        x[LS_WAIT] = 1_000;
        x[LS_BACK] = 400;
        x[LS_BACK_BYTES] = 8 << 10;
        x[LS_VIEWS] = 200;
        x[LS_MAP] = 1_200;
        x[LS_MAP_TILES] = 10;
        x[LS_MAP_CALLS] = 2;
        x[LS_UNMAP] = 800;
        x[LS_UNMAP_TILES] = 6;
        assert_eq!(
            lock_split_line(&x),
            "sparse lock split (4 submits): wait 0.250 ms, copies back 0.100 ms (2 KB), GPU-view tiles 0.050 ms, mappings 0.300 ms (2.5 tiles in 0.5 calls), unmappings 0.200 ms (1.5 tiles), rest 0.100 ms a submit (of 1.000 ms)"
        );
        // (parts counted over a submit that ended after the sums were taken: no negative rest)
        x[LS_ALL] = 0;
        assert!(lock_split_line(&x).contains("rest 0.000 ms"));
        assert!(lock_split_line(&[0; 11]).starts_with("sparse lock split (0 submits): wait 0.000 ms"));
    }

    /// The holders part of the line: only the holders with time, as averages a submit.
    #[test]
    fn the_holders_line_names_who_held_the_lock_and_whom_the_waits_were_behind() {
        let mut x = [0u64; 2 * HOLDERS + 2];
        x[Holder::Submit as usize] = 4_000;
        x[Holder::TileMaps as usize] = 800;
        x[HOLDERS + Holder::Settle as usize] = 2_000;
        x[HELD_AFTER_WAIT] = 400;
        assert_eq!(
            holders_line(&x, 4),
            "held by: submits 1.000, put-off mappings 0.200 ms a submit; waits behind: copies back settled 0.500 ms a submit; the submit's other waits for it: after stage 0.100, ranges 0.000 ms a submit"
        );
        assert!(holders_line(&[0; 2 * HOLDERS + 2], 0).starts_with("held by: none ms a submit; waits behind: none"));
    }

    /// A device of the test's own (a software one without a GPU).
    unsafe fn test_device() -> ID3D12Device {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
        let mut dev: Option<ID3D12Device> = None;
        if std::env::var_os("XCL_TEST_WARP").is_some() || D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
            D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
        }
        dev.expect("D3D12 device")
    }

    /// Whether `dev` is a software device (WARP).
    unsafe fn software(dev: &ID3D12Device) -> bool {
        use windows::Win32::Graphics::Dxgi::*;
        let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else { return false };
        let adapter: windows::core::Result<IDXGIAdapter1> = factory.EnumAdapterByLuid(dev.GetAdapterLuid());
        adapter.and_then(|a| a.GetDesc1()).is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0)
    }

    /// The small ones of an upload's runs go up in dispatches: each lands where it belongs (far
    /// into the window too, next to runs that go up with a copy), and what lies between them keeps
    /// what the GPU has there.
    #[test]
    fn small_runs_go_up_in_dispatches_and_leave_what_lies_between() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        const SIZE: usize = 3 << 30;
        unsafe {
            let dev = test_device();
            // (the software device makes a reserved buffer of 3 GiB and is removed by the first copies
            // into it, 0x887a0020 – and a process has one device an adapter: every test after it failed)
            if software(&dev) {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let mut reserved: Option<ID3D12Resource> = None;
            if dev.CreateReservedResource(&buffer_desc(SIZE as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved).is_err() {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let window = reserved.expect("reserved buffer");
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            let mut s = new_state(&dev, vec![(base, window.clone())]).expect("state");
            // (the 1 GiB a dispatch reaches from its base ends between the third and the fourth)
            let tiles = [base, base + TILE, base + (1 << 30) - TILE, base + (1 << 30), base + SIZE - TILE];
            let mut seed = 0x1234_5678_9ABC_DEF1u64;
            let mut random = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            for &t in &tiles {
                assert!(!VirtualAlloc(t as *const c_void, TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
                s.commit_range(t, TILE);
                for i in 0..TILE / 8 {
                    *(t as *mut u64).add(i) = random();
                }
            }
            // all of it goes up, then the CPU writes all of it again and only some runs go up
            let all: Vec<(usize, usize)> = tiles.iter().map(|&t| (t, TILE)).collect();
            s.upload_dirty(&all);
            let on_gpu: Vec<Vec<u8>> = tiles.iter().map(|&t| std::slice::from_raw_parts(t as *const u8, TILE).to_vec()).collect();
            let mut runs: Vec<(usize, usize)> = Vec::new();
            for &t in &tiles {
                for i in 0..TILE / 8 {
                    *(t as *mut u64).add(i) = random();
                }
                let mut o = 0;
                loop {
                    o += CHUNK * (1 + random() as usize % 9);
                    let n = if random() % 23 == 0 { SCATTER_RUN + CHUNK * (random() as usize % 40) } else { CHUNK * (1 + random() as usize % 12) };
                    if o + n > TILE {
                        break;
                    }
                    runs.push((t + o, n));
                    o += n;
                }
            }
            let small = runs.iter().filter(|r| r.1 < SCATTER_RUN).count() as u64;
            let counts = || (SCATTERED[0].load(Ordering::Relaxed), SCATTERED[1].load(Ordering::Relaxed));
            let before = counts();
            s.upload_dirty(&runs);
            let after = counts();
            assert!(small > 300 && runs.len() as u64 > small, "{small} small runs of {}", runs.len());
            assert!(after.0 - before.0 >= 3 && after.1 - before.1 >= small, "{} dispatches of {} runs", after.0 - before.0, after.1 - before.1);

            let mut d = buffer_desc((tiles.len() * TILE) as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            let mut back: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back).expect("readback buffer");
            let back = back.expect("readback buffer");
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            for (k, &t) in tiles.iter().enumerate() {
                list.CopyBufferRegion(&back, (k * TILE) as u64, &window, (t - base) as u64, TILE as u64);
            }
            list.Close().expect("Close");
            s.queue.ExecuteCommandLists(&[Some(list.cast().expect("list"))]);
            s.signal();
            s.wait_for(s.value);
            let mut p = std::ptr::null_mut();
            back.Map(0, None, Some(&mut p)).expect("Map");
            for (k, &t) in tiles.iter().enumerate() {
                let mut want = on_gpu[k].clone();
                for &(a, n) in runs.iter().filter(|r| r.0 >= t && r.0 < t + TILE) {
                    want[a - t..a - t + n].copy_from_slice(std::slice::from_raw_parts(a as *const u8, n));
                }
                let got = std::slice::from_raw_parts((p as *const u8).add(k * TILE), TILE);
                let wrong = (0..TILE / CHUNK).filter(|c| got[c * CHUNK..][..CHUNK] != want[c * CHUNK..][..CHUNK]).collect::<Vec<_>>();
                assert!(wrong.is_empty(), "tile {k}: chunks {:?}… of {} are not what they should be", &wrong[..wrong.len().min(8)], TILE / CHUNK);
            }
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// A copy back after a title submission does not wait for the title's queue: that queue can
    /// be waiting for work the title submits next, from the thread that is in here. The copy
    /// lands once the title's work is done (`drain_readbacks`). Before, the sparse queue
    /// GPU-waited for the title's queue and this thread waited for the copy (5 s, then on).
    #[test]
    fn a_copy_back_does_not_wait_for_the_title_queue() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let base = VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            assert!(!VirtualAlloc(base as *const c_void, TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc(TILE as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved).expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");
            let rbq_queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("readback queue");
            let rbq_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("readback fence");

            let mut s = State {
                dev: dev.clone(),
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rb_queue_fences: FxHashMap::default(),
                rbq: Some((rbq_queue, rbq_fence)),
                rbq_value: 0,
                rb_jobs: VecDeque::new(),
                rb_pool: Vec::new(),
                imm_jobs: VecDeque::new(),
                job_seq: 0,
                windows: vec![(base, reserved.clone())],
                pools: Vec::new(),
                free: Vec::new(),
                cooling: VecDeque::new(),
                retired: Vec::new(),
                retire_tried: std::time::Instant::now(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                kept: FxHashMap::default(),
                pending: Vec::new(),
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: BTreeSet::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: BTreeMap::new(),
                scratch_runs: Vec::new(),
                own_pages: Vec::new(),
                mirrors: FxHashMap::default(),
                mirror_src: FxHashMap::default(),
                src_runs: BTreeMap::new(),
                sources: &SOURCES,
                fill: Vec::new(),
                asks: Vec::new(),
                remaps: Vec::new(),
                later: VecDeque::new(),
                fill_pages: fill_pages(),
                mirrored: 0,
                mirror_cap: mirror_cap(),
                refused: FxHashSet::default(),
                no_memory: None,
                starved: Vec::new(),
                counts: MirrorCounts::default(),
                diag: ViewDiag::default(),
                poison: None,
            };
            s.commit_range(base, TILE);
            assert!(s.mapped.contains_key(&base));
            // (as before the title's submit)
            s.flush_pending();
            std::ptr::write_bytes(base as *mut u8, 0, 256);

            // the title's queue: blocked on a fence nothing has signalled yet, then a copy of
            // 0..64 into the tile (a GPU write the title's next CPU read expects)
            let title: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate");
            let mut up: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &{ let mut d = buffer_desc(256); d.Flags = D3D12_RESOURCE_FLAG_NONE; d }, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up)
                .expect("upload buffer");
            let up = up.expect("upload buffer");
            let mut p: *mut c_void = std::ptr::null_mut();
            up.Map(0, None, Some(&mut p)).expect("Map");
            for i in 0..64u32 {
                *(p as *mut u32).add(i as usize) = 0xC0DE_0000 | i;
            }
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            list.CopyBufferRegion(&reserved, 0, &up, 0, 256);
            list.Close().expect("close");
            title.Wait(&s.fence, s.value).expect("wait for the tile mapping");
            title.Wait(&gate, 1).expect("wait for the gate");
            title.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);

            let t0 = std::time::Instant::now();
            s.after_execute(&title, &[(base, 256)], &[]);
            let took = t0.elapsed();
            assert!(took < std::time::Duration::from_secs(1), "the copy back waited {took:?} for the title's queue");
            s.drain_readbacks(Drain::Finished);
            assert_eq!(*(base as *const u32), 0, "nothing lands before the title's work is done");

            // the title's queue goes on: the copy back lands once it is done
            gate.Signal(1).expect("gate signal");
            // as the title would: wait for its own work (here the fences its queue signals: the
            // first once its lists are done, the second once the copy behind them is)
            let t = std::time::Instant::now();
            loop {
                s.drain_readbacks(Drain::TitleDone);
                if *(base as *const u32) == 0xC0DE_0000 {
                    break;
                }
                assert!(t.elapsed() < std::time::Duration::from_secs(5), "the copy back did not land");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            for i in 0..64u32 {
                assert_eq!(*(base as *const u32).add(i as usize), 0xC0DE_0000 | i, "dword {i} after the copy back");
            }
            assert!(s.rb_jobs.is_empty(), "the copy back was consumed");
            assert_eq!(s.rb_pool.len(), 1, "its buffer is kept for the next one");
            drop(s);
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// `queue()` answers while the state's lock is held: a resource's last `Release` asks for
    /// it (`late_release::drain_known`) from inside a D3D12 call that the lock's holder can be
    /// waiting for (a list's `Reset` against an upload's allocator `Reset`, with `XCL_DRED=1`).
    #[test]
    fn the_queue_is_answered_while_the_state_is_locked() {
        let held = STATE.lock().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let asked = std::thread::spawn(move || {
            let _ = queue();
            let _ = tx.send(());
        });
        let answered = rx.recv_timeout(std::time::Duration::from_secs(5)).is_ok();
        drop(held);
        asked.join().unwrap();
        assert!(answered, "queue() waited for the sparse state's lock");
    }

    /// A submit gets the lock soon although another thread takes it again and again for 0.2 ms each.
    #[test]
    fn a_submit_is_let_in_before_the_commits() {
        use std::sync::atomic::AtomicUsize;
        let (m, waiting, stop) = (Mutex::new(0u64), AtomicUsize::new(0), AtomicBool::new(false));
        std::thread::scope(|s| {
            s.spawn(|| {
                while !stop.load(Ordering::Acquire) {
                    let mut g = lock_in_turn(&m, &waiting);
                    *g += 1;
                    let t = std::time::Instant::now();
                    while t.elapsed().as_micros() < 200 {
                        std::hint::spin_loop();
                    }
                }
            });
            while *m.lock().unwrap() == 0 {
                std::thread::yield_now();
            }
            let mut turns: Vec<u64> = Vec::new();
            for _ in 0..20 {
                let before = *lock_first(&m, &waiting);
                std::thread::sleep(std::time::Duration::from_millis(2));
                let after = *lock_first(&m, &waiting);
                // (the turns the other thread had while this one slept and waited)
                turns.push(after - before);
            }
            stop.store(true, Ordering::Release);
            turns.sort_unstable();
            // 2 ms of sleep are 10 turns; without the rule a waiting thread stood behind hundreds
            assert!(turns[10] < 60, "the other thread's turns before a submit had the lock: {turns:?}");
        });
        assert_eq!(waiting.load(Ordering::Acquire), 0);
    }

    /// A thread that only tries the lock (a fence look settling copies back) does not take it while
    /// a submit waits for it, when it gives way; without that it takes it at once. Its time giving
    /// way does not count against its budget, but it does not give way for ever.
    #[test]
    fn a_thread_that_tries_the_lock_gives_way_to_a_waiting_submit() {
        use std::sync::atomic::AtomicUsize;
        use std::time::Duration;
        let (m, waiting) = (Mutex::new(()), AtomicUsize::new(0));
        // nobody waits: had at once either way
        assert!(try_lock_giving_way(&m, &waiting, Duration::from_millis(5), true).is_some());
        assert!(try_lock_giving_way(&m, &waiting, Duration::from_millis(5), false).is_some());
        // a submit waits for 8 ms: the lock, free all along, is not taken in the first 5 ms
        waiting.store(1, Ordering::Release);
        let t0 = std::time::Instant::now();
        let got = std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(Duration::from_millis(8));
                waiting.store(0, Ordering::Release);
            });
            try_lock_giving_way(&m, &waiting, Duration::from_millis(5), true).is_some()
        });
        let took = t0.elapsed();
        assert!(got, "the lock was free once the submit was done");
        assert!(took >= Duration::from_millis(5), "taken after {took:?}, while the submit waited");
        // a submit that never stops waiting: it is given way to for the budget, then tried
        waiting.store(1, Ordering::Release);
        let t0 = std::time::Instant::now();
        assert!(try_lock_giving_way(&m, &waiting, Duration::from_millis(3), true).is_some());
        assert!(t0.elapsed() >= Duration::from_millis(3));
        assert!(try_lock_giving_way(&m, &waiting, Duration::from_millis(3), false).is_some());
        // held by another thread: given up after the budget (and the time given way)
        let held = m.lock().unwrap();
        let t0 = std::time::Instant::now();
        let got = std::thread::scope(|s| s.spawn(|| try_lock_giving_way(&m, &waiting, Duration::from_millis(3), true).is_some()).join().unwrap());
        assert!(!got);
        assert!(t0.elapsed() >= Duration::from_millis(6), "gave up after {:?}", t0.elapsed());
        drop(held);
    }

    /// A mapping put off does not wait for the title's next submit: a CPU signal sends it, and
    /// without one it goes a few milliseconds after the commit (work the title submitted before
    /// may reach the memory once a value the CPU writes lets it run).
    #[test]
    fn a_mapping_put_off_is_sent_without_a_submit() {
        if super::super::backend() != super::super::Backend::Real {
            return;
        }
        unsafe {
            let _ = super::super::real::device_raw();
            // (a device without sparse windows: commits wait for one, nothing is put off)
            if STATE.lock().unwrap().is_none() {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let Some((at, len)) = alloc(64 * TILE, TILE) else { return };
            let put_off = || STATE.lock().unwrap().as_ref().is_some_and(|s| s.pending.iter().any(|t| (at..at + len).contains(t)));
            let mapped = || STATE.lock().unwrap().as_ref().is_some_and(|s| (at..at + len).step_by(TILE).all(|t| s.mapped.contains_key(&t)));
            send_pending();
            assert!(mapped() && !put_off(), "a CPU signal sends what is put off");
            decommitted(at, len);
            VirtualFree(at as *mut c_void, len, MEM_DECOMMIT);
            // (as a submit does: the tiles unmapped, so the next commit needs mappings again)
            if let Some(s) = STATE.lock().unwrap().as_mut() {
                s.flush_parked();
            }
            assert!(!VirtualAlloc(at as *const c_void, len, MEM_COMMIT, PAGE_READWRITE).is_null());
            committed(at, len);
            let t0 = std::time::Instant::now();
            while put_off() {
                assert!(t0.elapsed().as_secs() < 5, "nothing sent the mappings");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert!(mapped());
            assert!(free(at));
        }
    }

    /// A block freed on one thread while another allocates: its address goes back to the
    /// windows only once it is decommitted (released first, the decommit came after another
    /// thread had the same address again: that thread's writes faulted).
    #[test]
    fn a_freed_block_is_decommitted_before_its_address_is_reused() {
        if crate::gpu::uma::get().is_none_or(|u| !u.has_sparse()) {
            return;
        }
        let threads: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| unsafe {
                    for i in 0..2000u64 {
                        let Some((p, n)) = alloc(TILE, TILE) else { continue };
                        std::ptr::write_volatile(p as *mut u64, i);
                        std::ptr::write_volatile((p + n - 8) as *mut u64, i);
                        assert_eq!(std::ptr::read_volatile(p as *const u64), i);
                        assert!(free(p));
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
    }

    /// `f` on another thread while this one holds the sparse state (as `before_execute` does
    /// around its D3D12 calls, which can wait for that thread): did `f` finish without waiting
    /// for the state? The thread runs before the state is taken, so its start is not timed.
    fn returns_while_the_state_is_held(f: impl FnOnce() + Send + 'static) -> bool {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let (tx, rx) = std::sync::mpsc::channel();
        let t = std::thread::spawn(move || {
            let _ = ready_tx.send(());
            let _ = go_rx.recv();
            f();
            let _ = tx.send(());
        });
        ready_rx.recv().unwrap();
        let held = STATE.lock().unwrap();
        go_tx.send(()).unwrap();
        let done = rx.recv_timeout(std::time::Duration::from_secs(5));
        drop(held);
        let joined = t.join();
        assert!(joined.is_ok(), "the thread panicked: {joined:?}");
        done.is_ok()
    }

    /// A placed buffer released while another thread holds the sparse state: the release does
    /// not wait for the state, and the next `lock_state` lets the buffer go.
    #[test]
    fn a_buffer_released_while_the_state_is_held_does_not_wait_for_it() {
        const RES: usize = 0x7ff0_dead_0000;
        assert!(returns_while_the_state_is_held(|| forget_buffer(RES)), "forget_buffer waited for the sparse state's lock");
        drop(lock_state());
        assert!(!FORGOTTEN.lock().unwrap().contains(&RES));
    }


    /// Freeing the last tile in use of a pool and taking a tile again takes the retired pool
    /// back, not a new 64 MB heap each time (#913: 288 pools, 18 GB asked of a 12 GB card).
    #[test]
    fn a_pool_retired_a_moment_ago_is_taken_back() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");
            let mut s = State {
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rb_queue_fences: FxHashMap::default(),
                rbq: None,
                rbq_value: 0,
                rb_jobs: VecDeque::new(),
                rb_pool: Vec::new(),
                imm_jobs: VecDeque::new(),
                job_seq: 0,
                windows: Vec::new(),
                pools: Vec::new(),
                free: Vec::new(),
                cooling: VecDeque::new(),
                retired: Vec::new(),
                retire_tried: std::time::Instant::now(),
                parked: FxHashMap::default(),
                kept: FxHashMap::default(),
                pending: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: BTreeSet::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: BTreeMap::new(),
                scratch_runs: Vec::new(),
                own_pages: Vec::new(),
                mirrors: FxHashMap::default(),
                mirror_src: FxHashMap::default(),
                src_runs: BTreeMap::new(),
                sources: &SOURCES,
                fill: Vec::new(),
                asks: Vec::new(),
                remaps: Vec::new(),
                later: VecDeque::new(),
                fill_pages: fill_pages(),
                mirrored: 0,
                mirror_cap: mirror_cap(),
                refused: FxHashSet::default(),
                no_memory: None,
                starved: Vec::new(),
                counts: MirrorCounts::default(),
                diag: ViewDiag::default(),
                poison: None,
            };

            let (first, reused) = s.take_tile().expect("a tile from a new pool");
            assert!(!reused, "a new pool's tile needs no clearing");
            s.free.push(first);
            s.retire_free_pools();
            assert_eq!(s.retired.len(), 1, "an all-free pool is retired");
            for round in 0..8 {
                let (tile, reused) = s.take_tile().expect("a tile");
                assert!(reused, "round {round}: a tile of a retired pool is cleared before use");
                s.free.push(tile);
                s.retire_free_pools();
                assert_eq!(s.retired.len(), 1, "round {round}: the retired pool was taken back, not a new one made");
                assert_eq!(s.pools.iter().filter(|p| p.is_some()).count(), 0, "round {round}");
            }
        }
    }

    #[test]
    fn aligned_tile_addresses_spread_across_hash_buckets() {
        use std::hash::BuildHasher;

        let hasher = BuildHasherDefault::<FxHasher>::default();
        let mut used = [false; 256];
        for tile in 0..used.len() {
            let address = (1usize << 32) + tile * TILE;
            let bucket = hasher.hash_one(address) as usize & (used.len() - 1);
            used[bucket] = true;
        }
        let occupied = used.iter().filter(|&&bucket| bucket).count();
        assert!(occupied >= used.len() / 2, "256 aligned addresses used {occupied} buckets");
    }

    /// Only the 64-byte chunks the CPU changed are sent again – the rest of the page may hold
    /// what the GPU wrote.
    #[test]
    fn only_changed_chunks_are_sent() {
        let mut page = vec![0u8; PAGE];
        let mut sent = ZERO_HASHES;
        assert!(changed_chunks(&page, &mut sent).is_empty(), "a zero page is what the GPU has");
        page[10] = 1;
        page[200] = 2;
        page[255] = 3;
        page[4000] = 4;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(0, 64), (192, 64), (3968, 64)]);
        assert!(changed_chunks(&page, &mut sent).is_empty(), "nothing changed since");
        page[256] = 5; // next to the chunk at 192: one run
        page[200] = 9;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(192, 128)]);
    }

    /// `scan_page` skips allocating a hash box on all-zero pages, allocates on the first write,
    /// and coalesces adjacent dirty chunks across consecutive pages.
    #[test]
    fn scan_page_lazy_alloc_and_coalesce() {
        let mut page = vec![0u8; PAGE];
        let mut slot: Option<Box<[u64; CHUNKS]>> = None;
        let mut out = Vec::new();
        scan_page(&page, 0x1_0000, &mut slot, &mut out);
        assert!(slot.is_none(), "all-zero page must not allocate hash box");
        assert!(out.is_empty());

        page[PAGE - 64] = 0xAA;
        scan_page(&page, 0x1_0000, &mut slot, &mut out);
        assert!(slot.is_some());
        assert_eq!(out, vec![(0x1_0FC0, 64)]);

        // Adjacent chunk at the start of the next page coalesces into a single run.
        let mut next = vec![0u8; PAGE];
        next[0] = 0xBB;
        let mut next_slot: Option<Box<[u64; CHUNKS]>> = None;
        scan_page(&next, 0x1_1000, &mut next_slot, &mut out);
        assert_eq!(out, vec![(0x1_0FC0, 128)]);
    }

    /// Ring-buffer planner keeps every chunk contiguous, 64-byte aligned, <= cap/2, and splits
    /// cleanly across the end of the ring buffer without wrapping mid-copy.
    #[test]
    fn ring_planner_handles_wrap_and_large_chunks() {
        let cap = 64 << 20;
        // Normal aligned chunk
        let (a, n) = plan_ring_chunk(0, 4096, cap);
        assert_eq!((a, n), (0, 4096));
        // Unaligned cursor aligns up to 64
        let (a, n) = plan_ring_chunk(17, 256, cap);
        assert_eq!((a, n), (64, 256));
        // Oversized chunk capped at cap / 2
        let (a, n) = plan_ring_chunk(0, cap * 2, cap);
        assert_eq!((a, n), (0, cap / 2));
        // Near end of ring with >= 4 KiB left: shrinks to fill tail
        let near = (cap - 8192) as u64;
        let (a, n) = plan_ring_chunk(near, 16384, cap);
        assert_eq!((a, n), (near, 8192));
        assert_eq!((a as usize % cap) + n, cap);
        // Tiny tail (< 4 KiB): wraps cleanly to 0 of next cycle
        let tiny = (cap - 512) as u64;
        let (a, n) = plan_ring_chunk(tiny, 4096, cap);
        assert_eq!((a, n), (cap as u64, 4096));
        assert_eq!(a as usize % cap, 0);
    }

    /// `query_tile_holes` marks reserved, free, and `PAGE_NOACCESS` pages as holes while clearing
    /// bits only for committed, accessible 4 KiB pages.
    #[test]
    fn query_tile_holes_detects_reserved_noaccess_and_committed_pages() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            // SAFETY: Reserving 2 tiles (128 KiB) with VirtualAlloc returns a 64 KiB-aligned VA.
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);

            // Initially every page in both reserved tiles is a hole.
            assert_eq!(query_tile_holes(base), 0xFFFF);
            assert_eq!(query_tile_holes(base + TILE), 0xFFFF);

            // Commit pages 4..8 as PAGE_READWRITE and page 9 as PAGE_NOACCESS.
            let p4 = VirtualAlloc((base + 4 * PAGE) as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert!(!p4.is_null());
            let p9 = VirtualAlloc((base + 9 * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_NOACCESS);
            assert!(!p9.is_null());

            // Only bits 4..8 are cleared; reserved [0..4), PAGE_NOACCESS [9], and reserved [10..16) remain holes.
            assert_eq!(query_tile_holes(base), 0xFFFF & !(0b1111u16 << 4));
            assert_eq!(query_tile_holes(base + TILE), 0xFFFF);

            // Commit pages 0..4 and confirm bits 0..4 clear too.
            let p0 = VirtualAlloc(base as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert!(!p0.is_null());
            assert_eq!(query_tile_holes(base), 0xFF00);

            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
            // Released (MEM_FREE) memory reports all holes.
            assert_eq!(query_tile_holes(base), 0xFFFF);
        }
    }

    /// Mapping tiles for a placed buffer over partially-committed/reserved memory must preserve
    /// uncommitted hole bits so `readback_ranges` / `sync_sent_range` do not access-violate,
    /// and later commits clear those hole bits so subsequent writes scan cleanly.
    #[test]
    fn ensure_mapped_with_query_holes_skips_uncommitted_pages_on_readback() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");

            // Reserve 2 tiles; commit only [base + 0x4000, base + 0x8000) to simulate a
            // non-64KiB-aligned placed buffer with uncommitted prefix [base, base + 0x4000) and
            // uncommitted second tile [base + TILE, base + 2*TILE).
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            assert!(!VirtualAlloc((base + 4 * PAGE) as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());

            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc((2 * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
                .expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");

            let mut s = State {
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rb_queue_fences: FxHashMap::default(),
                rbq: None,
                rbq_value: 0,
                rb_jobs: VecDeque::new(),
                rb_pool: Vec::new(),
                imm_jobs: VecDeque::new(),
                job_seq: 0,
                windows: vec![(base, reserved)],
                pools: Vec::new(),
                free: Vec::new(),
                cooling: VecDeque::new(),
                retired: Vec::new(),
                retire_tried: std::time::Instant::now(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                kept: FxHashMap::default(),
                pending: Vec::new(),
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: BTreeSet::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: BTreeMap::new(),
                scratch_runs: Vec::new(),
                own_pages: Vec::new(),
                mirrors: FxHashMap::default(),
                mirror_src: FxHashMap::default(),
                src_runs: BTreeMap::new(),
                sources: &SOURCES,
                fill: Vec::new(),
                asks: Vec::new(),
                remaps: Vec::new(),
                later: VecDeque::new(),
                fill_pages: fill_pages(),
                mirrored: 0,
                mirror_cap: mirror_cap(),
                refused: FxHashSet::default(),
                no_memory: None,
                starved: Vec::new(),
                counts: MirrorCounts::default(),
                diag: ViewDiag::default(),
                poison: None,
            };

            // Mirrors `place_buffer`: map both tiles probing real OS page commitment.
            s.ensure_mapped(base, 2 * TILE, |t| query_tile_holes(t));
            assert_eq!(s.mapped[&base].holes, 0xFF0F, "uncommitted prefix and suffix pages must stay holes");
            assert_eq!(s.mapped[&(base + TILE)].holes, 0xFFFF, "fully reserved tile must have all 16 holes set");

            // Reading back the full 2-tile span must not fault on uncommitted pages.
            s.readback_ranges(&[(base, 2 * TILE)]);

            // Now incrementally commit [base, base + 4*PAGE) and clear its hole bits (as `committed` does).
            assert!(!VirtualAlloc(base as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            let entry = s.mapped.get_mut(&base).unwrap();
            for i in 0..4 {
                entry.holes &= !(1u16 << i);
            }
            assert_eq!(entry.holes, 0xFF00);

            // Writing to the newly committed page is detected by `scan_page`.
            *(base as *mut u8) = 0x5A;
            let mut dirty = Vec::new();
            let page0 = std::slice::from_raw_parts(base as *const u8, PAGE);
            scan_page(page0, base, &mut entry.sent[0], &mut dirty);
            assert_eq!(dirty, vec![(base, CHUNK)]);

            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// A tile decommitted and committed again before the next submit gets its pool tile back
    /// with no new tile mapping (cleared instead). One that is not is unmapped and freed by
    /// `flush_parked` (before a submit), or when the free list runs out; a placed buffer's
    /// release does not free it while it is parked.
    /// Until every fence the quarantined tiles wait for has reached its mark (5 s at most each).
    unsafe fn wait_quarantined(s: &State) {
        for batch in &s.quarantined {
            for (f, v) in batch.marks.iter().flatten() {
                let t0 = std::time::Instant::now();
                while f.GetCompletedValue() < *v && t0.elapsed().as_secs() < 5 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        }
    }

    #[test]
    fn a_tile_decommitted_and_committed_again_keeps_its_video_memory() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc((2 * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
                .expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");

            let mut s = State {
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rb_queue_fences: FxHashMap::default(),
                rbq: None,
                rbq_value: 0,
                rb_jobs: VecDeque::new(),
                rb_pool: Vec::new(),
                imm_jobs: VecDeque::new(),
                job_seq: 0,
                windows: vec![(base, reserved)],
                pools: Vec::new(),
                free: Vec::new(),
                cooling: VecDeque::new(),
                retired: Vec::new(),
                retire_tried: std::time::Instant::now(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                kept: FxHashMap::default(),
                pending: Vec::new(),
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: BTreeSet::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: BTreeMap::new(),
                scratch_runs: Vec::new(),
                own_pages: Vec::new(),
                mirrors: FxHashMap::default(),
                mirror_src: FxHashMap::default(),
                src_runs: BTreeMap::new(),
                sources: &SOURCES,
                fill: Vec::new(),
                asks: Vec::new(),
                remaps: Vec::new(),
                later: VecDeque::new(),
                fill_pages: fill_pages(),
                mirrored: 0,
                mirror_cap: mirror_cap(),
                refused: FxHashSet::default(),
                no_memory: None,
                starved: Vec::new(),
                counts: MirrorCounts::default(),
                diag: ViewDiag::default(),
                poison: None,
            };

            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            assert_eq!(s.pending, vec![base], "a new tile's mapping waits for the next flush");
            assert_eq!(s.value, 0, "nothing sent yet");
            s.flush_pending();
            assert!(s.pending.is_empty());
            let v = s.value;
            assert!(v > 0, "a new tile is mapped");

            // decommitted: parked, still mapped, nothing sent to the GPU
            s.decommit_range(base, TILE);
            assert!(!s.mapped.contains_key(&base));
            assert_eq!(s.parked.get(&base), Some(&tile));
            assert!(!s.free.contains(&tile));
            assert_eq!(s.value, v, "no tile mapping changed");
            s.release_if_unused(base, tile);
            assert!(!s.free.contains(&tile) && s.cooling.is_empty(), "a parked tile is not free");

            assert!(s.runs.is_empty() && runs_of(&s).is_empty());
            // committed again: the same tile, cleared, no new mapping
            s.to_zero.clear();
            let gen = mapped_gen();
            s.commit_range(base, TILE);
            assert_eq!(s.runs, BTreeMap::from([(base, base + TILE)]));
            assert_eq!(s.mapped[&base].pool, tile);
            assert_eq!(s.mapped[&base].holes, 0);
            assert!(s.parked.is_empty());
            assert_eq!(s.to_zero, BTreeSet::from([(base, TILE)]));
            assert_eq!(s.value, v, "no tile mapping changed");
            assert!(mapped_gen() > gen, "readers of `mapped` see it changed");

            // decommitted and flushed (as before a submit): unmapped and quarantined until the
            // unmap fence completes, then cooling, not free yet
            s.decommit_range(base, TILE);
            assert!(s.flush_parked());
            assert!(s.parked.is_empty());
            assert!(s.quarantined.iter().any(|b| b.tiles.contains(&tile)));
            assert!(!s.cooling.iter().any(|c| c.0 == tile) && !s.free.contains(&tile));
            assert!(s.value > v, "the unmapping is sent");
            assert!(!s.flush_parked(), "nothing left to flush");
            s.wait_for(s.value);
            // (and the title queues other tests made, whose fences the quarantine waits for too)
            wait_quarantined(&s);
            assert!(s.drain_quarantined());
            assert!(s.cooling.iter().any(|c| c.0 == tile) && !s.free.contains(&tile));

            // no free tile left: neither the parked tile (mapped at its address until the next
            // flush) nor the one cooling is taken for the next commit, but a new pool's. Before,
            // the parked one was unmapped and taken at once.
            s.commit_range(base, TILE);
            let parked = s.mapped[&base].pool;
            assert_ne!(parked, tile, "the cooling tile is not taken");
            s.decommit_range(base, TILE);
            s.free.clear();
            s.commit_range(base + TILE, TILE);
            assert_eq!(s.parked.get(&base), Some(&parked), "still parked");
            let tile = s.mapped[&(base + TILE)].pool;
            assert_ne!(tile.0, parked.0, "a tile of a new pool");
            assert!(!s.mapped.contains_key(&base));
            assert!(s.flush_parked());
            assert!(!s.cooling.iter().any(|c| c.0 == parked), "unmapped at the flush, quarantined");
            s.wait_for(s.value);
            // (and the title queues other tests made, whose fences the quarantine waits for too)
            wait_quarantined(&s);
            assert!(s.drain_quarantined());
            assert_eq!(s.cooling.back().map(|c| c.0), Some(parked), "unmapped at the flush, then cooling");

            // a placed buffer uses the tile: the window keeps it when it is decommitted and
            // flushed, memory committed there again is the buffer's, and it is free only once
            // the buffer is gone and it was decommitted
            *s.pinned.entry(tile).or_default() += 1;
            let at = base + TILE;
            assert!(s.pending.contains(&at), "its mapping is not sent yet");
            s.decommit_range(at, TILE);
            assert!(!s.flush_parked(), "nothing to unmap");
            assert_eq!(s.kept.get(&at), Some(&tile));
            assert!(!s.free.contains(&tile));
            // kept before its mapping was sent: that stays pending, and the commit that takes the
            // tile back sends it
            let v = s.value;
            s.flush_pending();
            assert_eq!((s.value, &s.pending), (v, &vec![at]), "no mapping of a kept tile");
            s.commit_range(at, TILE);
            s.flush_pending();
            assert!(s.value > v && s.pending.is_empty(), "mapped with the commit");
            s.decommit_range(at, TILE);
            assert!(!s.flush_parked(), "nothing to unmap");
            assert_eq!(s.kept.get(&at), Some(&tile));
            let v = s.value;
            s.to_zero.clear();
            s.commit_range(at, TILE);
            assert_eq!(s.mapped[&at].pool, tile, "the buffer's tile");
            assert_eq!(s.to_zero, BTreeSet::from([(at, TILE)]));
            assert_eq!(s.value, v, "no tile mapping changed");
            s.decommit_range(at, TILE);
            assert!(!s.flush_parked());
            s.pinned.remove(&tile);
            s.release_if_unused(at, tile);
            assert!(s.kept.is_empty() && !s.cooling.iter().any(|c| c.0 == tile));
            assert!(s.flush_parked());
            s.wait_for(s.value);
            // (and the title queues other tests made, whose fences the quarantine waits for too)
            wait_quarantined(&s);
            assert!(s.drain_quarantined());
            assert!(s.cooling.iter().any(|c| c.0 == tile) && !s.free.contains(&tile));

            assert_eq!(s.runs, runs_of(&s));
            // committed and decommitted again before a flush: no mapping is sent for it
            s.flush_pending();
            s.flush_parked();
            let v = s.value;
            s.commit_range(base, TILE);
            assert_eq!(s.pending, vec![base]);
            s.decommit_range(base, TILE);
            s.flush_pending(); // as for an upload or readback before the title's next submit
            assert_eq!(s.value, v, "a tile gone before the flush is not mapped");
            assert_eq!(s.pending, vec![base], "parked, its mapping never sent: still pending");
            // taken back from `parked`: the flush now sends its mapping
            s.commit_range(base, TILE);
            s.flush_pending();
            assert!(s.pending.is_empty());
            assert!(s.value > v, "a tile parked before its mapping was sent is mapped");

            // neighbours are one run, a tile taken out of it leaves the other
            s.commit_range(base, 2 * TILE);
            assert_eq!(s.runs, BTreeMap::from([(base, base + 2 * TILE)]));
            s.decommit_range(base, TILE);
            assert_eq!(s.runs, BTreeMap::from([(base + TILE, base + 2 * TILE)]));
            s.decommit_range(base + TILE, PAGE);
            assert_eq!(s.runs, runs_of(&s), "a tile with pages left stays");
            s.flush_pending();

            s.wait_for(s.value);
            // (and the title queues other tests made, whose fences the quarantine waits for too)
            wait_quarantined(&s);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    struct SignalOnDrop(ID3D12Fence, u64);

    impl Drop for SignalOnDrop {
        fn drop(&mut self) {
            unsafe {
                let _ = self.0.Signal(self.1);
            }
        }
    }

    unsafe fn test_state_with_reserved(tiles: usize) -> (usize, State) {
        use windows_sys::Win32::System::Memory::MEM_RESERVE;

        let dev = test_device();
        let base = VirtualAlloc(std::ptr::null(), tiles * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
        assert!(base != 0 && base % TILE == 0);
        let mut reserved: Option<ID3D12Resource> = None;
        dev.CreateReservedResource(&buffer_desc((tiles * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
            .expect("CreateReservedResource");
        let reserved = reserved.expect("reserved buffer");
        let s = new_state(&dev, vec![(base, reserved)]).expect("state");
        (base, s)
    }

    /// `XCL_SPARSE_LAZY` on (or off) for this thread while it lives, and `take_tile` finding no
    /// video memory while `no_tiles` is set.
    struct Lazy;
    impl Lazy {
        fn on(on: bool) -> Lazy {
            LAZY_COMMITS_FOR_TEST.with(|c| c.set(Some(on)));
            Lazy
        }
        fn no_tiles(&self, none: bool) {
            NO_TILES.with(|c| c.set(none));
        }
    }
    impl Drop for Lazy {
        fn drop(&mut self) {
            LAZY_COMMITS_FOR_TEST.with(|c| c.set(None));
            NO_TILES.with(|c| c.set(false));
        }
    }

    /// A state of `tiles` tiles for the lazy tests, its window's CPU pages committed and the title's
    /// queue. None: no GPU whose unmapped tiles read zeros (the lazy path is left out there).
    unsafe fn lazy_state(tiles: usize) -> Option<(usize, State, ID3D12CommandQueue)> {
        let dev = test_device();
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).ok()?;
        if software(&dev) || o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
            return None;
        }
        let (base, s) = test_state_with_reserved(tiles);
        assert!(!VirtualAlloc(base as *const c_void, tiles * TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
        let queue: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).ok()?;
        Some((base, s, queue))
    }

    /// `len` bytes of the window at `va` as the GPU has them, after what was sent to it.
    unsafe fn window_bytes(s: &mut State, va: usize, len: usize) -> Vec<u8> {
        let mut d = buffer_desc(len as u64);
        d.Flags = D3D12_RESOURCE_FLAG_NONE;
        let mut back: Option<ID3D12Resource> = None;
        s.dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back).expect("readback buffer");
        let back = back.expect("readback buffer");
        let (base, window) = (s.windows[0].0, s.windows[0].1.clone());
        let alloc: ID3D12CommandAllocator = s.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
        let list: ID3D12GraphicsCommandList = s.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
        list.CopyBufferRegion(&back, 0, &window, (va - base) as u64, len as u64);
        list.Close().expect("Close");
        s.flush_pending();
        s.queue.ExecuteCommandLists(&[Some(list.cast().expect("list"))]);
        s.signal();
        s.wait_for(s.value);
        let mut p = std::ptr::null_mut();
        back.Map(0, None, Some(&mut p)).expect("Map");
        let out = std::slice::from_raw_parts(p as *const u8, len).to_vec();
        back.Unmap(0, None);
        out
    }

    /// Tile `t`'s CPU pages written with bytes made from `seed`; its pages, for a submit's dirty list.
    unsafe fn write_tile(t: usize, seed: u64) -> Vec<usize> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        for i in 0..TILE / 8 {
            *(t as *mut u64).add(i) = next(&mut x);
        }
        (t..t + TILE).step_by(PAGE).collect()
    }

    /// With `XCL_SPARSE_LAZY` a committed tile gets no video memory until the CPU's first bytes
    /// go up to it; they reach the GPU, and a later submit with nothing written leaves them there
    /// (a tile cleared one submit after its bytes went up would read zeros). A tile never written
    /// keeps none and reads zeros.
    #[test]
    fn a_lazy_tile_gets_memory_at_its_first_bytes_and_keeps_them() {
        let lazy = Lazy::on(true);
        unsafe {
            let Some((base, mut s, queue)) = lazy_state(2) else { return crate::gpu::recomp::runner::skip_test("no GPU with tiled resources tier 2") };
            s.commit_range(base, 2 * TILE);
            assert_eq!(s.mapped[&base].pool, LAZY, "committed: listed with no video memory");
            assert_eq!(s.mapped[&(base + TILE)].pool, LAZY);
            // a used pool tile on hand, so that its old bytes would show if it were not cleared
            s.no_memory = None;
            let dirty = write_tile(base, 7);
            s.before_execute_on(&queue, &dirty, PAGE);
            assert_ne!(s.mapped[&base].pool, LAZY, "its first bytes gave it a pool tile");
            assert_eq!(s.mapped[&(base + TILE)].pool, LAZY, "the tile never written has none");
            let cpu = std::slice::from_raw_parts(base as *const u8, TILE).to_vec();
            assert!(window_bytes(&mut s, base, TILE) == cpu, "the bytes reached the GPU");
            s.before_execute_on(&queue, &[], PAGE);
            s.before_execute_on(&queue, &[], PAGE);
            assert!(window_bytes(&mut s, base, TILE) == cpu, "and are still there two submits later");
            assert!(window_bytes(&mut s, base + TILE, TILE).iter().all(|&b| b == 0), "a tile with no memory reads zeros");
        }
        drop(lazy);
    }

    /// Bytes that found no video memory at an upload are owed and go up whole once there is some.
    #[test]
    fn bytes_that_found_no_memory_go_up_once_there_is_some() {
        let lazy = Lazy::on(true);
        unsafe {
            let Some((base, mut s, queue)) = lazy_state(1) else { return crate::gpu::recomp::runner::skip_test("no GPU with tiled resources tier 2") };
            s.commit_range(base, TILE);
            let dirty = write_tile(base, 11);
            lazy.no_tiles(true);
            s.before_execute_on(&queue, &dirty, PAGE);
            assert_eq!(s.mapped[&base].pool, LAZY, "no video memory to be had: still none");
            assert!(s.starved.contains(&base), "its bytes are owed");
            lazy.no_tiles(false);
            s.no_memory = None;
            s.before_execute_on(&queue, &[], PAGE);
            assert_ne!(s.mapped[&base].pool, LAZY, "a submit with nothing written gives it its memory");
            let cpu = std::slice::from_raw_parts(base as *const u8, TILE).to_vec();
            assert!(window_bytes(&mut s, base, TILE) == cpu, "and all its bytes");
        }
        drop(lazy);
    }

    /// A memory write the title records on a list gives the tile its memory before the list runs
    /// (`give_memory`); a placed buffer over a tile that finds none leaves it out of its mapping
    /// instead of mapping pool tile u32::MAX.
    #[test]
    fn a_recorded_write_gives_memory_and_a_buffer_never_maps_the_sentinel() {
        let lazy = Lazy::on(true);
        unsafe {
            let Some((base, mut s, _queue)) = lazy_state(2) else { return crate::gpu::recomp::runner::skip_test("no GPU with tiled resources tier 2") };
            s.commit_range(base, 2 * TILE);
            s.give_memory(&[(base + 8, 8)]);
            assert_ne!(s.mapped[&base].pool, LAZY, "the recorded write's tile got memory");
            assert_eq!(s.mapped[&(base + TILE)].pool, LAZY, "the other did not");
            lazy.no_tiles(true);
            let tiles = s.buffer_tiles(base, 2 * TILE);
            assert!(tiles.iter().all(|&(_, tile)| tile != LAZY), "no tile of a buffer is the sentinel: {tiles:?}");
            assert!(tiles.iter().any(|&(t, _)| t == base), "the tile with memory is the buffer's");
        }
        drop(lazy);
    }

    /// Decommitting a tile with no video memory frees and parks nothing; committed again it is
    /// listed again. With the switch off a commit maps at once, as before.
    #[test]
    fn a_lazy_tile_decommits_without_a_pool_tile_and_off_maps_at_commit() {
        unsafe {
            let lazy = Lazy::on(true);
            let Some((base, mut s, _queue)) = lazy_state(1) else { return crate::gpu::recomp::runner::skip_test("no GPU with tiled resources tier 2") };
            s.commit_range(base, TILE);
            let (free, parked) = (s.free.len(), s.parked.len());
            s.decommit_range(base, TILE);
            assert!(!s.mapped.contains_key(&base), "gone from the window");
            assert_eq!((s.free.len(), s.parked.len()), (free, parked), "no pool tile freed or parked");
            s.commit_range(base, TILE);
            assert_eq!(s.mapped[&base].pool, LAZY, "listed again");
            drop(lazy);
            let off = Lazy::on(false);
            let Some((base2, mut s2, _q2)) = lazy_state(1) else { return };
            s2.commit_range(base2, TILE);
            assert_ne!(s2.mapped[&base2].pool, LAZY, "off: mapped at the commit");
            drop(off);
        }
    }

    /// The tiles `flush_parked` quarantined, once the sparse queue and the title queues are past
    /// them (`drain_quarantined`), waited for up to 5 s.
    unsafe fn settle_quarantined(s: &mut State) {
        s.wait_for(s.value);
        let start = std::time::Instant::now();
        while !s.quarantined.is_empty() && start.elapsed() < std::time::Duration::from_secs(5) {
            if !s.drain_quarantined() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    /// `flush_parked` must keep unmapped pool tiles out of `free` (and `take_tile` must not reuse
    /// them) while the sparse queue's unmap fence is still in flight on the GPU.
    #[test]
    fn flush_parked_does_not_free_or_reuse_tiles_while_unmap_is_in_flight() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            s.wait_for(s.value);

            let gate: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate fence");
            let _unblock = SignalOnDrop(gate.clone(), 1);
            s.queue.Wait(&gate, 1).expect("gate sparse queue");

            s.decommit_range(base, TILE);
            s.free.clear();
            assert!(s.flush_parked());
            assert!(
                !s.free.contains(&tile) && !s.cooling.iter().any(|c| c.0 == tile),
                "unmapped tile must remain quarantined until unmap fence completes"
            );

            // While the unmap is still blocked on `gate`, committing another tile must not steal `tile`.
            s.commit_range(base + TILE, TILE);
            assert_ne!(s.mapped[&(base + TILE)].pool, tile, "in-flight unmapped tile must not be reused before GPU completion");

            gate.Signal(1).expect("release sparse queue");
            s.wait_for(s.value);
            assert!(s.drain_quarantined(), "completed unmap batch should drain after fence completion");
            assert!(
                s.cooling.iter().any(|c| c.0 == tile) && !s.free.contains(&tile),
                "unmapped tile must go on to cool (`cooling`) once unmap fence completes"
            );
            drop(_unblock);
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// `flush_parked` must keep unmapped pool tiles quarantined until every active title queue has
    /// finished its prior work, even after `s.queue` itself finishes unmapping.
    #[test]
    fn flush_parked_quarantines_tiles_until_title_queues_finish() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            let title_q: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
            assert!(s.note_queue(&title_q));

            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            s.wait_for(s.value);

            let title_gate: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("title gate fence");
            let _unblock = SignalOnDrop(title_gate.clone(), 1);
            title_q.Wait(&title_gate, 1).expect("gate title queue");

            s.decommit_range(base, TILE);
            s.free.clear();
            assert!(s.flush_parked());

            // Even after `s.queue` finishes `unmap_tiles`, `title_q` is still gated behind `title_gate`.
            s.wait_for(s.value);
            assert!(!s.drain_quarantined(), "tile must remain quarantined while title queue has work in flight");
            assert!(!s.free.contains(&tile) && !s.cooling.iter().any(|c| c.0 == tile));

            s.commit_range(base + TILE, TILE);
            assert_ne!(s.mapped[&(base + TILE)].pool, tile, "must not reuse tile still in use by title queue");

            title_gate.Signal(1).expect("release title queue");
            let start = std::time::Instant::now();
            while !s.drain_quarantined() && start.elapsed() < std::time::Duration::from_secs(5) {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            assert!(
                s.cooling.iter().any(|c| c.0 == tile) && !s.free.contains(&tile),
                "tile must go on to cool (`cooling`) after title queue completes"
            );
            drop(_unblock);
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// Sub-tile decommits (two 32 KiB halves, sixteen 4 KiB pages, or decommitting the last
    /// remaining committed page) must unmap the tile and return unpinned pool tiles to `free`
    /// once `holes` reaches `0xFFFF`, pruning stale `to_zero` entries for unmapped tiles.
    #[test]
    fn decommit_sub_tile_reclaims_pool_tile_when_all_pages_become_holes() {
        let mut mapped: FxHashMap<usize, TileEntry> = FxHashMap::default();
        let mut pinned: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let mut free: Vec<(u32, u32)> = Vec::new();
        let mut to_zero: BTreeSet<(usize, usize)> = BTreeSet::new();

        // 1. Two 32 KiB decommits across a full 64 KiB tile.
        let t0 = 0x10_0000usize;
        mapped.insert(t0, TileEntry::with_holes((0, 10), 0));
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t0, TILE / 2).is_empty());
        assert_eq!(mapped[&t0].holes, 0x00FF);
        assert_eq!(to_zero.len(), 8);
        assert!(free.is_empty());

        let unmapped = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t0 + TILE / 2, TILE / 2);
        assert_eq!(unmapped, vec![t0]);
        assert!(!mapped.contains_key(&t0));
        assert_eq!(free, vec![(0, 10)]);
        assert!(to_zero.is_empty(), "stale to_zero entries for the unmapped tile must be pruned");

        // 2. Sixteen individual 4 KiB decommits in reverse order.
        let t1 = 0x20_0000usize;
        free.clear();
        mapped.insert(t1, TileEntry::with_holes((0, 11), 0));
        for i in (1..PAGES_PER_TILE).rev() {
            let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t1 + i * PAGE, PAGE);
            assert!(u.is_empty(), "tile still has committed pages after step {i}");
        }
        assert!(free.is_empty());
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t1, PAGE);
        assert_eq!(u, vec![t1]);
        assert!(!mapped.contains_key(&t1));
        assert_eq!(free, vec![(0, 11)]);
        assert!(to_zero.is_empty());

        // 3. Pinned tile decommitted in two halves: unmapped from `mapped`, NOT returned to `free`.
        let t2 = 0x30_0000usize;
        free.clear();
        mapped.insert(t2, TileEntry::with_holes((0, 12), 0));
        pinned.insert((0, 12), 1);
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t2, TILE / 2).is_empty());
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t2 + TILE / 2, TILE / 2);
        assert_eq!(u, vec![t2]);
        assert!(!mapped.contains_key(&t2));
        assert!(free.is_empty(), "pinned pool tile must not return to free list");
        assert!(to_zero.is_empty());

        // 4. Single-committed-page tile decommit reclaims tile without dropping another tile's to_zero.
        let t3 = 0x40_0000usize;
        let t4 = 0x50_0000usize;
        mapped.insert(t3, TileEntry::with_holes((0, 13), 0xFFFF & !(1u16 << 2)));
        mapped.insert(t4, TileEntry::with_holes((0, 14), 0));
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t4, PAGE).is_empty());
        assert_eq!(to_zero, BTreeSet::from([(t4, PAGE)]));
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t3 + 2 * PAGE, PAGE);
        assert_eq!(u, vec![t3]);
        assert_eq!(free, vec![(0, 13)]);
        assert_eq!(to_zero, BTreeSet::from([(t4, PAGE)]), "unrelated mapped tile's to_zero must remain");

        // 5. The pieces as runs: neighbours together, pieces inside a whole tile with it.
        let t5 = 0x60_0000usize;
        to_zero.extend([(t5, TILE), (t5 + 3 * PAGE, PAGE), (t5 + TILE, TILE), (t5 + 2 * TILE + PAGE, PAGE)]);
        assert_eq!(zero_runs(to_zero), vec![(t4, PAGE), (t5, 2 * TILE), (t5 + 2 * TILE + PAGE, PAGE)]);
    }

    /// The pages in `early`, checking that its ranges are page-aligned, not empty, disjoint and
    /// merged (no two touch).
    fn early_pages(early: &BTreeMap<usize, usize>) -> std::collections::BTreeSet<usize> {
        let mut last_end = None;
        let mut pages = std::collections::BTreeSet::new();
        for (&a, &b) in early {
            assert!(a % PAGE == 0 && b % PAGE == 0 && a < b, "range {a:#x}..{b:#x}");
            assert!(last_end.is_none_or(|e| e < a), "range {a:#x}..{b:#x} touches the one before");
            last_end = Some(b);
            pages.extend((a..b).step_by(PAGE));
        }
        pages
    }

    /// xorshift64*: the same sequence on every run
    fn next(x: &mut u64) -> u64 {
        *x ^= *x >> 12;
        *x ^= *x << 25;
        *x ^= *x >> 27;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Before the device exists, commits and decommits leave the pages a plain page set would
    /// hold, with the live path's rule (a range counts every page it touches): random ranges,
    /// unaligned too, over 300 pages.
    #[test]
    fn early_commits_are_the_pages_a_page_set_holds() {
        let base = 0x4_2000_0000usize;
        for seed in 1..=8u64 {
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let (mut early, mut model) = (BTreeMap::new(), std::collections::BTreeSet::new());
            for _ in 0..4000 {
                let a = base + (next(&mut x) % (300 * PAGE as u64)) as usize;
                let n = match next(&mut x) % 4 {
                    0 => PAGE,
                    1 => 1 + (next(&mut x) % 64) as usize,
                    _ => 1 + (next(&mut x) % (24 * PAGE as u64)) as usize,
                };
                let (start, end) = page_span(a, n);
                let pages = (start..end).step_by(PAGE);
                if next(&mut x) % 3 == 0 {
                    early_remove(&mut early, start, end);
                    pages.for_each(|p| {
                        model.remove(&p);
                    });
                } else {
                    early_add(&mut early, start, end);
                    model.extend(pages);
                }
                assert_eq!(early_pages(&early), model, "seed {seed}");
            }
        }
    }

    /// The same pages as the list this replaced, for what that list handled exactly: one
    /// entry per committed page (`XMemMapPhysicalPagesScatter`, `home_again`) and page-aligned
    /// decommits of any length.
    #[test]
    fn early_commits_match_the_old_list_for_page_entries() {
        let base = 0x4_2000_0000usize;
        let mut x = 0x1234_5678_9ABC_DEF1u64;
        let (mut early, mut old): (BTreeMap<usize, usize>, Vec<(usize, usize)>) = (BTreeMap::new(), Vec::new());
        for _ in 0..20000 {
            let a = base + (next(&mut x) % 512) as usize * PAGE;
            if next(&mut x) % 2 == 0 {
                old.push((a, PAGE));
                early_add(&mut early, a, a + PAGE);
            } else {
                let n = (1 + next(&mut x) % 8) as usize * PAGE;
                old.retain(|&(y, m)| y + m <= a || y >= a + n);
                early_remove(&mut early, a, a + n);
            }
        }
        let old_pages: std::collections::BTreeSet<usize> = old.iter().map(|&(y, _)| y).collect();
        assert_eq!(early_pages(&early), old_pages);
    }

    /// A range committed in one call keeps its other pages when one page of it is decommitted
    /// (`commit_homes` commits several pages, `move_home_away` gives one back). The list this
    /// replaced dropped the whole entry, so `init` left those still-committed pages as holes.
    #[test]
    fn early_decommit_of_one_page_keeps_the_rest() {
        let a = 0x4_2000_0000usize;
        let mut early = BTreeMap::new();
        early_add(&mut early, a, a + 4 * TILE);
        early_remove(&mut early, a + TILE, a + 2 * TILE);
        assert_eq!(early.into_iter().collect::<Vec<_>>(), vec![(a, a + TILE), (a + 2 * TILE, a + 4 * TILE)]);
    }

    #[test]
    fn lazy_range_helpers_track_tile_overlap_holes_and_nonzero_pages() {
        let base = 0x5_0000_0000usize;
        let mut lazy = BTreeMap::new();
        // Partially cover tile 0 (pages 4..8) and fully cover tile 2.
        early_add(&mut lazy, base + 4 * PAGE, base + 8 * PAGE);
        early_add(&mut lazy, base + 2 * TILE, base + 3 * TILE);

        assert!(lazy_overlaps_tile(&lazy, base));
        assert!(!lazy_overlaps_tile(&lazy, base + TILE));
        assert!(lazy_overlaps_tile(&lazy, base + 2 * TILE));
        assert_eq!(lazy_tile_holes(&lazy, base), 0xFFFF & !(0b1111u16 << 4));
        assert_eq!(lazy_tile_holes(&lazy, base + TILE), 0xFFFF);
        assert_eq!(lazy_tile_holes(&lazy, base + 2 * TILE), 0);

        let mut page = vec![0u8; PAGE];
        assert!(!page_has_nonzero(&page));
        page[PAGE - 1] = 1;
        assert!(page_has_nonzero(&page));
    }

    /// Time for n 64 KiB pages committed one at a time and decommitted one at a time in
    /// reverse, before the device exists (`cargo test --release -p xcl --lib --
    /// --ignored --exact hle::d3d12x::sparse::tests::early_commit_timing --nocapture`,
    /// `XCL_EARLY_PAGES=16384,32768`). Run it alone: a test that made the device keeps a state.
    #[test]
    #[ignore]
    fn early_commit_timing() {
        if STATE.lock().unwrap().is_some() {
            return eprintln!("early_commit_timing: a sparse state exists – run it alone");
        }
        let sizes: Vec<usize> = std::env::var("XCL_EARLY_PAGES").ok().map_or(vec![4096, 8192, 16384, 32768], |v| v.split(',').map(|s| s.trim().parse().unwrap()).collect());
        let base = 0x4_2000_0000usize;
        for n in sizes {
            let t = std::time::Instant::now();
            for i in 0..n {
                unsafe { committed(base + i * TILE, TILE) };
            }
            let commit = t.elapsed();
            let t = std::time::Instant::now();
            for i in (0..n).rev() {
                unsafe { decommitted(base + i * TILE, TILE) };
            }
            let decommit = t.elapsed();
            println!("early: {n} pages: commit {:.1} ms, decommit {:.1} ms", commit.as_secs_f64() * 1e3, decommit.as_secs_f64() * 1e3);
        }
    }

    /// A sparse window of the test's own over a GPU view, and the CPU view the view shows, paired
    /// by `kernelx::mirrors` as a retail title's are (`tiles` tiles each, a size no other test
    /// pairs: pairs are made by size). The window's address space (4 GiB, a window's) is reserved
    /// with no access: a CPU touch of it faults the test. The CPU view is reserved, and committed
    /// where the test says. The window's buffer has `tiles` tiles.
    struct Views {
        s: State,
        dev: ID3D12Device,
        buf: ID3D12Resource,
        /// the title's queue its submits go to
        title: ID3D12CommandQueue,
        w: usize,
        c: usize,
    }

    impl Views {
        /// None: the device has no reserved buffers
        unsafe fn new(tiles: usize) -> Option<Views> {
            use windows_sys::Win32::System::Memory::MEM_RESERVE;
            let dev = test_device();
            let mut buf: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc((tiles * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut buf).ok()?;
            let buf = buf?;
            let w = VirtualAlloc(std::ptr::null(), crate::gpu::uma::WINDOW, MEM_RESERVE, PAGE_NOACCESS) as usize;
            let c = VirtualAlloc(std::ptr::null(), tiles * TILE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert!(w != 0 && c != 0 && w % TILE == 0 && c % TILE == 0);
            crate::hle::kernelx::mirrors::reserved(c, tiles * TILE, 0x2040_2000, PAGE_READWRITE);
            crate::hle::kernelx::mirrors::reserved(w, tiles * TILE, 0x2040_2000, 0x801);
            assert_eq!(cpu_of(w + 0x1_2345), c + 0x1_2345, "the views are paired");
            let title: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
            let mut s = new_state(&dev, vec![(w, buf.clone())]).expect("state");
            // (what its decommits are told of is the test's own)
            s.sources = Box::leak(Box::default());
            Some(Views { s, dev, buf, title, w, c })
        }

        /// Commit tiles `k..k + n` of the CPU view.
        unsafe fn commit(&self, k: usize, n: usize) {
            assert!(!VirtualAlloc((self.c + k * TILE) as *const c_void, n * TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
        }

        /// Write tiles `k..k + n` of the CPU view with bytes made from `seed`.
        unsafe fn write(&self, k: usize, n: usize, seed: u64) {
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            for i in 0..n * TILE / 8 {
                *((self.c + k * TILE) as *mut u64).add(i) = next(&mut x);
            }
        }

        /// The bytes of tile `k` of the CPU view.
        unsafe fn cpu_tile(&self, k: usize) -> Vec<u8> {
            std::slice::from_raw_parts((self.c + k * TILE) as *const u8, TILE).to_vec()
        }

        /// Tiles `0..n` of the GPU view asked for (`map_mirror`), taken by the test's state as a
        /// submit takes them (other tests' asks stay queued).
        fn ask(&mut self, n: usize) {
            let (lo, hi) = (self.w, self.w + n * TILE);
            map_mirror(lo, hi - lo);
            MIRROR_ASKS.lock().unwrap().retain(|&t| t < lo || t >= hi);
            self.s.asks.extend((lo..hi).step_by(TILE));
        }

        /// The tiles of `0..n` queued by `remapped`, taken by the test's state as a submit takes
        /// them (other tests' stay queued).
        fn take_remaps(&mut self, n: usize) {
            let (lo, hi) = (self.w, self.w + n * TILE);
            let mut q = REMAPS.lock().unwrap();
            self.s.remaps.extend(q.iter().copied().filter(|&t| t >= lo && t < hi));
            q.retain(|&t| t < lo || t >= hi);
        }

        /// A submit of the title, with the CPU pages `dirty` written since the last.
        unsafe fn submit(&mut self, dirty: &[usize]) {
            let title = self.title.clone();
            self.s.before_execute_on(&title, dirty, PAGE);
        }

        /// `len` bytes at `off` of `res` as the GPU has them, after what was sent to it.
        unsafe fn gpu_bytes(&mut self, res: &ID3D12Resource, off: usize, len: usize) -> Vec<u8> {
            let mut d = buffer_desc(len as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            let mut back: Option<ID3D12Resource> = None;
            self.dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back).expect("readback buffer");
            let back = back.expect("readback buffer");
            let alloc: ID3D12CommandAllocator = self.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = self.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            list.CopyBufferRegion(&back, 0, res, off as u64, len as u64);
            list.Close().expect("Close");
            self.s.flush_pending();
            self.s.queue.ExecuteCommandLists(&[Some(list.cast().expect("list"))]);
            self.s.signal();
            self.s.wait_for(self.s.value);
            let mut p = std::ptr::null_mut();
            back.Map(0, None, Some(&mut p)).expect("Map");
            let out = std::slice::from_raw_parts(p as *const u8, len).to_vec();
            back.Unmap(0, None);
            out
        }
    }

    impl Drop for Views {
        fn drop(&mut self) {
            use windows_sys::Win32::System::Memory::MEM_RELEASE;
            unsafe {
                self.s.wait_for(self.s.value);
                crate::hle::kernelx::mirrors::released(self.c);
                crate::hle::kernelx::mirrors::released(self.w);
                VirtualFree(self.c as *mut c_void, 0, MEM_RELEASE);
                VirtualFree(self.w as *mut c_void, 0, MEM_RELEASE);
            }
        }
    }

    /// The tiles of a GPU view are read and written at the CPU view they show, never at their own
    /// addresses (reserved with no access, as the title's are: a touch faults): the uploads'
    /// sources are in one piece only as far as the CPU view is, and immediates and copies back
    /// land in the CPU view. Before, the scan and the copies read the window's own addresses.
    #[test]
    fn a_mirrored_tile_is_read_and_written_at_its_cpu_view() {
        unsafe {
            let Some(mut v) = Views::new(37) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 5);
            v.write(0, 5, 1);
            v.s.asks.extend((0..4).map(|k| w + k * TILE));
            v.s.map_mirrors();
            for k in 0..4 {
                let e = &v.s.mapped[&(w + k * TILE)];
                assert_eq!((e.src, e.holes), (c + k * TILE, 0), "tile {k} shows its CPU tile");
            }
            assert_eq!(v.s.mirrored, 4);
            assert!(v.s.runs.is_empty() && runs_of(&v.s).is_empty(), "the view's own addresses are no CPU memory");
            // the CPU bytes of a run over the tiles are in one piece …
            assert_eq!(v.s.cpu_span(w + 0x100, 3 * TILE), (c + 0x100, 3 * TILE));
            // … up to a tile shown over another part of the CPU view …
            v.s.mapped.get_mut(&(w + 2 * TILE)).unwrap().src = c + 4 * TILE;
            assert_eq!(v.s.cpu_span(w + 0x100, 3 * TILE), (c + 0x100, 2 * TILE - 0x100));
            assert_eq!(v.s.cpu_span(w + 2 * TILE + 0x40, 0x80), (c + 4 * TILE + 0x40, 0x80));
            v.s.mapped.get_mut(&(w + 2 * TILE)).unwrap().src = c + 2 * TILE;
            // … or to one of the window's own (the one after the 4th has nothing: its address is
            // its own)
            assert_eq!(v.s.cpu_span(w + 3 * TILE, 2 * TILE), (c + 3 * TILE, TILE));

            // filled whole from the CPU view at the next submit
            v.submit(&[]);
            assert!(v.s.fill.is_empty());
            assert!(v.s.mapped[&w].sent.iter().all(|p| p.is_some()), "every page written went up");
            // an immediate of the GPU lands in the CPU view, and stays sent
            let gen = |s: &State, t: usize| s.mapped[&t].gen;
            v.s.store_imm(w + 2 * TILE + 0x40, 0x1122_3344_5566_7788, 8, gen(&v.s, w + 2 * TILE));
            assert_eq!(*((c + 2 * TILE + 0x40) as *const u64), 0x1122_3344_5566_7788);
            v.s.store_imm(w + 2 * TILE + 0x48, 0xAABB_CCDD, 4, gen(&v.s, w + 2 * TILE));
            assert_eq!(*((c + 2 * TILE + 0x48) as *const u32), 0xAABB_CCDD);
            assert!(v.s.chunk_clean(w + 2 * TILE + 0x40), "what landed is what the GPU has: not sent again");
            // so does a copy back …
            let back: Vec<u8> = (0..256u32).map(|i| i as u8 ^ 0x5A).collect();
            v.s.copy_back(back.as_ptr() as *mut u8, w + 3 * TILE + 0x200, 0, 256, gen(&v.s, w + 3 * TILE));
            assert_eq!(std::slice::from_raw_parts((c + 3 * TILE + 0x200) as *const u8, 256), &back[..]);
            // … but one recorded while the tile showed another part of the CPU view
            let before = gen(&v.s, w + TILE);
            v.s.drop_mirror(w + TILE);
            v.s.link_mirror(w + TILE, c + 4 * TILE);
            assert!(v.s.new_mirror(w + TILE, c + 4 * TILE, 0, false));
            let tile = v.cpu_tile(4);
            v.s.copy_back(back.as_ptr() as *mut u8, w + TILE, 0, 256, before);
            assert_eq!(v.cpu_tile(4), tile, "a copy back of what the tile showed before is left out");
        }
    }

    /// What the title's CPU writes has to reach the tiles over it: the watched ranges are the CPU
    /// tiles (one with no committed page yet too: a write there gives its tile video memory),
    /// never the GPU view's own addresses. Forgotten, the tiles give their video memory back.
    #[test]
    fn the_watch_ranges_of_mirrored_tiles_are_their_cpu_views() {
        unsafe {
            let Some(mut v) = Views::new(41) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 2);
            v.write(0, 2, 2);
            v.s.asks.extend((0..3).map(|k| w + k * TILE));
            let gen = mapped_gen();
            v.s.map_mirrors();
            assert!(mapped_gen() > gen, "readers of the ranges see them changed");
            assert_eq!(v.s.watched_ranges(), vec![(c, c + 3 * TILE)]);
            assert!(!v.s.mapped.contains_key(&(w + 2 * TILE)), "no committed page: no video memory yet");
            assert_eq!(v.s.mirrors[&(c + 2 * TILE)], vec![w + 2 * TILE]);
            assert_eq!(v.s.mirrored, 2);
            let gen = mapped_gen();
            v.s.forget_mirrors_in(c, c + 41 * TILE);
            assert!(mapped_gen() > gen);
            assert!(v.s.watched_ranges().is_empty() && v.s.mirrors.is_empty() && v.s.mirror_src.is_empty());
            assert_eq!(v.s.mirrored, 0);
            assert!(v.s.parked.contains_key(&w) && v.s.parked.contains_key(&(w + TILE)), "their video memory goes back");
            v.submit(&[]);
        }
    }

    /// The CPU view decommitted under a tile (told before Windows does it): its pages are holes,
    /// cleared on the GPU and not read again (a read would fault); a page written once committed
    /// again is one no more (a lazily committed page of the view). A tile with no committed page
    /// left gives its video memory back, and gets it again when its CPU tile is written.
    #[test]
    fn a_decommit_of_the_cpu_view_makes_holes_and_parks_the_tile() {
        unsafe {
            let Some(mut v) = Views::new(43) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 1);
            v.write(0, 1, 3);
            // the second tile: its first 8 pages committed, the rest not yet
            assert!(!VirtualAlloc((c + TILE) as *const c_void, 8 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::write_bytes((c + TILE) as *mut u8, 0x3C, 8 * PAGE);
            v.s.asks.extend([w, w + TILE]);
            v.s.map_mirrors();
            assert_eq!(v.s.mapped[&(w + TILE)].holes, 0xFF00);
            v.submit(&[]);
            assert!(v.s.mapped[&w].sent[3].is_some());

            v.s.source_decommitted_in(c + 3 * PAGE, PAGE);
            assert_ne!(VirtualFree((c + 3 * PAGE) as *mut c_void, PAGE, MEM_DECOMMIT), 0);
            assert_eq!(v.s.mapped[&w].holes, 1 << 3);
            assert!(v.s.mapped[&w].sent[3].is_none(), "the GPU holds zeros there");
            assert!(v.s.to_zero.contains(&(w + 3 * PAGE, PAGE)));
            // written before the decommit (the look reports it): asked again, still a hole, not read
            v.submit(&[c + 3 * PAGE]);
            assert_eq!((v.s.mapped[&w].holes, v.s.counts.rechecks), (1 << 3, 1));
            // a page of the second tile committed on its first write: no hole from that submit on
            assert!(!VirtualAlloc((c + TILE + 9 * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            *((c + TILE + 9 * PAGE + 0x80) as *mut u32) = 0x600D_F00D;
            v.submit(&[c + TILE + 9 * PAGE]);
            assert_eq!(v.s.mapped[&(w + TILE)].holes, 0xFF00 & !(1 << 9));
            assert!(v.s.mapped[&(w + TILE)].sent[9].is_some(), "and it went up");

            // the first tile decommitted whole: parked, still wanted over its CPU tile
            let pool = v.s.mapped[&w].pool;
            v.s.source_decommitted_in(c, TILE);
            assert_ne!(VirtualFree(c as *mut c_void, TILE, MEM_DECOMMIT), 0);
            assert!(!v.s.mapped.contains_key(&w));
            assert_eq!(v.s.parked.get(&w), Some(&pool));
            assert_eq!((v.s.mirrored, v.s.counts.given_back), (1, 1));
            assert_eq!(v.s.mirrors[&c], vec![w]);
            // committed and written again: back from parked, filled whole at the submit
            v.commit(0, 1);
            *((c + 5 * PAGE) as *mut u8) = 1;
            v.s.mirror_writes(&[c + 5 * PAGE], PAGE);
            assert_eq!((v.s.mapped[&w].pool, v.s.mapped[&w].holes), (pool, 0));
            assert!(v.s.fill.contains(&w) && v.s.to_zero.contains(&(w, TILE)));
            assert_eq!(v.s.mirrored, 2);
            v.submit(&[c + 5 * PAGE]);
            assert!(v.s.fill.is_empty() && v.s.mapped[&w].sent[5].is_some());
        }
    }

    /// `map_mirror` is called while the title records, on any thread: it does not wait for the
    /// state (held by a submit around its D3D12 calls), and asks for each tile once.
    #[test]
    fn map_mirror_does_not_wait_for_the_state() {
        // (an address nothing reserves: only the asking is looked at)
        const AT: usize = 0x7ff3_0000_0000;
        let asked = returns_while_the_state_is_held(|| {
            assert_eq!(map_mirror(AT + 0x10, 2 * TILE), 3, "three tiles queued");
            assert_eq!(map_mirror(AT + TILE, 3 * TILE), 1, "the fourth alone is new");
            assert_eq!(map_mirror(AT + 0x20, 0x100), 0, "a range in tiles asked for queues none");
            // (with the state held no submit takes them)
            let asks = MIRROR_ASKS.lock().unwrap();
            for k in 0..4 {
                assert_eq!(asks.iter().filter(|&&t| t == AT + k * TILE).count(), 1, "tile {k} is asked for once");
            }
        });
        assert!(asked, "map_mirror waited for the sparse state's lock");
        // (other tests forget tiles too: the count only goes up)
        let before = mirror_asks_forgotten();
        forget_mirrors(AT, 4 * TILE);
        assert!(!MIRROR_ASKED.contains(AT) && !MIRROR_ASKS.lock().unwrap().contains(&AT), "forgotten tiles may be asked for again");
        assert!(mirror_asks_forgotten() >= before + 4, "whoever remembers having asked for them is told");
    }

    /// A thread asks for a tile while `forget_mirrors` forgets it: once the forget is done the
    /// tile is queued or may be asked for again, and the thread is told to. Before, the asked set
    /// lost the tile first, the ask in between queued it anew and the queue lost it after: asked
    /// for, in no queue, never mapped (the GPU read zeros there).
    #[test]
    fn a_tile_asked_for_while_it_is_forgotten_is_not_lost() {
        // (an address nothing reserves: only the asking is looked at)
        const AT: usize = 0x7ff3_0010_0000;
        assert_eq!(map_mirror(AT, TILE), 1);
        let between = std::rc::Rc::new(std::cell::Cell::new(None));
        let b = between.clone();
        BETWEEN_FORGETS.with(|f| f.set(Some(Box::new(move || b.set(Some(map_mirror(AT, TILE)))))));
        let (counted, before) = (ASKS_FORGOTTEN.with(|n| n.get()), mirror_asks_forgotten());
        forget_mirrors(AT, TILE);
        assert_eq!(between.get(), Some(0), "asked for in between: still asked for, not queued anew");
        assert_eq!(ASKS_FORGOTTEN.with(|n| n.get()), counted + 1, "the forget after it is counted");
        assert!(mirror_asks_forgotten() > before);
        let queued = MIRROR_ASKS.lock().unwrap().contains(&AT);
        assert!(!MIRROR_ASKED.contains(AT) || queued, "asked for and in no queue: never mapped");
        assert_eq!(map_mirror(AT, TILE), 1, "asked for again: queued");
        forget_mirrors(AT, TILE);
    }

    /// An asked tile that `take_asks` leaves out (no pair: here past the GPU view's span) leaves
    /// the asked set uncounted, and may be asked for again. Counted, an address bound at every
    /// draw emptied every thread's memory of its asks at every submit, and every thread asked
    /// again.
    #[test]
    fn an_ask_left_out_is_not_counted_and_may_be_asked_for_again() {
        unsafe {
            let Some(mut v) = Views::new(127) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let g = v.w + 127 * TILE;
            assert_eq!(cpu_of(g), g, "no pair past the views");
            assert_eq!(map_mirror(g + 0x40, 0x100), 1);
            // (taken by the test's state as a submit takes them; other tests' asks stay queued)
            MIRROR_ASKS.lock().unwrap().retain(|&t| t != g);
            v.s.asks.push(g);
            let counted = ASKS_FORGOTTEN.with(|n| n.get());
            v.s.map_mirrors();
            assert!(v.s.asks.is_empty() && v.s.mirror_src.is_empty() && !v.s.mapped.contains_key(&g), "left out");
            assert_eq!(ASKS_FORGOTTEN.with(|n| n.get()), counted, "not counted");
            assert_eq!(map_mirror(g, TILE), 1, "asked for again");
            MIRROR_ASKS.lock().unwrap().retain(|&t| t != g);
            MIRROR_ASKED.remove(g);
        }
    }

    /// An ask a submit took that `forget_mirrors_in` drops (its CPU view forgotten) may be asked
    /// for again, and the threads that remember asking for it are told. Before, it stayed in the
    /// asked set: asked for again, it was not queued.
    #[test]
    fn an_ask_dropped_by_a_forget_may_be_asked_for_again() {
        unsafe {
            let Some(mut v) = Views::new(131) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            assert_eq!(map_mirror(w + 0x40, 0x100), 1);
            MIRROR_ASKS.lock().unwrap().retain(|&t| t != w);
            v.s.asks.push(w);
            let (counted, before) = (ASKS_FORGOTTEN.with(|n| n.get()), mirror_asks_forgotten());
            v.s.forget_mirrors_in(c, c + TILE);
            assert!(v.s.asks.is_empty(), "the ask over the CPU tile forgotten goes");
            assert_eq!(ASKS_FORGOTTEN.with(|n| n.get()), counted + 1, "and is counted");
            assert!(mirror_asks_forgotten() > before);
            assert_eq!(map_mirror(w, TILE), 1, "asked for again");
            MIRROR_ASKS.lock().unwrap().retain(|&t| t != w);
            MIRROR_ASKED.remove(w);
        }
    }

    /// Past `mirror_cap` tiles over CPU views get no video memory (counted once each); room
    /// comes back when a CPU tile is decommitted whole, and a refused tile gets it when its CPU
    /// tile is written.
    #[test]
    fn mirrored_tiles_stop_at_the_cap() {
        unsafe {
            let Some(mut v) = Views::new(47) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 4);
            v.write(0, 4, 5);
            v.s.mirror_cap = 2;
            v.s.asks.extend((0..4).map(|k| w + k * TILE));
            v.s.map_mirrors();
            assert_eq!(v.s.mirrored, 2);
            assert!(v.s.mapped.contains_key(&w) && v.s.mapped.contains_key(&(w + TILE)));
            assert!(!v.s.mapped.contains_key(&(w + 2 * TILE)) && !v.s.mapped.contains_key(&(w + 3 * TILE)));
            assert_eq!(v.s.counts.refused, 2);
            assert_eq!(v.s.mirror_src.len(), 4, "the refused ones are still wanted");
            v.s.mirror_writes(&[c + 2 * TILE], PAGE);
            assert_eq!((v.s.mirrored, v.s.counts.refused), (2, 2), "refused again, counted once");
            v.s.source_decommitted_in(c, TILE);
            assert_ne!(VirtualFree(c as *mut c_void, TILE, MEM_DECOMMIT), 0);
            assert_eq!(v.s.mirrored, 1);
            v.s.mirror_writes(&[c + 2 * TILE + PAGE], PAGE);
            assert!(v.s.mapped.contains_key(&(w + 2 * TILE)), "room again: written, it gets video memory");
            assert_eq!((v.s.mirrored, v.s.counts.refused, v.s.refused.len()), (2, 2, 1));
            v.submit(&[c + 2 * TILE + PAGE]);
        }
    }

    /// The holes of a CPU view's tiles are asked of Windows once for each region of pages in one
    /// state, not once a tile: VirtualQuery walks a region to its end, and a call a tile over a
    /// long committed range took the square of its length. They are what `query_tile_holes` says.
    #[test]
    fn the_holes_of_a_cpu_view_take_a_query_a_region() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        const TILES: usize = 64;
        unsafe {
            let c = VirtualAlloc(std::ptr::null(), TILES * TILE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert!(c != 0 && !VirtualAlloc(c as *const c_void, TILES * TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
            // (pages never touched are not in the working set: VirtualQuery answers for them)
            assert_ne!(VirtualFree((c + 10 * TILE + 5 * PAGE) as *mut c_void, PAGE, MEM_DECOMMIT), 0);
            *((c + 20 * TILE) as *mut u8) = 1;
            let tiles: Vec<usize> = (0..TILES).map(|k| c + k * TILE).collect();
            let q0 = VIRTUAL_QUERIES.with(|n| n.get());
            let holes = source_holes(&tiles);
            let queries = VIRTUAL_QUERIES.with(|n| n.get()) - q0;
            for (k, &t) in tiles.iter().enumerate() {
                assert_eq!(holes[k], query_tile_holes(t), "tile {k}");
            }
            assert_eq!(holes[10], 1 << 5);
            assert!(queries <= 3, "{queries} VirtualQuery calls for {TILES} tiles in 3 regions");
            assert_ne!(VirtualFree(c as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// A tile of a GPU view holds what its CPU view holds once it is asked for (the CPU wrote it
    /// before), and follows the CPU's writes, the many small ones too (dispatches) – far into the
    /// window and across the 1 GiB a dispatch reaches on a GPU. Before, the GPU read zeros there.
    #[test]
    fn a_gpu_view_tile_is_filled_from_its_cpu_view_and_follows_its_writes() {
        unsafe {
            let soft = software(&test_device());
            // (the software device takes small reserved buffers only)
            let tiles = if soft { 53 } else { (3 << 30) / TILE };
            let ks: [usize; 5] = if soft { [0, 1, 2, 3, 52] } else { [0, 1, (1 << 30) / TILE - 1, (1 << 30) / TILE, tiles - 1] };
            let Some(mut v) = Views::new(tiles) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            for &k in &ks {
                v.commit(k, 1);
                v.write(k, 1, 7 + k as u64);
            }
            v.s.asks.extend(ks.iter().map(|&k| v.w + k * TILE));
            v.submit(&[]);
            let buf = v.buf.clone();
            for &k in &ks {
                assert!(v.gpu_bytes(&buf, k * TILE, TILE) == v.cpu_tile(k), "tile {k} is filled from its CPU view");
            }
            // the CPU writes runs of each again
            let mut x = 0x1234_5678_9ABC_DEF1u64;
            let (mut dirty, mut small) = (Vec::new(), 0usize);
            for &k in &ks {
                let t = v.c + k * TILE;
                let mut o = 0;
                loop {
                    o += CHUNK * (1 + next(&mut x) as usize % 9);
                    let n = if next(&mut x) % 23 == 0 { SCATTER_RUN + CHUNK * (next(&mut x) as usize % 40) } else { CHUNK * (1 + next(&mut x) as usize % 12) };
                    if o + n > TILE {
                        break;
                    }
                    small += (n < SCATTER_RUN) as usize;
                    for i in 0..n / 8 {
                        *((t + o) as *mut u64).add(i) = next(&mut x);
                    }
                    dirty.extend(((t + o) & !(PAGE - 1)..t + o + n).step_by(PAGE));
                    o += n;
                }
            }
            dirty.sort_unstable();
            dirty.dedup();
            assert!(small > SCATTER_FROM * 4, "{small} small runs");
            let before = SCATTERED[1].load(Ordering::Relaxed);
            v.submit(&dirty);
            if !soft {
                assert!(SCATTERED[1].load(Ordering::Relaxed) - before >= SCATTER_FROM as u64, "the small runs went up in dispatches");
            }
            for &k in &ks {
                assert!(v.gpu_bytes(&buf, k * TILE, TILE) == v.cpu_tile(k), "tile {k} follows its CPU view");
            }
        }
    }

    /// What the title's lists write into a GPU view (a copy, an immediate) lands in the CPU view
    /// behind it, where the title's CPU reads it. Before, it was copied to the view's own
    /// addresses, which fault.
    #[test]
    fn a_gpu_write_into_a_gpu_view_lands_in_its_cpu_view() {
        unsafe {
            let Some(mut v) = Views::new(59) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 1);
            v.s.asks.push(w);
            v.submit(&[]);
            assert!(v.s.mapped.contains_key(&w));
            let mut up: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(256);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            v.dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up).expect("upload buffer");
            let up = up.expect("upload buffer");
            let mut p: *mut c_void = std::ptr::null_mut();
            up.Map(0, None, Some(&mut p)).expect("Map");
            for i in 0..64u32 {
                *(p as *mut u32).add(i as usize) = 0xC0DE_0000 | i;
            }
            let alloc: ID3D12CommandAllocator = v.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = v.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            list.CopyBufferRegion(&v.buf, 0x100, &up, 0, 256);
            list.Close().expect("close");
            let title = v.title.clone();
            title.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            v.s.after_execute(&title, &[(w + 0x100, 256)], &[(w + 0x800, 0x0123_4567_89AB_CDEF, 8)]);
            let t = std::time::Instant::now();
            while *((c + 0x100) as *const u32) != 0xC0DE_0000 || *((c + 0x800) as *const u64) == 0 {
                v.s.drain_readbacks(Drain::TitleDone);
                assert!(t.elapsed() < std::time::Duration::from_secs(5), "the copy back did not land");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            for i in 0..64u32 {
                assert_eq!(*((c + 0x100) as *const u32).add(i as usize), 0xC0DE_0000 | i, "dword {i}");
            }
            assert_eq!(*((c + 0x800) as *const u64), 0x0123_4567_89AB_CDEF, "the immediate");
        }
    }

    /// A buffer the title places in a GPU view shows the CPU view behind it: its tiles are the
    /// window's, filled from the CPU view (one with no committed page yet too, which the buffer
    /// needs: it shows the CPU's writes once they come). Before, its tiles were the view's own
    /// uncommitted pages, zeros for good.
    #[test]
    fn a_buffer_placed_in_a_gpu_view_shows_its_cpu_view() {
        unsafe {
            let Some(mut v) = Views::new(61) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 1);
            v.write(0, 1, 11);
            let mut res: Option<ID3D12Resource> = None;
            v.dev.CreateReservedResource(&buffer_desc(2 * TILE as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut res).expect("CreateReservedResource");
            let res = res.expect("reserved buffer");
            let tiles = v.s.buffer_tiles(w, 2 * TILE);
            assert_eq!(tiles.iter().map(|t| t.0).collect::<Vec<_>>(), vec![w, w + TILE], "both tiles, the one with no committed page too");
            assert_eq!((v.s.mapped[&w].src, v.s.mapped[&(w + TILE)].src), (c, c + TILE));
            assert_eq!(v.s.mapped[&(w + TILE)].holes, 0xFFFF);
            v.s.map_tiles(&res, w, &tiles);
            for &(_, tile) in &tiles {
                *v.s.pinned.entry(tile).or_default() += 1;
            }
            v.submit(&[]);
            assert!(v.gpu_bytes(&res, 0, TILE) == v.cpu_tile(0), "the buffer shows the CPU view");
            assert!(v.gpu_bytes(&res, TILE, TILE) == vec![0u8; TILE]);
            v.commit(1, 1);
            v.write(1, 1, 12);
            let dirty: Vec<usize> = (0..PAGES_PER_TILE).map(|i| c + TILE + i * PAGE).collect();
            v.submit(&dirty);
            assert!(v.gpu_bytes(&res, TILE, TILE) == v.cpu_tile(1), "and the CPU's writes to it");
        }
    }

    /// `n` physical pages of the title mapped at their home, one after the other (a pool for the
    /// queue's page mappings whose pages a tile may show): (the pages, the home). None: no unified
    /// memory.
    unsafe fn home_pages(n: usize) -> Option<(Vec<usize>, usize)> {
        use crate::hle::kernelx::{AllocateTitlePhysicalPages, MapTitlePhysicalPages};
        crate::gpu::uma::get()?;
        let (mut k, mut pages) = (n, vec![0usize; n]);
        assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut k, pages.as_mut_ptr()), 1);
        let home = MapTitlePhysicalPages(std::ptr::null_mut(), n, 0, 4, pages.as_mut_ptr()) as usize;
        assert!(home != 0 && home % TILE == 0);
        Some((pages, home))
    }

    /// The 64 KiB at `at` written with bytes made from `seed`; returns them.
    unsafe fn fill_tile(at: usize, seed: u64) -> Vec<u8> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        for i in 0..TILE / 8 {
            *(at as *mut u64).add(i) = next(&mut x);
        }
        std::slice::from_raw_parts(at as *const u8, TILE).to_vec()
    }

    /// One call of the queue's page-mapping method: `ranges` (kind, start, count) from GPU-view
    /// address `dest` on, in the pool of `pages` pages at `pool`.
    unsafe fn map_pages_at(dest: usize, ranges: &[[u32; 3]], pool: usize, pages: u32) {
        // (a batch: the first destination, the number of ranges, a pointer to them)
        let batch = [dest as u64, ranges.len() as u64, ranges.as_ptr() as u64];
        assert!(crate::hle::kernelx::mirrors::apply_batches(1, batch.as_ptr() as usize, pool, pages).is_some());
    }

    /// A tile of a GPU view shows the page the queue's page mappings put there, from the submit
    /// after the call: a mapped page, nothing for an unmapped one (no video memory, still asked
    /// for), the CPU view for one never named; the page's later writes, a remap, a page mapped
    /// where nothing was (with no new ask) and an unmap. No ask is forgotten on the way, and a
    /// range over two pages kept apart is not one piece on the CPU.
    #[test]
    fn a_tile_of_a_gpu_view_shows_the_page_mapped_there() {
        unsafe {
            let Some(mut v) = Views::new(163) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let Some((mut pages, h)) = home_pages(3) else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let (w, buf) = (v.w, v.buf.clone());
            let mut pool: Vec<Vec<u8>> = (0..3).map(|k| fill_tile(h + k * TILE, 40 + k as u64)).collect();
            // three tiles over their CPU view, asked for and filled
            v.commit(0, 3);
            v.write(0, 3, 44);
            v.ask(3);
            v.submit(&[]);
            assert!(v.s.mapped.contains_key(&(w + TILE)));
            let forgotten = ASKS_FORGOTTEN.with(|n| n.get());
            let shown = page_tiles()[0];

            // pool page 0 at view page 0, view page 1 unmapped, view page 2 left as it is
            map_pages_at(w, &[[0, 0, 1], [2, 0, 1]], h, 3);
            v.take_remaps(3);
            assert_eq!(v.s.remaps, vec![w, w + TILE], "the asked tiles of the changed pages");
            v.submit(&[]);
            assert!(v.gpu_bytes(&buf, 0, TILE) == pool[0], "the pool page");
            assert!(!v.s.mapped.contains_key(&(w + TILE)) && !v.s.mirror_src.contains_key(&(w + TILE)), "unmapped: no video memory");
            assert!(MIRROR_ASKED.contains(w + TILE), "and still asked for");
            assert!(v.gpu_bytes(&buf, 2 * TILE, TILE) == v.cpu_tile(2), "never named: its CPU tile");
            assert!(page_tiles()[0] >= shown + 2);
            // a write to the pool page goes up
            pool[0] = fill_tile(h, 50);
            v.submit(&(0..PAGES_PER_TILE).map(|i| h + i * PAGE).collect::<Vec<_>>());
            assert!(v.gpu_bytes(&buf, 0, TILE) == pool[0], "the pool page's write");

            // view page 0 remapped to pool page 2, pool page 1 mapped at view page 1 (asked for
            // while unmapped: no new ask)
            assert_eq!(map_mirror(w + TILE, 0x100), 0, "still asked for");
            map_pages_at(w, &[[0, 2, 1], [0, 1, 1]], h, 3);
            assert_eq!(super::super::view_want::cpu_run((w + TILE - 8) as u64, 16), None, "two pages kept apart");
            v.take_remaps(3);
            v.submit(&[]);
            assert!(v.gpu_bytes(&buf, 0, TILE) == pool[2], "the new page");
            assert!(v.gpu_bytes(&buf, TILE, TILE) == pool[1], "the page mapped where nothing was");
            // an unmap takes the video memory
            map_pages_at(w, &[[2, 0, 1]], h, 3);
            v.take_remaps(3);
            v.submit(&[]);
            assert!(!v.s.mapped.contains_key(&w) && MIRROR_ASKED.contains(w));
            assert_eq!(ASKS_FORGOTTEN.with(|n| n.get()), forgotten, "no ask forgotten on the way");

            // freed: the tile over a freed page shows nothing
            assert_eq!(crate::hle::kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), 3, pages.as_mut_ptr()), 1);
            v.take_remaps(3);
            v.submit(&[]);
            assert!(!v.s.mapped.contains_key(&(w + TILE)));
        }
    }

    /// A buffer placed over GPU-view tiles keeps its video memory while the pages under it
    /// change: a remap shows the new page on the same pool tile, an unmap zeros, and a buffer
    /// placed over an unmapped page shows the page once it is mapped there.
    #[test]
    fn a_buffer_over_a_remapped_tile_reads_the_new_page() {
        unsafe {
            let Some(mut v) = Views::new(167) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let Some((mut pages, h)) = home_pages(2) else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let w = v.w;
            let pool: Vec<Vec<u8>> = (0..2).map(|k| fill_tile(h + k * TILE, 60 + k as u64)).collect();
            // pool page 0 at view page 0, view page 1 unmapped; the buffer's tiles are asked for
            // (as a placed buffer's are, `view_want`)
            map_pages_at(w, &[[0, 0, 1], [2, 0, 1]], h, 2);
            v.ask(2);
            let mut res: Option<ID3D12Resource> = None;
            v.dev.CreateReservedResource(&buffer_desc(2 * TILE as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut res).expect("CreateReservedResource");
            let res = res.expect("reserved buffer");
            let tiles = v.s.buffer_tiles(w, 2 * TILE);
            assert_eq!(tiles.len(), 2);
            assert_eq!((v.s.mapped[&w].src, v.s.mapped[&(w + TILE)].src), (h, no_source()), "the unmapped one over nothing");
            v.s.map_tiles(&res, w, &tiles);
            for &(_, tile) in &tiles {
                *v.s.pinned.entry(tile).or_default() += 1;
            }
            v.submit(&[]);
            assert!(v.gpu_bytes(&res, 0, TILE) == pool[0], "the buffer shows the pool page");
            assert!(v.gpu_bytes(&res, TILE, TILE) == vec![0u8; TILE]);
            let (p0, p1) = (v.s.mapped[&w].pool, v.s.mapped[&(w + TILE)].pool);

            // remapped: the new page on the same pool tile
            map_pages_at(w, &[[0, 1, 1]], h, 2);
            v.take_remaps(2);
            v.submit(&[]);
            assert!(v.gpu_bytes(&res, 0, TILE) == pool[1], "the buffer reads the new page");
            assert_eq!(v.s.mapped[&w].pool, p0, "on the same pool tile");
            // unmapped: zeros, the pool tile kept
            map_pages_at(w, &[[2, 0, 1]], h, 2);
            v.take_remaps(2);
            v.submit(&[]);
            assert!(v.gpu_bytes(&res, 0, TILE) == vec![0u8; TILE], "unmapped: zeros");
            assert_eq!((v.s.mapped[&w].src, v.s.mapped[&w].pool), (no_source(), p0));
            // the page mapped under the buffer's second tile, placed while it was unmapped
            map_pages_at(w + TILE, &[[0, 0, 1]], h, 2);
            v.take_remaps(2);
            v.submit(&[]);
            assert!(v.gpu_bytes(&res, TILE, TILE) == pool[0], "shown once mapped");
            assert_eq!((v.s.mapped[&(w + TILE)].src, v.s.mapped[&(w + TILE)].pool), (h, p1));

            assert_eq!(crate::hle::kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), 2, pages.as_mut_ptr()), 1);
            v.take_remaps(2);
            v.submit(&[]);
        }
    }

    /// A tile follows its page when the page moves (`retarget`): the link moves, and its pool
    /// tile, gen and sent hashes stay, with nothing cleared or filled; a scan of the new place
    /// sends nothing when its bytes are the same and the chunk written when they differ; a copy
    /// back recorded before the move lands at the new place; a page never used, not committed at
    /// its new place, gives the video memory up.
    #[test]
    fn a_tile_follows_its_page_without_being_filled_again() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        unsafe {
            let Some(mut v) = Views::new(173) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let Some((mut pages, h)) = home_pages(1) else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let (w, buf) = (v.w, v.buf.clone());
            let bytes = fill_tile(h, 70);
            map_pages_at(w, &[[0, 0, 1]], h, 1);
            v.ask(1);
            v.submit(&[]);
            assert!(v.gpu_bytes(&buf, 0, TILE) == bytes, "the page at its home");
            // its bytes copied to another place (as kernelx moves a page), and the tile follows
            let r = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert!(r != 0 && r % TILE == 0 && !VirtualAlloc(r as *const c_void, TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::copy_nonoverlapping(h as *const u8, r as *mut u8, TILE);
            let e = &v.s.mapped[&w];
            let (pool, gen, sent) = (e.pool, e.gen, e.sent.clone());
            let (fills, followed) = (v.s.counts.fills, page_tiles()[1]);
            v.s.retarget(w, h, r, true);
            let e = &v.s.mapped[&w];
            assert_eq!((e.src, e.holes, e.pool, e.gen), (r, 0, pool, gen), "the link moves, the video memory stays");
            assert!(e.sent == sent, "and what was sent");
            assert_eq!(v.s.mirror_src[&w], r);
            assert!(!v.s.mirrors.contains_key(&h) && v.s.mirrors[&r] == vec![w]);
            assert!(v.s.fill.is_empty() && v.s.later.is_empty() && v.s.to_zero.range((w, 0)..(w + TILE, 0)).next().is_none(), "nothing cleared or filled");
            assert!(page_tiles()[1] > followed);
            v.s.retarget(w, h, r + TILE, true);
            assert_eq!(v.s.mirror_src[&w], r, "a tile linked elsewhere stays");

            // the new place's pages written (kernelx's known write): the same bytes, nothing sent
            let dirty: Vec<usize> = (0..PAGES_PER_TILE).map(|i| r + i * PAGE).collect();
            v.submit(&dirty);
            assert_eq!(v.s.counts.fills, fills, "no fill");
            assert!(v.s.mapped[&w].sent == sent, "nothing sent");
            // a chunk that differs goes up
            *((r + 3 * PAGE + 0x40) as *mut u64) ^= 0xFFFF;
            v.submit(&[r + 3 * PAGE]);
            assert!(v.s.mapped[&w].sent != sent);
            assert!(v.gpu_bytes(&buf, 0, TILE) == std::slice::from_raw_parts(r as *const u8, TILE), "the chunk written");
            // a copy back recorded before the move lands at the new place
            let back = [0x5Au8; 256];
            v.s.copy_back(back.as_ptr() as *mut u8, w + 0x200, 0, 256, gen);
            assert_eq!(std::slice::from_raw_parts((r + 0x200) as *const u8, 256), &back[..]);

            // a page never used: not committed at its new place, no video memory for it
            v.s.retarget(w, r, r + TILE, false);
            assert!(!v.s.mapped.contains_key(&w) && v.s.parked.contains_key(&w) && v.s.later.contains(&(w, r + TILE)));
            v.submit(&[]);
            assert!(!v.s.mapped.contains_key(&w) && v.s.mirror_src[&w] == r + TILE, "linked, with no video memory until its page is written");

            assert_eq!(crate::hle::kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), 1, pages.as_mut_ptr()), 1);
            v.take_remaps(1);
            assert_ne!(VirtualFree(r as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// A decommit of the CPU view under way – the state told, Windows not done yet – when a submit
    /// asks Windows again about pages of it written before (the look reports them): they stay
    /// holes and are not read, and a tile with no page left gets no video memory back. Before,
    /// Windows still had them: the submit read them while they were freed, and kept them as
    /// committed for good.
    #[test]
    fn pages_being_decommitted_stay_holes_for_a_submit() {
        unsafe {
            let Some(mut v) = Views::new(67) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 2);
            v.write(0, 2, 13);
            v.s.asks.extend([w, w + TILE]);
            v.submit(&[]);
            assert_eq!((v.s.mapped[&w].holes, v.s.mapped[&(w + TILE)].holes), (0, 0));
            // a page of the first tile and the whole second: the decommitter tells the state …
            let p = c + 3 * PAGE;
            let (one, shown) = freeing_begins(v.s.sources, p, PAGE);
            assert!(shown, "tiles show it");
            v.s.source_decommitted_in(p, PAGE);
            let (two, _) = freeing_begins(v.s.sources, c + TILE, TILE);
            v.s.source_decommitted_in(c + TILE, TILE);
            assert!(!v.s.mapped.contains_key(&(w + TILE)));
            // … a submit, with pages of both written before …
            v.submit(&[p, c + TILE + 2 * PAGE]);
            assert_eq!(v.s.mapped[&w].holes, 1 << 3, "Windows has the page still: a hole all the same");
            assert!(v.s.mapped[&w].sent[3].is_none(), "and not read");
            assert!(!v.s.mapped.contains_key(&(w + TILE)), "no page left: no video memory");
            // … then Windows
            assert_ne!(VirtualFree(p as *mut c_void, PAGE, MEM_DECOMMIT), 0);
            freeing_ends(v.s.sources, one);
            assert_ne!(VirtualFree((c + TILE) as *mut c_void, TILE, MEM_DECOMMIT), 0);
            freeing_ends(v.s.sources, two);
            assert!(lock_sources(v.s.sources).freeing.is_empty());
            v.submit(&[p, c + TILE + 2 * PAGE]);
            assert_eq!(v.s.mapped[&w].holes, 1 << 3);
            assert!(!v.s.mapped.contains_key(&(w + TILE)));
        }
    }

    /// A tile asked for while a decommit of its CPU view is under way (nothing showed that memory
    /// when the decommit began, so the state was not told): the pages being decommitted are holes
    /// in it and are not read, and a decommit that begins once it is linked finds it. Before,
    /// Windows had them still, and the submit filled the tile from them while they were freed.
    #[test]
    fn a_tile_linked_during_a_decommit_of_its_cpu_view_takes_those_pages_as_holes() {
        unsafe {
            let Some(mut v) = Views::new(71) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 1);
            v.write(0, 1, 17);
            let (span, shown) = freeing_begins(v.s.sources, c + 2 * PAGE, 2 * PAGE);
            assert!(!shown, "nothing shows it yet: the decommit does not wait for the state");
            v.s.asks.push(w);
            v.submit(&[]);
            assert_eq!(v.s.mapped[&w].holes, 0b1100);
            assert!(v.s.mapped[&w].sent[2].is_none() && v.s.mapped[&w].sent[3].is_none(), "not read");
            assert!(v.s.mapped[&w].sent[4].is_some(), "the rest went up");
            assert_ne!(VirtualFree((c + 2 * PAGE) as *mut c_void, 2 * PAGE, MEM_DECOMMIT), 0);
            freeing_ends(v.s.sources, span);
            let (span, shown) = freeing_begins(v.s.sources, c + 8 * PAGE, PAGE);
            assert!(shown, "linked: a decommit from here on tells the state");
            freeing_ends(v.s.sources, span);
        }
    }

    /// The pair of a tile asked for goes between its look up and its link (the CPU view released,
    /// and its `forget_mirrors` found nothing linked): the tile is not linked, and decommits of
    /// that memory do not wait for the state. Before, it stayed linked over memory that may be
    /// given out anew.
    #[test]
    fn a_tile_whose_pair_went_before_its_link_is_left_out() {
        unsafe {
            let Some(mut v) = Views::new(73) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 1);
            v.write(0, 1, 19);
            v.s.asks.push(w);
            BEFORE_LINK.with(|f| f.set(Some(Box::new(move || crate::hle::kernelx::mirrors::released(c)))));
            v.s.map_mirrors();
            assert!(v.s.mirror_src.is_empty() && v.s.mirrors.is_empty() && !v.s.mapped.contains_key(&w));
            assert!(lock_sources(v.s.sources).runs.is_empty());
            assert_eq!(v.s.mirrored, 0);
        }
    }

    /// `source_decommit` and `source_release` hold the pages as being freed while Windows frees
    /// them (the caller's call, whose result they return), and not after.
    #[test]
    fn a_decommit_of_cpu_memory_is_under_way_while_windows_does_it() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        unsafe {
            let a = VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            assert!(a != 0);
            let under_way = |s: (usize, usize)| lock_sources(&SOURCES).freeing.contains(&s);
            let r = source_decommit(a + PAGE + 0x10, PAGE, || {
                assert!(under_way((a + PAGE, a + 3 * PAGE)), "the pages it touches");
                VirtualFree((a + PAGE) as *mut c_void, 2 * PAGE, MEM_DECOMMIT)
            });
            assert_ne!(r, 0);
            assert!(!under_way((a + PAGE, a + 3 * PAGE)));
            let r = source_release(a, TILE, || {
                assert!(under_way((a, a + TILE)));
                VirtualFree(a as *mut c_void, 0, MEM_RELEASE)
            });
            assert_ne!(r, 0);
            assert!(!under_way((a, a + TILE)));
        }
    }

    /// A CPU view that lies in a sparse window itself (a title's reservation of 64 KiB pages)
    /// decommitted there: the tiles over it get the holes too, and a tile with no page left gives
    /// its video memory back; the pages are held as being freed while Windows frees them
    /// (`decommit`). Before, only the view's own tiles got them, and copies back, immediates and
    /// scans through the tiles over it touched the decommitted pages.
    #[test]
    fn a_decommit_of_a_cpu_view_in_a_window_makes_holes_in_the_tiles_over_it() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        unsafe {
            let Some(mut v) = Views::new(79) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            let mut own: Option<ID3D12Resource> = None;
            v.dev.CreateReservedResource(&buffer_desc((79 * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut own).expect("CreateReservedResource");
            // (after the GPU view's window: a tile in both is the view's)
            v.s.windows.push((c, own.expect("reserved buffer")));
            v.commit(0, 1);
            v.s.commit_range(c, TILE);
            v.write(0, 1, 23);
            v.s.asks.push(w);
            v.submit(&[]);
            assert_eq!((v.s.mapped[&c].src, v.s.mapped[&w].src, v.s.mapped[&w].holes), (0, c, 0));
            v.s.decommit_range(c + 3 * PAGE, PAGE);
            assert_ne!(VirtualFree((c + 3 * PAGE) as *mut c_void, PAGE, MEM_DECOMMIT), 0);
            assert_eq!(v.s.mapped[&c].holes, 1 << 3);
            assert_eq!(v.s.mapped[&w].holes, 1 << 3, "the tile over it has the hole too");
            assert!(v.s.mapped[&w].sent[3].is_none() && v.s.to_zero.contains(&(w + 3 * PAGE, PAGE)));
            v.s.decommit_range(c, TILE);
            assert_ne!(VirtualFree(c as *mut c_void, TILE, MEM_DECOMMIT), 0);
            assert!(!v.s.mapped.contains_key(&w) && v.s.parked.contains_key(&w), "no page left: its video memory goes back");
            assert_eq!(v.s.mirrored, 0);
            // (written before: the look reports it, and nothing reads it)
            v.submit(&[c + 5 * PAGE]);
            assert!(!v.s.mapped.contains_key(&w));

            let a = VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            assert!(a != 0);
            let r = decommit(a, PAGE, || {
                assert!(lock_sources(&SOURCES).freeing.contains(&(a, a + PAGE)), "being freed while Windows does it");
                VirtualFree(a as *mut c_void, PAGE, MEM_DECOMMIT)
            });
            assert_ne!(r, 0);
            assert!(!lock_sources(&SOURCES).freeing.contains(&(a, a + PAGE)));
            assert_ne!(VirtualFree(a as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// A copy back or an immediate recorded before the CPU memory a tile shows changed – a page
    /// decommitted and committed again, or the tile shown over another CPU tile – lands nowhere:
    /// the GPU's bytes were of the memory before. What is recorded now lands. Before, a copy
    /// back compared only the CPU address and an immediate nothing, and the old bytes went into
    /// the zero chunks of the new memory.
    #[test]
    fn a_copy_back_or_immediate_of_memory_since_changed_lands_nowhere() {
        unsafe {
            let Some(mut v) = Views::new(83) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 2);
            v.s.asks.extend([w, w + TILE]);
            v.submit(&[]);
            let (g0, g1) = (v.s.mapped[&w].gen, v.s.mapped[&(w + TILE)].gen);
            // a page of the first CPU tile decommitted, committed again and written in part: asked
            // again at the submit, it is no hole
            let p = c + 3 * PAGE;
            v.s.source_decommitted_in(p, PAGE);
            assert_ne!(VirtualFree(p as *mut c_void, PAGE, MEM_DECOMMIT), 0);
            assert!(!VirtualAlloc(p as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            *(p as *mut u64) = 0x600D;
            v.submit(&[p]);
            assert_eq!(v.s.mapped[&w].holes, 0);
            let back = vec![0xABu8; 256];
            v.s.copy_back(back.as_ptr() as *mut u8, w + 3 * PAGE, 0, 256, g0);
            assert_eq!(*(p as *const u64), 0x600D);
            assert!(std::slice::from_raw_parts((p + 8) as *const u8, 248).iter().all(|&b| b == 0), "the GPU's old bytes do not land in the new memory");
            // the second tile shown over the first CPU tile: an immediate of before lands nowhere …
            v.s.drop_mirror(w + TILE);
            v.s.link_mirror(w + TILE, c);
            assert!(v.s.new_mirror(w + TILE, c, 0, false));
            v.s.store_imm(w + TILE + 0x80, 0x1234_5678, 8, g1);
            assert_eq!(*((c + 0x80) as *const u64), 0);
            // … one of now does
            v.s.store_imm(w + TILE + 0x80, 0x1234_5678, 8, v.s.mapped[&(w + TILE)].gen);
            assert_eq!(*((c + 0x80) as *const u64), 0x1234_5678);
            // (a submission's immediates are recorded with the memory they are for)
            if v.s.rbq.is_some() {
                let title = v.title.clone();
                v.s.after_execute(&title, &[], &[(w + 0x100, 0x77, 8)]);
                assert_eq!(v.s.imm_jobs.back().map(|j| j.writes[0].3), Some(v.s.mapped[&w].gen));
                let t = std::time::Instant::now();
                while !v.s.imm_jobs.is_empty() {
                    v.s.drain_readbacks(Drain::TitleDone);
                    assert!(t.elapsed() < std::time::Duration::from_secs(5), "the immediate did not land");
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                assert_eq!(*((c + 0x100) as *const u64), 0x77);
            }
        }
    }

    /// A buffer placed over a tile of a GPU view at `mirror_cap` gets the tile all the same: it
    /// maps its tiles once, and the window tile could get video memory later that the buffer
    /// would not have. Before, it was left out of the buffer for good.
    #[test]
    fn a_buffer_placed_at_the_cap_gets_its_tiles() {
        unsafe {
            let Some(mut v) = Views::new(97) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 2);
            v.write(0, 2, 31);
            v.s.mirror_cap = 1;
            v.s.asks.push(w);
            v.s.map_mirrors();
            assert_eq!(v.s.mirrored, 1);
            let tiles = v.s.buffer_tiles(w + TILE, TILE);
            assert_eq!(tiles.iter().map(|t| t.0).collect::<Vec<_>>(), vec![w + TILE], "the buffer has its tile");
            assert_eq!(v.s.mapped[&(w + TILE)].src, c + TILE);
            assert_eq!((v.s.mirrored, v.s.counts.refused), (2, 0), "it counts");
            v.submit(&[]);
        }
    }

    /// A tile kept mapped for a placed buffer while its CPU tile was decommitted whole takes its
    /// video memory back once the CPU tile is committed and written again, at `mirror_cap` too:
    /// the memory is held already. Before, it was refused, and stayed mapped in the window and the
    /// buffer with the old bytes while the new ones never went up.
    #[test]
    fn a_kept_tile_is_taken_back_at_the_cap() {
        unsafe {
            let Some(mut v) = Views::new(101) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 2);
            v.write(0, 2, 37);
            v.s.asks.extend([w, w + TILE]);
            v.submit(&[]);
            let pool = v.s.mapped[&w].pool;
            // (a placed buffer uses it)
            *v.s.pinned.entry(pool).or_default() += 1;
            v.s.source_decommitted_in(c, TILE);
            assert_ne!(VirtualFree(c as *mut c_void, TILE, MEM_DECOMMIT), 0);
            v.s.flush_parked();
            assert_eq!((v.s.kept.get(&w), v.s.mirrored), (Some(&pool), 1));
            v.s.mirror_cap = 1;
            v.commit(0, 1);
            v.write(0, 1, 41);
            v.s.mirror_writes(&[c + PAGE], PAGE);
            assert_eq!(v.s.mapped.get(&w).map(|e| e.pool), Some(pool), "taken back");
            assert_eq!((v.s.mirrored, v.s.counts.refused), (2, 0));
            assert!(v.s.fill.contains(&w));
            v.submit(&[c + PAGE]);
        }
    }

    /// A tile refused at `mirror_cap` gets video memory at a submit once there is room again,
    /// filled with what its CPU view held (written once before, maybe never again); while there
    /// is none, a write to its CPU tile does not ask Windows about it. Before, it waited for a
    /// write after the room came back, and asked Windows at every write meanwhile.
    #[test]
    fn a_tile_refused_at_the_cap_gets_video_memory_when_there_is_room() {
        unsafe {
            let Some(mut v) = Views::new(103) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 2);
            v.write(0, 2, 43);
            // (the third CPU tile half committed: asked about, the rest takes a VirtualQuery)
            assert!(!VirtualAlloc((c + 2 * TILE) as *const c_void, 8 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::write_bytes((c + 2 * TILE) as *mut u8, 0x5A, 8 * PAGE);
            v.s.mirror_cap = 2;
            v.s.asks.extend([w, w + TILE, w + 2 * TILE]);
            v.submit(&[]);
            assert!(!v.s.mapped.contains_key(&(w + 2 * TILE)) && v.s.refused.contains(&(w + 2 * TILE)));
            let q0 = VIRTUAL_QUERIES.with(|n| n.get());
            v.submit(&[c + 2 * TILE]);
            assert_eq!(VIRTUAL_QUERIES.with(|n| n.get()), q0, "at the cap, Windows is not asked about it");
            assert!(!v.s.mapped.contains_key(&(w + 2 * TILE)));
            // room again (a CPU tile decommitted whole), and no write: it gets its video memory
            v.s.source_decommitted_in(c, TILE);
            assert_ne!(VirtualFree(c as *mut c_void, TILE, MEM_DECOMMIT), 0);
            assert_eq!(v.s.mirrored, 1);
            v.submit(&[]);
            assert_eq!(v.s.mapped.get(&(w + 2 * TILE)).map(|e| e.holes), Some(0xFF00));
            assert!(v.s.refused.is_empty() && v.s.mirrored == 2 && v.s.counts.refused == 1);
            assert!((0..8).all(|i| v.s.mapped[&(w + 2 * TILE)].sent[i].is_some()), "what the CPU view held went up");
        }
    }

    /// A tile that found no video memory is refused like one at the cap (not asked for again at
    /// every ask), and gets it at a submit once there is some. Before, it was not even recorded:
    /// it read zeros until its CPU tile was written again.
    #[test]
    fn a_tile_refused_for_want_of_video_memory_gets_it_later() {
        unsafe {
            let Some(mut v) = Views::new(107) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let w = v.w;
            v.commit(0, 1);
            v.write(0, 1, 47);
            // (as after a heap that could not be made)
            v.s.no_memory = Some(std::time::Instant::now());
            v.s.asks.push(w);
            v.submit(&[]);
            assert!(!v.s.mapped.contains_key(&w) && v.s.refused.contains(&w));
            assert_eq!(v.s.counts.refused, 1);
            v.s.no_memory = None;
            v.submit(&[]);
            assert!(v.s.mapped.contains_key(&w) && v.s.refused.is_empty());
            assert!(v.s.mapped[&w].sent.iter().all(|p| p.is_some()), "what the CPU view held went up");
        }
    }

    /// A pool tile freed is used again `COOL` after at the earliest, the oldest first and before
    /// a tile of `free`. Before, it went to `free` and was the next one taken.
    #[test]
    fn a_freed_pool_tile_cools_before_it_is_used_again() {
        let t0 = std::time::Instant::now();
        let ms = |n: u64| t0 + std::time::Duration::from_millis(n);
        let mut free: Vec<(u32, u32)> = Vec::new();
        let mut cooling: VecDeque<((u32, u32), std::time::Instant)> = VecDeque::new();
        cooling.push_back(((0, 5), ms(0)));
        cooling.push_back(((0, 6), ms(10)));
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(1999)), None, "freed 1999 ms ago: still cooling");
        assert!(!cooled(&cooling, ms(1999)));
        free.push((1, 9));
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(1999)), Some((1, 9)), "a free tile meanwhile");
        assert!(cooled(&cooling, ms(2000)));
        free.push((1, 8));
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(2005)), Some((0, 5)), "cooled: before a free tile");
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(2005)), Some((1, 8)), "the next one freed 10 ms later is not");
        cooling.push_back(((2, 1), ms(20)));
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(5000)), Some((0, 6)), "the oldest first");
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(5000)), Some((2, 1)));
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(5000)), None);
    }

    /// The tile mapping line gives the calls and tiles since the counts before, by kind of window.
    #[test]
    fn the_tile_mapping_line_counts_since_the_last() {
        let before = [1, 10, 2, 20, 3, 30, 4, 40];
        assert_eq!(tile_calls_line(&before, &before), None, "no call: no line");
        let now = [2, 74, 2, 20, 3, 30, 7, 41];
        assert_eq!(
            tile_calls_line(&now, &before).as_deref(),
            Some("sparse windows 1 calls mapping 64 tiles, 0 calls unmapping 0 tiles; GPU views 0 calls mapping 0 tiles, 3 calls unmapping 1 tiles")
        );
    }

    /// The GPU-view diagnostics: the poison pattern a switch names, the keep decision within the
    /// cap, the buckets of the submits a tile was away, and the line of the counts since the last.
    #[test]
    fn view_diag_switches_buckets_and_line() {
        assert_eq!(poison_pattern(None), None);
        assert_eq!(poison_pattern(Some("1")), Some(0x7FC0_DEAD));
        assert_eq!(poison_pattern(Some("0x0000BEEF")), Some(0xBEEF));
        assert_eq!(poison_pattern(Some("0x0")), None, "a zero pattern is no poison");
        assert_eq!(poison_pattern(Some("0xnope")), None);
        assert!(!keep_view_tile(false, 1, 10), "off: given back");
        assert!(keep_view_tile(true, 10, 10), "at the cap with this tile: kept");
        assert!(!keep_view_tile(true, 11, 10), "past it (a placed buffer's tiles): given back");
        let buckets: Vec<usize> = [0, 1, 2, 8, 9, 64, 65, u64::MAX].into_iter().map(back_bucket).collect();
        assert_eq!(buckets, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut before = [0u64; VS];
        before[VS_NOW] = 5;
        assert_eq!(view_stats_line(&before, &before), None, "nothing counted: no line");
        let mut now = [7, 3, 2, 1, 4, 0, 1, 0, 3 << 19, 20, 6, 9, 1, 12, 0];
        for (n, b) in now.iter_mut().zip(before.iter_mut()).take(VS_NOW) {
            *b = 1;
            *n += 1;
        }
        assert_eq!(
            view_stats_line(&now, &before).as_deref(),
            Some("GPU-view tiles (2 s): 7 asked for, 3 filled, 2 given back, 1 kept; back after 0-1 / 2-8 / 9-64 / more submits: 4 / 0 / 1 / 0 (1.5 MB filled again); 6 of 20 submits left asked tiles waiting; now 9 with video memory, 1 kept, 12 waiting, 0 refused")
        );
    }

    /// `XCL_VIEW_KEEP`: a tile whose CPU view is decommitted stays mapped where it is (kept, not
    /// parked, so no submit unmaps it), and is taken back from there once its CPU tile is written
    /// again, counted in the bucket of the submits it was away.
    #[test]
    fn a_kept_tile_stays_mapped_and_comes_back() {
        unsafe {
            let Some(mut v) = Views::new(191) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            (v.s.diag.keep, v.s.diag.stats) = (true, true);
            v.commit(0, 1);
            v.write(0, 1, 9);
            v.s.asks.push(w);
            v.s.map_mirrors();
            v.submit(&[]);
            let pool = v.s.mapped[&w].pool;
            v.s.source_decommitted_in(c, TILE);
            assert_ne!(VirtualFree(c as *mut c_void, TILE, MEM_DECOMMIT), 0);
            assert_eq!((v.s.kept.get(&w), v.s.parked.get(&w)), (Some(&pool), None));
            assert_eq!((v.s.mirrored, v.s.counts.given_back, v.s.diag.kept), (0, 0, 1));
            v.submit(&[]);
            assert_eq!(v.s.kept.get(&w), Some(&pool), "no submit unmaps it");
            v.commit(0, 1);
            *((c + 5 * PAGE) as *mut u8) = 1;
            v.s.mirror_writes(&[c + 5 * PAGE], PAGE);
            assert_eq!((v.s.mapped[&w].pool, v.s.kept.get(&w)), (pool, None), "back from kept");
            assert_eq!((v.s.diag.back, v.s.diag.back_bytes), ([1, 0, 0, 0], TILE as u64), "away one submit");
            v.submit(&[c + 5 * PAGE]);
        }
    }

    /// `XCL_VIEW_POISON`: a tile of the window without video memory reads the pattern, one with
    /// it what was filled there, and one given back the pattern again once a submit unmaps it.
    #[test]
    fn a_tile_without_video_memory_reads_the_poison() {
        unsafe {
            let Some(mut v) = Views::new(193) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.s.poison_windows(0x7FC0_DEAD);
            assert!(v.s.poison.is_some(), "a poison tile");
            let buf = v.buf.clone();
            let poison: Vec<u8> = std::iter::repeat_n(0x7FC0_DEADu32.to_le_bytes(), 8).flatten().collect();
            assert_eq!(v.gpu_bytes(&buf, 0, 32), poison);
            assert_eq!(v.gpu_bytes(&buf, 192 * TILE + 0x40, 32), poison, "every tile");
            v.commit(0, 1);
            v.write(0, 1, 11);
            v.s.asks.push(w);
            v.s.map_mirrors();
            v.submit(&[]);
            assert_eq!(v.gpu_bytes(&buf, 0, 32), v.cpu_tile(0)[..32].to_vec(), "filled");
            assert_eq!(v.gpu_bytes(&buf, TILE, 32), poison, "the next tile still has none");
            v.s.source_decommitted_in(c, TILE);
            assert_ne!(VirtualFree(c as *mut c_void, TILE, MEM_DECOMMIT), 0);
            v.submit(&[]);
            assert_eq!(v.gpu_bytes(&buf, 0, 32), poison, "given back");
        }
    }

    /// A pool goes when all its tiles are free or cooled: one with a tile still cooling stays.
    #[test]
    fn a_pool_with_a_tile_still_cooling_is_not_retired() {
        let t0 = std::time::Instant::now();
        let ms = |n: u64| t0 + std::time::Duration::from_millis(n);
        // pool 1 free but its last tile, freed at 0 ms; pool 0 free but its first, freed at 1500 ms
        let mut free: Vec<(u32, u32)> = (0..POOL_TILES - 1).map(|t| (1, t)).collect();
        free.extend((1..POOL_TILES).map(|t| (0, t)));
        let cooling = VecDeque::from([((1, POOL_TILES - 1), ms(0)), ((0, 0), ms(1500))]);
        assert!(free_heaps(&free, &cooling, 2, ms(1999)).is_empty(), "their last tiles are cooling");
        assert_eq!(free_heaps(&free, &cooling, 2, ms(2000)), vec![1], "pool 1's cooled");
        assert_eq!(free_heaps(&free, &cooling, 2, ms(3500)), vec![0, 1]);
        assert_eq!(free_heaps(&free[1..], &cooling, 2, ms(3500)), vec![0], "a tile of pool 1 in use");
    }

    /// A GPU-view tile given back and unmapped at a submit is not the video memory of the next
    /// tile asked for: that gets another, and the tile is used again once it cooled. Before, it
    /// was the next tile taken, mapped, cleared and filled on the sparse queue while the title's
    /// work submitted before could still read it.
    #[test]
    fn a_tile_given_back_is_not_used_again_before_it_cooled() {
        unsafe {
            let Some(mut v) = Views::new(113) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 3);
            v.write(0, 3, 59);
            v.s.asks.push(w);
            v.submit(&[]);
            let pool = v.s.mapped[&w].pool;
            v.s.forget_mirrors_in(c, c + TILE);
            assert_eq!(v.s.parked.get(&w), Some(&pool));
            v.submit(&[]);
            // (unmapped at the submit, and quarantined until the GPU is past it)
            settle_quarantined(&mut v.s);
            assert_eq!(v.s.cooling.iter().map(|t| t.0).collect::<Vec<_>>(), vec![pool], "unmapped, cooling");
            v.s.asks.push(w + TILE);
            v.submit(&[]);
            assert_ne!(v.s.mapped[&(w + TILE)].pool, pool, "another tile");
            // (as 2 s later)
            for t in v.s.cooling.iter_mut() {
                t.1 = t.1.checked_sub(COOL).expect("an earlier instant");
            }
            v.s.asks.push(w + 2 * TILE);
            v.submit(&[]);
            assert_eq!(v.s.mapped[&(w + 2 * TILE)].pool, pool, "cooled: used again");
            assert!(v.s.cooling.is_empty());
        }
    }

    /// A submit that frees no tile tries to retire pools once a tile cooled or a heap waits to
    /// go, `COOL` after the last try at the earliest.
    #[test]
    fn a_submit_with_nothing_freed_retires_pools_once_tiles_cooled() {
        let t0 = std::time::Instant::now();
        let ms = |n: u64| t0 + std::time::Duration::from_millis(n);
        // the last tile of a pool freed at 0 ms, by a flush that tried at 0 ms
        let cooling = VecDeque::from([((0, 0), ms(0))]);
        assert!(!retire_due(&cooling, false, ms(0), ms(1999)), "still cooling");
        assert!(retire_due(&cooling, false, ms(0), ms(2000)), "cooled");
        assert!(!retire_due(&cooling, false, ms(1000), ms(2000)), "tried 1 s ago");
        assert!(!retire_due(&VecDeque::new(), false, ms(0), ms(5000)), "nothing freed, nothing retired");
        assert!(retire_due(&VecDeque::new(), true, ms(0), ms(2000)), "a retired heap to drop");
        assert!(!retire_due(&VecDeque::new(), true, ms(500), ms(2000)));
    }

    /// A pool whose last tile a submit freed goes at a submit `COOL` later that frees none.
    /// Before, only a submit that freed tiles retired pools, while those it freed still cooled:
    /// the pool's 64 MB stayed until the title gave back another tile.
    #[test]
    fn a_pool_goes_once_its_last_tile_cooled_with_nothing_parked() {
        unsafe {
            let Some(mut v) = Views::new(113) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 3);
            v.write(0, 3, 61);
            v.s.asks.push(w);
            v.submit(&[]);
            let pool = v.s.mapped[&w].pool;
            v.s.forget_mirrors_in(c, c + TILE);
            v.submit(&[]);
            // (unmapped at the submit, and quarantined until the GPU is past it)
            settle_quarantined(&mut v.s);
            assert_eq!(v.s.cooling.iter().map(|t| t.0).collect::<Vec<_>>(), vec![pool], "unmapped, cooling");
            assert!(v.s.pools[pool.0 as usize].is_some(), "its last tile is cooling");
            v.submit(&[]);
            assert!(v.s.pools[pool.0 as usize].is_some(), "less than COOL later");
            // (as 2 s later)
            for t in v.s.cooling.iter_mut() {
                t.1 = t.1.checked_sub(COOL).expect("an earlier instant");
            }
            v.s.retire_tried = v.s.retire_tried.checked_sub(COOL).expect("an earlier instant");
            assert!(v.s.parked.is_empty());
            v.submit(&[]);
            assert!(v.s.pools[pool.0 as usize].is_none(), "retired with nothing parked");
            assert!(v.s.cooling.is_empty() && !v.s.free.iter().any(|t| t.0 == pool.0));
            assert_eq!(v.s.retired.len(), 1);
        }
    }

    /// A submit fills tiles from their CPU views up to `fill_pages` committed pages (not tiles):
    /// the rest wait for the next submits in their order, and tiles asked for later wait behind
    /// them. Before, a submit took 4096 tiles at once (256 MiB hashed and uploaded on the title's
    /// thread with the state's lock held).
    #[test]
    fn a_submit_fills_a_bounded_amount_of_committed_memory_in_order() {
        unsafe {
            let Some(mut v) = Views::new(109) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 6);
            v.write(0, 6, 53);
            // four CPU tiles with one committed page each
            for k in 6..10 {
                assert!(!VirtualAlloc((c + k * TILE) as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
                *((c + k * TILE) as *mut u64) = k as u64;
            }
            v.s.fill_pages = 2 * PAGES_PER_TILE;
            let shown = |s: &State| (0..10).filter(|&k| s.mapped.contains_key(&(w + k * TILE))).collect::<Vec<_>>();
            v.s.asks.extend((0..6).map(|k| w + k * TILE));
            v.submit(&[]);
            assert_eq!(shown(&v.s), vec![0, 1], "two whole tiles a submit");
            assert_eq!(v.s.mirror_src.len(), 6, "the rest are linked: their CPU tiles' writes are watched");
            // asked for after: behind the first ones
            v.s.asks.extend((6..10).map(|k| w + k * TILE));
            v.submit(&[]);
            assert_eq!(shown(&v.s), vec![0, 1, 2, 3]);
            v.submit(&[]);
            assert_eq!(shown(&v.s), vec![0, 1, 2, 3, 4, 5]);
            v.submit(&[]);
            assert_eq!(shown(&v.s), (0..10).collect::<Vec<_>>(), "four pages: in one submit");
            assert!(v.s.later.is_empty() && v.s.asks.is_empty());
            assert_eq!(v.s.counts.fill_bytes, (6 * PAGES_PER_TILE + 4) as u64 * PAGE as u64);
        }
    }

    /// A copy back the GPU is done with lands before a submit's tiles change: a page that was a
    /// hole when it was recorded, committed and written in part since, does not take what the GPU
    /// had there. Before, the submit cleared the hole first, and the copy put the GPU's bytes into
    /// the page's zero chunks.
    #[test]
    fn copies_back_done_land_before_the_tiles_change() {
        unsafe {
            let Some(mut v) = Views::new(89) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            assert!(!VirtualAlloc(c as *const c_void, 8 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            v.s.asks.push(w);
            v.submit(&[]);
            assert_eq!(v.s.mapped[&w].holes, 0xFF00);
            // a copy back of the whole tile, done on the GPU
            let mut job = v.s.rb_job(TILE, D3D12_COMMAND_LIST_TYPE_DIRECT).expect("readback job");
            std::ptr::write_bytes(job.ptr, 0xAB, TILE);
            job.copies.push((w, 0, TILE, v.s.mapped[&w].gen));
            let fence = || v.dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            job.fences = Some((fence(), fence()));
            v.s.job_seq += 1;
            job.seq = v.s.job_seq;
            v.s.rb_jobs.push_back(job);
            RB_PENDING.fetch_add(1, Ordering::Release);
            // a page committed and written in part since
            let p = c + 9 * PAGE;
            assert!(!VirtualAlloc(p as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            *(p as *mut u64) = 0x600D;
            v.submit(&[p]);
            assert!(v.s.rb_jobs.is_empty(), "the copy landed");
            assert_eq!(*((c + 0x40) as *const u8), 0xAB, "in the pages committed then");
            assert_eq!(v.s.mapped[&w].holes, 0xFF00 & !(1 << 9));
            assert_eq!(*(p as *const u64), 0x600D);
            assert!(std::slice::from_raw_parts((p + 8) as *const u8, PAGE - 8).iter().all(|&b| b == 0), "not in the page committed since");
        }
    }
}
