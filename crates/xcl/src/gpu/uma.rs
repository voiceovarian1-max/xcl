//! Console unified memory (UMA). On the console CPU and GPU share memory and shaders reach
//! it through plain addresses in SGPRs. XCL keeps the title's graphics memory in regions
//! that D3D12 can open as heaps (`ID3D12Device3::OpenExistingHeapFromAddress`), so the GPU
//! works on them directly.
//!
//! A heap is smaller than 4 GiB, so memory is split into windows at `base + k · 4 GiB`.
//! The aperture has 4 windows by default and can grow to 8 or 16 with heap-backed shaders
//! (`XCL_HEAP_WINDOWS=1` + `XCL_UMA_WINDOWS=8|16`). The base is aligned to the aperture size,
//! so shaders compute the window index from `(address >> 32) - (base >> 32)` and the byte
//! offset from the low 32 bits. Allocations never straddle windows. With 16 windows, 6 of them
//! hold GPU views of the title's CPU memory (`Uma::view_reserve`), memory only the GPU reaches.
//!
//! Measured on Windows 11 + RX 9070 (2026-11-20): the heap must cover a fully committed
//! `VirtualAlloc` allocation (a reservation with partial commit is rejected); up to
//! 4095 MiB works (exactly 4 GiB is rejected), opening 4 GiB takes ~0.5 s.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE};

/// VirtualAlloc: track CPU writes (GetWriteWatch)
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// allocation granularity (as VirtualAlloc)
pub const GRAIN: usize = 0x1_0000;
/// largest region D3D12 opens as a heap
pub const MAX_SIZE: usize = (4096 << 20) - GRAIN;
/// distance between windows
pub const WINDOW: usize = 1 << 32;
/// Legacy layout and the number of memory UAVs in the ABI root signature.
pub const MAX_WINDOWS: usize = 4;
/// Maximum number of windows addressable through the shared descriptor heap (`XCL_HEAP_WINDOWS=1`).
pub const MAX_HEAP_WINDOWS: usize = 16;

/// Maximum sparse VRAM windows reserved in runtime layouts (3 × 4 GiB = 12 GiB VA).
/// Keeping sparse VA capped at 12 GiB prevents huge startup `MEM_64K_PAGES` title heaps from
/// being pulled into sparse VRAM (which causes multi-minute tile-mapping stalls and multi-GB
/// sparse pool bloat), while still giving expanded 8/16-window layouts extra coherent windows.
const MAX_RUNTIME_SPARSE_WINDOWS: usize = 3;

/// View windows of the 16-window layout (`WindowKind::View`, 24 GiB): GPU views of the title's
/// CPU memory go there (`Uma::view_reserve`). A retail title reserves 24 GiB of 64 KiB pages as
/// a CPU view and the same again as a GPU-only view of it, and gives the GPU only addresses of
/// the second; outside the windows every shader read of it is 0.
pub const VIEW_WINDOWS: usize = 6;

/// View windows of a runtime layout of `windows` windows (none below 16: no room for them).
fn view_windows(windows: usize) -> usize {
    if windows == MAX_HEAP_WINDOWS { VIEW_WINDOWS } else { 0 }
}

/// Maximum number of committed coherent windows in an expanded runtime layout.
/// Always leaves 3 sparse VRAM windows (12 GiB) and the view windows: 1 coherent for 4-window,
/// 5 for 8-window, 7 for 16-window.
fn max_coherent_windows(windows: usize) -> usize {
    if windows > MAX_WINDOWS {
        windows.saturating_sub(MAX_RUNTIME_SPARSE_WINDOWS + view_windows(windows)).max(1)
    } else {
        1
    }
}

/// Maximum total committed coherent bytes for a runtime layout of `windows` windows.
fn max_coherent_bytes(windows: usize) -> usize {
    max_coherent_windows(windows) * MAX_SIZE
}

/// Split `total_coh` coherent bytes across `1..=max_coherent_windows(windows)` committed windows.
fn coherent_window_sizes(size: usize, coherent: usize, windows: usize) -> Vec<usize> {
    let max_coh = max_coherent_windows(windows);
    let total_coh = size
        .min(coherent.max(RUNTIME_COMMITTED_CAP))
        .clamp(GRAIN, max_coh * MAX_SIZE);
    let coh_wins = total_coh.div_ceil(MAX_SIZE).clamp(1, max_coh);
    let mut coh_sizes = Vec::with_capacity(coh_wins);
    let mut rem = total_coh;
    for _ in 0..coh_wins {
        let chunk = ((rem.min(MAX_SIZE) + GRAIN - 1) & !(GRAIN - 1)).clamp(GRAIN, MAX_SIZE);
        coh_sizes.push(chunk);
        rem = rem.saturating_sub(chunk);
    }
    coh_sizes
}

/// Exclusive upper bound for candidate 4 GiB window slot indices.
/// Expanded apertures stay below the 40-bit console GPU VA limit (`1 TiB`).
fn slot_limit(windows: usize) -> usize {
    if windows > MAX_WINDOWS { 1 << (40 - 32) } else { 512 }
}

/// Requested runtime UMA window count (`XCL_UMA_WINDOWS=4|8|16`, default 4).
/// Layouts wider than 4 windows require `XCL_HEAP_WINDOWS=1`.
pub(crate) fn configured_runtime_windows() -> usize {
    let Ok(val) = std::env::var("XCL_UMA_WINDOWS") else { return MAX_WINDOWS };
    let n = val.trim().parse::<usize>().unwrap_or(0);
    if !matches!(n, MAX_WINDOWS | 8 | MAX_HEAP_WINDOWS) {
        crate::xlog_warn!("UMA: XCL_UMA_WINDOWS={val:?} must be 4, 8, or 16 – using {MAX_WINDOWS}");
        return MAX_WINDOWS;
    }
    if n > MAX_WINDOWS && !crate::gpu::recomp::pso::heap_windows() {
        crate::xlog_warn!("UMA: XCL_UMA_WINDOWS={n} requires XCL_HEAP_WINDOWS=1 – using {MAX_WINDOWS}");
        return MAX_WINDOWS;
    }
    n
}

/// bytes of sparse window 0 only 32-bit descriptor heaps get (`Uma::desc_range`)
const DESC_HEADROOM: usize = 512 << 20;
/// cap on the eager committed coherent window in `for_runtime` (per-draw args, readback buffers)
const RUNTIME_COMMITTED_CAP: usize = 512 << 20;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WindowKind {
    None,
    Committed(usize),
    Sparse,
    /// GPU views of the title's CPU memory (`view_reserve`): reserved PAGE_NOACCESS, never
    /// committed nor write-watched – the CPU never touches them. The GPU reaches them through
    /// the window's tiled buffer (`d3d12x::sparse`), whose tiles get video memory on demand.
    View,
}

pub struct Uma {
    base: usize,
    kinds: Vec<WindowKind>,
    /// VirtualAlloc base pointers owned by this Uma (freed on Drop)
    win_allocs: Vec<usize>,
    inner: Mutex<Inner>,
    /// free ranges of the sparse windows (address → length) and the reservations in them
    sparse_free: Mutex<(BTreeMap<usize, usize>, BTreeMap<usize, usize>)>,
    /// the same for the view windows (`view_reserve`)
    views: Mutex<(BTreeMap<usize, usize>, BTreeMap<usize, usize>)>,
    /// the committed windows are not the GPU's memory: it has copies of them (`set_shadowed`)
    shadowed: AtomicBool,
    /// XCL's own pool in window 0 for `Arena` blocks: (start, end, free blocks). The title
    /// cannot take it – a title that commits all of unified memory at start (Unicorn
    /// Overlord: 2046 of 2048 MB) left no room for draw arguments, and draws were skipped.
    pool: Mutex<(usize, usize, Vec<usize>)>,
    /// windows remade as driver heaps (`hle::d3d12x::uma_driver`), one bit each
    driver: std::sync::atomic::AtomicU32,
}

/// bytes of XCL's arena pool (`Uma::new`)
pub const POOL: usize = 64 << 20;

#[derive(Default)]
struct Inner {
    /// free ranges: offset → length (coalesced)
    free: BTreeMap<usize, usize>,
    /// live allocations: offset → length
    used: BTreeMap<usize, usize>,
}

/// Carve an `align`-aligned block of `size` bytes from `free` within `[lo_bound, hi_bound)`.
/// When `single_window` is true and `size <= WINDOW`, the returned block never straddles a
/// 4 GiB window boundary.
fn take_sparse_in(
    g: &mut (BTreeMap<usize, usize>, BTreeMap<usize, usize>),
    lo_bound: usize,
    hi_bound: usize,
    size: usize,
    align: usize,
    single_window: bool,
) -> Option<usize> {
    if lo_bound >= hi_bound {
        return None;
    }
    let first_fit = |(&o, &l): (&usize, &usize)| {
        let lo = o.max(lo_bound);
        let hi = o.saturating_add(l).min(hi_bound);
        if lo >= hi {
            return None;
        }
        let mut at = (lo.checked_add(align - 1)?) & !(align - 1);
        while at.checked_add(size).is_some_and(|e| e <= hi) {
            if !single_window || size > WINDOW || at / WINDOW == (at + size - 1) / WINDOW {
                return Some((o, l, at));
            }
            let next_win = (at / WINDOW + 1).checked_mul(WINDOW)?;
            at = (next_win.checked_add(align - 1)?) & !(align - 1);
        }
        None
    };
    let first = g.0.first_key_value()?;
    let found = first_fit(first).or_else(|| {
        // Disjoint free ranges mean only the immediate predecessor can cross lo_bound.
        // Check it first, then search the starts inside the bounds in address order.
        g.0.range(..lo_bound).next_back().and_then(first_fit)
            .or_else(|| g.0.range(lo_bound..hi_bound).find_map(first_fit))
    });
    let (start, len, at) = found?;
    g.0.remove(&start);
    if at > start {
        g.0.insert(start, at - start);
    }
    if at + size < start + len {
        g.0.insert(at + size, start + len - (at + size));
    }
    g.1.insert(at, size);
    Some(at)
}

/// The reservation of `take_sparse_in` at `p` goes back to the free ranges (merged with its
/// neighbours): its bytes.
fn give_back(g: &mut (BTreeMap<usize, usize>, BTreeMap<usize, usize>), p: usize) -> Option<usize> {
    let size = g.1.remove(&p)?;
    let (mut start, mut len) = (p, size);
    if let Some((&o, &l)) = g.0.range(..p).next_back() {
        if o + l == p {
            g.0.remove(&o);
            start = o;
            len += l;
        }
    }
    if let Some(l) = g.0.remove(&(p + size)) {
        len += l;
    }
    g.0.insert(start, len);
    Some(size)
}

/// The reservation of `take_sparse_in` holding `p`: (start, bytes).
fn reservation_at(g: &(BTreeMap<usize, usize>, BTreeMap<usize, usize>), p: usize) -> Option<(usize, usize)> {
    g.1.range(..=p).next_back().map(|(&a, &l)| (a, l)).filter(|&(a, l)| p < a + l)
}

/// `VirtualAlloc` of `len` bytes exactly at `at` (false: Windows has something there, or put
/// it elsewhere – then nothing stays reserved).
fn reserve_exactly(at: usize, len: usize, flags: u32, protect: u32) -> bool {
    let p = unsafe { VirtualAlloc(at as *const std::ffi::c_void, len, flags, protect) };
    if p as usize != at {
        if !p.is_null() {
            unsafe { VirtualFree(p, 0, MEM_RELEASE) };
        }
        return false;
    }
    true
}

impl Uma {
    /// Commit `size` bytes (split into as few equal windows as fit, at most 4) at the first
    /// free 16 GiB-aligned address from 16 GiB up, with remaining windows reserved as sparse.
    pub fn new(size: usize) -> Option<Uma> {
        let size = ((size + GRAIN - 1) & !(GRAIN - 1)).max(GRAIN);
        let n = size.div_ceil(MAX_SIZE).clamp(1, MAX_WINDOWS);
        let mut u = Uma::with_windows(n, (size.div_ceil(n) + GRAIN - 1) & !(GRAIN - 1))?;
        u.reserve_sparse_windows();
        u.reserve_pool(POOL);
        Some(u)
    }

    /// XCL's arena pool: `bytes` at the end of the first committed window taken now, before
    /// the title can (a quarter of that window at most); the title's allocations still start
    /// at the window's start.
    pub fn reserve_pool(&self, bytes: usize) {
        let Some(win) = self.kinds.iter().position(|w| matches!(w, WindowKind::Committed(_))) else { return };
        let win_size = self.window(win).1;
        let bytes = (bytes.min(win_size / 4) / ARENA_BLOCK) * ARENA_BLOCK;
        if bytes == 0 {
            return;
        }
        let at = {
            let mut g = self.inner.lock().unwrap();
            // the free range of the committed window that reaches furthest, its last `bytes`
            let Some((start, len)) = g.free.iter().filter(|(&o, &l)| o / WINDOW == win && l >= bytes).max_by_key(|(&o, &l)| o + l).map(|(&o, &l)| (o, l)) else { return };
            let at = (start + len - bytes) & !(GRAIN - 1);
            if at < start {
                return;
            }
            g.free.remove(&start);
            if at > start {
                g.free.insert(start, at - start);
            }
            if at + bytes < start + len {
                g.free.insert(at + bytes, start + len - at - bytes);
            }
            g.used.insert(at, bytes);
            at
        };
        let p = self.base + at;
        unsafe { std::ptr::write_bytes(p as *mut u8, 0, bytes) };
        *self.pool.lock().unwrap() = (p, p + bytes, (p..p + bytes).step_by(ARENA_BLOCK).rev().collect());
    }

    /// Diagnostic: how the committed window stands: free bytes in it in total, the largest free
    /// run, the bytes in use, and XCL's pool blocks left.
    pub fn committed_state(&self) -> (usize, usize, usize, usize) {
        let Some(win) = self.kinds.iter().position(|w| matches!(w, WindowKind::Committed(_))) else { return (0, 0, 0, 0) };
        let g = self.inner.lock().unwrap();
        let mine = |o: usize| o / WINDOW == win;
        let (mut free, mut largest) = (0usize, 0usize);
        for (_, &l) in g.free.iter().filter(|(&o, _)| mine(o)) {
            free += l;
            largest = largest.max(l);
        }
        let used = g.used.iter().filter(|(&o, _)| mine(o)).map(|(_, &l)| l).sum();
        drop(g);
        (free, largest, used, self.pool.lock().unwrap().2.len())
    }

    /// A block of `ARENA_BLOCK` bytes from XCL's pool (None: the pool is used up or absent).
    fn pool_block(&self) -> Option<usize> {
        self.pool.lock().unwrap().2.pop()
    }

    /// Runtime layout: windows 0..3 (12 GiB) reserved as write-watched sparse VRAM windows
    /// (window 0 holds 32-bit-addressed descriptor heaps + buffers + XMem, windows 1..2 hold
    /// large title reservations and overflow), and window 3 as a smaller committed coherent
    /// window (`OpenExistingHeapFromAddress`) for per-draw arguments and CPU readback buffers.
    pub fn for_runtime(size: usize) -> Option<Uma> {
        Self::for_runtime_with(size, RUNTIME_COMMITTED_CAP)
    }

    /// `for_runtime` with a coherent window of up to `coherent` bytes (`coherent_hint`).
    pub fn for_runtime_with(size: usize, coherent: usize) -> Option<Uma> {
        Self::reserve_runtime(size, coherent, MAX_WINDOWS).or_else(|| Self::new(size))
    }

    /// `for_runtime_with` using `windows` total shader-visible windows: up to 3 sparse VRAM
    /// windows followed by `coh_wins` committed coherent windows sized to hold `coherent` bytes
    /// (up to `max_coherent_windows(windows)`), then, with 16, `VIEW_WINDOWS` view windows.
    pub fn for_runtime_windows(size: usize, coherent: usize, windows: usize) -> Option<Uma> {
        let windows = if matches!(windows, MAX_WINDOWS | 8 | MAX_HEAP_WINDOWS) { windows } else { MAX_WINDOWS };
        Self::reserve_runtime(size, coherent, windows)
    }

    /// The layout: sparse windows (at most `MAX_RUNTIME_SPARSE_WINDOWS`), then the coherent
    /// windows, then (16 windows only) the view windows. With view windows the whole aperture
    /// is reserved – the coherent windows' uncommitted tails and the windows after the views
    /// too, PAGE_NOACCESS – so Windows never places anything at an address shaders read as a
    /// window (it read the memory of whatever was there, or of window 0).
    fn reserve_runtime(size: usize, coherent: usize, windows: usize) -> Option<Uma> {
        let coh_sizes = coherent_window_sizes(size, coherent, windows);
        let coh_wins = coh_sizes.len();
        let sparse_wins = (windows - coh_wins).min(MAX_RUNTIME_SPARSE_WINDOWS);
        let sparse_len = sparse_wins * WINDOW;
        let view_wins = view_windows(windows);
        let views_at = sparse_wins + coh_wins;
        let whole = view_wins > 0;
        if views_at + view_wins > windows {
            return None;
        }
        for slot in (windows..slot_limit(windows)).step_by(windows) {
            let base = slot << 32;
            if !reserve_exactly(base, sparse_len, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) {
                continue;
            }
            let mut win_allocs = Vec::with_capacity(4 + 2 * coh_wins);
            win_allocs.push(base);
            let mut ok = true;
            for (c, &c_size) in coh_sizes.iter().enumerate() {
                let c_base = base + (sparse_wins + c) * WINDOW;
                if !reserve_exactly(c_base, c_size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) {
                    ok = false;
                    break;
                }
                win_allocs.push(c_base);
                if whole && c_size < WINDOW {
                    if !reserve_exactly(c_base + c_size, WINDOW - c_size, MEM_RESERVE, PAGE_NOACCESS) {
                        ok = false;
                        break;
                    }
                    win_allocs.push(c_base + c_size);
                }
            }
            if ok && whole {
                // the view windows, then the rest of the aperture: one reservation each
                for (lo, hi) in [(views_at, views_at + view_wins), (views_at + view_wins, windows)] {
                    if lo == hi {
                        continue;
                    }
                    if !reserve_exactly(base + lo * WINDOW, (hi - lo) * WINDOW, MEM_RESERVE, PAGE_NOACCESS) {
                        ok = false;
                        break;
                    }
                    win_allocs.push(base + lo * WINDOW);
                }
            }
            if !ok {
                for p in win_allocs {
                    unsafe { VirtualFree(p as *mut std::ffi::c_void, 0, MEM_RELEASE) };
                }
                continue;
            }
            let mut inner = Inner::default();
            let mut kinds = vec![WindowKind::None; windows];
            kinds[..sparse_wins].fill(WindowKind::Sparse);
            for (c, &c_size) in coh_sizes.iter().enumerate() {
                let k = sparse_wins + c;
                inner.free.insert(k * WINDOW, c_size);
                kinds[k] = WindowKind::Committed(c_size);
            }
            kinds[views_at..views_at + view_wins].fill(WindowKind::View);
            let mut sparse_free: (BTreeMap<usize, usize>, BTreeMap<usize, usize>) = Default::default();
            sparse_free.0.insert(base, sparse_len);
            let mut views: (BTreeMap<usize, usize>, BTreeMap<usize, usize>) = Default::default();
            if view_wins > 0 {
                views.0.insert(base + views_at * WINDOW, view_wins * WINDOW);
            }
            let u = Uma {
                base,
                kinds,
                win_allocs,
                inner: Mutex::new(inner),
                sparse_free: Mutex::new(sparse_free),
                views: Mutex::new(views),
                shadowed: AtomicBool::new(false),
                pool: Mutex::new((0, 0, Vec::new())),
                driver: Default::default(),
            };
            u.reserve_pool(POOL);
            return Some(u);
        }
        None
    }

    /// The windows after the committed ones as one write-watched reservation (sparse windows).
    fn reserve_sparse_windows(&mut self) {
        let committed = self.windows();
        if committed >= self.kinds.len() {
            return;
        }
        let start = self.base + committed * WINDOW;
        let len = (self.kinds.len() - committed) * WINDOW;
        let p = unsafe { VirtualAlloc(start as *const std::ffi::c_void, len, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) };
        if p as usize == start {
            self.kinds[committed..].fill(WindowKind::Sparse);
            self.win_allocs.push(start);
            self.sparse_free.lock().unwrap().0.insert(start, len);
        } else if !p.is_null() {
            unsafe { VirtualFree(p, 0, MEM_RELEASE) };
        }
    }

    /// The range of sparse window 0 kept for 32-bit descriptor heaps (`sparse_alloc_in(0)`):
    /// after the window's first 64 KiB, which stay unused (a console address of 0 reads as
    /// null). The title's memory never takes it, and it sits at the start of the window so the
    /// title still gets one contiguous range of 11.5 GiB: XDefiant reserves 8 GiB + 4 MiB first
    /// thing, more than windows 1..2 hold, and with the range at the top of window 0 that
    /// reservation took all of it – no descriptor heap could be made, every frame was black.
    fn desc_range(&self) -> Option<(usize, usize)> {
        self.is_sparse_window(0).then(|| (self.base + GRAIN, self.base + GRAIN + DESC_HEADROOM))
    }

    /// The first sparse address the title's memory may use (after `desc_range`).
    fn title_sparse_start(&self) -> usize {
        self.desc_range().map_or(self.base, |(_, end)| end)
    }

    /// The end of the last sparse window: reservations and allocations in the sparse windows
    /// stay below it, never in the view windows (a CPU view of 24 GiB there would be tiled
    /// video memory again).
    fn sparse_end(&self) -> usize {
        self.base + self.kinds.iter().rposition(|w| matches!(w, WindowKind::Sparse)).map_or(0, |k| k + 1) * WINDOW
    }

    /// Whether window `k` is a sparse VRAM window.
    pub fn is_sparse_window(&self, k: usize) -> bool {
        self.kinds.get(k).is_some_and(|w| matches!(w, WindowKind::Sparse))
    }

    /// Whether window `k` is a committed coherent window.
    pub fn is_committed_window(&self, k: usize) -> bool {
        self.kinds.get(k).is_some_and(|w| matches!(w, WindowKind::Committed(_)))
    }

    /// Whether window `k` is a view window (GPU views of the title's CPU memory, `view_reserve`).
    pub fn is_view_window(&self, k: usize) -> bool {
        self.kinds.get(k).is_some_and(|w| matches!(w, WindowKind::View))
    }

    /// Whether `p` is in a view window, reserved for a GPU view or not. Not memory the CPU can
    /// touch: `contains`, `is_sparse` and `is_committed` are false there.
    pub fn is_view(&self, p: usize) -> bool {
        self.window_of(p).is_some_and(|k| self.is_view_window(k))
    }

    /// Address space for a GPU view of `size` bytes in the view windows, starting at a window's
    /// start (so a view of 24 GiB fits the 6 windows whole). None: no view windows (layouts
    /// below 16 windows), or no room left.
    pub fn view_reserve(&self, size: usize) -> Option<usize> {
        let lo = self.kinds.iter().position(|w| matches!(w, WindowKind::View))?;
        let hi = self.kinds.iter().rposition(|w| matches!(w, WindowKind::View))? + 1;
        let size = size.max(1).checked_next_multiple_of(GRAIN)?;
        take_sparse_in(&mut self.views.lock().unwrap(), self.base + lo * WINDOW, self.base + hi * WINDOW, size, WINDOW, false)
    }

    /// The bytes of the view reservation starting at `p` (it is gone after this).
    pub fn view_release(&self, p: usize) -> Option<usize> {
        give_back(&mut self.views.lock().unwrap(), p)
    }

    /// The view reservation holding `p`: (start, bytes).
    pub fn view_reservation(&self, p: usize) -> Option<(usize, usize)> {
        reservation_at(&self.views.lock().unwrap(), p)
    }

    /// Whether this Uma has any sparse VRAM windows.
    pub fn has_sparse(&self) -> bool {
        self.kinds.iter().any(|w| matches!(w, WindowKind::Sparse))
    }

    /// Windows translated shaders address (`mem`, `mem1` …): full configured aperture for
    /// sparse runtime layouts (keeping shader cache keys aligned with `XCL_UMA_WINDOWS=8|16`
    /// even when trailing slots are unallocated), or highest active window + 1 otherwise.
    pub fn shader_windows(&self) -> usize {
        if self.has_sparse() {
            self.kinds.len()
        } else {
            self.kinds.iter().rposition(|w| !matches!(w, WindowKind::None)).map_or(0, |i| i + 1)
        }
    }

    /// Whether `p` is in a sparse window.
    pub fn is_sparse(&self, p: usize) -> bool {
        self.window_of(p).is_some_and(|k| self.is_sparse_window(k))
    }

    /// Whether `p` is in a committed window.
    pub fn is_committed(&self, p: usize) -> bool {
        self.window_of(p).is_some_and(|k| self.is_committed_window(k))
    }

    /// Window `k` is a driver heap now (`hle::d3d12x::uma_driver`).
    pub fn set_driver_window(&self, k: usize) {
        self.driver.fetch_or(1 << k, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether window `k` is a driver heap (`set_driver_window`).
    pub fn is_driver_window(&self, k: usize) -> bool {
        k < 32 && self.driver.load(std::sync::atomic::Ordering::Relaxed) & (1 << k) != 0
    }

    /// Whether `[p, p + len)` lies in a committed coherent window – memory that is committed and
    /// read-write as a whole, so it needs no `VirtualQuery` (which walks a region to its end to
    /// size it: on a 4 GiB window ~40 ms a call, a retail game's DirectStorage requests crawled).
    pub fn in_committed_window(&self, p: usize, len: usize) -> bool {
        len > 0 && self.is_committed(p) && p.checked_add(len - 1).is_some_and(|e| self.window_of(e) == self.window_of(p))
    }

    /// The GPU could not open the committed windows as heaps (`OpenExistingHeapFromAddress`
    /// not implemented: D3DMetal): each has a buffer in video memory instead, a copy that
    /// `d3d12x::sparse` keeps in step like a sparse window's tiles.
    pub fn set_shadowed(&self) {
        self.shadowed.store(true, Ordering::Release);
    }

    pub fn shadowed(&self) -> bool {
        self.shadowed.load(Ordering::Acquire)
    }

    /// Whether the GPU works on a copy of the memory at `p` (a sparse window, or a committed
    /// one that is shadowed): what the GPU writes there the CPU sees only after a readback.
    pub fn is_gpu_copy(&self, p: usize) -> bool {
        self.is_sparse(p) || (self.shadowed() && self.is_committed(p))
    }

    /// Address space for a title reservation of `size` bytes in the sparse windows (64 KiB
    /// aligned, first fit; not committed). Prefers windows 1+ so window 0 stays uncluttered for
    /// small allocations, falling back to all of the title's sparse space (never `desc_range`).
    pub fn sparse_reserve(&self, size: usize) -> Option<usize> {
        if !self.has_sparse() {
            return None;
        }
        let size = (size.max(1) + GRAIN - 1) & !(GRAIN - 1);
        let mut g = self.sparse_free.lock().unwrap();
        let end = self.sparse_end();
        let lo = self.title_sparse_start();
        if let Some(at) = take_sparse_in(&mut g, (self.base + WINDOW).max(lo), end, size, GRAIN, false) {
            return Some(at);
        }
        take_sparse_in(&mut g, lo, end, size, GRAIN, false)
    }

    /// Reserve an aligned range strictly inside sparse window `k` (window 0: 32-bit descriptor
    /// heaps, from `desc_range` first, then whatever of the window is free). Returns
    /// `(address, rounded_size)`.
    pub fn sparse_alloc_in(&self, k: usize, size: usize, align: usize) -> Option<(usize, usize)> {
        if !self.is_sparse_window(k) {
            return None;
        }
        let size = (size.max(1) + GRAIN - 1) & !(GRAIN - 1);
        let align = align.max(GRAIN).next_power_of_two();
        let (w_start, w_len) = self.window(k);
        let mut g = self.sparse_free.lock().unwrap();
        let desc = if k == 0 { self.desc_range() } else { None };
        if let Some((lo, hi)) = desc {
            if let Some(at) = take_sparse_in(&mut g, lo, hi, size, align, true) {
                return Some((at, size));
            }
        }
        let lo = desc.map_or(w_start, |(lo, _)| lo);
        take_sparse_in(&mut g, lo, w_start + w_len, size, align, true).map(|at| (at, size))
    }

    /// Reserve an aligned range in the sparse windows that never straddles a 4 GiB window
    /// boundary (when `size <= WINDOW`): window 0 first, then windows 1+, never `desc_range`.
    /// Returns `(address, rounded_size)`.
    pub fn sparse_alloc(&self, size: usize, align: usize) -> Option<(usize, usize)> {
        if !self.has_sparse() {
            return None;
        }
        let size = size.max(1).checked_next_multiple_of(GRAIN)?;
        let align = align.max(GRAIN).checked_next_power_of_two()?;
        let mut g = self.sparse_free.lock().unwrap();
        let end = self.sparse_end();
        let lo = self.title_sparse_start();
        let above_0 = (self.base + WINDOW).max(lo);
        if size <= WINDOW {
            if self.is_sparse_window(0) {
                if let Some(at) = take_sparse_in(&mut g, lo, self.base + WINDOW, size, align, true) {
                    return Some((at, size));
                }
            }
            take_sparse_in(&mut g, above_0, end, size, align, true).map(|at| (at, size))
        } else {
            take_sparse_in(&mut g, above_0, end, size, align, false)
                .or_else(|| take_sparse_in(&mut g, lo, end, size, align, false))
                .map(|at| (at, size))
        }
    }

    /// `sparse_alloc` for memory shaders never address with 32 bits: windows 1+ first (as
    /// `sparse_reserve`), window 0 only when they have no room – it stands for the console's low
    /// 4 GiB, where a title's own descriptor heaps must lie (`console_heap::title_memory`).
    /// Above a window the same as `sparse_alloc`.
    pub fn sparse_alloc_above(&self, size: usize, align: usize) -> Option<(usize, usize)> {
        if size <= WINDOW && self.has_sparse() {
            let size = size.max(1).checked_next_multiple_of(GRAIN)?;
            let align = align.max(GRAIN).checked_next_power_of_two()?;
            let above_0 = (self.base + WINDOW).max(self.title_sparse_start());
            if let Some(at) = take_sparse_in(&mut self.sparse_free.lock().unwrap(), above_0, self.sparse_end(), size, align, true) {
                return Some((at, size));
            }
        }
        self.sparse_alloc(size, align)
    }

    /// Diagnostic: the free bytes of sparse window `k` the title may take (not `desc_range`), in
    /// all and the largest run.
    pub fn sparse_free_in(&self, k: usize) -> (usize, usize) {
        let (lo, hi) = (self.base + k * WINDOW, self.base + (k + 1) * WINDOW);
        let lo = if k == 0 { lo.max(self.title_sparse_start()) } else { lo };
        let g = self.sparse_free.lock().unwrap();
        let runs = g.0.iter().map(|(&o, &l)| (o.max(lo), (o + l).min(hi))).filter(|&(a, e)| a < e).map(|(a, e)| e - a);
        runs.fold((0, 0), |(all, big), n| (all + n, big.max(n)))
    }

    /// The bytes of the sparse reservation starting at `p` (it is gone after this).
    pub fn sparse_release(&self, p: usize) -> Option<usize> {
        give_back(&mut self.sparse_free.lock().unwrap(), p)
    }

    /// The sparse reservation holding `p`: (start, bytes).
    pub fn sparse_reservation(&self, p: usize) -> Option<(usize, usize)> {
        reservation_at(&self.sparse_free.lock().unwrap(), p)
    }

    /// Bytes currently reserved/allocated in sparse windows.
    pub fn sparse_used(&self) -> usize {
        self.sparse_free.lock().unwrap().1.values().sum()
    }

    /// `windows` committed windows of `size` bytes each.
    pub fn with_windows(windows: usize, size: usize) -> Option<Uma> {
        let size = ((size + GRAIN - 1) & !(GRAIN - 1)).clamp(GRAIN, MAX_SIZE);
        let windows = windows.clamp(1, MAX_HEAP_WINDOWS);
        let aperture = windows.next_power_of_two().max(MAX_WINDOWS);
        for slot in (aperture..slot_limit(aperture)).step_by(aperture) {
            let base = slot << 32;
            let mut got = Vec::new();
            for k in 0..windows {
                let want = base + k * WINDOW;
                // write-watched: textures the title keeps here are uploaded again after CPU writes
                let p = unsafe { VirtualAlloc(want as *const std::ffi::c_void, size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) };
                if p.is_null() {
                    break;
                }
                got.push(p);
            }
            if got.len() == windows {
                let mut inner = Inner::default();
                let mut kinds = vec![WindowKind::None; aperture];
                let mut win_allocs = Vec::with_capacity(windows + 1);
                for k in 0..windows {
                    inner.free.insert(k * WINDOW, size);
                    kinds[k] = WindowKind::Committed(size);
                    win_allocs.push(got[k] as usize);
                }
                return Some(Uma {
                    base,
                    kinds,
                    win_allocs,
                    inner: Mutex::new(inner),
                    sparse_free: Mutex::new(Default::default()),
                    views: Mutex::new(Default::default()),
                    shadowed: AtomicBool::new(false),
                    pool: Mutex::new((0, 0, Vec::new())),
                    driver: Default::default(),
                });
            }
            for p in got {
                unsafe { VirtualFree(p, 0, MEM_RELEASE) };
            }
        }
        None
    }

    pub fn base(&self) -> usize {
        self.base
    }

    /// usable bytes across all committed windows
    pub fn size(&self) -> usize {
        self.kinds
            .iter()
            .map(|w| match *w {
                WindowKind::Committed(s) => s,
                _ => 0,
            })
            .sum()
    }

    /// number of committed windows
    pub fn windows(&self) -> usize {
        self.kinds.iter().filter(|w| matches!(w, WindowKind::Committed(_))).count()
    }

    /// (address, bytes) of window `k` (a sparse or view window: all 4 GiB)
    pub fn window(&self, k: usize) -> (usize, usize) {
        let len = match self.kinds.get(k).copied().unwrap_or(WindowKind::None) {
            WindowKind::Committed(s) => s,
            WindowKind::Sparse | WindowKind::View => WINDOW,
            WindowKind::None => 0,
        };
        (self.base + k * WINDOW, len)
    }

    /// Underlying write-watched `VirtualAlloc` regions `(base, size)` for `GetWriteWatch`:
    /// contiguous sparse windows belong to a single reservation, while each committed window
    /// is its own allocation (view windows are not write-watched: the CPU never writes there).
    pub fn watch_regions(&self) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = Vec::with_capacity(self.kinds.len());
        let mut k = 0;
        while k < self.kinds.len() {
            match self.kinds[k] {
                WindowKind::None | WindowKind::View => k += 1,
                WindowKind::Committed(s) => {
                    out.push((self.base + k * WINDOW, s));
                    k += 1;
                }
                WindowKind::Sparse => {
                    let start = self.base + k * WINDOW;
                    let mut len = 0usize;
                    while k < self.kinds.len() && matches!(self.kinds[k], WindowKind::Sparse) {
                        len += WINDOW;
                        k += 1;
                    }
                    out.push((start, len));
                }
            }
        }
        out
    }

    pub fn window_of(&self, p: usize) -> Option<usize> {
        let k = p.checked_sub(self.base)? / WINDOW;
        let (start, len) = self.window(k);
        (k < self.kinds.len() && len > 0 && p - start < len).then_some(k)
    }

    /// Whether `p` is unified memory the CPU can address (a committed or sparse window; never
    /// a view window, which only the GPU reaches).
    pub fn contains(&self, p: usize) -> bool {
        self.window_of(p).is_some_and(|k| !self.is_view_window(k))
    }

    /// Whether `p` is memory the title has here: a block handed out in a committed window, or
    /// one of its reservations in a sparse or view window. The windows' ranges are wide (4 GiB
    /// each) and a PC resource's GPU address can fall in one by number – that address is not
    /// the window's memory (a title's dynamic vertex buffer, a PC upload buffer, was read from
    /// an empty sparse window: nothing drew).
    pub fn is_title_address(&self, p: usize) -> bool {
        if self.is_sparse(p) {
            return self.sparse_reservation(p).is_some();
        }
        if self.is_view(p) {
            return self.view_reservation(p).is_some();
        }
        if !self.is_committed(p) {
            return false;
        }
        let off = p - self.base;
        let g = self.inner.lock().unwrap();
        g.used.range(..=off).next_back().is_some_and(|(&o, &l)| off < o + l)
    }

    /// Zeroed block of `size` bytes aligned to `align` (at least 64 KiB), first fit (window 0
    /// first).
    #[track_caller]
    pub fn alloc(&self, size: usize, align: usize) -> Option<*mut u8> {
        self.alloc_where(size, align, None, true)
    }

    /// `alloc` for reserved address space: not zeroed (a large reservation is not touched –
    /// what the title commits or maps there it writes; never-used memory is zero anyway).
    #[track_caller]
    pub fn reserve(&self, size: usize, align: usize) -> Option<*mut u8> {
        self.alloc_where(size, align, None, false)
    }

    /// `alloc` inside window `k` (memory the console addresses with 32 bits, e.g. descriptor
    /// heaps: window 0).
    #[track_caller]
    pub fn alloc_in(&self, k: usize, size: usize, align: usize) -> Option<*mut u8> {
        self.alloc_where(size, align, Some(k), true)
    }

    /// `XCL_TRACE_MEM`: each committed-window allocation of 1 MB or more and each one that
    /// fails, with the XCL code that asked for it (what fills the window).
    #[track_caller]
    fn trace(&self, size: usize, got: Option<usize>, used: usize) {
        let at = std::panic::Location::caller();
        match got {
            Some(p) => crate::xlog_info!("UMA: {} KB at {p:#x} for {}:{} ({} MB used)", size >> 10, at.file(), at.line(), used >> 20),
            None => crate::xlog_info!("UMA: {} KB for {}:{} – no room ({} MB used)", size >> 10, at.file(), at.line(), used >> 20),
        }
    }

    fn tracing() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("XCL_TRACE_MEM").is_some())
    }

    #[track_caller]
    fn alloc_where(&self, size: usize, align: usize, window: Option<usize>, zero: bool) -> Option<*mut u8> {
        let size = size.max(1).checked_next_multiple_of(GRAIN)?;
        let align = align.max(GRAIN).checked_next_power_of_two()?;
        let mut g = self.inner.lock().unwrap();
        let found = g.free.iter().filter(|(&o, _)| window.is_none_or(|k| o / WINDOW == k)).find_map(|(&o, &l)| {
            let at = self.base.checked_add(o)?.checked_next_multiple_of(align)? - self.base;
            (at.checked_add(size).is_some_and(|end| end <= o + l)).then_some((o, l, at))
        });
        let Some((start, len, at)) = found else {
            if Self::tracing() {
                let used = g.used.values().sum();
                drop(g);
                self.trace(size, None, used);
            }
            return None;
        };
        g.free.remove(&start);
        if at > start {
            g.free.insert(start, at - start);
        }
        if at + size < start + len {
            g.free.insert(at + size, start + len - at - size);
        }
        g.used.insert(at, size);
        let used = if Self::tracing() && size >= 1 << 20 { Some(g.used.values().sum()) } else { None };
        drop(g);
        let p = (self.base + at) as *mut u8;
        if let Some(used) = used {
            self.trace(size, Some(p as usize), used);
        }
        if zero {
            unsafe { std::ptr::write_bytes(p, 0, size) };
        }
        Some(p)
    }

    /// The bytes of the block from `alloc` at `p` (None: not one of ours).
    pub fn block_len(&self, p: usize) -> Option<usize> {
        if !self.is_committed(p) {
            return None;
        }
        {
            let g = self.pool.lock().unwrap();
            if p >= g.0 && p < g.1 {
                return Some(ARENA_BLOCK);
            }
        }
        self.inner.lock().unwrap().used.get(&(p - self.base)).copied()
    }

    /// Release a block from `alloc` (false: not one of ours).
    pub fn free(&self, p: usize) -> bool {
        if !self.is_committed(p) {
            return false;
        }
        {
            // a block of XCL's pool goes back to the pool
            let mut g = self.pool.lock().unwrap();
            if p >= g.0 && p < g.1 {
                g.2.push(p);
                return true;
            }
        }
        let mut g = self.inner.lock().unwrap();
        let off = p - self.base;
        let Some(orig) = g.used.remove(&off) else { return false };
        let (mut start, mut len) = (off, orig);
        // merge with the free neighbours
        if let Some((&o, &l)) = g.free.range(..off).next_back() {
            if o + l == off {
                g.free.remove(&o);
                start = o;
                len += l;
            }
        }
        if let Some(l) = g.free.remove(&(off + orig)) {
            len += l;
        }
        g.free.insert(start, len);
        true
    }

    /// Bytes handed out.
    pub fn used(&self) -> usize {
        self.inner.lock().unwrap().used.values().sum()
    }
}

impl Drop for Uma {
    fn drop(&mut self) {
        for &p in &self.win_allocs {
            unsafe {
                VirtualFree(p as *mut std::ffi::c_void, 0, MEM_RELEASE);
            }
        }
    }
}

/// Small pieces of committed unified memory for XCL's own per-draw data (argument tables,
/// user SGPRs): a block of `ARENA_BLOCK` bytes cut into aligned pieces, a new block when one
/// is used up. `Uma::alloc` hands out whole 64 KiB grains – three of those per draw filled
/// 2 GB within seconds in a scene of thousands of draws (Minecraft's world), and the draws
/// that found no room were skipped (the world flickered). The blocks go to `blocks`; their
/// owner frees them when the GPU is done (a command allocator's reset) and starts a fresh
/// arena.
#[derive(Default)]
pub struct Arena {
    at: usize,
    end: usize,
}

/// block size of an `Arena`
pub const ARENA_BLOCK: usize = 4 * GRAIN;

impl Arena {
    /// `n` bytes at an `align`-aligned address in committed unified memory (not zeroed), or
    /// None when committed unified memory is full.
    pub fn take(&mut self, u: &Uma, n: usize, align: usize, blocks: &mut Vec<usize>) -> Option<*mut u8> {
        let align = align.max(16).next_power_of_two();
        let n = n.max(1);
        let at = (self.at + align - 1) & !(align - 1);
        if self.at != 0 && at + n <= self.end {
            self.at = at + n;
            return Some(at as *mut u8);
        }
        if n > ARENA_BLOCK / 2 {
            // a large piece: a block of its own (the current one stays in use)
            let p = u.alloc(n, align)?;
            blocks.push(p as usize);
            return Some(p);
        }
        // XCL's pool first, then any free part of committed unified memory
        let p = match u.pool_block() {
            Some(p) => p,
            None => u.alloc(ARENA_BLOCK, GRAIN)? as usize,
        };
        blocks.push(p);
        self.at = p + n;
        self.end = p + ARENA_BLOCK;
        Some(p as *mut u8)
    }
}

static GLOBAL: OnceLock<Option<Uma>> = OnceLock::new();

/// A process that holds no title memory (a shader preparation worker): `get` is None from here
/// on, and nothing is reserved or committed for it (a run's region commits its coherent window).
/// Before any `get`.
pub fn leave_out() {
    let _ = GLOBAL.set(None);
}

/// The title's graphics memory XCL places in the coherent window (`note_coherent_need`) is
/// known only once it asks for it, and the window is committed when it is made: its size is
/// what the title put in it in its last run (`<data>/cache/uma_coherent*.txt`) and XCL's own
/// 512 MB, at most a window. `XCL_COHERENT_MB` chooses it.
///
/// Which of the title's GPU heaps go there is `XCL_COHERENT_HEAPS`'s
/// (`kernelx::coherent_heaps_route`), so a size is kept for each setting, in a file of its own
/// (`coherent_file_name`, and per console): a size measured with every heap in the window does not apply to a
/// run that sends them to the sparse windows. A retail title's 4582 MB of heaps, stored once,
/// made windows of 4095 + 998 MB in runs that put at most 105 MB in them, each shadowed by as
/// much video memory with `XCL_UMA_SHADOW=1`.
fn coherent_file(route: &str) -> std::path::PathBuf {
    crate::state::data_dir().join("cache").join(coherent_file_name(route, crate::state::config().is_series_s()))
}

/// The file in `<data>/cache` that keeps the coherent window's size for `route`, one number:
/// `uma_coherent.txt` for every heap in the window, the file and format builds before the
/// per-setting sizes read and write in the same data folder (a text they cannot parse gave them
/// a 512 MB window, too small for a retail title's heaps), and `uma_coherent_<setting>.txt`
/// (`:` as `_`) for the other settings, which those builds never read.
/// The Series S profile has files of its own (`_lockhart` before `.txt`, as xcl-dev/xcl#1110):
/// a title makes smaller heaps there (a retail game 2.9 GB for 4.5), and the Series X size kept
/// in the same data folder committed a window it never filled.
fn coherent_file_name(route: &str, series_s: bool) -> String {
    let device = if series_s { "_lockhart" } else { "" };
    if route == ALL_HEAPS {
        return format!("uma_coherent{device}.txt");
    }
    format!("uma_coherent_{}{device}.txt", route.replace(':', "_"))
}

/// The setting of `XCL_COHERENT_HEAPS` that puts every GPU heap in the coherent window (none),
/// as `kernelx::coherent_heaps_route` names it.
pub const ALL_HEAPS: &str = "all";

/// The bytes a `coherent_file` keeps (0: none), read as builds before read `uma_coherent.txt`.
fn stored_need(text: &str) -> usize {
    text.trim().parse::<usize>().unwrap_or(0)
}

/// Whether this process is running under Wine on an AMD GPU (`amdgpu`, PCI vendor `0x1002`).
/// On `amdgpu`, a multi-GiB `userptr` BO imported via `OpenExistingHeapFromAddress` stalls
/// every Vulkan submission (~150 ms/frame), whereas NVIDIA Linux does not hit that stall and
/// benefits from a larger coherent window.
fn wine_amd_gpu() -> bool {
    static AMD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AMD.get_or_init(|| {
        if !crate::util::under_wine() {
            return false;
        }
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE};
        let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }) else {
            return false;
        };
        let mut i = 0u32;
        while let Ok(adapter) = unsafe { factory.EnumAdapters1(i) } {
            i += 1;
            if let Ok(desc) = unsafe { adapter.GetDesc1() } {
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) == 0 {
                    return desc.VendorId == 0x1002;
                }
            }
        }
        false
    })
}

/// Bytes the coherent window(s) are made with for a runtime layout of `windows` windows.
pub fn coherent_hint_for(windows: usize) -> usize {
    let cap = max_coherent_bytes(windows);
    if let Some(mb) = std::env::var("XCL_COHERENT_MB").ok().and_then(|v| v.parse::<usize>().ok()) {
        return (mb << 20).clamp(GRAIN, cap);
    }
    // Under Wine/vkd3d-proton on Linux (amdgpu), a multi-GiB userptr BO imported via
    // OpenExistingHeapFromAddress stalls every GPU submission (~150 ms/frame); keep the
    // coherent window at 512 MiB and place overflow title GPU heaps in the sparse VRAM windows.
    if wine_amd_gpu() {
        return RUNTIME_COMMITTED_CAP;
    }
    let route = crate::hle::kernelx::coherent_heaps_route();
    let title = std::fs::read_to_string(coherent_file(&route)).map_or(0, |s| stored_need(&s));
    // only with the commit for all of it: the title's ~13 GB, video memory for its render targets
    // and textures (WDDM charges commit for each video-memory allocation too) and the window
    // itself – a retail game took ~18 GB; with less free it runs without (its GPU heaps outside
    // unified memory: no image) rather than taking the system's commit
    if title > 0 && free_commit() < NEEDS_FOR_COHERENT_HEAPS {
        crate::xlog_warn!(
            "UMA: {} MB of commit free, under {} MB: the title's GPU heaps stay outside unified memory (a larger page file gives them the coherent window)",
            free_commit() >> 20,
            NEEDS_FOR_COHERENT_HEAPS >> 20
        );
        return RUNTIME_COMMITTED_CAP;
    }
    (title + RUNTIME_COMMITTED_CAP).min(cap)
}

/// Bytes the coherent window is made with (legacy 4-window cap).
pub fn coherent_hint() -> usize {
    coherent_hint_for(MAX_WINDOWS)
}

/// Commit free before the title starts for its GPU heaps to go to the coherent window.
const NEEDS_FOR_COHERENT_HEAPS: usize = 17 << 30;

fn free_commit() -> usize {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
        return usize::MAX;
    }
    m.ullAvailPageFile as usize
}

/// bytes of graphics memory the title asked the coherent window for in this run
static NOTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// `coherent_file` is read and written under it
static COHERENT_FILE: Mutex<()> = Mutex::new(());
/// A run this long has made its GPU heaps (a retail title made all of them in its first 7 s):
/// what it put in the coherent window by then may lower the stored size (`settled_need`). A run
/// that ends sooner only ever grows it: one cut short before its heaps would leave the next
/// run's window too small for them.
const SETTLED: std::time::Duration = std::time::Duration::from_secs(60);

/// The size to store for `route` when `before` is stored and the run put `run` in the coherent
/// window by `SETTLED` (None: `before` stays). Only a setting that sends heaps to the sparse
/// windows lowers it, where a size measured with more in the window cost video memory for
/// nothing, and by half at most per run. With every heap in the window it only grows: a run that
/// stays in the menu has asked for a part of the heaps a level makes by then, and from a size
/// lowered to it the heaps of the next run that loads one would not fit and go to plain memory
/// (no image).
fn settled_need(route: &str, before: usize, run: usize) -> Option<usize> {
    (route.starts_with("sparse") && run < before).then(|| run.max(before / 2))
}

/// The title has asked for `bytes` of graphics memory XCL places in the coherent window so far
/// in this run (0: its GPU heaps all went to the sparse windows): the next run makes the window
/// that large (this one has the size it was made with).
pub fn note_coherent_need(bytes: usize) {
    let prev = NOTED.fetch_max(bytes, std::sync::atomic::Ordering::Relaxed);
    static SETTLE: std::sync::Once = std::sync::Once::new();
    SETTLE.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-uma-coherent".into()).spawn(|| {
            std::thread::sleep(SETTLED);
            settle_coherent_need();
        });
    });
    if bytes <= prev {
        return;
    }
    let _g = COHERENT_FILE.lock().unwrap();
    let route = crate::hle::kernelx::coherent_heaps_route();
    let f = coherent_file(&route);
    if bytes > std::fs::read_to_string(&f).map_or(0, |s| stored_need(&s)) {
        if let Some(d) = f.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let _ = std::fs::write(&f, bytes.to_string());
        let cap = max_coherent_bytes(configured_runtime_windows());
        crate::xlog_info!("UMA: the title's graphics memory for the coherent window: {} MB (up to {} MB from the next run)", bytes >> 20, (bytes + RUNTIME_COMMITTED_CAP).min(cap) >> 20);
    }
}

/// `SETTLED` into the run: the size stored for this run's `XCL_COHERENT_HEAPS` comes down
/// towards what the run put in the coherent window (`settled_need`).
fn settle_coherent_need() {
    let _g = COHERENT_FILE.lock().unwrap();
    let run = NOTED.load(std::sync::atomic::Ordering::Relaxed);
    let route = crate::hle::kernelx::coherent_heaps_route();
    let f = coherent_file(&route);
    let before = std::fs::read_to_string(&f).map_or(0, |s| stored_need(&s));
    if let Some(next) = settled_need(&route, before, run) {
        let _ = std::fs::write(&f, next.to_string());
        crate::xlog_info!(
            "UMA: this run put {} MB of the title's graphics memory in the coherent window, {} MB were stored for '{route}' (XCL_COHERENT_HEAPS): the next run makes it {} MB",
            run >> 20,
            before >> 20,
            (next + RUNTIME_COMMITTED_CAP).min(max_coherent_bytes(configured_runtime_windows())) >> 20
        );
    }
}

impl Uma {
    /// The first committed coherent window (None: none).
    pub fn committed_window(&self) -> Option<usize> {
        (0..self.kinds.len()).find(|&k| self.is_committed_window(k))
    }
}

/// The title's graphics memory (created on first use; `XCL_UMA_MB`, default 2048;
/// `XCL_UMA_WINDOWS=4|8|16`, default 4, with 8/16 requiring `XCL_HEAP_WINDOWS=1`).
pub fn get() -> Option<&'static Uma> {
    GLOBAL
        .get_or_init(|| {
            let mb = std::env::var("XCL_UMA_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(2048);
            let req = configured_runtime_windows();
            let coherent = coherent_hint_for(req);
            let u = if req > MAX_WINDOWS {
                Uma::for_runtime_windows((mb << 20).max(coherent), coherent, req).or_else(|| {
                    crate::xlog_error!(
                        "UMA: XCL_UMA_WINDOWS={req} asked for, but {req} aligned windows ({} GiB) could not be reserved below 1 TiB – falling back to {MAX_WINDOWS}: {}",
                        req * 4,
                        if view_windows(req) > 0 {
                            "GPU views of the title's memory stay outside the windows and shaders read 0 there (an empty 3D world)"
                        } else {
                            "the extra windows are not there"
                        }
                    );
                    let fallback = coherent_hint_for(MAX_WINDOWS);
                    Uma::for_runtime_with((mb << 20).max(fallback), fallback)
                })
            } else {
                Uma::for_runtime_with((mb << 20).max(coherent), coherent)
            };
            match &u {
                Some(u) => {
                    let sparse_n = (0..u.shader_windows()).filter(|&k| u.is_sparse_window(k)).count();
                    let coh_n = u.windows();
                    let view_n = (0..u.shader_windows()).filter(|&k| u.is_view_window(k)).count();
                    crate::xlog_info!(
                        "UMA: {} sparse VRAM window(s) ({} GiB VA) + {} coherent window(s) ({} MB) + {} view window(s) for GPU views at {:#x}",
                        sparse_n,
                        sparse_n * 4,
                        coh_n,
                        u.size() >> 20,
                        view_n,
                        u.base()
                    );
                }
                None => crate::xlog_warn!("UMA: could not reserve {mb} MB at a 4 GiB-aligned address – plain allocations"),
            }
            u
        })
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Arena blocks come from XCL's pool while the title holds the rest of window 0, and go
    /// back to it when freed.
    #[test]
    fn arena_blocks_come_from_the_pool_when_the_title_took_the_rest() {
        let u = Uma::with_windows(1, 8 << 20).unwrap();
        u.reserve_pool(4 * ARENA_BLOCK);
        // the pool is the window's end: the title's first allocation is still its start
        assert_eq!(u.pool.lock().unwrap().1, u.base() + (8 << 20));
        assert_eq!(u.alloc(GRAIN, GRAIN).map(|p| p as usize), Some(u.base()));
        // the title takes everything else
        let mut title = Vec::new();
        while let Some(p) = u.alloc(GRAIN, GRAIN) {
            title.push(p as usize);
        }
        assert!(u.alloc_in(0, ARENA_BLOCK, GRAIN).is_none());
        let (mut a, mut blocks) = (Arena::default(), Vec::new());
        for _ in 0..4 * (ARENA_BLOCK / 256) {
            a.take(&u, 200, 256, &mut blocks).expect("a piece from the pool");
        }
        assert_eq!(blocks.len(), 4);
        assert!(a.take(&u, 200, 256, &mut blocks).is_none(), "pool used up, the rest is the title's");
        for b in blocks.drain(..) {
            assert!(u.free(b));
        }
        let mut a = Arena::default();
        assert!(a.take(&u, 200, 256, &mut blocks).is_some(), "freed blocks are back in the pool");
    }

    /// Many small pieces share a block (aligned, disjoint); a large one gets its own.
    #[test]
    fn arena_cuts_blocks_into_aligned_pieces() {
        let u = Uma::with_windows(1, 4 << 20).unwrap();
        let (mut a, mut blocks) = (Arena::default(), Vec::new());
        let mut got = Vec::new();
        for k in 0..1000usize {
            let n = 16 + (k % 7) * 40;
            let p = a.take(&u, n, 256, &mut blocks).unwrap() as usize;
            assert_eq!(p % 256, 0);
            assert_eq!(u.window_of(p), Some(0));
            got.push((p, n));
        }
        got.sort_unstable();
        assert!(got.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0), "pieces overlap");
        // 1000 pieces of at most 256 aligned bytes: 1000 · 256 B = 4 blocks, not 1000 grains
        assert_eq!(blocks.len(), (1000 * 256usize).div_ceil(ARENA_BLOCK));
        let big = a.take(&u, ARENA_BLOCK, 256, &mut blocks).unwrap() as usize;
        assert_eq!(blocks.last(), Some(&big));
        for b in blocks {
            assert!(u.free(b));
        }
        assert_eq!(u.used(), 0);
    }

    #[test]
    fn windows_are_4_gib_apart_and_allocations_stay_inside() {
        let u = Uma::with_windows(2, 4 << 20).unwrap();
        assert_eq!(u.base() % (MAX_WINDOWS * WINDOW), 0, "16 GiB alignment");
        assert_eq!((u.windows(), u.size()), (2, 8 << 20));
        let a = u.alloc(3 << 20, 0).unwrap() as usize;
        let b = u.alloc(3 << 20, 0).unwrap() as usize; // no room left in window 0
        assert_eq!(u.window_of(a), Some(0));
        assert_eq!(u.window_of(b), Some(1));
        assert_eq!(b, u.base() + WINDOW);
        assert_eq!((b >> 32) & 3, 1, "window from the high address bits");
        assert!(u.alloc_in(0, 2 << 20, 0).is_none(), "window 0 full");
        assert!(u.alloc(2 << 20, 0).is_none());
        unsafe { *(b as *mut u32) = 7 };
        assert!(u.free(a) && u.free(b));
        assert!(!u.contains(u.base() + (4 << 20)) && !u.contains(u.base() + 2 * WINDOW));
    }

    #[test]
    fn alloc_free_coalesce() {
        // without XCL's arena pool (`Uma::new` takes it at the start of window 0)
        let u = Uma::with_windows(1, 16 << 20).unwrap();
        assert_eq!(u.base() & 0xFFFF_FFFF, 0, "4 GiB alignment");
        let a = u.alloc(100, 0).unwrap() as usize;
        let b = u.alloc(0x2_0000, 0).unwrap() as usize;
        let c = u.alloc(0x1_0000, 0x10_0000).unwrap() as usize;
        assert_eq!(a, u.base());
        assert_eq!(b, u.base() + GRAIN);
        assert_eq!(c % 0x10_0000, 0, "alignment");
        assert_eq!(u.used(), 4 * GRAIN);
        unsafe { *(b as *mut u32) = 0xDEAD };
        assert!(u.free(b) && !u.free(b) && !u.free(12345));
        assert!(u.free(a));
        // a + b coalesced: a 3-grain block fits at the base again, zeroed
        let d = u.alloc(3 * GRAIN, 0).unwrap() as usize;
        assert_eq!(d, u.base());
        assert_eq!(unsafe { *((d + GRAIN) as *const u32) }, 0);
        assert!(u.free(c) && u.free(d));
        assert_eq!(u.used(), 0);
        // everything merged back into one range
        assert!(u.alloc(16 << 20, 0).is_some());
    }

    /// Only handed-out blocks are the title's memory: an address in a window's range that is
    /// not one (a PC GPU address by number) is not.
    #[test]
    fn title_addresses_are_blocks_handed_out() {
        let u = Uma::new(16 << 20).unwrap();
        let a = u.alloc(0x2_0000, 0).unwrap() as usize;
        assert!(u.is_title_address(a) && u.is_title_address(a + 0x1_FFFF));
        assert!(!u.is_title_address(a + 0x2_0000) && !u.is_title_address(u.base() + (8 << 20)));
        assert!(!u.is_title_address(u.base() - 1) && !u.is_title_address(u.base() + (32 << 20)));
        assert!(u.free(a));
        assert!(!u.is_title_address(a));
    }

    /// XDefiant's first reservation is 8 GiB + 4 MiB, more than windows 1..2 hold: it has to go
    /// across window 0 without taking the room its 32-bit descriptor heaps need there.
    #[test]
    fn a_reservation_larger_than_windows_1_and_2_leaves_window_0_room_for_descriptor_heaps() {
        let u = Uma::for_runtime(16 << 20).unwrap();
        let big = 0x2_0041_0000;
        let r = u.sparse_reserve(big).unwrap();
        let heap = 442_368 * 32;
        let (d, ld) = u.sparse_alloc_in(0, heap, 256).expect("a descriptor heap in window 0");
        assert_eq!(u.window_of(d), Some(0));
        assert!(d + ld <= r || d >= r + big, "the heap is not inside the title's reservation");
        assert_ne!(d, u.base(), "no descriptor table at console address 0");
        let (x, _) = u.sparse_alloc(0x2_0000, 0).expect("the title's graphics memory after its reservation");
        assert!(x >= r + big || x + 0x2_0000 <= r);
        assert_eq!(u.sparse_release(x), Some(0x2_0000));
        assert_eq!(u.sparse_release(d), Some(ld));
        assert_eq!(u.sparse_release(r), Some(big));
        assert_eq!(u.sparse_used(), 0);
    }

    #[test]
    fn sparse_first_fit_keeps_the_range_crossing_the_lower_bound() {
        let lo = WINDOW;
        let hi = lo + 10 * GRAIN;
        let mut ranges = (
            BTreeMap::from([
                (GRAIN, GRAIN),
                (lo - GRAIN, 4 * GRAIN),
                (lo + 4 * GRAIN, 4 * GRAIN),
                (hi, 4 * GRAIN),
            ]),
            BTreeMap::new(),
        );

        // The range that starts before lo wins over the later range.
        assert_eq!(take_sparse_in(&mut ranges, lo, hi, 2 * GRAIN, GRAIN, true), Some(lo));
        assert_eq!(ranges.0.get(&(lo - GRAIN)), Some(&GRAIN));
        assert_eq!(ranges.0.get(&(lo + 2 * GRAIN)), Some(&GRAIN));
        // The remaining one-grain hole is too small. The next aligned fit is later.
        assert_eq!(take_sparse_in(&mut ranges, lo, hi, 2 * GRAIN, 4 * GRAIN, true), Some(lo + 4 * GRAIN));
        let after = ranges.clone();
        assert_eq!(take_sparse_in(&mut ranges, lo, hi, 3 * GRAIN, GRAIN, true), None);
        assert_eq!(ranges, after);
    }

    #[test]
    fn sparse_first_fit_skips_a_window_boundary_only_when_requested() {
        let lo = WINDOW - GRAIN;
        let hi = WINDOW + 2 * GRAIN;
        let ranges = BTreeMap::from([(GRAIN, GRAIN), (lo - GRAIN, 4 * GRAIN), (hi + GRAIN, GRAIN)]);
        let mut single = (ranges.clone(), BTreeMap::new());
        let mut cross = (ranges, BTreeMap::new());
        assert_eq!(take_sparse_in(&mut single, lo, hi, 2 * GRAIN, GRAIN, true), Some(WINDOW));
        assert_eq!(take_sparse_in(&mut cross, lo, hi, 2 * GRAIN, GRAIN, false), Some(lo));
    }

    #[test]
    fn runtime_layout_places_sparse_in_0_to_2_and_committed_in_3() {
        let u = Uma::for_runtime(16 << 20).unwrap();
        assert_eq!(u.base() % (MAX_WINDOWS * WINDOW), 0);
        assert_eq!(u.shader_windows(), 4);
        assert_eq!(u.windows(), 1);
        assert!(u.is_sparse_window(0) && u.is_sparse_window(1) && u.is_sparse_window(2));
        assert!(u.is_committed_window(3));

        // Committed allocations land in window 3
        let c = u.alloc(0x2_0000, 0).unwrap() as usize;
        assert_eq!(u.window_of(c), Some(3));
        assert!(u.is_committed(c) && !u.is_sparse(c) && u.is_title_address(c));
        assert!(u.free(c));

        // First sparse_alloc lands in window 0, after the descriptor range
        let (a, la) = u.sparse_alloc(0x2_0000, 0).unwrap();
        assert_eq!(a, u.base() + GRAIN + DESC_HEADROOM);
        assert_eq!(la, 0x2_0000);
        assert_eq!(u.window_of(a), Some(0));
        assert!(u.is_sparse(a) && u.is_title_address(a));

        // sparse_alloc_in(0, ...) stays inside window 0
        let (d, ld) = u.sparse_alloc_in(0, 4096, 256).unwrap();
        assert_eq!(u.window_of(d), Some(0));
        assert_eq!(ld, GRAIN);

        // Large sparse_reserve prefers window 1+
        let r = u.sparse_reserve(256 << 20).unwrap();
        assert_eq!(u.window_of(r), Some(1));

        // Release and coalesce
        assert_eq!(u.sparse_release(a), Some(la));
        assert_eq!(u.sparse_release(d), Some(ld));
        assert_eq!(u.sparse_release(r), Some(256 << 20));
        assert_eq!(u.sparse_used(), 0);
        assert!(!u.is_title_address(a));

        // Allocations <= 4 GiB never straddle a window boundary, and DESC_HEADROOM is preserved
        let fill_w0 = WINDOW - DESC_HEADROOM - GRAIN;
        let (x, _) = u.sparse_alloc(fill_w0, GRAIN).unwrap();
        assert_eq!(x, u.base() + GRAIN + DESC_HEADROOM);
        let (y, _) = u.sparse_alloc(GRAIN, GRAIN).unwrap();
        assert_eq!(u.window_of(y), Some(1), "overflows to window 1 before touching window 0 descriptor headroom");
        let (desc, _) = u.sparse_alloc_in(0, GRAIN, GRAIN).unwrap();
        assert_eq!(u.window_of(desc), Some(0), "descriptor headroom in window 0 remains available");
        assert_eq!(u.sparse_release(x), Some(fill_w0));
        assert_eq!(u.sparse_release(y), Some(GRAIN));
        assert_eq!(u.sparse_release(desc), Some(GRAIN));
        assert_eq!(u.sparse_used(), 0);
        // Contiguous sparse windows 0..2 are reported as one OS watch region; window 3 as another.
        assert_eq!(u.watch_regions(), vec![(u.base(), 3 * WINDOW), (u.base() + 3 * WINDOW, 16 << 20)]);
    }

    /// Whether Windows has nothing at `p` (MEM_FREE: it may place an allocation there).
    fn free_at(p: usize) -> bool {
        use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_FREE};
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = unsafe { VirtualQuery(p as *const std::ffi::c_void, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
        n != 0 && m.State == MEM_FREE
    }

    #[test]
    fn runtime_layout_supports_eight_and_sixteen_heap_windows() {
        for windows in [8usize, MAX_HEAP_WINDOWS] {
            let u = Uma::for_runtime_windows(16 << 20, RUNTIME_COMMITTED_CAP, windows).expect("expanded UMA aperture");
            let aperture = windows * WINDOW;
            assert_eq!(u.base() % aperture, 0, "base must be {windows}*4 GiB aligned");
            assert!(u.base() + aperture <= (1usize << 40), "expanded aperture must fit below 40-bit VA limit");
            assert_eq!(u.shader_windows(), windows);
            assert_eq!(u.windows(), 1);
            // Sparse VA stays capped at 3 windows (12 GiB) so huge startup heaps stay off sparse VRAM.
            for k in 0..MAX_RUNTIME_SPARSE_WINDOWS {
                assert!(u.is_sparse_window(k), "window {k} should be sparse");
            }
            assert_eq!(u.committed_window(), Some(MAX_RUNTIME_SPARSE_WINDOWS));
            // 8 windows: the trailing windows stay unreserved; 16: views right after the coherent
            // window, the rest kept empty (and reserved: `sixteen_windows_reserve_the_whole_aperture`)
            let views = MAX_RUNTIME_SPARSE_WINDOWS + 1..MAX_RUNTIME_SPARSE_WINDOWS + 1 + view_windows(windows);
            for k in (MAX_RUNTIME_SPARSE_WINDOWS + 1)..windows {
                assert!(!u.is_sparse_window(k) && !u.is_committed_window(k), "window {k} holds no CPU memory");
                assert_eq!(u.is_view_window(k), views.contains(&k), "window {k}: a view window in 16 only");
            }
            let c = u.alloc(0x2_0000, 0).unwrap() as usize;
            assert_eq!(u.window_of(c), Some(MAX_RUNTIME_SPARSE_WINDOWS));
            assert!(u.free(c));
            // Window 1..2 hold large reservations while preserving window 0 descriptor headroom; >12 GiB overflows cleanly.
            let big = 2 * WINDOW;
            let r = u.sparse_reserve(big).expect("8 GiB reservation in sparse windows 1..2");
            assert_eq!(u.window_of(r), Some(1));
            assert!(u.sparse_reserve(4 * WINDOW).is_none(), "reservations beyond 12 GiB sparse cap must overflow to host RAM");
            let (d, ld) = u.sparse_alloc_in(0, GRAIN, GRAIN).expect("window 0 descriptor headroom intact");
            assert_eq!(u.window_of(d), Some(0));
            assert_eq!(u.sparse_release(d), Some(ld));
            assert_eq!(u.sparse_release(r), Some(big));
            assert_eq!(u.sparse_used(), 0);
            assert_eq!(
                u.watch_regions(),
                vec![
                    (u.base(), MAX_RUNTIME_SPARSE_WINDOWS * WINDOW),
                    (u.base() + MAX_RUNTIME_SPARSE_WINDOWS * WINDOW, 16 << 20),
                ]
            );
        }
    }

    /// The Series X run's plan (`XCL_UMA_WINDOWS=16`, 4095 MB, a 512 MB coherent window) and where
    /// its first graphics memory goes: 3 sparse windows, the coherent one, 6 view windows, 6
    /// empty, and its GPU heaps of 1172 / 2199 / 563 / 596 MB and its 38 MB view heap at the
    /// addresses that run logged (0xf2e00000 into window 0). Nothing of the title memory size
    /// goes into the plan: the Series S profile gets the same one.
    #[test]
    fn the_series_x_plan_is_pinned() {
        let u = Uma::for_runtime_windows(4095 << 20, RUNTIME_COMMITTED_CAP, MAX_HEAP_WINDOWS).expect("16-window aperture");
        let b = u.base();
        assert_eq!(b % (MAX_HEAP_WINDOWS * WINDOW), 0);
        use WindowKind::*;
        let mut kinds = vec![Sparse, Sparse, Sparse, Committed(512 << 20)];
        kinds.extend([View; VIEW_WINDOWS]);
        kinds.extend([None; 6]);
        assert_eq!(u.kinds, kinds);
        assert_eq!((u.shader_windows(), u.windows(), u.size()), (16, 1, 512 << 20));
        assert_eq!(u.desc_range(), Some((b + GRAIN, b + GRAIN + DESC_HEADROOM)));
        assert_eq!((u.title_sparse_start(), u.sparse_end()), (b + 0x2001_0000, b + 3 * WINDOW));
        for (p, k) in [(0xf2e0_0000, Some(0)), (WINDOW + 0x140_0000, Some(1)), (3 * WINDOW, Some(3)), (3 * WINDOW + (512 << 20), Option::None), (4 * WINDOW, Some(4)), (10 * WINDOW, Option::None)] {
            assert_eq!(u.window_of(b + p), k, "{p:#x}");
        }
        // the GPU heaps (2 MB pages, rounded as `kernelx::VirtualAlloc` does), then the view heap
        let at: Vec<usize> = [1172usize, 2199, 563, 596].iter().map(|&mb| u.sparse_alloc((mb << 20).next_multiple_of(2 << 20), 2 << 20).unwrap().0 - b).collect();
        assert_eq!(at, [0x2020_0000, 0x6960_0000, WINDOW, WINDOW + (564 << 20)]);
        assert_eq!(u.sparse_alloc(0x260_0000, GRAIN).map(|(a, _)| a - b), Some(0xf2e0_0000));
        // the home of the Series X run's page pool (99776 pages: more than a window) goes where it went
        let pool = 99776 * GRAIN;
        let above = u.sparse_alloc_above(pool, GRAIN);
        assert_eq!(above.map(|(a, n)| (a - b, n)), Some((WINDOW + (1160 << 20), pool)));
        assert_eq!(u.sparse_release(above.unwrap().0), Some(pool));
        assert_eq!(u.sparse_alloc(pool, GRAIN), above);
    }

    /// Memory no shader addresses with 32 bits keeps out of window 0 while windows 1+ have room,
    /// and gets window 0 when they have none; above a window it goes as `sparse_alloc`'s.
    #[test]
    fn sparse_alloc_above_keeps_window_0_for_the_title() {
        let u = Uma::for_runtime_windows(16 << 20, RUNTIME_COMMITTED_CAP, MAX_HEAP_WINDOWS).expect("16-window aperture");
        let b = u.base();
        let (h, n) = u.sparse_alloc_above(664 << 20, GRAIN).unwrap();
        assert_eq!((h, n), (b + WINDOW, 664 << 20));
        // window 0 still has the room a 38 MB view heap needs after 2.9 GB of GPU heaps
        let heaps = u.sparse_alloc(2892 << 20, 2 << 20).unwrap();
        assert_eq!(u.window_of(heaps.0), Some(0));
        assert_eq!(u.sparse_alloc(0x260_0000, GRAIN).map(|(a, _)| u.window_of(a)), Some(Some(0)));
        let used = (2892 << 20) + 0x260_0000;
        assert_eq!(u.sparse_free_in(0), (WINDOW - 0x2001_0000 - used, WINDOW - 0x2020_0000 - used));
        assert_eq!(u.sparse_free_in(1), (WINDOW - (664 << 20), WINDOW - (664 << 20)));
        // windows 1..2 full: window 0
        let rest = u.sparse_reserve(2 * WINDOW - (664 << 20)).unwrap();
        assert_eq!(u.window_of(rest), Some(1));
        let (w0, _) = u.sparse_alloc_above(GRAIN, GRAIN).unwrap();
        assert_eq!(u.window_of(w0), Some(0));
        assert!(w0 >= u.title_sparse_start(), "never the descriptor headroom");
        assert!(u.sparse_alloc_above(WINDOW + GRAIN, GRAIN).is_none(), "no room above a window");
    }

    /// With 16 windows nothing of Windows' lands in the aperture: shaders read every window of
    /// it, and a window without memory of its own once read whatever Windows had put there.
    #[test]
    fn sixteen_windows_reserve_the_whole_aperture() {
        let u = Uma::for_runtime_windows(16 << 20, RUNTIME_COMMITTED_CAP, MAX_HEAP_WINDOWS).expect("16-window aperture");
        let base = u.base();
        let coh = u.committed_window().unwrap();
        let (c, c_len) = u.window(coh);
        // each window's start and end, the coherent window's uncommitted tail
        let mut probes: Vec<usize> = (0..MAX_HEAP_WINDOWS).flat_map(|k| [base + k * WINDOW, base + (k + 1) * WINDOW - 1]).collect();
        probes.push(c + c_len);
        for &p in &probes {
            assert!(!free_at(p), "{:#x} (window {}) is free: Windows may place memory there", p, (p - base) / WINDOW);
        }
        // the view windows and the empty ones are no CPU memory and nobody's write-watched region
        for k in coh + 1..MAX_HEAP_WINDOWS {
            let p = base + k * WINDOW + 0x1234;
            assert!(!u.contains(p) && !u.is_sparse(p) && !u.is_committed(p) && !u.is_gpu_copy(p) && !u.is_title_address(p), "window {k}");
            assert!(!u.is_sparse_window(k) && !u.is_committed_window(k));
            assert_eq!(u.window_of(p).is_some(), u.is_view_window(k), "window {k}: only view windows are windows of their own");
        }
        assert!(u.watch_regions().iter().all(|&(a, n)| a + n <= c + c_len), "{:x?}", u.watch_regions());
    }

    /// GPU views go to the view windows, window-aligned: one of 24 GiB fills them, and its
    /// release gives the range back whole. The CPU view never goes there.
    #[test]
    fn view_windows_hold_gpu_views() {
        let u = Uma::for_runtime_windows(16 << 20, RUNTIME_COMMITTED_CAP, MAX_HEAP_WINDOWS).expect("16-window aperture");
        let first = (0..MAX_HEAP_WINDOWS).find(|&k| u.is_view_window(k)).expect("view windows");
        assert_eq!((first..MAX_HEAP_WINDOWS).filter(|&k| u.is_view_window(k)).count(), VIEW_WINDOWS);
        let g = u.view_reserve(VIEW_WINDOWS * WINDOW).expect("24 GiB in the view windows");
        assert_eq!(g, u.base() + first * WINDOW);
        assert!(u.view_reserve(GRAIN).is_none(), "the views are full");
        for x in [0, 0x1234, 5 * WINDOW + 0x10_0000, VIEW_WINDOWS * WINDOW - 1] {
            assert!(u.is_view(g + x) && u.is_title_address(g + x), "{x:#x}");
            assert_eq!(u.window_of(g + x), Some(first + x / WINDOW));
            assert!(!u.contains(g + x) && !u.is_sparse(g + x), "{x:#x}: the CPU cannot touch a GPU view");
            assert_eq!(u.view_reservation(g + x), Some((g, VIEW_WINDOWS * WINDOW)));
        }
        // a CPU view of the same size does not fit the sparse windows and never takes the views
        assert!(u.sparse_reserve(VIEW_WINDOWS * WINDOW).is_none());
        assert!(u.sparse_alloc(2 * WINDOW, GRAIN).is_some_and(|(a, n)| a + n <= u.base() + MAX_RUNTIME_SPARSE_WINDOWS * WINDOW));
        assert_eq!(u.view_release(g), Some(VIEW_WINDOWS * WINDOW));
        assert_eq!(u.view_release(g), None);
        assert!(!u.is_title_address(g) && u.is_view(g));
        // smaller views start at a window each, and their room comes back merged
        let a = u.view_reserve(GRAIN).unwrap();
        let b = u.view_reserve(WINDOW + GRAIN).unwrap();
        assert_eq!((a, b), (g, g + WINDOW));
        assert_eq!(u.view_release(a), Some(GRAIN));
        assert_eq!(u.view_release(b), Some(WINDOW + GRAIN));
        assert_eq!(u.view_reserve(VIEW_WINDOWS * WINDOW), Some(g));
        // no view windows below 16
        let four = Uma::for_runtime(16 << 20).unwrap();
        assert!(four.view_reserve(GRAIN).is_none() && (0..MAX_WINDOWS).all(|k| !four.is_view_window(k)));
    }

    #[test]
    fn expanded_heap_window_shaders_translate_and_validate() {
        use crate::gpu::dxil::kernel::{translate, KernelOptions};
        use crate::gpu::rdna2::encode;
        use crate::gpu::recomp::{self, Options, HEAP_WINDOWS};
        let mut code = Vec::new();
        code.extend(encode("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 4), ("SBASE", 0), ("OFFSET", 0), ("SOFFSET", 125)], None).unwrap());
        code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        for w in [8u32, MAX_HEAP_WINDOWS as u32] {
            // HLSL path accepts 8/16 with heap_windows, rejects without.
            assert!(recomp::translate(&code, &Options { mem_windows: w, heap_windows: false, ..Default::default() }).is_err());
            let t = recomp::translate(&code, &Options { mem_windows: w, heap_windows: true, ..Default::default() }).unwrap();
            assert!(t.hlsl.contains(&format!("xcl_w >= {w}u")), "HLSL must range-check against {w} windows");
            assert!(!t.hlsl.contains("mem4"), "root signature ABI stays at 4 root UAVs");
            // Direct DXIL path validates cleanly with 8/16 | HEAP_WINDOWS.
            translate(&code, KernelOptions { wave64: true, split: false, mem_windows: w | HEAP_WINDOWS, sampler_base: 0 }).unwrap();
        }
    }

    #[test]
    fn expanded_runtime_layout_splits_large_coherent_heaps_across_multiple_windows() {
        // Legacy 4-window layout stays capped at 1 coherent window (4095 MiB).
        assert_eq!(coherent_window_sizes(6 << 30, 6 << 30, MAX_WINDOWS), vec![MAX_SIZE]);
        // 8- and 16-window layouts split >4 GiB coherent heaps across up to 5 / 7 committed windows
        // with exact remainder sizing while keeping sparse windows capped at 3 (12 GiB) and, with
        // 16, the 6 view windows free.
        let want = MAX_SIZE + (512 << 20);
        for w in [8usize, MAX_HEAP_WINDOWS] {
            assert_eq!(coherent_window_sizes(want, want, w), vec![MAX_SIZE, 512 << 20]);
            assert_eq!(coherent_window_sizes(5 * MAX_SIZE, 5 * MAX_SIZE, w).len(), 5);
        }
        assert_eq!(coherent_window_sizes(20 << 30, 20 << 30, 8).len(), 5);
        assert_eq!(coherent_window_sizes(20 << 30, 20 << 30, MAX_HEAP_WINDOWS).len(), 6);
        // a coherent need past 7 windows (as a `coherent_file` may keep) stops at 7: windows
        // 10..15 stay the views'
        assert_eq!(coherent_window_sizes(40 << 30, 40 << 30, MAX_HEAP_WINDOWS), vec![MAX_SIZE; 7]);
        assert_eq!(max_coherent_windows(MAX_WINDOWS), 1);
        assert_eq!(max_coherent_windows(8), 5);
        assert_eq!(max_coherent_windows(MAX_HEAP_WINDOWS), MAX_HEAP_WINDOWS - MAX_RUNTIME_SPARSE_WINDOWS - VIEW_WINDOWS);
        assert_eq!(max_coherent_bytes(MAX_HEAP_WINDOWS), 7 * MAX_SIZE);
        // Multi-window Uma::alloc seamlessly spills into the 2nd committed window.
        let u = Uma::with_windows(2, 8 << 20).unwrap();
        let a = u.alloc(6 << 20, GRAIN).unwrap() as usize;
        let b = u.alloc(6 << 20, GRAIN).unwrap() as usize;
        assert_eq!(u.window_of(a), Some(0));
        assert_eq!(u.window_of(b), Some(1));
        assert!(u.free(a) && u.free(b));
    }

    /// The coherent window's size is kept per `XCL_COHERENT_HEAPS` setting: the 4582 MB a run
    /// with every GPU heap in the window stored in `uma_coherent.txt` does not size the window of
    /// a run that sends them to the sparse windows, which has a file of its own. The file of every
    /// heap stays one number, the only text builds before this one read in the same data folder.
    #[test]
    fn the_stored_coherent_need_is_the_one_of_the_heaps_setting() {
        let x = |route| coherent_file_name(route, false);
        assert_eq!(x(ALL_HEAPS), "uma_coherent.txt", "the file builds before read and write");
        assert_eq!(x("sparse:16"), "uma_coherent_sparse_16.txt", "no ':' in a Windows file name");
        assert_eq!(x("sparse"), "uma_coherent_sparse.txt");
        assert_eq!(x("off"), "uma_coherent_off.txt");
        // the Series S profile's own files, beside the Series X ones
        assert_eq!(coherent_file_name(ALL_HEAPS, true), "uma_coherent_lockhart.txt");
        assert_eq!(coherent_file_name("sparse:16", true), "uma_coherent_sparse_16_lockhart.txt");
        let names: Vec<String> = [false, true].iter().flat_map(|&s| [ALL_HEAPS, "off", "sparse", "sparse:16", "sparse:64"].map(|r| coherent_file_name(r, s))).collect();
        assert!(names.iter().enumerate().all(|(i, n)| !names[..i].contains(n)), "one file per setting and console: {names:?}");
        // what is written is what builds before wrote (`bytes.to_string()`) and parse
        // (`s.trim().parse::<usize>()`)
        let stored = (4582usize << 20).to_string();
        assert_eq!(stored.trim().parse::<usize>(), Ok(4582 << 20));
        assert_eq!(stored_need(&stored), 4582 << 20);
        assert_eq!(stored_need(&format!("{stored}\r\n")), 4582 << 20);
        assert_eq!(stored_need(""), 0);
        // a text that is no number reads as none, as builds before read it: it is written over
        // with a number at the run's first GPU heap
        assert_eq!(stored_need("4804575232 all\n105000000 sparse:16\n"), 0);
    }

    /// A run lowers the stored coherent size only with a setting that sends GPU heaps to the
    /// sparse windows, and by half at most: with every heap in the window, a run that stayed in
    /// the menu (1 GB asked of a full run's 4582 MB) leaves it as it is.
    #[test]
    fn only_the_sparse_heaps_settings_lower_the_stored_coherent_need() {
        let full = 4582usize << 20;
        assert_eq!(settled_need(ALL_HEAPS, full, 1 << 30), None, "the next run that loads a level needs all of it");
        assert_eq!(settled_need(ALL_HEAPS, full, 0), None);
        assert_eq!(settled_need("off", full, 0), None);
        assert_eq!(settled_need("sparse:16", full, 105 << 20), Some(full / 2), "half at most per run");
        assert_eq!(settled_need("sparse:16", full / 2, 105 << 20), Some(full / 4));
        assert_eq!(settled_need("sparse:16", 200 << 20, 105 << 20), Some(105 << 20));
        assert_eq!(settled_need("sparse", 64 << 20, 0), Some(32 << 20));
        assert_eq!(settled_need("sparse:16", 105 << 20, 105 << 20), None, "nothing to lower");
        assert_eq!(settled_need("sparse:16", 105 << 20, 200 << 20), None, "growing is note_coherent_need's");
    }
}
