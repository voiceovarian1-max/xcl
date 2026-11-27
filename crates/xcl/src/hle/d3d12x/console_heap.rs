//! Console descriptor heaps (CBV/SRV/UAV and samplers).
//!
//! On the console a descriptor is GPU-readable memory: 32 bytes per CBV/SRV/UAV slot (T# /
//! V#), 16 per sampler (S#), and a descriptor table in a root argument is a 32-bit GPU address
//! (shaders load it with the high half 0 – amd_fidelityfx_upscaler_xbox.dll, HANDOFF "FSR4").
//! XCL gives console code heaps whose slots live in unified memory (so the table addresses
//! console code computes from the heap start work as they are in the recompiler's `mem`
//! model) and keeps PC descriptors beside each: view slot k → an SRV and, next to it, a UAV;
//! sampler slot k → PC sampler k. A slot holds the descriptor in the XCL shader ABI
//! (`desc::xcl_image_words`: T# word 0 = the SRV's index in the shader-visible PC heap;
//! `desc::xcl_sampler_words`: S# word 3 [11:0] = k; V# with its unified-memory address).
//!
//! View descriptors are values on the console: titles create them in a CPU-only heap and copy
//! them into the shader-visible one as memory (the console header's inline
//! CopyDescriptorsSimple; XDefiant copies every texture that way), and a descriptor table may
//! be in any heap. So all console view heaps share one shader-visible PC heap (`Global`):
//! each gets a range there, T# word 0 is an index into it – a T# copied anywhere still names
//! its PC descriptor – and that heap is what lists bind. The title's own PC heap stays as a
//! CPU-only mirror (where views are made, copies come from and UAV clears take their CPU
//! handle). A heap gets its place in the shared heap a page of `PAGE_SLOTS` slots at a time,
//! when a slot of that page is first used: titles make large heaps and fill few slots
//! (Minecraft: eight view heaps of 73 727 slots – 147 454 PC descriptors each, more in all
//! than the 1 000 000 an NVIDIA shared heap holds – with a few hundred views in them). A heap
//! left out of the shared heap would be lost: lists bind only the shared heap for views.
//! Without the shared heap (it could not be made) a heap keeps its own PC heap: SRV at 2k,
//! UAV at 2k + 1.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// bytes of a CBV/SRV/UAV slot (T# / V#) and of a sampler slot (S#)
pub const SLOT: u32 = 32;
pub const SAMPLER_SLOT: u32 = 16;

/// Whose memory a heap's slots are in.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Memory {
    /// a sparse VRAM allocation in window 0
    Sparse,
    /// committed system RAM
    Uma,
    /// the title's (its allocation hook gave it: `kernelx::runtime_alloc`)
    Title,
}

struct Heap {
    obj: usize,
    shadow: usize,
    n: u32,
    pc_cpu: u64,
    /// 0: not shader visible
    pc_gpu: u64,
    inc: u32,
    sampler: bool,
    /// views in the shared heap: the index there of each page's first SRV (`UNMAPPED`: the
    /// page has no place yet); empty: the heap's views are not in the shared heap
    pages: Vec<u32>,
    /// shader visible as the title made it
    visible: bool,
    memory: Memory,
}

impl Heap {
    fn slot_bytes(&self) -> u32 {
        if self.sampler {
            SAMPLER_SLOT
        } else {
            SLOT
        }
    }
}

static HEAPS: Mutex<Vec<Heap>> = Mutex::new(Vec::new());

/// The shader-visible PC heap of all console view heaps.
struct Global {
    dev: ID3D12Device,
    heap: ID3D12DescriptorHeap,
    cpu: u64,
    gpu: u64,
    inc: u32,
    cap: u32,
    next: u32,
    /// released ranges (first, count)
    free: Vec<(u32, u32)>,
    /// a CPU-only heap of `NULL_BLOCK` null views (even SRV, odd UAV) and its start: copied over
    /// every range of the shared heap that is not handed out (`fill_nulls`)
    nulls: Option<(ID3D12DescriptorHeap, u64)>,
}

// D3D12 descriptor heaps and devices are free-threaded
unsafe impl Send for Global {}

static GLOBAL: Mutex<Option<Global>> = Mutex::new(None);
static GLOBAL_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// descriptors of the shared heap at most (the GPU may take more: OPTIONS19)
pub const GLOBAL_MAX: u32 = 1_000_000;
// (one million: what every resource binding tier 2+ GPU takes. The translated shaders accept a
// T# index only below it (`gpu::dxil::lower_image::VIEW_HEAP_MAX_SRV`), so the heap must never be
// smaller than that bound: an index past the heap's end hangs NVIDIA GPUs, which took 1 000 000
// while the bound was 2^21 – a raw console T# of a texture at 256–512 MB passed it.)

/// null views in the CPU-only heap the shared heap's free ranges are filled from
const NULL_BLOCK: u32 = 4096;

/// Copy null views (`nulls`, a CPU-only heap of `NULL_BLOCK` of them, even SRV, odd UAV) over
/// descriptors `from..to` of the heap at `cpu`, a block at a time; an odd `from` starts from
/// the block's first UAV, so every even descriptor stays an SRV and every odd one a UAV, as the
/// SRV / UAV pair of a console slot.
unsafe fn fill_nulls(dev: &ID3D12Device, cpu: u64, inc: u32, from: u32, to: u32, nulls: u64) {
    let mut at = from;
    while at < to {
        let skew = at % 2;
        let n = (NULL_BLOCK - skew).min(to - at);
        copy_views(dev, n, cpu + at as u64 * inc as u64, nulls + skew as u64 * inc as u64);
        at += n;
    }
}

/// CopyDescriptorsSimple of `n` view descriptors, past XCL's hook of an adopted device (which
/// looks up the console heaps for each call: `HEAPS`, then `GLOBAL`), so that it can be made
/// with `SLOT_WRITES` held.
unsafe fn copy_views(dev: &ID3D12Device, n: u32, dst: u64, src: u64) {
    let raw = dev.as_raw();
    if super::real::is_adopted(raw) {
        let f: unsafe extern "system" fn(*mut std::ffi::c_void, u32, u64, u64, u32) = std::mem::transmute(super::real::orig_fn(raw, 24));
        return f(raw, n, dst, src, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32);
    }
    dev.CopyDescriptorsSimple(n, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: dst as usize }, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: src as usize }, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

/// Which PC resource the descriptor at each index of the shared heap names, and the other way
/// round – so that the views of a resource that is destroyed can be made null views.
///
/// A title keeps T#s (each with an index in the shared heap) in its own memory, and its shaders
/// read them from there. On the console a T# of a released texture still names memory (leftover
/// bytes, nothing faults), and a title can sample a texture it has released (a material table
/// not updated yet, a sample of weight zero). On the PC the slot would keep a descriptor of a
/// destroyed resource: a shader that samples it touches an unmapped address.
struct Names {
    /// index → the resource its descriptor names (0: none, a null view, or not known; `GONE`: the
    /// null view `resource_gone` wrote there); the truth
    slots: Vec<AtomicUsize>,
    /// resource → indexes that named it (some may name another since: `slots` says)
    by: super::sharded::Sharded<Vec<u32>>,
}

impl Names {
    fn new(n: usize) -> Names {
        Names { slots: (0..n).map(|_| AtomicUsize::new(0)).collect(), by: super::sharded::Sharded::new() }
    }

    fn name(&self, i: u32) -> usize {
        self.slots.get(i as usize).map_or(0, |s| s.load(Relaxed))
    }

    /// The descriptor at `i` names `res` now (0: none; `GONE`: it was made the null view of a
    /// destroyed resource's). With `SLOT_WRITES` held, along with the write of the descriptor.
    fn note(&self, i: u32, res: usize) {
        let Some(s) = self.slots.get(i as usize) else { return };
        if s.swap(res, Relaxed) == res || res == 0 || res == GONE {
            return;
        }
        self.by.with(res, |v| {
            v.push(i);
            // a resource whose views are made again and again: only the indexes that still name it
            if v.len() >= 64 && v.len().is_power_of_two() {
                v.retain(|&j| self.name(j) == res);
                v.sort_unstable();
                v.dedup();
            }
        });
    }

    fn clear(&self, from: u32, to: u32) {
        for i in from..to {
            if let Some(s) = self.slots.get(i as usize) {
                s.store(0, Relaxed);
            }
        }
    }

    /// The indexes whose descriptor may name `res` (None: none ever did – the common case, one
    /// lookup), forgotten here.
    fn candidates(&self, res: usize) -> Option<Vec<u32>> {
        self.by.remove(res)
    }

    /// Whether the descriptor at `i` still names `res`: then it is `GONE` from now on (the caller
    /// writes a null view there, with `SLOT_WRITES` held). One overwritten since names another
    /// resource and stays.
    fn claim(&self, i: u32, res: usize) -> bool {
        self.slots.get(i as usize).is_some_and(|s| s.compare_exchange(res, GONE, Relaxed, Relaxed).is_ok())
    }
}

/// What `Names` (and `names_of`) give for a descriptor `resource_gone` made a null view. The
/// title's own CPU-only mirror slot still holds the destroyed resource's view there, so a copy
/// of that slot writes the null view in the shared heap instead of the mirror's (`publish_range`).
pub const GONE: usize = usize::MAX;

/// (made with the first view written to the shared heap: 8 bytes an index)
static NAMES: std::sync::OnceLock<Names> = std::sync::OnceLock::new();

/// Writes of descriptors into the shared heap and of what they name (`NAMES`), one at a time:
/// a null view for a destroyed resource must not come over a view another thread just wrote.
/// Taken before `GLOBAL` or `HEAPS`, never with them held.
static SLOT_WRITES: Mutex<()> = Mutex::new(());

fn names() -> &'static Names {
    NAMES.get_or_init(|| Names::new(GLOBAL_MAX as usize))
}

/// What null views are copied from: the device, the shared heap's start and increment, the
/// null views' start (None: no shared heap, or no null views).
fn null_source() -> Option<(ID3D12Device, u64, u32, u64)> {
    GLOBAL.lock().unwrap().as_ref().and_then(|s| s.nulls.as_ref().map(|n| (s.dev.clone(), s.cpu, s.inc, n.1)))
}

/// The PC resource `res` is destroyed next: every descriptor of the shared heap that still names
/// it becomes the null view of its kind (SRV at even indexes, UAV at odd). The number of them.
/// A resource that never had a view in the shared heap costs one lookup.
pub fn resource_gone(res: usize) -> usize {
    let Some(n) = NAMES.get() else { return 0 };
    let Some(candidates) = n.candidates(res) else { return 0 };
    let Some((dev, cpu, inc, nulls)) = null_source() else { return 0 };
    let _w = SLOT_WRITES.lock().unwrap();
    let mut nulled = 0;
    for i in candidates {
        if n.claim(i, res) {
            unsafe { fill_nulls(&dev, cpu, inc, i, i + 1, nulls) };
            nulled += 1;
        }
    }
    nulled
}

/// view slots per page of a heap's place in the shared heap (2 PC descriptors each)
pub const PAGE_SLOTS: u32 = 256;
/// a page with no place in the shared heap yet
const UNMAPPED: u32 = u32::MAX;

/// PC descriptors of page `p` of a heap of `n` slots: 2 per slot, and the last page only as
/// long as the heap (a heap of a few slots takes a few descriptors, not a whole page's).
fn page_descriptors(n: u32, p: u32) -> u32 {
    2 * n.saturating_sub(p * PAGE_SLOTS).clamp(1, PAGE_SLOTS)
}

/// A first-fit range of `count` in `free` (released ranges), else at `next` below `cap`. A freed
/// range ending at the cursor is contiguous with the tail: take it and bump over into the tail.
fn take_range(free: &mut Vec<(u32, u32)>, next: &mut u32, cap: u32, count: u32) -> Option<u32> {
    if let Some(i) = free.iter().position(|&(_, n)| n >= count) {
        let (f, n) = free[i];
        if n == count {
            free.swap_remove(i);
        } else {
            free[i] = (f + count, n - count);
        }
        return Some(f);
    }
    // no single freed range is big enough: one ending at the cursor continues into the tail
    if let Some(i) = free.iter().position(|&(f, n)| f + n == *next && (n as u64 + (cap - *next) as u64) >= count as u64) {
        let (f, n) = free[i];
        free.swap_remove(i);
        *next += count - n;
        return Some(f);
    }
    if cap - *next < count {
        return None;
    }
    *next += count;
    Some(*next - count)
}

/// Descriptors of the shared heap on `dev`: what it takes (OPTIONS19; 1 000 000 without it), at
/// most `GLOBAL_MAX`.
unsafe fn global_cap(dev: &ID3D12Device) -> u32 {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS19::default();
    let max = if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS19, &mut o as *mut _ as *mut std::ffi::c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.MaxViewDescriptorHeapSize > 0 {
        o.MaxViewDescriptorHeapSize
    } else {
        1_000_000
    };
    max.min(GLOBAL_MAX)
}

/// The shared shader-visible heap, made on first use. False: there is none (it could not be
/// made) – view heaps then keep their own.
pub unsafe fn shared_heap(dev: &ID3D12Device) -> bool {
    {
        let g = GLOBAL.lock().unwrap();
        if g.is_some() {
            return true;
        }
        if GLOBAL_FAILED.load(Relaxed) {
            return false;
        }
    }
    // Made and filled without `GLOBAL` held: the device may be adopted, and its descriptor hooks
    // look up the console heaps (`HEAPS`, then `GLOBAL`). Two threads that get here at once both
    // make one; the first to store it wins.
    let cap = global_cap(dev);
    let d = D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: cap, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 };
    let heap = match dev.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&d) {
        Ok(h) => h,
        Err(e) => {
            GLOBAL_FAILED.store(true, Relaxed);
            crate::xlog_warn!("console descriptor heaps: no shared shader-visible heap of {cap} ({e}) – each keeps its own");
            return false;
        }
    };
    let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    let cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr as u64;
    let start = std::time::Instant::now();
    null_views(dev, cpu, inc);
    // D3D12 leaves a new heap's descriptors undefined: every one XCL has not handed out holds a
    // null view, so an index a shader takes from a stale or garbage T# reads zeros, never a
    // descriptor of whatever the memory held (the memory windows' views come over theirs next)
    let d = D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: NULL_BLOCK, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE, NodeMask: 0 };
    let nulls = dev.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&d).ok().map(|h| {
        let at = h.GetCPUDescriptorHandleForHeapStart().ptr as u64;
        null_views_in(dev, at, inc, NULL_BLOCK as u64);
        fill_nulls(dev, cpu, inc, crate::gpu::recomp::MEM_HEAP_BASE, cap, at);
        (h, at)
    });
    crate::xlog_info!(
        "console descriptor heaps: one shader-visible PC heap of {cap} view descriptors for all of them ({})",
        if nulls.is_some() { format!("null views in the rest, {} ms", start.elapsed().as_millis()) } else { "no null views past the first: no CPU-only heap for them".into() }
    );
    if cap < GLOBAL_MAX {
        crate::xlog_warn!(
            "console descriptor heaps: the shared view heap has {cap} descriptors, fewer than the {GLOBAL_MAX} translated shaders allow for: a texture whose T# index is from {cap} to {} reads past the heap's end (their guard does not cover that)",
            GLOBAL_MAX - 1
        );
    }
    let mut g = GLOBAL.lock().unwrap();
    if g.is_none() {
        *g = Some(Global {
            dev: dev.clone(),
            cpu,
            gpu: heap.GetGPUDescriptorHandleForHeapStart().ptr,
            inc,
            heap,
            cap,
            // the memory windows' raw UAVs (`set_window_views`) come first
            next: crate::gpu::recomp::MEM_HEAP_BASE + crate::gpu::uma::MAX_HEAP_WINDOWS as u32,
            free: Vec::new(),
            nulls,
        });
    }
    true
}

/// Null views at descriptors `0..MEM_HEAP_BASE` of the shared heap (an SRV / UAV pair per
/// console slot: even SRV, odd UAV), where shaders land for a T# of zeros (word 0, the PC index,
/// is 0): they read zeros there rather than whatever an unwritten descriptor holds.
unsafe fn null_views(dev: &ID3D12Device, cpu: u64, inc: u32) {
    null_views_in(dev, cpu, inc, crate::gpu::recomp::MEM_HEAP_BASE as u64)
}

/// `n` null views from `cpu`: even SRV, odd UAV (`null_views`).
unsafe fn null_views_in(dev: &ID3D12Device, cpu: u64, inc: u32, n: u64) {
    for k in 0..n {
        let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + k * inc as u64) as usize };
        if k % 2 == 0 {
            let d = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } },
            };
            dev.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&d), at);
        } else {
            let d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_UAV::default() },
            };
            dev.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&d), at);
        }
    }
}

static WINDOW_VIEWS_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the memory windows could not be put in the shared heap (`set_window_views`):
/// translated shaders then switch over root UAVs (`pso::heap_windows`).
pub fn window_views_failed() -> bool {
    WINDOW_VIEWS_FAILED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The memory windows' buffers `(resource, bytes)` as raw UAVs at `MEM_HEAP_BASE + k` of the
/// shared heap, made now if need be (`Options::heap_windows`: shaders take window k from there).
/// Active windows (`bytes > 0`) in wider layouts (8 / 16) must all have a valid buffer; a slot
/// without one (`(0, 0)`: a window with no memory) gets a null raw UAV – shaders read 0 there and
/// their stores go nowhere. (It once aliased window 0: what was at such an address read window
/// 0's bytes, and GPU writes there overwrote its descriptor heaps.)
pub unsafe fn set_window_views(dev: &ID3D12Device, windows: &[(usize, usize)]) -> bool {
    let count = windows.len().max(crate::gpu::uma::MAX_WINDOWS);
    let complete = count <= crate::gpu::uma::MAX_WINDOWS || windows.iter().all(|&(raw, bytes)| raw != 0 || bytes == 0);
    let ok = count <= crate::gpu::uma::MAX_HEAP_WINDOWS && complete && windows.first().is_some_and(|w| w.0 != 0) && shared_heap(dev) && {
        let (device, cpu, inc) = {
            let g = GLOBAL.lock().unwrap();
            let s = g.as_ref().unwrap();
            (s.dev.clone(), s.cpu, s.inc)
        };
        // The device may be adopted: its UAV hook re-enters descriptor bookkeeping.
        for k in 0..count {
            let (raw, bytes) = windows.get(k).copied().unwrap_or((0, 0));
            let raw = raw as *mut std::ffi::c_void;
            let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + (crate::gpu::recomp::MEM_HEAP_BASE as u64 + k as u64) * inc as u64) as usize };
            match ID3D12Resource::from_raw_borrowed(&raw) {
                Some(r) => device.CreateUnorderedAccessView(r, None::<&ID3D12Resource>, Some(&crate::gpu::recomp::runner::raw_window_uav(bytes as u64)), at),
                None => device.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(&crate::gpu::recomp::runner::raw_window_uav(crate::gpu::uma::MAX_SIZE as u64)),
                    at,
                ),
            }
        }
        true
    };
    if !ok {
        WINDOW_VIEWS_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
        crate::xlog_warn!("memory windows: could not bind {count} window(s) in the shared descriptor heap");
    }
    ok
}

/// `count` descriptors of the shared heap: the index of the first. None: no shared heap, or
/// no room left in it.
fn reserve_global(count: u32) -> Option<u32> {
    let mut g = GLOBAL.lock().unwrap();
    let s = g.as_mut()?;
    let (cap, mut next) = (s.cap, s.next);
    let r = take_range(&mut s.free, &mut next, cap, count);
    s.next = next;
    let used = in_use(s.next, &s.free);
    if r.is_some() {
        let placed = PAGES_PLACED.fetch_add(1, Relaxed) + 1;
        let marks = marks_passed(used, cap);
        if marks > FILL_SAID.fetch_max(marks, Relaxed) {
            crate::xlog_info!("console descriptor heaps: the shared heap is past {} % full: {used} of its {cap} descriptors in use, {placed} pages placed", FILL_MARKS[marks - 1]);
        }
    } else {
        let n = NO_ROOM.fetch_add(1, Relaxed) + 1;
        if n.is_power_of_two() {
            crate::xlog_warn!(
                "console descriptor heaps: the shared heap ({cap}) is full – a page of slots found no place there ({n} times so far; {used} descriptors in use, {} pages placed): those slots name the null views at 0 / 1, shaders see none of their views",
                PAGES_PLACED.load(Relaxed)
            );
        }
    }
    r
}

/// Pages that asked for a place in the shared heap and found no room (each try counts).
static NO_ROOM: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Pages of console heaps with a place in the shared heap now.
static PAGES_PLACED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// How many of `FILL_MARKS` the log has said the shared heap passed.
static FILL_SAID: AtomicUsize = AtomicUsize::new(0);

/// How full the shared heap is said to be in the log (percent), each once, the first time.
const FILL_MARKS: [u32; 3] = [50, 75, 90];

/// Descriptors of the shared heap handed out: those below the cursor `next` but the released
/// ranges in `free` (the null views and the memory windows at its start count).
fn in_use(next: u32, free: &[(u32, u32)]) -> u32 {
    next.saturating_sub(free.iter().map(|&(_, n)| n).sum())
}

/// How many of `FILL_MARKS` `used` descriptors of a shared heap of `cap` are past.
fn marks_passed(used: u32, cap: u32) -> usize {
    FILL_MARKS.iter().filter(|&&m| used as u64 * 100 > m as u64 * cap as u64).count()
}

/// The place in the shared heap of the page slot `k` of `h` is in – (index of the page's
/// first SRV there, the page's first slot) – given now if the page has none yet. None: the
/// heap is not in the shared heap, or that is full.
fn page_of(h: &mut Heap, k: u32) -> Option<(u32, u32)> {
    let p = (k / PAGE_SLOTS) as usize;
    let mut first = *h.pages.get(p)?;
    if first == UNMAPPED {
        first = reserve_global(page_descriptors(h.n, p as u32))?;
        h.pages[p] = first;
    }
    Some((first, p as u32 * PAGE_SLOTS))
}

/// The shared range starting at `first` (`count` descriptors) is free again. Its views (of the
/// released heap's resources, which may be gone) are made null views first, before another
/// heap can be given the range: no shader that still has an index there reads them.
pub fn release_global(first: u32, count: u32) {
    if count == 0 {
        return;
    }
    if let Some((dev, cpu, inc, nulls)) = null_source() {
        // (the range is still the released heap's)
        let _w = SLOT_WRITES.lock().unwrap();
        unsafe { fill_nulls(&dev, cpu, inc, first, first + count, nulls) };
        if let Some(n) = NAMES.get() {
            n.clear(first, first + count);
        }
    }
    if let Some(s) = GLOBAL.lock().unwrap().as_mut() {
        merge_release(&mut s.free, first, count);
    }
}

/// Add a released range to `free`, merging the ranges it touches: adjacent releases become
/// one range, so first-fit (and the cursor-straddle in `take_range`) sees the whole run.
fn merge_release(free: &mut Vec<(u32, u32)>, first: u32, count: u32) {
    free.push((first, count));
    free.sort_unstable();
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(free.len());
    for (f, n) in free.drain(..) {
        match out.last_mut() {
            Some(last) if f <= last.0 + last.1 => last.1 = last.1.max(f + n - last.0),
            _ => out.push((f, n)),
        }
    }
    *free = out;
}

/// The shader-visible PC heap lists bind for view descriptors (None: none shared).
pub fn global_heap() -> Option<ID3D12DescriptorHeap> {
    GLOBAL.lock().unwrap().as_ref().map(|s| s.heap.clone())
}

/// Whether `obj` (a title's view heap) has its descriptors in the shared heap.
pub fn in_global(obj: usize) -> bool {
    HEAPS.lock().unwrap().iter().any(|h| h.obj == obj && !h.sampler && !h.pages.is_empty())
}

/// Memory for a heap of `bytes` from the title's allocation hook – where the title expects the
/// heap then. Only what descriptor tables can address (window 0) and is committed; anything else
/// goes back to the title.
unsafe fn title_memory(u: &crate::gpu::uma::Uma, bytes: usize) -> Option<usize> {
    use crate::hle::kernelx::{runtime_alloc, runtime_free, D3D_DESCRIPTOR_HEAP};
    let p = runtime_alloc(bytes, D3D_DESCRIPTOR_HEAP)?;
    let readable = crate::gpu::shader::readable(p, bytes);
    if title_memory_fits(u, p, bytes, readable) {
        return Some(p);
    }
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
        let (free, largest) = u.sparse_free_in(0);
        crate::xlog_warn!(
            "console descriptor heap: the title's memory {p:#x} ({bytes:#x} bytes) not used (window {:?}, committed {readable}; window 0 has {} MB free, {} MB at most in one run)",
            u.window_of(p),
            free >> 20,
            largest >> 20
        );
    }
    runtime_free(p, D3D_DESCRIPTOR_HEAP);
    None
}

/// Whether the title's `[p, p + bytes)` can hold a heap's slots: committed (`readable`) and all
/// in window 0. A descriptor table is a 32-bit address, high half 0, which shaders and
/// `words_at` read in window 0 – at `base + (p as u32)`. In another window the tables would name
/// other bytes, and the substitute heap the title then gets does not start at its memory (a
/// retail game asserts that it does).
fn title_memory_fits(u: &crate::gpu::uma::Uma, p: usize, bytes: usize, readable: bool) -> bool {
    readable && bytes > 0 && u.window_of(p) == Some(0) && p.checked_add(bytes - 1).is_some_and(|e| u.window_of(e) == Some(0))
}

/// PC samplers of a console sampler heap of `n` slots: a shader-visible one gets the 2 048 a
/// shader-visible PC sampler heap holds at most, whatever `n` is. Translated shaders index the
/// bound sampler heap with S# word 3 [10:0] (0..2047), and a title builds S#s itself (a border
/// color pointer there): with only `n` PC samplers an index from `n` on read past the heap's end.
/// The samplers past `n` are the zero sampler; the console heap keeps its `n` slots.
pub fn pc_samplers(n: u32, visible: bool) -> u32 {
    if visible {
        n.max(D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE)
    } else {
        n
    }
}

/// `obj` is a PC heap from `pc_cpu` / `pc_gpu` (increment `inc`; `pc_gpu` 0 when not shader
/// visible) with 2·`n` view descriptors or `n` samplers: give it `n` console slots in
/// unified memory. `shared`: its views go to the shared heap (`shared_heap`), a page at a
/// time as slots are used; `visible`: what the title asked for.
pub fn register(obj: usize, n: u32, pc_cpu: u64, pc_gpu: u64, inc: u32, sampler: bool, shared: bool, visible: bool) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    let bytes = n.max(1) * if sampler { SAMPLER_SLOT } else { SLOT };
    // Descriptor tables are 32-bit console addresses (high dword 0): must live in window 0.
    // On the sparse-window path, `sparse::alloc_in(0, ...)` commits fresh zeroed pages so we
    // avoid touching unused pages with `write_bytes` (which would mark them dirty for write-watch).
    let (p, memory) = if let Some(p) = unsafe { title_memory(u, bytes as usize) } {
        (p as *mut u8, Memory::Title)
    } else if u.is_sparse_window(0) {
        let Some((addr, _)) = (unsafe { super::sparse::alloc_in(0, bytes as usize, 256) }) else {
            return false;
        };
        (addr as *mut u8, Memory::Sparse)
    } else {
        let Some(p) = u.alloc_in(0, bytes as usize, 256) else { return false };
        unsafe { std::ptr::write_bytes(p, 0, bytes as usize) };
        (p, Memory::Uma)
    };
    {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 48 {
            crate::xlog_info!(
                "console descriptor heap {obj:#x}: {n} {} slots at {} {:#x}, {}",
                if sampler { "sampler" } else { "view" },
                match memory {
                    Memory::Sparse => "sparse VRAM",
                    Memory::Uma => "UMA",
                    Memory::Title => "the title's memory",
                },
                p as usize - u.base(),
                if visible { "shader visible" } else { "CPU only" }
            );
        }
    }
    let pages = if shared && !sampler && GLOBAL.lock().unwrap().is_some() {
        vec![UNMAPPED; n.div_ceil(PAGE_SLOTS).max(1) as usize]
    } else if shared && sampler && GLOBAL_SAMPLER.lock().unwrap().is_some() {
        vec![UNMAPPED; n.div_ceil(SAMPLER_PAGE_SLOTS).max(1) as usize]
    } else {
        Vec::new()
    };
    HEAPS.lock().unwrap().push(Heap { obj, shadow: p as usize, n, pc_cpu, pc_gpu, inc, sampler, pages, visible, memory });
    true
}

/// The heap object was released for good: its slots go back to unified memory.
pub fn unregister(obj: usize) {
    let gone = {
        let mut heaps = HEAPS.lock().unwrap();
        heaps.iter().position(|h| h.obj == obj).map(|i| heaps.swap_remove(i))
    };
    if let Some(h) = gone {
        for (p, &f) in h.pages.iter().enumerate().filter(|&(_, &f)| f != UNMAPPED) {
            if h.sampler {
                release_global_sampler(f, sampler_page_descriptors(h.n, p as u32));
            } else {
                release_global(f, page_descriptors(h.n, p as u32));
                PAGES_PLACED.fetch_sub(1, Relaxed);
            }
        }
        match h.memory {
            Memory::Sparse => {
                unsafe { super::sparse::free(h.shadow) };
            }
            Memory::Uma => {
                if let Some(u) = crate::gpu::uma::get() {
                    u.free(h.shadow);
                }
            }
            Memory::Title => unsafe { crate::hle::kernelx::runtime_free(h.shadow, crate::hle::kernelx::D3D_DESCRIPTOR_HEAP) },
        }
    }
}

/// Whether heap `obj` holds samplers, and the 32-bit console address of its slots.
pub fn base(obj: usize) -> Option<(bool, u32)> {
    HEAPS.lock().unwrap().iter().find(|h| h.obj == obj).map(|h| (h.sampler, h.shadow as u32))
}

/// Start of the console slots of heap `obj` (CPU handle = GPU handle on the console).
pub fn start(obj: usize) -> Option<u64> {
    HEAPS.lock().unwrap().iter().find(|h| h.obj == obj).map(|h| h.shadow as u64)
}

/// A console slot: where its descriptor goes and its PC descriptors.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub at: usize,
    /// views: PC heap index of the SRV (the UAV is at + 1); samplers: of the sampler
    pub index: u32,
    /// PC CPU handles of the SRV / UAV where views are made (the title's heap)
    pub pc_srv: u64,
    pub pc_uav: u64,
    /// PC GPU handle of the UAV (0: heap not shader visible)
    pub pc_gpu_uav: u64,
    pub sampler: bool,
    /// CPU handle of the SRV in the shared heap (0: the heap is not in it)
    pub shared_srv: u64,
    /// the slot's number in its heap
    pub k: u32,
}

/// The slot behind a console CPU handle (None: not one of XCL's console heaps). A view slot
/// of a heap in the shared heap gets its page's place there now if the page has none yet; one
/// whose page finds no room there gets index 0, the null SRV / UAV pair.
pub fn slot(cpu: u64) -> Option<Slot> {
    let mut heaps = HEAPS.lock().unwrap();
    let h = heaps.iter_mut().find(|h| cpu >= h.shadow as u64 && cpu < (h.shadow + (h.n * h.slot_bytes()) as usize) as u64)?;
    let k = ((cpu - h.shadow as u64) / h.slot_bytes() as u64) as u32;
    if h.sampler {
        let shared = sampler_page_of(h, k).and_then(|(first, k0)| {
            let g = GLOBAL_SAMPLER.lock().unwrap();
            let g = g.as_ref()?;
            let i = first + (k - k0);
            Some((i, g.cpu + i as u64 * g.inc as u64))
        });
        // If this sampler heap is hosted in `GLOBAL_SAMPLER` (`!h.pages.is_empty()`), an exhausted
        // shared heap must map overflowed slots to reserved zero-sampler slot 0 rather than
        // aliasing unrelated heap entries at local index `k`.
        let (index, shared_srv) = shared.unwrap_or_else(|| (sampler_unmapped_index(!h.pages.is_empty(), k), 0));
        return Some(Slot { at: h.shadow + (k * SAMPLER_SLOT) as usize, index, pc_srv: h.pc_cpu + (k * h.inc) as u64, pc_uav: 0, pc_gpu_uav: 0, sampler: true, shared_srv, k });
    }
    let uav = ((2 * k + 1) * h.inc) as u64;
    let shared = page_of(h, k).and_then(|(first, k0)| {
        let g = GLOBAL.lock().unwrap();
        let g = g.as_ref()?;
        let i = first + 2 * (k - k0);
        Some((i, g.gpu + (i as u64 + 1) * g.inc as u64, g.cpu + i as u64 * g.inc as u64))
    });
    // A heap meant for the shared heap whose page found no room there: lists bind only the
    // shared heap, where 2k is another heap's view, a memory window's raw UAV (k from 2 to 9) or
    // past the heap's end; its T#s name the null views at 0 / 1 instead, and nothing is copied.
    let own = if h.pages.is_empty() { (2 * k, if h.pc_gpu == 0 { 0 } else { h.pc_gpu + uav }, 0) } else { (0, 0, 0) };
    let (index, pc_gpu_uav, shared_srv) = shared.unwrap_or(own);
    Some(Slot { at: h.shadow + (k * SLOT) as usize, index, pc_srv: h.pc_cpu + (2 * k * h.inc) as u64, pc_uav: h.pc_cpu + uav, pc_gpu_uav, sampler: false, shared_srv, k })
}

/// The view (or sampler) just made in slot `s` (its SRV, or with `uav` its UAV) of the PC
/// resource `res` (0: a null view; samplers: 0) goes to the shared heap.
pub unsafe fn publish(s: &Slot, uav: bool, res: usize) {
    if s.shared_srv == 0 {
        return;
    }
    if s.sampler {
        if let Some(g) = GLOBAL_SAMPLER.lock().unwrap().as_ref() {
            g.dev.CopyDescriptorsSimple(
                1,
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.shared_srv as usize },
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.pc_srv as usize },
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            );
        }
        return;
    }
    let Some((dev, inc)) = GLOBAL.lock().unwrap().as_ref().map(|g| (g.dev.clone(), g.inc)) else { return };
    let (dst, src) = if uav { (s.shared_srv + inc as u64, s.pc_uav) } else { (s.shared_srv, s.pc_srv) };
    let _w = SLOT_WRITES.lock().unwrap();
    copy_views(&dev, 1, dst, src);
    names().note(s.index + uav as u32, res);
}

/// The PC resources the SRV and the UAV of view slot `s` name in the shared heap (0: not known,
/// a sampler, or the slot has no place there): what `publish_range` takes for a copy of it.
pub fn names_of(s: &Slot) -> (usize, usize) {
    if s.sampler {
        return (0, 0);
    }
    match (s.shared_srv, NAMES.get()) {
        (0, _) | (_, None) => (0, 0),
        (_, Some(n)) => (n.name(s.index), n.name(s.index + 1)),
    }
}

/// `n` view or sampler slots from `first` were copied (in the title's heap): they go to the shared
/// heap, a run per page (a heap's pages lie apart there). `names`: what the source view slots' SRV
/// and UAV name (`names_of`), one pair a slot (none for samplers). A source descriptor that is
/// `GONE` gets the null view in the shared heap, not the copy of the destroyed resource's view
/// the title's heap still holds.
pub unsafe fn publish_range(first: &Slot, n: u32, names: &[(usize, usize)]) {
    let page_slots = if first.sampler { SAMPLER_PAGE_SLOTS } else { PAGE_SLOTS };
    let step = if first.sampler { SAMPLER_SLOT } else { SLOT };
    // views: the shared heap's device, taken once (no shared heap: nothing to publish)
    let dev = if first.sampler {
        None
    } else {
        let Some(d) = GLOBAL.lock().unwrap().as_ref().map(|g| g.dev.clone()) else { return };
        Some(d)
    };
    // the null views, taken before `SLOT_WRITES` when a source descriptor is `GONE`
    let nulls = if names.iter().any(|&(srv, uav)| srv == GONE || uav == GONE) { null_source() } else { None };
    let mut done = 0;
    while done < n {
        let s = if done == 0 { Some(*first) } else { slot((first.at + (done * step) as usize) as u64) };
        let Some(s) = s else { return };
        let run = (page_slots - s.k % page_slots).min(n - done);
        if s.shared_srv != 0 {
            if s.sampler {
                if let Some(g) = GLOBAL_SAMPLER.lock().unwrap().as_ref() {
                    g.dev.CopyDescriptorsSimple(
                        run,
                        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.shared_srv as usize },
                        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.pc_srv as usize },
                        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                    );
                }
            } else if let Some(dev) = &dev {
                let _w = SLOT_WRITES.lock().unwrap();
                copy_views(dev, 2 * run, s.shared_srv, s.pc_srv);
                let m = self::names();
                for j in 0..run {
                    let (srv, uav) = names.get((done + j) as usize).copied().unwrap_or((0, 0));
                    for (i, res) in [(s.index + 2 * j, srv), (s.index + 2 * j + 1, uav)] {
                        if let Some((ndev, cpu, inc, at)) = nulls.as_ref().filter(|_| res == GONE) {
                            fill_nulls(ndev, *cpu, *inc, i, i + 1, *at);
                        }
                        m.note(i, res);
                    }
                }
            }
        }
        done += run;
    }
}

/// view slot address → the resource its view names (`XCL_TRACE_FRAME`, `XCL_DUMP_TARGETS`, the page budget)
static RES: Mutex<Option<std::collections::HashMap<usize, usize>>> = Mutex::new(None);

/// The view in slot `s` names `res` (0: none).
pub fn note_resource(s: &Slot, res: usize) {
    // (the page budget keeps the textures views name: `vram::viewed`)
    let budget = super::vram::budget_on();
    if s.sampler || !(budget || super::present::trace_frame() || super::targets::enabled() || super::capture::enabled()) {
        return;
    }
    let mut g = RES.lock().unwrap();
    let m = g.get_or_insert_with(Default::default);
    let old = if res == 0 { m.remove(&s.at) } else { m.insert(s.at, res) };
    if budget && old != Some(res) {
        super::vram::viewed(old, res);
    }
}

/// The resource of the view slot at UMA address `at`.
pub fn resource_at(at: usize) -> Option<usize> {
    RES.lock().unwrap().as_ref()?.get(&at).copied()
}

/// The PC resource the descriptor at index `i` of the shared heap names (0: none or not known;
/// `GONE`: the null view of a destroyed resource) – what a T# with that index in word 0 samples,
/// whichever slot the T# lies in (`XCL_TRACE_FRAME` diagnostics).
pub fn named_by_index(i: u32) -> usize {
    NAMES.get().map_or(0, |n| n.name(i))
}

pub fn write(s: &Slot, words: &[u32]) {
    let n = words.len().min(if s.sampler { 4 } else { 8 });
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), s.at as *mut u32, n) };
}

/// Copy a console slot's descriptor to another slot of the same kind, the PC heap index
/// in it (T# word 0, S# word 3 [11:0]) renumbered for the destination. Buffer V# carry
/// their address and stay as they are.
pub fn copy_words(src: &Slot, dst: &Slot) {
    let n = if src.sampler { 4 } else { 8 };
    let mut w = [0u32; 8];
    unsafe { std::ptr::copy_nonoverlapping(src.at as *const u32, w.as_mut_ptr(), n) };
    if src.sampler {
        w[3] = (w[3] & !0xFFF) | (dst.index & 0xFFF);
    } else if w[3] >> 28 >= 8 {
        w[0] = dst.index;
    }
    write(dst, &w[..n]);
    if !src.sampler {
        note_resource(dst, resource_at(src.at).unwrap_or(0));
    }
}

/// `XCL_TRACE_FRAME`: what the slots of each shader-visible view heap hold – images whose
/// PC index is the slot's own (made or copied through the device) or another (copied by the
/// title itself, as memory), buffers, empty.
pub fn census() {
    let heaps = HEAPS.lock().unwrap();
    // samplers: S# word 3 [11:0] is the PC index in the sampler heap the slot belongs to
    for h in heaps.iter().filter(|h| h.sampler) {
        let (mut empty, mut own, mut other) = (0u32, 0u32, 0u32);
        let mut examples: Vec<(u32, u32)> = Vec::new();
        for k in 0..h.n {
            let w = unsafe { std::slice::from_raw_parts((h.shadow + (k * SAMPLER_SLOT) as usize) as *const u32, 4) };
            if w.iter().all(|&x| x == 0) {
                empty += 1;
            } else if w[3] & 0xFFF == sampler_own_index(&h.pages, k) {
                own += 1;
            } else {
                other += 1;
                if examples.len() < 8 {
                    examples.push((k, w[3] & 0xFFF));
                }
            }
        }
        crate::xlog_info!(
            "frame trace: sampler heap {:#x} ({} slots, {}): {own} with their own index, {other} with another (slot, index: {examples:?}), {empty} empty",
            h.obj,
            h.n,
            if h.visible { "shader visible" } else { "CPU only" }
        );
    }
    for h in heaps.iter().filter(|h| !h.sampler) {
        let (mut empty, mut own, mut other, mut buffers) = (0u32, 0u32, 0u32, 0u32);
        let mut examples: Vec<(u32, u32)> = Vec::new();
        let own_index = |k: u32| match h.pages.get((k / PAGE_SLOTS) as usize) {
            Some(&f) if f != UNMAPPED => f + 2 * (k % PAGE_SLOTS),
            Some(_) => 0,
            None => 2 * k,
        };
        for k in 0..h.n {
            let w = unsafe { std::slice::from_raw_parts((h.shadow + (k * SLOT) as usize) as *const u32, 8) };
            if w.iter().all(|&x| x == 0) {
                empty += 1;
            } else if w[3] >> 28 >= 8 {
                if w[0] == own_index(k) {
                    own += 1;
                } else {
                    other += 1;
                    if examples.len() < 8 {
                        examples.push((k, w[0]));
                    }
                }
            } else {
                buffers += 1;
            }
        }
        crate::xlog_info!(
            "frame trace: view heap {:#x} ({} slots, {}): {own} images with their own PC index, {other} with another (slot, index: {examples:?}), {buffers} buffers, {empty} empty",
            h.obj,
            h.n,
            if h.visible { "shader visible" } else { "CPU only" }
        );
    }
}

/// The descriptor at a console GPU address (low 32 bits, as in a descriptor table), if it
/// is in one of the console heaps.
pub fn words_at(addr32: u32) -> Option<[u32; 8]> {
    let u = crate::gpu::uma::get()?;
    let s = slot((u.base() + addr32 as usize) as u64)?;
    let mut w = [0u32; 8];
    unsafe { std::ptr::copy_nonoverlapping(s.at as *const u32, w.as_mut_ptr(), if s.sampler { 4 } else { 8 }) };
    Some(w)
}

// ---------------------------------------------------------------------------------------
// Shared shader-visible PC sampler heap (`GLOBAL_SAMPLER`, up to D3D12's 2048 limit):
// allows 16-byte S# descriptors copied via raw CPU `memcpy` between CPU-only and
// shader-visible sampler heaps to keep valid PC sampler indices without CopyDescriptors*.
// ---------------------------------------------------------------------------------------

/// D3D12 maximum shader-visible sampler heap size.
const GLOBAL_SAMPLER_MAX: u32 = 2048;
/// Sampler slots allocated per page from `GLOBAL_SAMPLER` (small page size minimizes waste across
/// many small sampler heaps while staying within D3D12's 2048-sampler hardware limit).
const SAMPLER_PAGE_SLOTS: u32 = 8;

/// Fallback index for a sampler slot when no `GLOBAL_SAMPLER` page is mapped:
/// hosted heaps (`hosted == true`, downgraded to CPU-only) use reserved slot `0` (immutable zero
/// sampler) so exhaustion never aliases another heap's sampler at `k`; unhosted heaps keep `k`.
fn sampler_unmapped_index(hosted: bool, k: u32) -> u32 {
    if hosted { 0 } else { k }
}

/// Expected PC sampler index for slot `k` of a sampler heap with shared-page table `pages`.
fn sampler_own_index(pages: &[u32], k: u32) -> u32 {
    match pages.get((k / SAMPLER_PAGE_SLOTS) as usize) {
        Some(&f) if f != UNMAPPED => f + (k % SAMPLER_PAGE_SLOTS),
        _ => sampler_unmapped_index(!pages.is_empty(), k),
    }
}

struct GlobalSampler {
    dev: ID3D12Device,
    heap: ID3D12DescriptorHeap,
    cpu: u64,
    inc: u32,
    cap: u32,
    next: u32,
    free: Vec<(u32, u32)>,
}

unsafe impl Send for GlobalSampler {}

static GLOBAL_SAMPLER: Mutex<Option<GlobalSampler>> = Mutex::new(None);
static GLOBAL_SAMPLER_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn sampler_page_descriptors(n: u32, p: u32) -> u32 {
    n.saturating_sub(p * SAMPLER_PAGE_SLOTS).clamp(1, SAMPLER_PAGE_SLOTS)
}

/// Shared shader-visible PC sampler heap (2048 descriptors), created lazily on first sampler heap.
pub unsafe fn shared_sampler_heap(dev: &ID3D12Device) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    if std::env::var("XCL_SHARED_SAMPLER_HEAP").is_ok_and(|v| v.trim() == "0") {
        return false;
    }
    let mut g = GLOBAL_SAMPLER.lock().unwrap();
    if g.is_none() {
        if GLOBAL_SAMPLER_FAILED.load(Relaxed) {
            return false;
        }
        let d = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            NumDescriptors: GLOBAL_SAMPLER_MAX,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        let heap: ID3D12DescriptorHeap = match dev.CreateDescriptorHeap(&d) {
            Ok(h) => h,
            Err(e) => {
                GLOBAL_SAMPLER_FAILED.store(true, Relaxed);
                crate::xlog_warn!("console descriptor heaps: no shared shader-visible sampler heap ({e}) – each keeps its own");
                return false;
            }
        };
        let cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr as u64;
        let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        let zero = zero_sampler();
        for k in 0..GLOBAL_SAMPLER_MAX as u64 {
            dev.CreateSampler(&zero, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + k * inc as u64) as usize });
        }
        crate::xlog_info!("console descriptor heaps: one shader-visible PC sampler heap of {GLOBAL_SAMPLER_MAX} descriptors for all sampler heaps");
        *g = Some(GlobalSampler {
            dev: dev.clone(),
            heap,
            cpu,
            inc,
            cap: GLOBAL_SAMPLER_MAX,
            // Reserve slot 0 as the immutable point-wrap zero sampler for empty/zeroed S# entries.
            next: 1,
            free: Vec::new(),
        });
    }
    true
}

fn reserve_global_sampler(count: u32) -> Option<u32> {
    let mut g = GLOBAL_SAMPLER.lock().unwrap();
    let s = g.as_mut()?;
    let (cap, mut next) = (s.cap, s.next);
    let r = take_range(&mut s.free, &mut next, cap, count);
    s.next = next;
    if r.is_none() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
            crate::xlog_warn!("console descriptor heaps: the shared sampler heap ({cap}) is full – overflowed sampler slots map to reserved slot 0");
        }
    }
    r
}

fn sampler_page_of(h: &mut Heap, k: u32) -> Option<(u32, u32)> {
    let p = (k / SAMPLER_PAGE_SLOTS) as usize;
    let mut first = *h.pages.get(p)?;
    if first == UNMAPPED {
        first = reserve_global_sampler(sampler_page_descriptors(h.n, p as u32))?;
        h.pages[p] = first;
    }
    Some((first, p as u32 * SAMPLER_PAGE_SLOTS))
}

fn release_global_sampler(first: u32, count: u32) {
    if count == 0 {
        return;
    }
    if let Some(s) = GLOBAL_SAMPLER.lock().unwrap().as_mut() {
        merge_release(&mut s.free, first, count);
    }
}

/// The shared shader-visible PC sampler heap lists bind (None: no shared sampler heap).
pub fn global_sampler_heap() -> Option<ID3D12DescriptorHeap> {
    GLOBAL_SAMPLER.lock().unwrap().as_ref().map(|s| s.heap.clone())
}

/// Whether `obj` (a title's sampler heap) has its descriptors in the shared sampler heap.
pub fn in_global_sampler(obj: usize) -> bool {
    HEAPS.lock().unwrap().iter().any(|h| h.obj == obj && h.sampler && !h.pages.is_empty())
}

/// The zero sampler: what an S# of zeros is on the console (point filtering, wrap, mip 0 only),
/// and what a shader gets whose S# XCL takes for a heap entry it is not (entry 0).
/// `XCL_ZERO_SAMPLER_LINEAR=1` (a diagnostic): linear filtering, clamp, every mip instead; a
/// picture that changes with it samples through the zero sampler.
pub(super) fn zero_sampler() -> D3D12_SAMPLER_DESC {
    static LINEAR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let linear = *LINEAR.get_or_init(|| {
        let on = std::env::var("XCL_ZERO_SAMPLER_LINEAR").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_info!("D3D12.X: XCL_ZERO_SAMPLER_LINEAR=1 - the zero sampler is linear, clamp, all mips (a diagnostic)");
        }
        on
    });
    let address = if linear { D3D12_TEXTURE_ADDRESS_MODE_CLAMP } else { D3D12_TEXTURE_ADDRESS_MODE_WRAP };
    D3D12_SAMPLER_DESC {
        Filter: if linear { D3D12_FILTER_MIN_MAG_MIP_LINEAR } else { D3D12_FILTER_MIN_MAG_MIP_POINT },
        AddressU: address,
        AddressV: address,
        AddressW: address,
        MipLODBias: 0.0,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        BorderColor: [0.0; 4],
        MinLOD: 0.0,
        MaxLOD: if linear { 1000.0 } else { 0.0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every T# index the translated shaders accept, and its UAV at `tw0 + 1`, is inside the
    /// shared heap this GPU gets: an index past the heap's end hangs NVIDIA GPUs (whose heap is
    /// 1 000 000, while the shaders accepted indices up to 2^21 - 2).
    #[test]
    fn the_shaders_t_sharp_index_bound_fits_the_shared_heap() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        let mut dev: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev) }.is_err() {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        let cap = unsafe { global_cap(&dev.unwrap()) };
        let bound = crate::gpu::dxil::lower_image::VIEW_HEAP_MAX_SRV;
        // accepted: tw0 < bound, so the highest UAV index is bound - 1 + 1 = bound
        assert!(bound < cap, "shaders accept indices up to {bound}, the shared heap holds {cap}");
    }

    /// The title's memory holds a heap only when committed and all in window 0: where a Series X
    /// run's view heap was (0xf2e00000 into window 0), not where a Series S run's was (window 1,
    /// 0x1400000 in), nor across the end of window 0, nor outside unified memory.
    #[test]
    fn title_memory_holds_a_heap_only_in_window_0() {
        use crate::gpu::uma::{Uma, WINDOW};
        let u = Uma::for_runtime_windows(16 << 20, 512 << 20, 16).expect("16-window aperture");
        let b = u.base();
        let view_heap = (1_245_184 * SLOT) as usize; // 0x2600000
        assert!(title_memory_fits(&u, b + 0xf2e0_0000, view_heap, true));
        assert!(!title_memory_fits(&u, b + 0xf2e0_0000, view_heap, false), "not committed");
        assert!(!title_memory_fits(&u, b + WINDOW + 0x140_0000, view_heap, true));
        assert!(!title_memory_fits(&u, b + WINDOW - 0x1_0000, view_heap, true), "into window 1");
        assert!(title_memory_fits(&u, b + WINDOW - view_heap, view_heap, true), "the end of window 0");
        // the coherent window, a view window, plain memory
        for p in [b + 3 * WINDOW, b + 4 * WINDOW, 0xf2e0_0000] {
            assert!(!title_memory_fits(&u, p, 0x1_0000, true), "{p:#x}");
        }
        assert!(!title_memory_fits(&u, b + 0x2001_0000, 0, true));
    }

    /// Ranges of the shared heap: released ones first (first fit, split), then new ones, and
    /// none past its size.
    #[test]
    fn shared_heap_ranges() {
        let (mut free, mut next) = (Vec::new(), 0u32);
        assert_eq!(take_range(&mut free, &mut next, 100, 40), Some(0));
        assert_eq!(take_range(&mut free, &mut next, 100, 40), Some(40));
        assert_eq!(take_range(&mut free, &mut next, 100, 40), None, "20 left");
        free.push((0, 40));
        assert_eq!(take_range(&mut free, &mut next, 100, 30), Some(0));
        assert_eq!(free, vec![(30, 10)]);
        assert_eq!(take_range(&mut free, &mut next, 100, 20), Some(80));
        assert_eq!(take_range(&mut free, &mut next, 100, 10), Some(30));
        assert!(free.is_empty());
    }

    /// How full the shared heap is: descriptors below the cursor but the released ones, and the
    /// marks (50, 75, 90 %) that use is past.
    #[test]
    fn shared_heap_use_and_the_marks_it_passes() {
        assert_eq!(in_use(100, &[]), 100);
        assert_eq!(in_use(100, &[(20, 10), (50, 6)]), 84);
        for (used, marks) in [(0, 0), (500, 0), (501, 1), (750, 1), (751, 2), (900, 2), (901, 3), (1000, 3)] {
            assert_eq!(marks_passed(used, 1000), marks, "{used} of 1000");
        }
        assert_eq!(marks_passed(1 << 21, 1 << 21), 3, "no overflow at the largest heap");
    }

    #[test]
    fn a_freed_range_at_the_cursor_continues_into_the_never_used_tail() {
        // the false-full shape: four freed slots end at the cursor and four more were never
        // used; a request for eight is contiguous across the cursor, not exhaustion
        let (mut free, mut next) = (vec![(8u32, 4u32)], 12u32);
        assert_eq!(take_range(&mut free, &mut next, 16, 8), Some(8), "freed range plus tail is one run");
        assert_eq!(next, 16);
        assert!(free.is_empty());
    }

    #[test]
    fn released_ranges_touching_one_another_merge_into_one() {
        let mut free = vec![(0u32, 4), (16, 4)];
        merge_release(&mut free, 4, 8);
        assert_eq!(free, vec![(0, 12), (16, 4)]);
        merge_release(&mut free, 12, 4);
        assert_eq!(free, vec![(0, 20)], "the merge chains across the whole run");
        merge_release(&mut free, 30, 2);
        assert_eq!(free, vec![(0, 20), (30, 2)]);
    }

    #[test]
    fn window_views_do_not_hold_global_during_device_calls() {
        const CHILD: &str = "XCL_TEST_WINDOW_VIEWS_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::console_heap::tests::window_views_do_not_hold_global_during_device_calls", "--nocapture"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            return;
        }
        use std::ffi::c_void;
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
        static UNLOCKED: AtomicBool = AtomicBool::new(true);
        static CALLS: AtomicU32 = AtomicU32::new(0);
        // slot k (bit k) got a null resource
        static NULLS: AtomicU32 = AtomicU32::new(0);
        #[repr(C)]
        struct Object { vt: *const usize }
        unsafe extern "system" fn retain(_this: *mut c_void) -> u32 { 1 }
        unsafe extern "system" fn create_uav(_this: *mut c_void, resource: *mut c_void, _counter: *mut c_void, _desc: *const D3D12_UNORDERED_ACCESS_VIEW_DESC, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
            // The adopted device hook can re-enter descriptor bookkeeping.
            UNLOCKED.fetch_and(GLOBAL.try_lock().is_ok(), Ordering::Relaxed);
            CALLS.fetch_add(1, Ordering::Relaxed);
            if resource.is_null() {
                let k = (at.ptr - 0x1000) / 32 - crate::gpu::recomp::MEM_HEAP_BASE as usize;
                NULLS.fetch_or(1 << k, Ordering::Relaxed);
            }
        }
        let mut vt = [0usize; 20];
        vt[1] = retain as usize;
        vt[2] = retain as usize;
        vt[19] = create_uav as usize;
        let mut device = Object { vt: vt.as_ptr() };
        let mut heap = Object { vt: vt.as_ptr() };
        let mut resource = Object { vt: vt.as_ptr() };
        unsafe {
            let dev = ID3D12Device::from_raw(&mut device as *mut _ as *mut c_void);
            *GLOBAL.lock().unwrap() = Some(Global {
                dev: dev.clone(),
                heap: ID3D12DescriptorHeap::from_raw(&mut heap as *mut _ as *mut c_void),
                cpu: 0x1000, gpu: 0x2000, inc: 32, cap: 1024,
                next: crate::gpu::recomp::MEM_HEAP_BASE + crate::gpu::uma::MAX_HEAP_WINDOWS as u32, free: Vec::new(), nulls: None,
            });
            let window = (&mut resource as *mut _ as usize, 0x10000);
            // the root-UAV slots past the windows there are: null views, not window 0 again
            assert!(set_window_views(&dev, &[window]));
            assert_eq!(CALLS.swap(0, Ordering::Relaxed), crate::gpu::uma::MAX_WINDOWS as u32);
            assert_eq!(NULLS.swap(0, Ordering::Relaxed), 0b1110);
            let mut windows = [window; crate::gpu::uma::MAX_HEAP_WINDOWS];
            assert!(set_window_views(&dev, &windows));
            assert_eq!(CALLS.swap(0, Ordering::Relaxed), windows.len() as u32);
            assert_eq!(NULLS.swap(0, Ordering::Relaxed), 0);
            // A window without memory `(0, 0)` (16 windows: the ones after the views) gets a null
            // view; an active window that failed to open `(0, bytes > 0)` is rejected.
            windows[7] = (0, 0);
            windows[15] = (0, 0);
            assert!(set_window_views(&dev, &windows));
            assert_eq!(CALLS.swap(0, Ordering::Relaxed), windows.len() as u32);
            assert_eq!(NULLS.swap(0, Ordering::Relaxed), 1 << 7 | 1 << 15);
            windows[7] = (0, 0x10000);
            assert!(!set_window_views(&dev, &windows));
            assert_eq!(CALLS.load(Ordering::Relaxed), 0);
            assert!(UNLOCKED.load(Ordering::Relaxed), "device callback ran while GLOBAL was locked");
            *GLOBAL.lock().unwrap() = None;
        }
    }

    /// A T# of zeros (an empty slot) names descriptors 0 (SRV) and 1 (UAV): those must be
    /// null views, never a memory window's raw buffer UAV (a texture read of that hangs the GPU).
    #[test]
    fn zero_tsharp_does_not_reach_window_views() {
        use std::ffi::c_void;
        use std::sync::Mutex as M;
        static SEEN: M<Vec<(bool, usize, usize)>> = M::new(Vec::new());
        #[repr(C)]
        struct Object { vt: *const usize }
        unsafe extern "system" fn srv(_this: *mut c_void, r: *mut c_void, _d: *const D3D12_SHADER_RESOURCE_VIEW_DESC, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
            SEEN.lock().unwrap().push((false, r as usize, at.ptr));
        }
        unsafe extern "system" fn uav(_this: *mut c_void, r: *mut c_void, _c: *mut c_void, _d: *const D3D12_UNORDERED_ACCESS_VIEW_DESC, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
            SEEN.lock().unwrap().push((true, r as usize, at.ptr));
        }
        let base = crate::gpu::recomp::MEM_HEAP_BASE;
        assert!(base >= 2, "window views overlap the SRV / UAV pair of a zero T#");
        assert_eq!(base % 2, 0, "window views must not split a console slot's SRV / UAV pair");
        let mut vt = [0usize; 20];
        vt[18] = srv as usize;
        vt[19] = uav as usize;
        let mut device = Object { vt: vt.as_ptr() };
        unsafe {
            let dev = std::mem::ManuallyDrop::new(ID3D12Device::from_raw(&mut device as *mut _ as *mut c_void));
            null_views(&dev, 0x1000, 32);
        }
        let seen = SEEN.lock().unwrap().clone();
        assert_eq!(seen.len(), base as usize);
        for (k, &(is_uav, r, at)) in seen.iter().enumerate() {
            assert_eq!((is_uav, r, at), (k % 2 == 1, 0, 0x1000 + 32 * k), "descriptor {k}");
        }
    }

    /// A shader-visible sampler heap has a PC sampler at every index a translated shader forms
    /// (S# word 3 & 0x7FF), whatever the title asked for (a retail title: 128); a CPU-only one
    /// as many as its slots. D3D12 makes a shader-visible sampler heap of that size.
    #[test]
    fn every_sampler_index_a_shader_forms_is_in_the_pc_heap() {
        for n in [1, 16, 128, 2047, 2048] {
            assert_eq!(pc_samplers(n, true), 2048, "{n} slots");
            assert!(u32::MAX & 0x7FF < pc_samplers(n, true), "{n} slots: the highest index a shader forms");
            assert_eq!(pc_samplers(n, false), n, "{n} slots, CPU only");
        }
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        let dev = unsafe {
            let mut dev: Option<ID3D12Device> = None;
            match CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>()) {
                Ok(warp) if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_ok() => dev.unwrap(),
                _ => return crate::gpu::recomp::runner::skip_test("no WARP device"),
            }
        };
        let d = D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, NumDescriptors: pc_samplers(128, true), Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 };
        let made = unsafe { dev.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&d) };
        assert!(made.is_ok(), "a shader-visible sampler heap of {}: {made:?}", d.NumDescriptors);
    }

    /// Every descriptor of a range that is not handed out gets a null view, an SRV at even and a
    /// UAV at odd indices (a range starting at an odd one too), a block at a time.
    #[test]
    fn free_ranges_of_the_shared_heap_get_null_views() {
        use std::ffi::c_void;
        use std::sync::Mutex as M;
        static COPIES: M<Vec<(u32, usize, usize)>> = M::new(Vec::new());
        #[repr(C)]
        struct Object { vt: *const usize }
        unsafe extern "system" fn copy(_this: *mut c_void, n: u32, dst: D3D12_CPU_DESCRIPTOR_HANDLE, src: D3D12_CPU_DESCRIPTOR_HANDLE, ty: D3D12_DESCRIPTOR_HEAP_TYPE) {
            assert_eq!(ty, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            COPIES.lock().unwrap().push((n, dst.ptr, src.ptr));
        }
        let mut vt = [0usize; 25];
        vt[24] = copy as usize;
        let mut device = Object { vt: vt.as_ptr() };
        let (heap, nulls, inc) = (0x10_0000u64, 0x90_0000u64, 32u32);
        for (from, to) in [(16u32, 3 * NULL_BLOCK + 5), (7, 9), (2 * NULL_BLOCK - 1, 2 * NULL_BLOCK + 1)] {
            COPIES.lock().unwrap().clear();
            unsafe {
                let dev = std::mem::ManuallyDrop::new(ID3D12Device::from_raw(&mut device as *mut _ as *mut c_void));
                fill_nulls(&dev, heap, inc, from, to, nulls);
            }
            let mut at = from;
            for &(n, dst, src) in COPIES.lock().unwrap().iter() {
                assert_eq!(dst as u64, heap + at as u64 * inc as u64, "{from}..{to}: contiguous");
                let k = (src as u64 - nulls) / inc as u64;
                assert_eq!(k % 2, at as u64 % 2, "{from}..{to}: an SRV over an SRV, a UAV over a UAV");
                assert!(k + n as u64 <= NULL_BLOCK as u64, "{from}..{to}: within the null block");
                at += n;
            }
            assert_eq!(at, to, "{from}..{to}: all of it");
        }
    }

    /// What each index of the shared heap names: a destroyed resource's views (also copies of
    /// them) are claimed for null views, an index overwritten since by another resource's view
    /// is not, and a resource that never had a view costs one lookup and claims nothing.
    #[test]
    fn a_destroyed_resources_views_are_claimed_and_no_other() {
        let (a, b, c) = (0xA000usize, 0xB000usize, 0xC000usize);
        let n = Names::new(64);
        n.note(4, a); // SRV of slot 2
        n.note(5, a); // its UAV
        n.note(8, a);
        n.note(8, b); // overwritten by a view of b
        // a copy of slot 2 to slot 6 (`names_of` of the source, then `publish_range`)
        let (srv, uav) = (n.name(4), n.name(5));
        n.note(12, srv);
        n.note(13, uav);
        n.note(20, a);
        n.clear(20, 22); // a released range: null views there already
        assert!(n.candidates(c).is_none(), "a resource without views");
        let mut claimed: Vec<u32> = n.candidates(a).unwrap().into_iter().filter(|&i| n.claim(i, a)).collect();
        claimed.sort_unstable();
        assert_eq!(claimed, vec![4, 5, 12, 13], "a's views and their copies; not 8 (b's now), not 20 (nulled)");
        assert_eq!((n.name(4), n.name(8), n.name(12)), (GONE, b, GONE));
        assert!(n.candidates(a).is_none(), "forgotten");
        // b goes: its one view
        let claimed: Vec<u32> = n.candidates(b).unwrap().into_iter().filter(|&i| n.claim(i, b)).collect();
        assert_eq!(claimed, vec![8]);
    }

    /// A resource whose views are made again and again into the same few slots keeps a list of
    /// those slots, not one entry for each view.
    #[test]
    fn views_made_again_do_not_grow_a_resources_list() {
        let (a, b) = (0xA000usize, 0xB000usize);
        let n = Names::new(64);
        for k in 0..10_000u32 {
            n.note(k % 4, a);
            n.note(k % 4, b);
        }
        n.note(1, a);
        let v = n.candidates(a).unwrap();
        assert!(v.len() < 128, "{} entries", v.len());
        assert_eq!(v.iter().filter(|&&i| n.claim(i, a)).count(), 1, "only slot 1 is a's now");
        assert!(n.candidates(b).unwrap().len() < 128);
    }

    #[test]
    fn slots_copy_and_release() {
        let Some(u) = crate::gpu::uma::get() else { return };
        let rel = |p: u64| (p as usize - u.base()) as u32;
        // views: 4 slots over a PC heap at 0x1000 (increment 32, GPU 0x5000); samplers: 2 at 0x9000
        assert!(register(0xA1, 4, 0x1000, 0x5000, 32, false, false, true));
        assert!(register(0xA2, 2, 0x9000, 0, 16, true, false, true));
        let (v, sm) = (start(0xA1).unwrap(), start(0xA2).unwrap());
        assert_eq!(u.window_of(v as usize), Some(0), "view heap must live in window 0 (32-bit table addressing)");
        assert_eq!(u.window_of(sm as usize), Some(0), "sampler heap must live in window 0");
        assert!(u.is_sparse(v as usize) && u.is_sparse(sm as usize), "descriptor heaps backed by sparse VRAM");
        let s1 = slot(v + 32).unwrap();
        assert_eq!((s1.index, s1.pc_srv, s1.pc_uav, s1.pc_gpu_uav), (2, 0x1040, 0x1060, 0x5060));
        // a 2D T# from slot 1 to slot 3: word 0 = the destination's SRV index
        write(&s1, &[2, 0x0470_0000, 0, 9 << 28, 0, 0, 0, 0]);
        let s3 = slot(v + 96).unwrap();
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap()[..4], [6, 0x0470_0000, 0, 9 << 28]);
        // a buffer V# keeps its address
        write(&s1, &[0x1234, 0, 64, 0x3101_6FAC, 0, 0, 0, 0]);
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap()[0], 0x1234);
        // a raytracing BVH SRSRC keeps its base address and box-sort flag verbatim
        let bvh = crate::gpu::desc::bvh_words(0x12_3456_7800, true);
        write(&s1, &[bvh[0], bvh[1], bvh[2], bvh[3], 0, 0, 0, 0]);
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap()[..4], bvh);
        // samplers: 16-byte slots, S# word 3 [11:0] = sampler index
        let (q0, q1) = (slot(sm).unwrap(), slot(sm + 16).unwrap());
        assert!(q1.sampler && q1.index == 1 && q1.pc_srv == 0x9010);
        write(&q0, &[1, 2, 3, 0xABC0_0000]);
        copy_words(&q0, &q1);
        assert_eq!(words_at(rel(sm + 16)).unwrap()[..4], [1, 2, 3, 0xABC0_0001]);
        assert!(slot(sm + 32).is_none(), "past the last sampler slot");
        unregister(0xA1);
        unregister(0xA2);
        assert!(start(0xA1).is_none() && slot(v).is_none() && slot(sm).is_none());
    }

    /// The shared heap full: the slots of a page that finds no place there name the null views
    /// at 0 / 1, never 2k (in the shared heap, the only view heap lists bind, that is another
    /// heap's view, a memory window's raw UAV for k from 2 to 9, or past its end); a T# copied
    /// there too. A page placed before keeps its place, and one placed once room is free again
    /// gets its own. Every page that found no room is counted.
    #[test]
    fn a_slot_whose_page_finds_no_room_names_the_null_views() {
        const CHILD: &str = "XCL_TEST_SHARED_HEAP_FULL_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::console_heap::tests::a_slot_whose_page_finds_no_room_names_the_null_views", "--nocapture"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            return;
        }
        let Some(u) = crate::gpu::uma::get() else { return };
        use std::ffi::c_void;
        #[repr(C)]
        struct Object { vt: *const usize }
        unsafe extern "system" fn retain(_this: *mut c_void) -> u32 { 1 }
        let mut vt = [0usize; 4];
        vt[1] = retain as usize;
        vt[2] = retain as usize;
        let mut device = Object { vt: vt.as_ptr() };
        let mut heap = Object { vt: vt.as_ptr() };
        // room for one page of 256 slots past the null views and the memory windows
        let base = crate::gpu::recomp::MEM_HEAP_BASE + crate::gpu::uma::MAX_HEAP_WINDOWS as u32;
        unsafe {
            *GLOBAL.lock().unwrap() = Some(Global {
                dev: ID3D12Device::from_raw(&mut device as *mut _ as *mut c_void),
                heap: ID3D12DescriptorHeap::from_raw(&mut heap as *mut _ as *mut c_void),
                cpu: 0x10_0000, gpu: 0x20_0000, inc: 32, cap: base + 2 * PAGE_SLOTS,
                next: base, free: Vec::new(), nulls: None,
            });
        }
        let rel = |p: u64| (p as usize - u.base()) as u32;
        assert!(register(0xC1, 2 * PAGE_SLOTS, 0x30_0000, 0, 32, false, true, true));
        assert!(register(0xC2, 16, 0x40_0000, 0, 32, false, true, true));
        let (a, b) = (start(0xC1).unwrap(), start(0xC2).unwrap());
        let placed = slot(a + 2 * SLOT as u64).unwrap();
        assert_eq!(placed.index, base + 4, "page 0 of the first heap takes the room there is");
        assert_ne!(placed.shared_srv, 0);
        let k = PAGE_SLOTS + 2;
        let s = slot(a + (k * SLOT) as u64).unwrap();
        assert_eq!((s.index, s.shared_srv, s.pc_gpu_uav), (0, 0, 0), "slot {k} of the first heap: its page found no room");
        assert_eq!(s.pc_srv, 0x30_0000 + (2 * k * 32) as u64, "views are still made in the title's heap");
        for k in 0..16 {
            let s = slot(b + (k * SLOT) as u64).unwrap();
            assert_eq!((s.index, s.shared_srv), (0, 0), "slot {k} of the second heap: no room for its page");
        }
        assert_eq!(NO_ROOM.load(Relaxed), 17, "every page that found no room, each time");
        assert_eq!((PAGES_PLACED.load(Relaxed), FILL_SAID.load(Relaxed)), (1, FILL_MARKS.len()), "one page placed; the heap is past every mark");
        // a 2D T# copied into a slot without a place names the null views
        write(&placed, &[base + 4, 0x0470_0000, 0, 9 << 28, 0, 0, 0, 0]);
        copy_words(&placed, &s);
        assert_eq!(words_at(rel(a + (k * SLOT) as u64)).unwrap()[..4], [0, 0x0470_0000, 0, 9 << 28]);
        assert_eq!(slot(a + 2 * SLOT as u64).unwrap().index, base + 4, "a placed page keeps its place");
        // the first heap goes: the second heap's page gets the room it left
        unregister(0xC1);
        let s = slot(b + 2 * SLOT as u64).unwrap();
        assert_eq!(s.index, base + 4);
        assert_ne!(s.shared_srv, 0);
        assert_eq!(PAGES_PLACED.load(Relaxed), 1, "the first heap's page went with it");
        unregister(0xC2);
        assert_eq!(PAGES_PLACED.load(Relaxed), 0);
        *GLOBAL.lock().unwrap() = None;
    }

    /// Minecraft's view heaps: eight of 73 727 slots, 147 454 PC descriptors each – more in all
    /// than an NVIDIA shared heap holds (1 000 000). Each gets a place there for the pages it
    /// uses; before, the seventh and eighth got none and their views were lost (flicker).
    #[test]
    fn large_heaps_get_shared_pages_as_used() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        if crate::gpu::uma::get().is_none() {
            return;
        }
        let dev = unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            dev.expect("D3D12 device")
        };
        assert!(unsafe { shared_heap(&dev) });
        const N: u32 = 73_727;
        let objs: Vec<usize> = (0..8).map(|i| 0xB000 + i).collect();
        for &o in &objs {
            assert!(register(o, N, 0x10_0000, 0, 32, false, true, true));
            assert!(in_global(o));
        }
        let mut seen = Vec::new();
        for &o in &objs {
            let v = start(o).unwrap();
            // slot 0, the last of page 0, the first of page 1 and the heap's last slot
            let s: Vec<Slot> = [0, PAGE_SLOTS - 1, PAGE_SLOTS, N - 1].iter().map(|&k| slot(v + (k * SLOT) as u64).unwrap()).collect();
            assert!(s.iter().all(|x| x.shared_srv != 0), "heap {o:#x}: a slot without a place in the shared heap");
            assert_eq!(s[1].index, s[0].index + 2 * (PAGE_SLOTS - 1), "a page is contiguous");
            assert_eq!(s[0].pc_srv, 0x10_0000, "views are still made in the title's heap");
            assert_eq!(s[3].pc_srv, 0x10_0000 + (2 * (N - 1) * 32) as u64);
            for x in &s {
                assert!(!seen.contains(&x.index), "two slots share the PC index {}", x.index);
                seen.push(x.index);
            }
            // the same slot again: the same place
            assert_eq!(slot(v).unwrap().index, s[0].index);
        }
        for &o in &objs {
            unregister(o);
        }
        // a heap smaller than a page takes 2 descriptors per slot, as it did when a heap's
        // whole range was reserved (a title with many small heaps must not fill the shared heap)
        assert_eq!((page_descriptors(4, 0), page_descriptors(N, 0), page_descriptors(N, N / PAGE_SLOTS)), (8, 2 * PAGE_SLOTS, 2 * (N % PAGE_SLOTS)));
        assert!(register(0xB100, 4, 0x10_0000, 0, 32, false, true, true) && register(0xB101, 4, 0x20_0000, 0, 32, false, true, true));
        let (a, b) = (slot(start(0xB100).unwrap()).unwrap(), slot(start(0xB101).unwrap()).unwrap());
        assert!(a.shared_srv != 0 && b.shared_srv != 0);
        assert_eq!(a.index.abs_diff(b.index), 8, "two heaps of 4 slots lie 8 descriptors apart");
        unregister(0xB100);
        unregister(0xB101);
    }

    #[test]
    fn shared_sampler_pages_preserve_zero_slot_and_clamp_tails() {
        assert_eq!(sampler_page_descriptors(8, 0), 8);
        assert_eq!(sampler_page_descriptors(SAMPLER_PAGE_SLOTS, 0), SAMPLER_PAGE_SLOTS);
        assert_eq!(sampler_page_descriptors(SAMPLER_PAGE_SLOTS + 5, 0), SAMPLER_PAGE_SLOTS);
        assert_eq!(sampler_page_descriptors(SAMPLER_PAGE_SLOTS + 5, 1), 5);
        // Simulate GLOBAL_SAMPLER reserving slot 0 for the immutable zero sampler.
        let (mut free, mut next) = (Vec::new(), 1u32);
        let a = take_range(&mut free, &mut next, GLOBAL_SAMPLER_MAX, sampler_page_descriptors(64, 0)).unwrap();
        let b = take_range(&mut free, &mut next, GLOBAL_SAMPLER_MAX, sampler_page_descriptors(64, 1)).unwrap();
        assert_eq!(a, 1);
        assert_eq!(b, 1 + SAMPLER_PAGE_SLOTS);
        assert!(b + SAMPLER_PAGE_SLOTS <= GLOBAL_SAMPLER_MAX);
        // Exhaustion fallback: hosted sampler heaps map overflowed slots to reserved slot 0; unhosted keep local k.
        assert_eq!(sampler_unmapped_index(true, 42), 0);
        assert_eq!(sampler_unmapped_index(false, 42), 42);
    }

    #[test]
    fn release_global_sampler_merges_adjacent_ranges_and_ignores_zero_count() {
        const CHILD: &str = "XCL_TEST_RELEASE_GLOBAL_SAMPLER_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::console_heap::tests::release_global_sampler_merges_adjacent_ranges_and_ignores_zero_count", "--nocapture"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            return;
        }
        use std::ffi::c_void;
        #[repr(C)]
        struct Object { vt: *const usize }
        unsafe extern "system" fn retain(_this: *mut c_void) -> u32 { 1 }
        let mut vt = [0usize; 4];
        vt[1] = retain as usize;
        vt[2] = retain as usize;
        let mut device = Object { vt: vt.as_ptr() };
        let mut heap = Object { vt: vt.as_ptr() };
        unsafe {
            *GLOBAL_SAMPLER.lock().unwrap() = Some(GlobalSampler {
                dev: ID3D12Device::from_raw(&mut device as *mut _ as *mut c_void),
                heap: ID3D12DescriptorHeap::from_raw(&mut heap as *mut _ as *mut c_void),
                cpu: 0x1000, inc: 32, cap: 17, next: 1, free: Vec::new(),
            });
            assert_eq!(reserve_global_sampler(4), Some(1));
            assert_eq!(reserve_global_sampler(4), Some(5));
            assert_eq!(reserve_global_sampler(4), Some(9));
            assert_eq!(reserve_global_sampler(4), Some(13));
            assert_eq!(reserve_global_sampler(1), None);
            // Adjacent out-of-order releases coalesce and zero-count releases are ignored.
            release_global_sampler(5, 4);
            release_global_sampler(1, 4);
            release_global_sampler(1, 0);
            assert_eq!(GLOBAL_SAMPLER.lock().unwrap().as_ref().unwrap().free, vec![(1, 8)]);
            assert_eq!(reserve_global_sampler(8), Some(1));
            // Coalesced range ending at cursor straddles cleanly into the never-used tail.
            {
                let mut g = GLOBAL_SAMPLER.lock().unwrap();
                let s = g.as_mut().unwrap();
                s.next = 13;
                s.free.clear();
            }
            release_global_sampler(9, 2);
            release_global_sampler(11, 2);
            assert_eq!(reserve_global_sampler(8), Some(9));
            let g = GLOBAL_SAMPLER.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert_eq!(s.next, 17);
            assert!(s.free.is_empty());
            drop(g);
            *GLOBAL_SAMPLER.lock().unwrap() = None;
        }
    }

    #[test]
    fn in_global_excludes_shared_sampler_heaps() {
        const CHILD: &str = "XCL_TEST_IN_GLOBAL_SAMPLER_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::console_heap::tests::in_global_excludes_shared_sampler_heaps", "--nocapture"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            return;
        }
        use std::ffi::c_void;
        #[repr(C)]
        struct Object { vt: *const usize }
        unsafe extern "system" fn retain(_this: *mut c_void) -> u32 { 1 }
        let mut vt = [0usize; 4];
        vt[1] = retain as usize;
        vt[2] = retain as usize;
        let mut device = Object { vt: vt.as_ptr() };
        let mut view_heap = Object { vt: vt.as_ptr() };
        let mut sampler_heap = Object { vt: vt.as_ptr() };
        unsafe {
            let dev = ID3D12Device::from_raw(&mut device as *mut _ as *mut c_void);
            *GLOBAL.lock().unwrap() = Some(Global {
                dev: dev.clone(),
                heap: ID3D12DescriptorHeap::from_raw(&mut view_heap as *mut _ as *mut c_void),
                cpu: 0x1000, gpu: 0x2000, inc: 32, cap: 1024, next: 64, free: Vec::new(), nulls: None,
            });
            *GLOBAL_SAMPLER.lock().unwrap() = Some(GlobalSampler {
                dev,
                heap: ID3D12DescriptorHeap::from_raw(&mut sampler_heap as *mut _ as *mut c_void),
                cpu: 0x4000, inc: 32, cap: 17, next: 1, free: Vec::new(),
            });
            if crate::gpu::uma::get().is_some() {
                assert!(register(0xC100, 16, 0x9000, 0, 16, true, true, true));
                assert!(register(0xC101, 16, 0x1000, 0, 32, false, true, true));
                assert!(register(0xC102, 4, 0xA000, 0, 16, true, false, true));
                assert!(!in_global(0xC100) && in_global_sampler(0xC100));
                assert!(in_global(0xC101) && !in_global_sampler(0xC101));
                assert!(!in_global(0xC102) && !in_global_sampler(0xC102));
                unregister(0xC100);
                unregister(0xC101);
                unregister(0xC102);
                // Adjacent 4-slot sampler heaps coalesce in GLOBAL_SAMPLER on unregister.
                assert!(register(0xC110, 4, 0x9000, 0, 16, true, true, true));
                assert!(register(0xC111, 4, 0x9100, 0, 16, true, true, true));
                assert_eq!(slot(start(0xC110).unwrap()).unwrap().index, 1);
                assert_eq!(slot(start(0xC111).unwrap()).unwrap().index, 5);
                unregister(0xC110);
                unregister(0xC111);
                assert_eq!(GLOBAL_SAMPLER.lock().unwrap().as_ref().unwrap().free, vec![(1, 8)]);
            } else {
                let mut heaps = HEAPS.lock().unwrap();
                heaps.push(Heap { obj: 0xC100, shadow: 0, n: 16, pc_cpu: 0, pc_gpu: 0, inc: 16, sampler: true, pages: vec![UNMAPPED; 2], visible: true, memory: Memory::Uma });
                heaps.push(Heap { obj: 0xC101, shadow: 0, n: 16, pc_cpu: 0, pc_gpu: 0, inc: 32, sampler: false, pages: vec![UNMAPPED], visible: true, memory: Memory::Uma });
                drop(heaps);
                assert!(!in_global(0xC100) && in_global_sampler(0xC100));
                assert!(in_global(0xC101) && !in_global_sampler(0xC101));
                HEAPS.lock().unwrap().clear();
            }
            *GLOBAL_SAMPLER.lock().unwrap() = None;
            *GLOBAL.lock().unwrap() = None;
        }
    }

    #[test]
    fn sampler_own_index_tracks_mapped_pages_and_fallbacks() {
        // Hosted shared sampler heap with two mapped pages at GLOBAL_SAMPLER indices 1 and 17.
        let pages = [1u32, 17u32];
        assert_eq!(sampler_own_index(&pages, 0), 1);
        assert_eq!(sampler_own_index(&pages, SAMPLER_PAGE_SLOTS - 1), SAMPLER_PAGE_SLOTS);
        assert_eq!(sampler_own_index(&pages, SAMPLER_PAGE_SLOTS), 17);
        assert_eq!(sampler_own_index(&pages, SAMPLER_PAGE_SLOTS + 5), 22);
        // Unmapped or out-of-range page in a hosted shared sampler heap maps to reserved slot 0.
        assert_eq!(sampler_own_index(&[UNMAPPED], 0), 0);
        assert_eq!(sampler_own_index(&[1, UNMAPPED], SAMPLER_PAGE_SLOTS + 1), 0);
        assert_eq!(sampler_own_index(&[1], SAMPLER_PAGE_SLOTS * 2), 0);
        // Unhosted sampler heap (empty pages) keeps local slot index k.
        assert_eq!(sampler_own_index(&[], 0), 0);
        assert_eq!(sampler_own_index(&[], 42), 42);
    }
}
