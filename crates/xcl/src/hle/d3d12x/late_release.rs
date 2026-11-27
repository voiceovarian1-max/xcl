//! The title's last `Release` of a resource, put off until the GPU is past the work in flight.
//!
//! D3D12 on the PC asks that an object outlives the GPU work that uses it. A console title
//! keeps to that by its own count of finished frames – but XCL's submits are not the title's:
//! texture uploads go out with a later submit, tile mappings on a queue of XCL's own. A retail
//! title's final `Release` of a resource then came while a queue still had work on it (debug
//! layer: "CORRUPTION: … is referenced by GPU operations in-flight", id 921).
//!
//! The last reference is held here instead. Each batch of held resources gets a fence value
//! signalled on every queue with work on it; the batch goes once all of them have passed it.
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// a batch is not closed more often than this (a `Signal` a queue each time)
const SEAL_EVERY: Duration = Duration::from_millis(10);
/// how often the thread here looks for batches that can go
const TICK: Duration = Duration::from_millis(4);

/// What is let go when the queues are past it.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Dead {
    /// an adopted resource's last reference (its original Release ends it)
    Resource(usize),
    /// a pipeline state XCL owns (`hold_pipeline`)
    Pipeline(usize),
    /// a heap XCL owns (`hold_heap`)
    Heap(usize),
    /// a reference of XCL's own to a superseded target, whose tiles then leave the shared pages
    /// (`hold_unmap`)
    Unmap(usize),
}

struct Batch {
    dead: Vec<Dead>,
    /// (fence, value) a queue: all reached = nothing submitted before the batch closed is left
    marks: Vec<(ID3D12Fence, u64)>,
}

#[derive(Default)]
struct Held {
    /// queue → its fence here and the last value signalled on it
    fences: HashMap<usize, (ID3D12Fence, u64)>,
    open: Vec<Dead>,
    sealed: VecDeque<Batch>,
    last_seal: Option<Instant>,
}

// (the interface pointers are only used under the lock)
unsafe impl Send for Held {}

impl Held {
    fn is_empty(&self) -> bool {
        self.open.is_empty() && self.sealed.is_empty()
    }

    /// Close the open batch (at most every `SEAL_EVERY`, or `seal_now`) with a fence value on each of `queues`,
    /// and take out the batches every queue is past: what to release now.
    unsafe fn drain(&mut self, queues: &[ID3D12CommandQueue], seal_now: bool) -> Vec<Dead> {
        if !self.open.is_empty() && (seal_now || self.last_seal.is_none_or(|t| t.elapsed() >= SEAL_EVERY)) {
            let mut marks = Vec::with_capacity(queues.len());
            let mut ok = true;
            for q in queues {
                let key = q.as_raw() as usize;
                if !self.fences.contains_key(&key) {
                    let mut dev: Option<ID3D12Device> = None;
                    let made = q.GetDevice(&mut dev).ok().and(dev).and_then(|d| super::real::create_raw_fence(&d).ok());
                    let Some(f) = made else {
                        ok = false;
                        break;
                    };
                    self.fences.insert(key, (f, 0));
                }
                let (fence, value) = self.fences.get_mut(&key).unwrap();
                if q.Signal(&*fence, *value + 1).is_err() {
                    ok = false;
                    break;
                }
                *value += 1;
                marks.push((fence.clone(), *value));
            }
            // (no fence or no signal: the device is gone – the batch stays open, and held)
            if ok {
                self.last_seal = Some(Instant::now());
                let dead = std::mem::take(&mut self.open);
                self.sealed.push_back(Batch { dead, marks });
            }
        }
        let mut gone = Vec::new();
        while self.sealed.front().is_some_and(|b| b.marks.iter().all(|(f, v)| super::real::fence_completed(&f) >= *v)) {
            gone.extend(self.sealed.pop_front().unwrap().dead);
        }
        gone
    }
}

static HELD: Mutex<Option<Held>> = Mutex::new(None);

/// Whether last releases are held (`XCL_LATE_RELEASE=0` turns it off).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_LATE_RELEASE").is_ok_and(|v| v.trim() == "0"))
}

/// how long a last `Release` itself waits for queues with nothing to do to say so
const IDLE_WAIT: Duration = Duration::from_micros(300);
/// how long queues that had work then are taken to have some: the next releases leave theirs to the
/// thread at once (a title that streams releases many at a time, each with a `Signal` on every queue
/// and `IDLE_WAIT` of its own)
const BUSY_FOR: Duration = Duration::from_millis(5);

/// Whether queues that were found busy are still taken to be.
#[derive(Default)]
struct Busy(Mutex<Option<Instant>>);

impl Busy {
    fn is(&self, now: Instant) -> bool {
        self.0.lock().unwrap().is_some_and(|until| now < until)
    }

    fn found(&self, now: Instant) {
        *self.0.lock().unwrap() = Some(now + BUSY_FOR);
    }
}

static BUSY: Busy = Busy(Mutex::new(None));

/// `res` (an adopted resource) holds one reference that is now this module's. With `pinned` (it may
/// be over pages that are freed next) and no work in flight it goes before this returns (the queues
/// pass a fence value at once): a title that frees the memory under a resource right after its last
/// `Release` finds it unpinned, as before. Otherwise the thread here lets it go – the driver takes
/// ~100 us to end a texture, and a title ends thousands of them when it changes places.
pub unsafe fn hold(res: usize, pinned: bool) {
    HELD.lock().unwrap().get_or_insert_with(Held::default).open.push(Dead::Resource(res));
    let start = Instant::now();
    let busy = !pinned || BUSY.is(start);
    if !busy {
        drain_known(true);
        while start.elapsed() < IDLE_WAIT && HELD.lock().unwrap().as_ref().is_some_and(|h| !h.is_empty()) {
            std::hint::spin_loop();
            drain_known(false);
        }
        if HELD.lock().unwrap().as_ref().is_none_or(Held::is_empty) {
            return;
        }
        BUSY.found(Instant::now());
    }
    {
        static N: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
        let n = N[0].fetch_add(1, Ordering::Relaxed) + 1;
        let left = N[1].fetch_add(busy as u64, Ordering::Relaxed) + busy as u64;
        if n.is_power_of_two() && n >= 1024 {
            crate::xlog_info!("D3D12.X: {n} last releases left to the thread so far, {left} of them without a look at the queues");
        }
    }
    start_thread();
}

/// A thread of its own lets the held ones go: the title may release a resource and then wait
/// for its memory without another submit (an imported heap pins the pages under its resources –
/// unmapping them stood still while one was held "until the next submit").
fn start_thread() {
    static THREAD: std::sync::Once = std::sync::Once::new();
    THREAD.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-late-release".into()).spawn(|| loop {
            std::thread::sleep(TICK);
            unsafe { drain_known(false) };
        });
    });
}

/// A pipeline state XCL owns (raw pointer, 0 = none) is let go: the compute pass of a two-pass
/// pipeline, the real pipeline behind a title's stand-in. Lists in flight may bind it (the title
/// released its own pipeline, not XCL's, and D3D12 asks that a pipeline outlives the work that
/// runs it), so it is released as a resource's last reference is: once every queue is past a
/// fence value signalled after this. `XCL_LATE_RELEASE=0`: now.
pub fn hold_pipeline(p: usize) {
    if p == 0 {
        return;
    }
    if !enabled() {
        return super::pso_async::release(p);
    }
    HELD.lock().unwrap().get_or_insert_with(Held::default).open.push(Dead::Pipeline(p));
    start_thread();
}

/// A heap XCL owns that reserved resources mapped tiles of (`vram`'s page pools) is let go: a
/// resource the title released maps them until it goes here, and work queued before still reads
/// them – with the heap gone under that work the GPU faults. So it goes as such a resource does,
/// and not before those held earlier: once every queue is past a fence value signalled after this.
/// `XCL_LATE_RELEASE=0`: now.
pub fn hold_heap(heap: ID3D12Heap) {
    if !enabled() {
        return drop(heap);
    }
    HELD.lock().unwrap().get_or_insert_with(Held::default).open.push(Dead::Heap(heap.into_raw() as usize));
    start_thread();
}

/// A target superseded by a newer placement over its pages (`placements`) leaves the shared pages
/// once every queue is past a fence value signalled after this: work submitted before may still
/// draw to it or read it. `XCL_LATE_RELEASE=0`: now.
pub fn hold_unmap(res: ID3D12Resource) {
    if !enabled() {
        unsafe { super::placements::unmap_if_superseded(&res) };
        return drop(res);
    }
    HELD.lock().unwrap().get_or_insert_with(Held::default).open.push(Dead::Unmap(res.into_raw() as usize));
    start_thread();
}

/// Release what every queue XCL knows of is past: the title's, the tile-mapping one, the
/// upload ones.
pub unsafe fn drain_known(seal_now: bool) {
    if HELD.lock().unwrap().as_ref().is_none_or(Held::is_empty) {
        return;
    }
    let mut queues: Vec<ID3D12CommandQueue> = Vec::new();
    for raw in super::present::known_queues() {
        let raw = raw as *mut c_void;
        if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&raw) {
            queues.push(q.clone());
        }
    }
    queues.extend(super::sparse::queue());
    queues.extend(super::title_textures::upload_queues());
    let gone = match HELD.lock().unwrap().as_mut() {
        Some(h) => h.drain(&queues, seal_now),
        None => return,
    };
    // (outside the lock: a release may end other objects, whose releases come back here)
    for dead in gone {
        let res = match dead {
            Dead::Resource(res) => res,
            Dead::Pipeline(p) => {
                super::pso_async::release(p);
                continue;
            }
            Dead::Heap(h) => {
                drop(ID3D12Heap::from_raw(h as *mut c_void));
                continue;
            }
            Dead::Unmap(r) => {
                // (then the reference goes: the title's last one may have gone meanwhile)
                let r = ID3D12Resource::from_raw(r as *mut c_void);
                super::placements::unmap_if_superseded(&r);
                continue;
            }
        };
        // (its views in the shared heap become null views before the object goes)
        super::real::resource_ending(res);
        // A placed buffer's tiles stay pinned until here: the held object maps them, and work
        // queued before its release reads them through it. Unpinned right before the object goes
        // (the queues are past that work), not after: then another thread could have made a
        // buffer at the same address, whose tiles this would unpin. (`forget_buffer` takes no
        // lock of this module's and does not wait for the sparse state's.)
        super::sparse::forget_buffer(res);
        super::placements::gone(res);
        let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(super::real::orig_fn(res as *mut c_void, 2));
        f(res as *mut c_void);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

    /// A held resource stays while a queue has not got past the work submitted before it was
    /// released, and goes once the queue has.
    #[test]
    fn a_held_resource_goes_when_the_queue_is_past_its_batch() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            // the queue stands still, as with work in flight: it waits for a fence nobody has signalled
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Wait(&gate, 1).unwrap();
            let mut res: Option<ID3D12Resource> = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 4096,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut res).unwrap();
            let raw = res.unwrap().into_raw() as usize;

            let mut held = Held::default();
            held.open.push(Dead::Resource(raw));
            let queues = [queue.clone()];
            assert!(held.drain(&queues, false).is_empty(), "the queue has not passed the batch's fence value");
            std::thread::sleep(Duration::from_millis(100));
            assert!(held.drain(&queues, false).is_empty(), "still in flight: still held");
            assert!(!held.is_empty());

            gate.Signal(1).unwrap();
            let start = Instant::now();
            let mut gone = Vec::new();
            while gone.is_empty() && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(5));
                gone = held.drain(&queues, false);
            }
            assert_eq!(gone, vec![Dead::Resource(raw)], "released once the queue is past the batch");
            assert!(held.is_empty());
            drop(ID3D12Resource::from_raw(raw as *mut c_void));
        }
    }

    /// Queues found busy are taken to be for a moment, and looked at again after it.
    #[test]
    fn busy_queues_are_looked_at_again_after_a_moment() {
        let busy = Busy::default();
        let t = Instant::now();
        assert!(!busy.is(t), "never looked at");
        busy.found(t);
        assert!(busy.is(t) && busy.is(t + BUSY_FOR / 2));
        assert!(!busy.is(t + BUSY_FOR));
        busy.found(t + BUSY_FOR);
        assert!(busy.is(t + BUSY_FOR + BUSY_FOR / 2));
    }

    /// With no queue at all nothing can be in flight: the next drain lets the batch go.
    #[test]
    fn without_queues_a_batch_goes_at_once() {
        let mut held = Held::default();
        held.open.push(Dead::Resource(0x1000));
        held.open.push(Dead::Pipeline(0x2000));
        assert_eq!(unsafe { held.drain(&[], false) }, vec![Dead::Resource(0x1000), Dead::Pipeline(0x2000)]);
        assert!(held.is_empty());
    }

    /// A pipeline state XCL lets go goes into a batch (a list in flight may bind it), and the
    /// thread here releases it once the queues are past – through its COM Release. (That a batch
    /// waits for a busy queue is `a_held_resource_goes_when_the_queue_is_past_its_batch`.)
    #[test]
    fn a_pipeline_xcl_lets_go_is_released_by_the_thread() {
        use std::sync::atomic::AtomicU32;
        #[repr(C)]
        struct Fake {
            vt: *const usize,
            refs: AtomicU32,
        }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            (*(this as *const Fake)).refs.fetch_add(1, Ordering::SeqCst) + 1
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            (*(this as *const Fake)).refs.fetch_sub(1, Ordering::SeqCst) - 1
        }
        let vt = [0usize, add_ref as usize, release as usize];
        let fake = Box::new(Fake { vt: vt.as_ptr(), refs: AtomicU32::new(1) });
        let p = &*fake as *const Fake as usize;
        hold_pipeline(p);
        assert!(enabled());
        // (a queue XCL knows of may still be busy: the batch waits for its fence)
        let start = Instant::now();
        while fake.refs.load(Ordering::SeqCst) != 0 && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(fake.refs.load(Ordering::SeqCst), 0, "released once the queues are past");
        hold_pipeline(0);
    }
}
