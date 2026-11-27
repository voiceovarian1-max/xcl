//! A title's command allocator Reset does not take recordings the title still submits.
//!
//! On the PC a list's recording lives in its allocator: after `allocator.Reset()` a list that
//! recorded on it must not be submitted again ("The command allocator was reset after the
//! command list was recorded", debug layer id 541 – the runtime removes the device,
//! 0x887a0001). A retail title does exactly that, which its console lets it: one recording,
//! submitted at 61.6 s, at 63.2 s on a second queue and at 67.8 s again, with the allocator
//! it had been given reset at 67.2 s.
//!
//! So the title's allocator is a name for a PC allocator that changes: at the title's Reset,
//! when lists still hold recordings on the PC allocator in use (or the GPU has not finished
//! them), that one is put aside as it is and the title's allocator goes on with another – one
//! put aside earlier whose lists have all been reset since and whose submits are done, or a new
//! one. A list reset on the title's allocator records on the PC allocator in use. A list holds
//! one recording, so no more allocators are put aside than there are lists.
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

#[derive(Default)]
struct Backing {
    /// PC lists whose recording is on this allocator
    users: HashSet<usize>,
    /// queue → (fence, value after the last submit of one of them)
    marks: HashMap<usize, (ID3D12Fence, u64)>,
}

impl Backing {
    /// The GPU has passed every submit of a recording on this allocator.
    unsafe fn idle(&self) -> bool {
        self.marks.values().all(|(f, v)| super::real::fence_completed(f) >= *v)
    }

    unsafe fn free(&self) -> bool {
        self.users.is_empty() && self.idle()
    }
}

#[derive(Default)]
struct State {
    /// title allocator → the PC allocator in use (absent: itself)
    cur: HashMap<usize, usize>,
    /// title allocator → PC allocators put aside
    aside: HashMap<usize, Vec<usize>>,
    backings: HashMap<usize, Backing>,
    of_list: HashMap<usize, usize>,
    /// title allocator → the type of the lists on it
    types: HashMap<usize, i32>,
    fences: HashMap<usize, (ID3D12Fence, u64)>,
    made: u64,
    swaps: u64,
    /// PC list → its last submit (queue, fence, value after it, recording number then)
    last_submit: HashMap<usize, LastSubmit>,
    /// PC list → recordings so far (a Reset or creation starts the next one)
    recordings: HashMap<usize, u64>,
    resubmits: u64,
}

struct LastSubmit {
    queue: usize,
    fence: ID3D12Fence,
    value: u64,
    recording: u64,
}

unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// `XCL_ALLOC_SWAP=0` turns it off.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_ALLOC_SWAP").is_ok_and(|v| v.trim() == "0"))
}

fn move_list(s: &mut State, list: usize, backing: usize) {
    if let Some(old) = s.of_list.insert(list, backing) {
        if old != backing {
            if let Some(b) = s.backings.get_mut(&old) {
                b.users.remove(&list);
            }
        }
    }
    s.backings.entry(backing).or_default().users.insert(list);
}

/// The PC list's final Release ended its recording. Submitted work still holds the backing
/// through `marks`; removing the list must not remove those fences or the other users.
pub(super) fn forget_list(list: usize) {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    s.last_submit.remove(&list);
    s.recordings.remove(&list);
    unpin(s, list);
}

/// The PC list's recording ended but the list stays (a split segment moved to the spares): its
/// allocator is no longer held for it. Its last submit stays known, so it is not handed out
/// again while the GPU still runs that submit (`in_flight`).
pub(super) fn unpin_list(list: usize) {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    unpin(s, list);
}

fn unpin(s: &mut State, list: usize) {
    if let Some(backing) = s.of_list.remove(&list) {
        if let Some(b) = s.backings.get_mut(&backing) {
            b.users.remove(&list);
        }
    }
}

/// Reset `list` on the title's allocator `alloc`, committing its new recording association
/// only after native Reset succeeds. No pool lock is held across the driver call.
pub fn reset_list(list: usize, alloc: usize, ty: i32, reset: impl FnOnce(usize) -> HRESULT) -> HRESULT {
    let backing = current(alloc);
    let hr = reset(backing);
    if hr < 0 || !enabled() || alloc == 0 {
        return hr;
    }
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    if ty >= 0 {
        s.types.insert(alloc, ty);
    }
    move_list(s, list, backing);
    *s.recordings.entry(list).or_default() += 1;
    hr
}

#[cfg(test)]
pub(super) use recording_backing as recording_backing_for_test;

pub(super) fn recording_backing(list: usize) -> Option<usize> {
    STATE.lock().unwrap().as_ref()?.of_list.get(&list).copied()
}

/// The title's allocator `alloc` stands for the PC allocator `pc` (as after a Reset put its own
/// aside); `pc == alloc` ends that.
#[cfg(test)]
pub(super) fn stand_in_for_test(alloc: usize, pc: usize) {
    STATE.lock().unwrap().get_or_insert_with(State::default).cur.insert(alloc, pc);
}

/// The PC allocator the title's allocator `alloc` stands for now.
pub fn current(alloc: usize) -> usize {
    if !enabled() {
        return alloc;
    }
    STATE.lock().unwrap().as_ref().and_then(|s| s.cur.get(&alloc).copied()).unwrap_or(alloc)
}

/// `list` (of type `ty`) was made recording on the PC allocator `raw` itself
/// (CreateCommandList).
pub fn created_on(list: usize, raw: usize, ty: i32) {
    if !enabled() || raw == 0 {
        return;
    }
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    s.types.entry(raw).or_insert(ty);
    move_list(s, list, raw);
    *s.recordings.entry(list).or_default() += 1;
}

/// Diagnostic (#1452): `lists` are about to be submitted to `queue`. A PC list whose previous
/// submit the GPU has not finished must not be submitted again on the PC (debug layer id 553,
/// "the command queue fence has not advanced past previous executions of the command list");
/// a console lets a title do it. Logged (the first 8, then at powers of two) with whether it is
/// the same recording and the same queue. Every submitted PC list is seen (`executed` marks them all).
pub unsafe fn note_resubmits(queue: *mut c_void, lists: &[*mut c_void]) -> usize {
    if !enabled() {
        return 0;
    }
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return 0 };
    let mut found = 0;
    for &l in lists {
        let l = l as usize;
        let Some(last) = s.last_submit.get(&l) else { continue };
        if super::real::fence_completed(&last.fence) >= last.value {
            continue;
        }
        let same_recording = s.recordings.get(&l).copied().unwrap_or(0) == last.recording;
        let same_queue = last.queue == queue as usize;
        found += 1;
        s.resubmits += 1;
        let n = s.resubmits;
        if n <= 8 || n.is_power_of_two() {
            crate::xlog_warn!(
                "D3D12.X: a command list is submitted while its previous submit is still on the GPU ({}, {}; #{n} so far) – the PC does not allow it (debug layer id 553)",
                if same_recording { "the same recording" } else { "a new recording" },
                if same_queue { "the same queue" } else { "another queue" },
            );
        }
    }
    found
}

/// How long a submit waits for the previous submits of its lists (`XCL_RESUBMIT_WAIT_MS`, 2 000;
/// 0: not at all, as before).
fn resubmit_wait() -> std::time::Duration {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    std::time::Duration::from_millis(*MS.get_or_init(|| std::env::var("XCL_RESUBMIT_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2000)))
}

/// `lists` are about to be submitted: the ones whose previous submit the GPU has not finished
/// (`note_resubmits`) wait for it first, `limit` at most for all of them; then they go as they
/// are. On the PC a list must not be executed again before its previous execution is done (debug
/// layer id 553); a console lets a title do it, and a retail title did it about once a second in
/// its world – 128 times before its device was removed with shader memory faults, and not once in
/// a run that ended well. Returns (lists waited for, of them not done in time).
pub unsafe fn wait_for_previous_submits(lists: &[*mut c_void]) -> (usize, usize) {
    wait_for_previous_submits_within(lists, resubmit_wait())
}

unsafe fn wait_for_previous_submits_within(lists: &[*mut c_void], limit: std::time::Duration) -> (usize, usize) {
    if !enabled() || limit.is_zero() {
        return (0, 0);
    }
    // (the fences are taken out: nothing is waited for with STATE held)
    let pending: Vec<(ID3D12Fence, u64)> = {
        let g = STATE.lock().unwrap();
        let Some(s) = g.as_ref() else { return (0, 0) };
        lists
            .iter()
            .filter_map(|&l| s.last_submit.get(&(l as usize)))
            .filter(|l| super::real::fence_completed(&l.fence) < l.value)
            .map(|l| (l.fence.clone(), l.value))
            .collect()
    };
    if pending.is_empty() {
        return (0, 0);
    }
    let start = std::time::Instant::now();
    let mut late = 0;
    for (fence, value) in &pending {
        // (XCL's own fence, read without the title's hook; a removed device reads as the largest
        // value: no wait then. Such a submit is milliseconds from done: a short poll)
        let mut polls = 0u32;
        while super::real::fence_completed(fence) < *value {
            if start.elapsed() >= limit {
                late += 1;
                break;
            }
            if polls < 64 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            polls += 1;
        }
    }
    static WAITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = WAITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if late > 0 {
        crate::xlog_warn!(
            "D3D12.X: {late} of {} previous submits of lists submitted again were not done after {} ms – they go as they are (#{n} such submits)",
            pending.len(),
            limit.as_millis()
        );
    } else if n <= 8 || n.is_power_of_two() {
        crate::xlog_info!(
            "D3D12.X: a submit waited {:.1} ms for the previous submits of {} of its lists (#{n} such submits)",
            start.elapsed().as_secs_f64() * 1e3,
            pending.len()
        );
    }
    (pending.len(), late)
}

/// Whether the GPU has not finished the last submit of the PC list `list` (#1452: it must not be
/// submitted again before that; `title_list` records the title's next recording elsewhere).
/// (`fence_completed`: XCL's own fence, without the title's hook, STATE held)
pub fn in_flight(list: usize) -> bool {
    let g = STATE.lock().unwrap();
    g.as_ref().and_then(|s| s.last_submit.get(&list)).is_some_and(|l| unsafe { super::real::fence_completed(&l.fence) } < l.value)
}

/// `lists` were just submitted to `queue`.
pub unsafe fn executed(queue: *mut c_void, lists: &[*mut c_void]) {
    if !enabled() || queue.is_null() {
        return;
    }
    // (every submit is marked, also of lists this module does not track – the PC lists behind
    // console lists – so note_resubmits sees them all)
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let key = queue as usize;
    if !s.fences.contains_key(&key) {
        let mut dev: Option<ID3D12Device> = None;
        let Some(f) = q.GetDevice(&mut dev).ok().and(dev).and_then(|d| super::real::create_raw_fence(&d).ok()) else { return };
        s.fences.insert(key, (f, 0));
    }
    let (fence, value) = {
        let (f, v) = s.fences.get_mut(&key).unwrap();
        if q.Signal(&*f, *v + 1).is_err() {
            return;
        }
        *v += 1;
        (f.clone(), *v)
    };
    for &l in lists {
        if let Some(&b) = s.of_list.get(&(l as usize)) {
            s.backings.entry(b).or_default().marks.insert(key, (fence.clone(), value));
        }
        let recording = s.recordings.get(&(l as usize)).copied().unwrap_or(0);
        s.last_submit.insert(l as usize, LastSubmit { queue: key, fence: fence.clone(), value, recording });
    }
}

/// The lists recording on the PC allocators of `alloc` hold them no more: the allocator a title
/// list keeps for its put-off dispatches goes with the list, before the list's segments that
/// recorded on it are released (`list_x::on_release`), and `forget` gives back only what nothing
/// holds.
pub(super) fn unpin_users(alloc: usize) {
    let mut g = STATE.lock().unwrap();
    if let Some(s) = g.as_mut() {
        unpin_users_in(s, alloc);
    }
}

fn unpin_users_in(s: &mut State, alloc: usize) {
    let mine: Vec<usize> = s.cur.get(&alloc).copied().into_iter().chain(s.aside.get(&alloc).into_iter().flatten().copied()).chain([alloc]).collect();
    for b in mine {
        if let Some(b) = s.backings.get_mut(&b) {
            for l in std::mem::take(&mut b.users) {
                s.of_list.remove(&l);
            }
        }
    }
}

/// Whether the GPU may still run a recording made on the title's allocator `alloc`.
pub(super) unsafe fn allocator_in_flight(alloc: usize) -> bool {
    let g = STATE.lock().unwrap();
    let Some(s) = g.as_ref() else { return false };
    let mine = s.cur.get(&alloc).into_iter().chain(s.aside.get(&alloc).into_iter().flatten()).chain([&alloc]);
    let busy = mine.filter_map(|b| s.backings.get(b)).any(|b| !b.idle());
    busy
}

/// The title's allocator `alloc` is gone (its last Release): nothing of it stays here – a later
/// allocator may get its address, and must not go on with this one's PC allocators. The ones
/// made here go with it (those lists still record on are left alone: not freed, not tracked).
/// What ended recordings on `alloc` itself used goes back when the GPU is done with it, as at a
/// Reset: nothing else would give it back (only a Reset of a later allocator at its address).
pub unsafe fn forget(alloc: usize) {
    let (own, made): (bool, Vec<usize>) = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        s.types.remove(&alloc);
        let mut mine: Vec<usize> = s.aside.remove(&alloc).unwrap_or_default();
        mine.extend(s.cur.remove(&alloc));
        mine.push(alloc);
        // (before the loop: `alloc` may be in `mine` more than once, and its backing goes at the
        // first)
        let own = s.backings.get(&alloc).is_none_or(|x| x.free());
        let mut made = Vec::new();
        for b in mine {
            let free = s.backings.get(&b).is_none_or(|x| x.free());
            if let Some(gone) = s.backings.remove(&b) {
                for l in gone.users {
                    s.of_list.remove(&l);
                }
            }
            if b != alloc && free {
                made.push(b);
            }
        }
        (own, made)
    };
    // (outside the lock: their Release comes back here)
    if own {
        super::console_list::allocator_reset(alloc as *mut c_void);
    }
    for b in made {
        // what ended recordings on it used goes with it
        super::console_list::allocator_reset(b as *mut c_void);
        drop(ID3D12CommandAllocator::from_raw(b as *mut c_void));
    }
}

/// The title resets its allocator `alloc`. `None`: reset it as it is (nothing holds a
/// recording on it, or this is off). `Some(hr)`: done here – the PC allocator in use was put
/// aside with its recordings and another one took its place.
///
/// What ended recordings left in unified memory is held per PC allocator
/// (`console_list::hold_until_reset`). It goes back here for the PC allocators of `alloc` whose
/// submits the GPU has passed – not for one put aside because the GPU still runs what was
/// recorded on it (#1358).
pub unsafe fn title_reset(alloc: *mut c_void) -> Option<HRESULT> {
    title_reset_with(alloc, |raw| super::real::reset_pc_allocator(raw))
}

/// `title_reset` with the PC Reset supplied (tests).
unsafe fn title_reset_with(alloc: *mut c_void, reset: impl FnMut(*mut c_void) -> HRESULT) -> Option<HRESULT> {
    if !enabled() {
        return None;
    }
    let (hr, done) = {
        let mut g = STATE.lock().unwrap();
        let s = g.get_or_insert_with(State::default);
        let hr = title_reset_in(s, alloc, reset);
        (hr, done_with(s, alloc as usize))
    };
    // (outside the lock)
    for b in done {
        super::console_list::allocator_reset(b as *mut c_void);
    }
    hr
}

/// The PC allocators of the title's allocator `a` (the one in use and those put aside) whose
/// submits the GPU has passed: the recordings on them that have ended need their memory no more.
unsafe fn done_with(s: &State, a: usize) -> Vec<usize> {
    let cur = s.cur.get(&a).copied().unwrap_or(a);
    std::iter::once(cur)
        .chain(s.aside.get(&a).into_iter().flatten().copied())
        .filter(|b| s.backings.get(b).is_none_or(|x| x.idle()))
        .collect()
}

/// The allocator transition, with the PC Reset supplied so failure/retry needs no GPU in tests.
unsafe fn title_reset_in(s: &mut State, alloc: *mut c_void, mut reset: impl FnMut(*mut c_void) -> HRESULT) -> Option<HRESULT> {
    let a = alloc as usize;
    let c = s.cur.get(&a).copied().unwrap_or(a);
    if s.backings.get(&c).is_none_or(|b| b.free()) {
        if let Some(b) = s.backings.get_mut(&c) {
            b.marks.clear();
        }
        if c == a {
            return None;
        }
        return Some(reset(c as *mut c_void));
    }
    let ty = *s.types.get(&a)?;
    // one put aside earlier that nothing uses any more, or a new one
    let reuse = s.aside.get(&a).and_then(|v| v.iter().copied().find(|f| s.backings.get(f).is_none_or(|b| b.free())));
    let next = match reuse {
        Some(f) => {
            // Keep ownership until Reset succeeds so a failure can be retried.
            let hr = reset(f as *mut c_void);
            if hr < 0 {
                return Some(hr);
            }
            s.aside.get_mut(&a).unwrap().retain(|&x| x != f);
            if let Some(b) = s.backings.get_mut(&f) {
                b.marks.clear();
            }
            f
        }
        None => {
            let this = ID3D12CommandAllocator::from_raw_borrowed(&alloc)?;
            let mut dev: Option<ID3D12Device> = None;
            this.GetDevice(&mut dev).ok()?;
            let made: ID3D12CommandAllocator = dev?.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE(ty)).ok()?;
            let raw = made.into_raw();
            // (made through the title's device it is adopted already)
            if !super::real::is_adopted(raw) {
                super::real::adopt(raw, super::real::Kind::CommandAllocator);
            }
            s.made += 1;
            raw as usize
        }
    };
    s.aside.entry(a).or_default().push(c);
    s.cur.insert(a, next);
    s.swaps += 1;
    if s.swaps.is_power_of_two() {
        let aside: usize = s.aside.values().map(|v| v.len()).sum();
        crate::xlog_info!(
            "D3D12.X: {} allocator resets so far with recordings still held or on the GPU – another PC allocator took its place ({} made, {aside} put aside now)",
            s.swaps,
            s.made
        );
    }
    Some(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// When a title list goes, the segments that recorded its put-off dispatches on its own
    /// allocator hold neither the PC allocator it stands for now nor the ones put aside: `forget`
    /// then releases those the GPU is done with (they were left behind, with what was held on them).
    #[test]
    fn the_own_allocator_of_a_list_that_goes_is_held_by_none_of_its_segments() {
        let (own, now, aside, other) = (0x1000usize, 0x2000usize, 0x3000usize, 0x4000usize);
        let (seg, old_seg, elsewhere) = (0x5000usize, 0x6000usize, 0x7000usize);
        let mut s = State::default();
        s.cur.insert(own, now);
        s.aside.insert(own, vec![aside]);
        move_list(&mut s, seg, now);
        move_list(&mut s, old_seg, aside);
        move_list(&mut s, elsewhere, other);
        assert!(!unsafe { s.backings[&now].free() });
        unpin_users_in(&mut s, own);
        assert!(unsafe { s.backings[&now].free() && s.backings[&aside].free() }, "nothing holds them");
        assert!(!s.of_list.contains_key(&seg) && !s.of_list.contains_key(&old_seg));
        assert_eq!(s.of_list.get(&elsewhere), Some(&other), "a list on another allocator keeps its hold");
        assert!(s.backings[&other].users.contains(&elsewhere));
    }

    #[test]
    fn a_spare_whose_reset_fails_stays_available_for_retry() {
        let (alloc, spare, list) = (0x1000usize, 0x2000usize, 0x3000usize);
        let mut s = State::default();
        s.types.insert(alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        move_list(&mut s, list, alloc);
        s.aside.insert(alloc, vec![spare]);
        s.backings.insert(spare, Backing::default());
        let mut calls = Vec::new();

        for _ in 0..2 {
            let hr = unsafe {
                title_reset_in(&mut s, alloc as *mut c_void, |raw| {
                    calls.push(raw as usize);
                    crate::util::E_FAIL
                })
            };
            assert_eq!(hr, Some(crate::util::E_FAIL));
            assert_eq!(s.aside.get(&alloc), Some(&vec![spare]), "the failed candidate is still owned and reusable");
            assert!(!s.cur.contains_key(&alloc), "the current backing did not change");
            assert_eq!(s.of_list.get(&list), Some(&alloc));
            assert!(s.backings[&alloc].users.contains(&list), "the old recording is still held");
            assert_eq!((s.made, s.swaps), (0, 0));
        }

        let hr = unsafe {
            title_reset_in(&mut s, alloc as *mut c_void, |raw| {
                calls.push(raw as usize);
                0
            })
        };
        assert_eq!(hr, Some(0));
        assert_eq!(calls, vec![spare; 3], "all attempts used the same spare without creating another");
        assert_eq!(s.cur.get(&alloc), Some(&spare));
        assert_eq!(s.aside.get(&alloc), Some(&vec![alloc]), "only the successful reset switches backings");
        assert_eq!(s.of_list.get(&list), Some(&alloc), "the earlier recording stays on its original backing");
        assert_eq!((s.made, s.swaps), (0, 1));
    }

    /// Final-release cleanup drops only the recording's hold: another live list and a pending
    /// GPU fence each still prevent reuse, then completion makes the backing reusable.
    #[test]
    fn released_lists_leave_only_live_recordings_and_gpu_work_holding_the_backing() {
        use std::cell::Cell;
        use std::mem::{offset_of, size_of};

        #[repr(C)]
        struct Fence {
            vtable: *const usize,
            value: Cell<u64>,
            releases: Cell<u32>,
        }
        unsafe extern "system" fn completed(this: *mut c_void) -> u64 {
            (*this.cast::<Fence>()).value.get()
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            let f = &*this.cast::<Fence>();
            f.releases.set(f.releases.get() + 1);
            0
        }
        extern "system" fn unexpected_call() { std::process::abort(); }

        const SLOT: usize = size_of::<usize>();
        let mut vtable = [unexpected_call as usize; size_of::<ID3D12Fence_Vtbl>() / SLOT];
        vtable[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / SLOT] = completed as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.Release) / SLOT] = release as usize;
        let mut fence = Fence { vtable: vtable.as_ptr(), value: Cell::new(0), releases: Cell::new(0) };
        let mut ids = Box::new([0u8; 3]);
        let alloc = ids.as_mut_ptr() as usize;
        let (first, second) = (alloc + 1, alloc + 2);
        created_on(first, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        created_on(second, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        {
            let mut g = STATE.lock().unwrap();
            let b = g.as_mut().unwrap().backings.get_mut(&alloc).unwrap();
            // The map owns one synthetic fence reference, dropped by forget below.
            let f = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut fence).cast()) };
            b.marks.insert(1, (f, 7));
        }

        super::super::list_x::on_release(first);
        let first_removed = {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            !s.of_list.contains_key(&first) && s.of_list.get(&second) == Some(&alloc)
                && s.backings[&alloc].users == HashSet::from([second])
                && !unsafe { s.backings[&alloc].free() }
        };
        super::super::list_x::on_release(second);
        super::super::list_x::on_release(first); // duplicate cleanup is harmless
        let pending_still_held = {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            !s.of_list.contains_key(&second) && s.backings[&alloc].users.is_empty()
                && s.backings[&alloc].marks.len() == 1 && !unsafe { s.backings[&alloc].free() }
        };
        fence.value.set(7);
        let reusable = unsafe { STATE.lock().unwrap().as_ref().unwrap().backings[&alloc].free() };
        // Clean up the synthetic COM pointer before assertions, including on the old code.
        unsafe { forget(alloc) };
        assert!(first_removed, "another live recording must remain while the released list goes");
        assert!(pending_still_held, "final list Release must retain the submitted-work fence");
        assert!(reusable, "a released list cannot keep a completed backing busy");
        assert_eq!(fence.releases.get(), 1);
    }

    /// What an ended recording left in unified memory stays while the GPU has not passed the
    /// submits on its PC allocator – also when the title's Reset puts that allocator aside and
    /// goes on with another (#1358) – and goes back at the first Reset after the GPU has.
    #[test]
    fn ended_recordings_keep_their_memory_until_the_gpu_passes_their_allocator() {
        use std::cell::Cell;
        use std::mem::{offset_of, size_of};

        #[repr(C)]
        struct Fence {
            vtable: *const usize,
            value: Cell<u64>,
        }
        unsafe extern "system" fn completed(this: *mut c_void) -> u64 {
            (*this.cast::<Fence>()).value.get()
        }
        unsafe extern "system" fn release(_: *mut c_void) -> u32 {
            0
        }
        extern "system" fn unexpected_call() { std::process::abort(); }

        const SLOT: usize = size_of::<usize>();
        let mut vtable = [unexpected_call as usize; size_of::<ID3D12Fence_Vtbl>() / SLOT];
        vtable[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / SLOT] = completed as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.Release) / SLOT] = release as usize;
        let mut fence = Fence { vtable: vtable.as_ptr(), value: Cell::new(0) };
        let mut ids = Box::new([0u8; 4]);
        let alloc = ids.as_mut_ptr() as usize;
        let (spare, list, block) = (alloc + 1, alloc + 2, alloc + 3);
        // a list recorded on the title's allocator, was submitted, and is gone; the GPU has not
        // run the submit yet
        created_on(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            // the map owns one synthetic fence reference, dropped below
            let f = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut fence).cast()) };
            s.backings.get_mut(&alloc).unwrap().marks.insert(1, (f, 1));
            // a PC allocator put aside earlier that nothing uses any more
            s.aside.insert(alloc, vec![spare]);
        }
        forget_list(list);
        super::super::console_list::hold_until_reset(alloc, vec![block]);

        let mut resets = Vec::new();
        let first = unsafe { title_reset_with(alloc as *mut c_void, |raw| { resets.push(raw as usize); 0 }) };
        let stand_in = current(alloc);
        let held = super::super::console_list::take_pending_for_test(alloc);
        super::super::console_list::hold_until_reset(alloc, held.clone());
        fence.value.set(1);
        let second = unsafe { title_reset_with(alloc as *mut c_void, |raw| { resets.push(raw as usize); 0 }) };
        let after = super::super::console_list::take_pending_for_test(alloc);
        // (before the assertions: nothing synthetic stays in the pool)
        {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            for b in [alloc, spare] {
                s.backings.remove(&b);
                s.cur.remove(&b);
                s.aside.remove(&b);
                s.types.remove(&b);
            }
        }
        assert_eq!((first, stand_in), (Some(0), spare), "the GPU still runs the submit: the spare takes the place");
        assert_eq!(held, vec![block], "the memory of a recording the GPU still runs is held over the Reset");
        assert_eq!((second, resets), (Some(0), vec![spare, spare]));
        assert!(after.is_empty(), "the GPU passed the submit: the memory goes back");
    }

    /// The title's Reset (through `real::alloc_reset`) while the GPU has not run a submit of a
    /// recording on its allocator: the allocator is put aside, and what the recording left in
    /// unified memory stays until the GPU has passed it (#1358).
    #[test]
    fn a_reset_that_puts_the_allocator_aside_keeps_what_the_gpu_still_reads() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let a = alloc.as_raw();
            if !super::super::real::is_adopted(a) {
                super::super::real::adopt(a, super::super::real::Kind::CommandAllocator);
            }
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let l = list.as_raw() as usize;
            created_on(l, a as usize, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            list.Close().unwrap();
            // the queue runs the list only once `gate` is signalled
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Wait(&gate, 1).unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            executed(queue.as_raw(), &[l as *mut c_void]);
            // the list's recording ends; what it put in unified memory is held for the allocator
            forget_list(l);
            let block = l + 1;
            super::super::console_list::hold_until_reset(a as usize, vec![block]);

            alloc.Reset().unwrap();
            let stand_in = current(a as usize);
            let held = super::super::console_list::take_pending_for_test(a as usize);
            super::super::console_list::hold_until_reset(a as usize, held.clone());

            gate.Signal(1).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&done, 1).unwrap();
            let start = std::time::Instant::now();
            while done.GetCompletedValue() < 1 && start.elapsed().as_secs() < 10 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            let ran = done.GetCompletedValue() >= 1;
            alloc.Reset().unwrap();
            let after = super::super::console_list::take_pending_for_test(a as usize);
            drop(list);
            drop(alloc);

            assert_ne!(stand_in, a as usize, "the GPU had not run the submit: another PC allocator took the place");
            assert_eq!(held, vec![block], "the memory of a recording the GPU had not run is held over the Reset");
            assert!(ran, "the queue finishes");
            assert!(after.is_empty(), "the GPU passed the submit: the next Reset gives the memory back");
        }
    }

    /// A list recorded once and submitted again after the title reset its allocator: the
    /// recording is still there (the device is not removed), the title's allocator goes on with
    /// another PC allocator, and the first one is used again once the list has moved on.
    #[test]
    fn a_recording_outlives_the_reset_of_the_titles_allocator() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let a = alloc.as_raw();
            if !super::super::real::is_adopted(a) {
                super::super::real::adopt(a, super::super::real::Kind::CommandAllocator);
            }
            let list: ID3D12GraphicsCommandList = {
                let _alloc_guard = super::super::real::UnadoptGuard::new(a);
                dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap()
            };
            let l = list.as_raw();
            created_on(l as usize, a as usize, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            list.Close().unwrap();
            let lists = [Some(list.cast::<ID3D12CommandList>().unwrap())];
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done = |v: u64| {
                queue.Signal(&fence, v).unwrap();
                let start = std::time::Instant::now();
                while fence.GetCompletedValue() < v && start.elapsed().as_secs() < 10 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                assert!(fence.GetCompletedValue() >= v, "the queue finishes");
            };

            queue.ExecuteCommandLists(&lists);
            executed(queue.as_raw(), &[l]);
            done(1);

            // the title resets its allocator (through the hook: `real::alloc_reset`) …
            alloc.Reset().unwrap();
            let second = current(a as usize);
            assert_ne!(second, a as usize, "the list still holds its recording: another PC allocator takes the place");
            // … and submits the recording again
            queue.ExecuteCommandLists(&lists);
            executed(queue.as_raw(), &[l]);
            done(2);
            assert!(dev.GetDeviceRemovedReason().is_ok(), "the recording was still there");

            // the list is reset on the title's allocator: it records on the PC allocator in use
            let second_alloc = ID3D12CommandAllocator::from_raw_borrowed(&(second as *mut c_void)).unwrap().clone();
            assert_eq!(reset_list(l as usize, a as usize, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |pc| {
                assert_eq!(pc, second);
                let _alloc_guard = super::super::real::UnadoptGuard::new(pc as *mut c_void);
                list.Reset(&second_alloc, None).map_or_else(|e| e.code().0, |_| 0)
            }), 0);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&lists);
            executed(queue.as_raw(), &[l]);
            done(3);

            // the next reset: the first PC allocator holds nothing now and is used again
            alloc.Reset().unwrap();
            assert_eq!(current(a as usize), a as usize, "the allocator put aside is free again");
            assert!(dev.GetDeviceRemovedReason().is_ok());

            // the title's allocator goes while another PC allocator stands for it: nothing of
            // it stays (the next allocator at its address is its own)
            assert_eq!(reset_list(l as usize, a as usize, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |pc| {
                assert_eq!(pc, a as usize);
                let _alloc_guard = super::super::real::UnadoptGuard::new(pc as *mut c_void);
                list.Reset(&alloc, None).map_or_else(|e| e.code().0, |_| 0)
            }), 0);
            list.Close().unwrap();
            alloc.Reset().unwrap();
            assert_ne!(current(a as usize), a as usize);
            drop(lists);
            drop(list);
            drop(alloc);
            assert_eq!(current(a as usize), a as usize, "a released allocator leaves nothing behind");
            assert!(!STATE.lock().unwrap().as_ref().unwrap().types.contains_key(&(a as usize)));
        }
    }

    /// A list submitted again while its previous submit is still on the GPU is counted (and
    /// logged); once that submit is done, or after the list is forgotten, it is not.
    #[test]
    fn a_resubmit_while_the_previous_submit_runs_is_noticed() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            // a fence the GPU has not reached stands for a submit still running
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut ids = Box::new([0u8; 2]);
            let (alloc, list) = (ids.as_mut_ptr() as usize, ids.as_mut_ptr() as usize + 1);
            let queue = 0x10usize as *mut c_void;
            created_on(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            {
                let mut g = STATE.lock().unwrap();
                let s = g.as_mut().unwrap();
                let recording = s.recordings[&list];
                s.last_submit.insert(list, LastSubmit { queue: queue as usize, fence: fence.clone(), value: 1, recording });
            }
            let while_running = note_resubmits(queue, &[list as *mut c_void]);
            fence.Signal(1).unwrap();
            let after_done = note_resubmits(queue, &[list as *mut c_void]);
            {
                // running again, then the list goes: nothing is left to compare with
                let mut g = STATE.lock().unwrap();
                g.as_mut().unwrap().last_submit.get_mut(&list).unwrap().value = 2;
            }
            forget_list(list);
            let after_forget = note_resubmits(queue, &[list as *mut c_void]);
            forget(alloc);
            assert_eq!(while_running, 1, "a submit while the previous one runs must be noticed");
            assert_eq!(after_done, 0, "a finished previous submit is not a resubmit");
            assert_eq!(after_forget, 0, "a forgotten list has no previous submit");
        }
    }

    /// A list whose previous submit still runs is waited for before it is submitted again: until
    /// that submit is done, or the limit; a list with nothing running is not waited for.
    #[test]
    fn a_resubmit_waits_for_the_previous_submit() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            // a fence the GPU has not reached stands for a submit still running
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut ids = Box::new([0u8; 2]);
            let (alloc, list) = (ids.as_mut_ptr() as usize, ids.as_mut_ptr() as usize + 1);
            created_on(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            {
                let mut g = STATE.lock().unwrap();
                let s = g.as_mut().unwrap();
                let recording = s.recordings[&list];
                s.last_submit.insert(list, LastSubmit { queue: 0x10, fence: fence.clone(), value: 1, recording });
            }
            let lists = [list as *mut c_void];
            // it never finishes: the limit ends the wait, and says so
            let t = std::time::Instant::now();
            assert_eq!(wait_for_previous_submits_within(&lists, std::time::Duration::from_millis(30)), (1, 1));
            assert!(t.elapsed() >= std::time::Duration::from_millis(30));
            // no limit: no wait
            assert_eq!(wait_for_previous_submits_within(&lists, std::time::Duration::ZERO), (0, 0));
            // it finishes meanwhile: the wait ends then, far before the limit
            let signal = fence.clone();
            let done = std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(20));
                signal.Signal(1).unwrap();
            });
            let t = std::time::Instant::now();
            assert_eq!(wait_for_previous_submits_within(&lists, std::time::Duration::from_secs(20)), (1, 0));
            assert!(t.elapsed() < std::time::Duration::from_secs(10), "the wait ends when the submit is done");
            done.join().unwrap();
            // nothing runs now: nothing to wait for
            assert_eq!(wait_for_previous_submits_within(&lists, std::time::Duration::from_secs(20)), (0, 0));
            forget_list(list);
            forget(alloc);
        }
    }
}
