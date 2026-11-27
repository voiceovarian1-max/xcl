//! Diagnostic (`XCL_ACTIVITY`, and `pso_async`'s pipeline line): the time the title's threads
//! spend in some of XCL's calls while it loads, summed over the threads.
//!
//! With `XCL_LOOK_STATS=2` also the calls a frame is made of on the thread that makes the title's
//! submits (its submits, presents and waits), and every call counted on that thread apart: what
//! it does in a frame besides submitting, added up.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub enum Call {
    PlacedTexture,
    PlacedBuffer,
    PhysicalPages,
    Memory,
    Pipeline,
    /// a bind's wait for a pipeline still being built (`pso_async::resolve_wait`)
    PipelineWait,
    /// the title's ExecuteCommandLists, whole (`submit_cost::done`; `XCL_LOOK_STATS=2` only, as
    /// the ones below)
    Submit,
    /// PresentX, or a present of the fallback's at a console method
    Present,
    /// a present's wait for the copy of the frame before
    PresentCopyWait,
    WaitFrameEvent,
    /// the title's blocking waits for a fence (SetEventOnCompletion without an event)
    FenceWait,
    /// copies back landed for the title's look at a fence (`sparse::settle_readbacks_within`)
    FenceSettle,
    TexturePages,
    TextureTargets,
    TextureRegister,
    WindowMap,
    WindowUnmap,
    PageMap,
    PageUnmap,
    Readable,
    ReadbackNow,
    SettleReadbacks,
}

pub const CALLS: [(Call, &str); 22] = [
    (Call::PlacedTexture, "placed textures"),
    (Call::PlacedBuffer, "placed buffers"),
    (Call::PhysicalPages, "physical page calls"),
    (Call::Memory, "memory calls"),
    (Call::Pipeline, "pipelines"),
    (Call::PipelineWait, "waits for pipelines being built"),
    (Call::Submit, "ExecuteCommandLists"),
    (Call::Present, "presents"),
    (Call::PresentCopyWait, "waits of a present for the last frame's copy"),
    (Call::WaitFrameEvent, "WaitFrameEventX"),
    (Call::FenceWait, "blocking fence waits"),
    (Call::FenceSettle, "copies back settled at fence looks"),
    (Call::TexturePages, "of them: texture pages"),
    (Call::TextureTargets, "of them: video-memory targets"),
    (Call::TextureRegister, "of them: kept in step"),
    (Call::WindowMap, "sparse window maps"),
    (Call::WindowUnmap, "sparse window unmaps"),
    (Call::PageMap, "texture page maps"),
    (Call::PageUnmap, "texture page unmaps"),
    (Call::Readable, "readable checks outside unified memory"),
    (Call::ReadbackNow, "CPU reads of GPU-written memory (readback_now)"),
    (Call::SettleReadbacks, "readback settles"),
];

/// calls, microseconds
static COST: [[AtomicU64; 2]; CALLS.len()] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; CALLS.len()];
/// the same on the submitting thread (`SUBMITTER`)
static ON_SUBMITTER: [[AtomicU64; 2]; CALLS.len()] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; CALLS.len()];
/// the thread that makes the title's submits (0: none seen, or `XCL_LOOK_STATS` is not 2)
static SUBMITTER: AtomicU32 = AtomicU32::new(0);

/// Counts the call it stands in when it goes.
pub struct Timer {
    pub call: Call,
    start: Instant,
}

pub fn time(call: Call) -> Timer {
    Timer { call, start: Instant::now() }
}

impl Drop for Timer {
    fn drop(&mut self) {
        add(self.call, self.start.elapsed());
    }
}

fn this_thread() -> u32 {
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

/// A call that took `d`.
pub fn add(call: Call, d: Duration) {
    let us = d.as_micros() as u64;
    let c = &COST[call as usize];
    c[0].fetch_add(1, Ordering::Relaxed);
    c[1].fetch_add(us, Ordering::Relaxed);
    let s = SUBMITTER.load(Ordering::Relaxed);
    if s != 0 && s == this_thread() {
        let c = &ON_SUBMITTER[call as usize];
        c[0].fetch_add(1, Ordering::Relaxed);
        c[1].fetch_add(us, Ordering::Relaxed);
    }
}

/// The calling thread made one of the title's submits: the first thread to is the submitting
/// thread from then on (a retail title's world made every submit and present on one thread).
/// Called with `XCL_LOOK_STATS=2` only.
pub fn note_submitter() {
    let _ = SUBMITTER.compare_exchange(0, this_thread(), Ordering::Relaxed, Ordering::Relaxed);
}

/// (calls, microseconds) so far, in the order of `CALLS`.
pub fn totals() -> [(u64, u64); CALLS.len()] {
    std::array::from_fn(|k| (COST[k][0].load(Ordering::Relaxed), COST[k][1].load(Ordering::Relaxed)))
}

/// `totals` of the submitting thread
pub fn submitter_totals() -> [(u64, u64); CALLS.len()] {
    std::array::from_fn(|k| (ON_SUBMITTER[k][0].load(Ordering::Relaxed), ON_SUBMITTER[k][1].load(Ordering::Relaxed)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_call_is_counted_with_its_time() {
        // (in the order of the enum: a name stands for its own count)
        assert!(CALLS.iter().enumerate().all(|(k, c)| c.0 as usize == k));
        let before = totals()[Call::Memory as usize];
        {
            let mut t = time(Call::Pipeline);
            t.call = Call::Memory;
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let after = totals()[Call::Memory as usize];
        assert!(after.0 > before.0 && after.1 >= before.1 + 2000);
    }

    #[test]
    fn the_submitting_thread_has_its_own_count() {
        // (no other test makes a submitter: this thread is it)
        note_submitter();
        let me = this_thread();
        assert_eq!(SUBMITTER.load(Ordering::Relaxed), me);
        std::thread::spawn(note_submitter).join().unwrap();
        assert_eq!(SUBMITTER.load(Ordering::Relaxed), me, "the first thread stays the submitting one");
        let k = Call::FenceSettle as usize;
        let (all, mine) = (totals()[k], submitter_totals()[k]);
        add(Call::FenceSettle, Duration::from_micros(300));
        std::thread::spawn(|| add(Call::FenceSettle, Duration::from_micros(50))).join().unwrap();
        let (all2, mine2) = (totals()[k], submitter_totals()[k]);
        assert_eq!((all2.0 - all.0, all2.1 - all.1), (2, 350));
        assert_eq!((mine2.0 - mine.0, mine2.1 - mine.1), (1, 300));
    }
}
