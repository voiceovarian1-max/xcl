//! Pre-entry shader preparation in the same HWND that later owns the DXGI swap chain.
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use windows::Win32::Foundation::{HWND, LRESULT, WPARAM};
use windows_sys::Win32::Foundation::{RECT, WAIT_OBJECT_0};
use windows_sys::Win32::Graphics::Gdi::*;
use windows_sys::Win32::System::Threading::WaitForSingleObject;
use windows_sys::Win32::UI::WindowsAndMessaging::*;
use crate::config::Boot;
use crate::shaders::{LaunchPreparation, prepare_for_launch};
use crate::util::wide;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static SCREEN: Mutex<Option<Screen>> = Mutex::new(None);
const READY: u32 = WM_APP + 0x40;

struct Screen {
    progress: Arc<LaunchPreparation>,
    start: Instant,
    clock: StallClock,
    ready: bool,
}

impl Screen {
    fn release(&mut self) -> bool {
        self.ready = !self.progress.cancel.load(Ordering::Relaxed);
        self.ready
    }
}

pub(super) fn active() -> bool { ACTIVE.load(Ordering::Acquire) }

pub(crate) fn before_guest(boot: &Boot) -> Result<(), String> {
    // This handle was opened by the launcher on itself, then inherited by this host.
    let parent = std::env::var("XCL_PREPARATION_PARENT").ok()
        .and_then(|s| s.parse::<usize>().ok()).filter(|&h| h != 0)
        .map(|h| unsafe { OwnedHandle::from_raw_handle(h as _) });
    let menu = std::env::var_os("XCL_PREPARATION_MENU").is_some();
    std::env::remove_var("XCL_PREPARATION_PARENT");
    std::env::remove_var("XCL_PREPARATION_MENU");
    let root = &boot.game_root;
    if boot.config.graphics.eq_ignore_ascii_case("null") ||
        (!root.join("shaders.rpf").try_exists().map_err(|e| e.to_string())? && !root.join("shaders").is_dir()) {
        return Ok(());
    }
    if left_out(std::env::var("XCL_PREPARE").ok().as_deref()) {
        crate::xlog_info!("Shader preparation left out (XCL_PREPARE=0): the title builds what it uses as it goes");
        return Ok(());
    }
    let only = Some(crate::hle::d3d12x::pso_async::earlier_used_shaders())
        .filter(|used| !used.is_empty() && !std::env::var("XCL_PREPARE_ALL").is_ok_and(|v| v.trim() == "1"));
    if let Some(used) = &only {
        crate::xlog_info!("Shader preparation: only the {} shaders earlier runs used (XCL_PREPARE_ALL=1: every record)", used.len());
    }
    let progress = Arc::new(LaunchPreparation { adapter: boot.config.d3d12_adapter.clone(), only, ..Default::default() });
    let hwnd = if super::enabled() {
        *SCREEN.lock().unwrap() = Some(Screen { progress: progress.clone(), start: Instant::now(), clock: StallClock::default(), ready: false });
        ACTIVE.store(true, Ordering::Release);
        let hwnd = super::ensure_window().ok_or("Could not create the presentation window for shader preparation")?;
        crate::xlog_info!("Shader preparation window {:#x}", hwnd.0 as usize);
        Some(hwnd)
    } else { None };
    let result = std::thread::scope(|scope| {
        let worker = scope.spawn(|| prepare_for_launch(root, &progress));
        // (a line every 5 s: a log that stands still for long reads as a frozen process)
        let (began, mut said) = (Instant::now(), Instant::now());
        while !worker.is_finished() {
            if said.elapsed() >= Duration::from_secs(5) {
                said = Instant::now();
                let workers = progress.workers.load(Ordering::Relaxed);
                let graphics = progress.graphics.load(Ordering::Relaxed);
                crate::xlog_info!(
                    "Shader preparation: {} of {} {} {} after {} s",
                    progress.done.load(Ordering::Relaxed),
                    progress.total.load(Ordering::Relaxed),
                    if graphics { "graphics pipelines" } else { "records" },
                    if graphics { format!("built by {workers} workers") } else if workers != 0 { format!("checked for pipelines by {workers} workers") } else { "translated".to_string() },
                    began.elapsed().as_secs()
                );
            }
            if let Some(parent) = &parent {
                if unsafe { WaitForSingleObject(parent.as_raw_handle(), 80) } == WAIT_OBJECT_0 {
                    // The launcher was stopped during preparation. Do not leave an orphan compile or start the title.
                    std::process::exit(2);
                }
            } else { std::thread::sleep(Duration::from_millis(80)); }
        }
        worker.join().map_err(|_| "Shader preparation worker stopped unexpectedly")?
    });
    let mut result = if progress.cancel.load(Ordering::Relaxed) { Err("Shader preparation cancelled".into()) } else { result };
    if let Some(hwnd) = hwnd {
        if result.is_ok() {
            // Complete painting on the window's owner thread before the guest can draw.
            if unsafe { SendMessageW(hwnd.0, READY, 0, 0) } == 0 {
                result = Err("Shader preparation cancelled".into());
            }
        } else if let Err(e) = &result {
            if menu && !progress.cancel.load(Ordering::Relaxed) {
                unsafe { MessageBoxW(hwnd.0, wide(e).as_ptr(), wide("XCL – Shader preparation failed").as_ptr(), MB_OK | MB_ICONERROR); }
            }
        }
        ACTIVE.store(false, Ordering::Release);
        *SCREEN.lock().unwrap() = None;
    }
    result
}

pub(super) unsafe fn on_message(hwnd: HWND, msg: u32, wp: WPARAM) -> Option<LRESULT> {
    if !active() { return None; }
    match msg {
        WM_PAINT => {
            let mut guard = SCREEN.lock().unwrap();
            if let Some(s) = guard.as_mut() { paint(hwnd, s); }
        }
        WM_ERASEBKGND => return Some(LRESULT(1)),
        WM_TIMER if wp.0 == super::COMPILE_TIMER => { InvalidateRect(hwnd.0, std::ptr::null(), 0); }
        WM_CLOSE => {
            if let Some(s) = SCREEN.lock().unwrap().as_ref() { s.progress.cancel.store(true, Ordering::Relaxed); }
            InvalidateRect(hwnd.0, std::ptr::null(), 0);
        }
        READY => {
            {
                let mut view = SCREEN.lock().unwrap();
                let s = view.as_mut()?;
                if !s.release() { return Some(LRESULT(0)); }
            }
            RedrawWindow(hwnd.0, std::ptr::null(), std::ptr::null_mut(), RDW_INVALIDATE | RDW_UPDATENOW);
            crate::hle::kbm::attach(hwnd.0);
            ACTIVE.store(false, Ordering::Release);
            return Some(LRESULT(1));
        }
        _ => return None,
    }
    Some(LRESULT(0))
}

unsafe fn fill(dc: HDC, rect: RECT, color: u32) {
    let brush = CreateSolidBrush(color);
    FillRect(dc, &rect, brush);
    DeleteObject(brush);
}

unsafe fn text(dc: HDC, rect: RECT, value: &str, size: i32, weight: i32, color: u32, align: u32) {
    let font = CreateFontW(-size, 0, 0, 0, weight, 0, 0, 0, DEFAULT_CHARSET as _, 0, 0, CLEARTYPE_QUALITY as _, 0, wide("Segoe UI").as_ptr());
    let old = SelectObject(dc, font);
    SetTextColor(dc, color);
    let utf = wide(value);
    let mut rect = rect;
    DrawTextW(dc, utf.as_ptr(), (utf.len()-1) as i32, &mut rect, DT_SINGLELINE | DT_VCENTER | DT_NOPREFIX | DT_END_ELLIPSIS | align);
    SelectObject(dc, old);
    DeleteObject(font);
}

unsafe fn paint(hwnd: HWND, s: &mut Screen) {
    let mut ps: PAINTSTRUCT = std::mem::zeroed();
    let dest = BeginPaint(hwnd.0, &mut ps);
    let mut rc = RECT::default();
    GetClientRect(hwnd.0, &mut rc);
    if rc.right > 0 && rc.bottom > 0 {
        let dc = CreateCompatibleDC(dest);
        let bitmap = CreateCompatibleBitmap(dest, rc.right, rc.bottom);
        let old = SelectObject(dc, bitmap);
        SetBkMode(dc, TRANSPARENT as i32);
        fill(dc, rc, 0x201815);
        let scale = |px: i32| (px * GetDeviceCaps(dc, LOGPIXELSY as i32).max(96)) / 96;
        let width = scale(560).min((rc.right - scale(40)).max(1));
        let x = (rc.right - width) / 2;
        let y = ((rc.bottom - scale(150)) / 2).max(0);
        let row = |top, bottom| RECT { left: x, right: x+width, top: y+scale(top), bottom: y+scale(bottom) };
        let done = s.progress.done.load(Ordering::Relaxed);
        let total = s.progress.total.load(Ordering::Relaxed);
        let pct = percent(done, total, s.ready);
        let (quiet, last) = s.clock.sample(done, s.start.elapsed());
        let workers = s.progress.workers.load(Ordering::Relaxed);
        let graphics = workers != 0 && s.progress.graphics.load(Ordering::Relaxed);
        let heading = if s.progress.cancel.load(Ordering::Relaxed) { "Cancelling preparation" } else if graphics { "Building graphics pipelines" } else if workers != 0 { "Building pipelines" } else { "Preparing shaders" };
        let mut title = row(0,48);
        title.right -= scale(84);
        text(dc, title, heading, scale(31), 600, 0xf7f5f1, DT_LEFT);
        text(dc, row(0,48), &format!("{pct}%"), scale(27), 600, 0xdfd09a, DT_RIGHT);
        let bar = row(65,72);
        fill(dc, bar, 0x4b4038);
        fill(dc, RECT {right: x+(width as i64 * pct as i64/100) as i32, ..bar}, 0xbdb458);
        let count = if total == 0 { "Reading shader records…".into() }
            else if graphics { format!("{done} of {total} graphics pipelines built by {workers} worker processes") }
            else if workers != 0 { format!("{done} of {total} records checked by {workers} worker processes") }
            else { format!("{done} of {total} records processed") };
        text(dc, row(87,112), &count, scale(17), 400, 0xe1dcd7, DT_LEFT);
        let mut times = if workers != 0 { format!("Large pipelines compile in the driver: {quiet} s since the last record") } else { format!("No shader progress for {quiet} s") };
        if let Some(ago) = last { times.push_str(&format!("    ·    Last stall ended {ago} s ago")); }
        text(dc, row(117,145), &times, scale(14), 400, 0xb4aaa4, DT_LEFT);
        BitBlt(dest, 0, 0, rc.right, rc.bottom, dc, 0, 0, SRCCOPY);
        SelectObject(dc, old);
        DeleteObject(bitmap);
        DeleteDC(dc);
    }
    EndPaint(hwnd.0, &ps);
}

#[derive(Default)]
struct StallClock {
    done: usize,
    advanced_at: Duration,
    last_stall_ended: Option<Duration>,
}

impl StallClock {
    fn sample(&mut self, done: usize, now: Duration) -> (u64, Option<u64>) {
        if done != self.done {
            if now - self.advanced_at >= Duration::from_secs(5) {
                self.last_stall_ended = Some(now);
            }
            self.done = done;
            self.advanced_at = now;
        }
        ((now - self.advanced_at).as_secs(), self.last_stall_ended.map(|end| (now - end).as_secs()))
    }
}

fn percent(done: usize, total: usize, ready: bool) -> usize {
    if ready { 100 } else { done.saturating_mul(100).checked_div(total).unwrap_or(0).min(99) }
}

/// `XCL_PREPARE=0` (`value`): the title starts at once and builds what it uses as it goes,
/// standing still for those builds, as before the preparation (the switch nvidia-dev has).
fn left_out(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim() == "0")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only `XCL_PREPARE=0` leaves the preparation out; unset or any other value prepares.
    #[test]
    fn only_xcl_prepare_0_leaves_the_preparation_out() {
        assert!(left_out(Some("0")) && left_out(Some(" 0 ")));
        assert!(!left_out(None) && !left_out(Some("1")) && !left_out(Some("")) && !left_out(Some("no")));
    }

    #[test]
    fn clock_advances_during_a_stall_and_remembers_recovery() {
        let mut clock = StallClock::default();
        let s = Duration::from_secs;
        assert_eq!(clock.sample(0, s(1)), (1, None));
        assert_eq!(clock.sample(0, s(5)), (5, None));
        assert_eq!(clock.sample(2, s(6)), (0, Some(0)));
        assert_eq!(clock.sample(2, s(8)), (2, Some(2)));
        assert_eq!(clock.sample(3, s(9)), (0, Some(3)));
        assert_eq!(clock.sample(3, s(15)), (6, Some(9)));
        assert_eq!(clock.sample(4, s(16)), (0, Some(0)));
    }

    #[test]
    fn close_before_the_ready_message_cancels_the_handoff() {
        let progress = Arc::new(LaunchPreparation::default());
        let mut screen = Screen { progress: progress.clone(), start: Instant::now(), clock: StallClock::default(), ready: false };
        progress.cancel.store(true, Ordering::Relaxed);
        assert!(!screen.release());
        assert!(!screen.ready);
    }

    #[test]
    fn only_a_successful_join_releases_the_last_percent() {
        assert_eq!(percent(0, 0, false), 0);
        assert_eq!(percent(1, 4, false), 25);
        assert_eq!(percent(4, 4, false), 99);
        assert_eq!(percent(4, 4, true), 100);
    }
}
