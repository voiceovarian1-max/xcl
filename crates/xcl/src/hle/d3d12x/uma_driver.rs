//! A coherent UMA window in the driver's memory instead of the title's own pages opened with
//! `OpenExistingHeapFromAddress`. On NVIDIA, GPU loads and atomics on the imported pages ran 8-22x
//! slower than on a driver CUSTOM L0 heap, and a whole-frame replay ran window 3's programs 4-270x
//! faster from such a heap. The title keeps its addresses: the window's pages are copied out, the
//! window is released, every other large free stretch of address space is reserved for a moment,
//! so the driver's CPU mapping of the new heap can only land in the hole the window left, and the
//! pages go back. When the mapping lands anywhere else, the window is made again and imported as
//! before. The heap is made with write watch (`D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH`), so the window
//! stays in `Uma::watch_regions` and the title's CPU writes reach its textures as before; a driver
//! that refuses it, or whose mapping `GetWriteWatch` cannot read, keeps the import.
//!
//! `XCL_UMA_DRIVER`: `custom` or `1` on, `0` off; unset: on for an NVIDIA adapter outside Wine.

use std::ffi::c_void;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_FREE, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS,
    PAGE_READWRITE,
};

const MEM_WRITE_WATCH: u32 = 0x20_0000;

#[link(name = "kernel32")]
extern "system" {
    fn GetWriteWatch(flags: u32, base: *const c_void, size: usize, addresses: *mut *mut c_void, count: *mut usize, granularity: *mut u32) -> u32;
}

/// Whether the coherent windows of `device` become driver heaps (see the module's comment).
pub(super) fn wanted(device: &ID3D12Device) -> bool {
    decide(std::env::var("XCL_UMA_DRIVER").ok().as_deref(), crate::util::under_wine, || vendor(device))
}

/// `wanted` from the switch's value, whether the process runs under Wine and the adapter's vendor.
fn decide(switch: Option<&str>, wine: impl FnOnce() -> bool, vendor: impl FnOnce() -> Option<u32>) -> bool {
    match switch.map(str::trim) {
        Some("custom" | "1") => true,
        Some("0") => false,
        _ => !wine() && vendor() == Some(0x10DE),
    }
}

/// PCI vendor id of `device`'s adapter.
fn vendor(device: &ID3D12Device) -> Option<u32> {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter, IDXGIFactory4};
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
        let adapter: IDXGIAdapter = factory.EnumAdapterByLuid(device.GetAdapterLuid()).ok()?;
        Some(adapter.GetDesc().ok()?.VendorId)
    }
}

/// The written pages of `[base, base + size)` that hold anything but zeros, with their bytes.
unsafe fn written_pages(base: usize, size: usize) -> Result<Vec<(usize, Vec<u8>)>, String> {
    let mut out = Vec::new();
    let mut buf: Vec<*mut c_void> = vec![std::ptr::null_mut(); 1 << 16];
    let end = base + size;
    let mut at = base;
    while at < end {
        let (mut n, mut page) = (buf.len(), 0u32);
        let status = GetWriteWatch(0, at as *const c_void, end - at, buf.as_mut_ptr(), &mut n, &mut page);
        if status != 0 {
            return Err(format!("GetWriteWatch at {at:#x} failed ({status})"));
        }
        for &p in &buf[..n] {
            let s = std::slice::from_raw_parts(p as *const u8, page as usize);
            if s.iter().any(|&b| b != 0) {
                out.push((p as usize, s.to_vec()));
            }
        }
        if n < buf.len() {
            break;
        }
        at = buf[n - 1] as usize + page as usize;
    }
    Ok(out)
}

/// Reserves every free stretch of 1 GiB or more (but its first 256 MiB, room for the driver's own
/// small mappings) outside `[hole, hole + len)`.
unsafe fn groom(hole: usize, len: usize) -> Vec<usize> {
    let (top, mut a, mut got) = (0x7FFF_FFFE_0000usize, 0x1_0000usize, Vec::new());
    while a < top {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(a as *const c_void, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) == 0 {
            break;
        }
        let (base, end) = (m.BaseAddress as usize, (m.BaseAddress as usize + m.RegionSize).min(top));
        a = end;
        if m.State != MEM_FREE || m.RegionSize < 1 << 30 {
            continue;
        }
        let s = ((base + 0xFFFF) & !0xFFFF) + (256 << 20);
        for (s, e) in [(s, end.min(hole)), (s.max(hole + len), end)] {
            let mut s = s;
            while s < e {
                let n = (e - s).min(1 << 40);
                let p = VirtualAlloc(s as *const c_void, n, MEM_RESERVE, PAGE_NOACCESS);
                if !p.is_null() {
                    got.push(p as usize);
                }
                s += n;
            }
        }
    }
    got
}

unsafe fn ungroom(got: Vec<usize>) {
    for p in got {
        VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
    }
}

/// Window `[base, base + size)` (a whole `VirtualAlloc` allocation of its own) remade as a driver
/// heap mapped at `base`, its pages as they were and write-watched: the heap and a buffer over all
/// of it, mapped for good. On failure the window is the title's memory again, its pages as they
/// were.
pub(super) unsafe fn remake(device: &ID3D12Device, base: usize, size: usize) -> Result<(ID3D12Heap, ID3D12Resource), String> {
    remake_with_after_release(device, base, size, || Ok(()))
}

unsafe fn remake_with_after_release(
    device: &ID3D12Device,
    base: usize,
    size: usize,
    after_release: impl FnOnce() -> Result<(), String>,
) -> Result<(ID3D12Heap, ID3D12Resource), String> {
    let t0 = std::time::Instant::now();
    let pages = written_pages(base, size)?;
    let kept: usize = pages.iter().map(|p| p.1.len()).sum();
    if VirtualFree(base as *mut c_void, 0, MEM_RELEASE) == 0 {
        return Err(format!("VirtualFree of the window at {base:#x} failed"));
    }
    let restore = |why: String| -> String {
        let p = VirtualAlloc(base as *const c_void, size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
        if p as usize != base {
            let os_error = windows_sys::Win32::Foundation::GetLastError();
            crate::xlog_error!(
                "D3D12.X: coherent window could not be restored at {base:#x} ({size} bytes, got {p:?}, Windows error {os_error}) after {why}; stopping before guest memory is used"
            );
            crate::log::flush();
            windows_sys::Win32::System::Threading::TerminateProcess(
                windows_sys::Win32::System::Threading::GetCurrentProcess(),
                0x7C1_0003,
            );
            std::process::abort();
        }
        for (a, bytes) in &pages {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), *a as *mut u8, bytes.len());
        }
        why
    };
    let groomed = groom(base, size);
    let reservations = groomed.len();
    let made = (|| -> Result<(ID3D12Heap, ID3D12Resource, usize), String> {
        after_release()?;
        let properties = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        // write-watched as the title's own window was: its CPU writes to textures there are found so
        let flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH;
        let desc = D3D12_HEAP_DESC { SizeInBytes: size as u64, Properties: properties, Alignment: 0x1_0000, Flags: flags };
        let mut heap: Option<ID3D12Heap> = None;
        device.CreateHeap(&desc, &mut heap).map_err(|e| format!("CreateHeap (write-watched): {e}"))?;
        let heap = heap.ok_or("CreateHeap")?;
        let buf_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buf: Option<ID3D12Resource> = None;
        device.CreatePlacedResource(&heap, 0, &buf_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf).map_err(|e| format!("CreatePlacedResource: {e}"))?;
        let buf = buf.ok_or("CreatePlacedResource")?;
        let mut p: *mut c_void = std::ptr::null_mut();
        buf.Map(0, Some(&D3D12_RANGE::default()), Some(&mut p)).map_err(|e| format!("Map: {e}"))?;
        Ok((heap, buf, p as usize))
    })();
    ungroom(groomed);
    // (`GetWriteWatch` reads the mapping: the window can stay in `Uma::watch_regions`)
    let watched = || {
        let (mut one, mut n, mut page) = ([std::ptr::null_mut::<c_void>(); 1], 1usize, 0u32);
        GetWriteWatch(0, base as *const c_void, size, one.as_mut_ptr(), &mut n, &mut page) == 0
    };
    match made {
        Ok((heap, buf, p)) if p == base && watched() => {
            for (a, bytes) in &pages {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), *a as *mut u8, bytes.len());
            }
            crate::xlog_info!(
                "D3D12.X: the coherent window at {base:#x} ({} MB) is a driver CUSTOM write-back L0 heap mapped at its address, write-watched ({reservations} reservations, {} pages kept, {} KB, {} ms)",
                size >> 20,
                pages.len(),
                kept >> 10,
                t0.elapsed().as_millis()
            );
            Ok((heap, buf))
        }
        Ok((heap, buf, p)) => {
            buf.Unmap(0, None);
            drop((buf, heap));
            Err(restore(if p == base {
                "GetWriteWatch cannot read the heap's mapping".to_string()
            } else {
                format!("Map gave {p:#x}, not the window's {base:#x} ({reservations} reservations)")
            }))
        }
        Err(e) => Err(restore(e)),
    }
}

/// `heap` and `buf` as the raw pointers `open_uma` keeps.
pub(super) fn raw(heap: ID3D12Heap, buf: ID3D12Resource) -> (usize, u64, usize) {
    let va = unsafe { buf.GetGPUVirtualAddress() };
    (buf.into_raw() as usize, va, heap.into_raw() as usize)
}

#[cfg(test)]
mod tests {
    use super::decide;

    /// On by default for NVIDIA outside Wine only; the switch overrides either way.
    #[test]
    fn driver_windows_by_default_on_nvidia_outside_wine() {
        assert!(decide(None, || false, || Some(0x10DE)));
        assert!(!decide(None, || true, || Some(0x10DE)));
        assert!(!decide(None, || false, || Some(0x1002)));
        assert!(!decide(None, || false, || None));
        assert!(!decide(Some("0"), || false, || Some(0x10DE)));
        assert!(decide(Some("custom"), || true, || Some(0x1002)));
        assert!(decide(Some(" 1 "), || false, || None));
    }

    #[test]
    fn failed_write_watch_does_not_release_the_original_window() {
        use super::*;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        const SIZE: usize = 1 << 16;
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            std::ptr::write_bytes(base as *mut u8, 0x5a, SIZE);
            let mut dev = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
                return crate::gpu::recomp::runner::skip_test("no D3D12 device");
            }
            // A regular VirtualAlloc block lacks MEM_WRITE_WATCH, forcing GetWriteWatch to fail.
            assert!(remake(&dev.unwrap(), base, SIZE).is_err());
            let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            assert_ne!(VirtualQuery(base as *const c_void, &mut info, std::mem::size_of_val(&info)), 0);
            assert_eq!(info.AllocationBase as usize, base);
            assert_eq!(info.State, MEM_COMMIT);
            assert!(std::slice::from_raw_parts(base as *const u8, SIZE).iter().all(|&b| b == 0x5a));
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn failed_restore_stops_before_importing_an_unbacked_window() {
        use super::*;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        const SIZE: usize = 1 << 16;
        const CODE: i32 = 0x7C1_0003;
        if std::env::var_os("XCL_TEST_RESTORE_COLLISION").is_some() {
            unsafe {
                let mut dev = None;
                D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).unwrap();
                let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
                assert_ne!(base, 0);
                (base as *mut u8).write(0x5a);
                let result = remake_with_after_release(&dev.unwrap(), base, SIZE, || {
                    let occupied = VirtualAlloc(base as *const c_void, SIZE, MEM_RESERVE, PAGE_NOACCESS);
                    assert_eq!(occupied as usize, base);
                    Err("synthetic post-release failure".into())
                });
                eprintln!("continued after failed restore: {result:?}");
            }
            return;
        }
        let mut dev: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev) }.is_err() {
            return crate::gpu::recomp::runner::skip_test("no D3D12 device");
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "hle::d3d12x::uma_driver::tests::failed_restore_stops_before_importing_an_unbacked_window", "--nocapture"])
            .env("XCL_TEST_RESTORE_COLLISION", "1")
            .output()
            .unwrap();
        let transcript = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        assert_eq!(output.status.code(), Some(CODE), "{transcript}");
        assert!(transcript.contains("coherent window could not be restored"), "{transcript}");
        assert!(!transcript.contains("continued after failed restore"), "{transcript}");
    }

    /// (probe) ns per loop iteration of GPU atomics and loads on one buffer in each memory home:
    /// video memory, the title's own pages opened with `OpenExistingHeapFromAddress` (as XCL opens
    /// the coherent window), the same with write watch, and a driver CUSTOM write-back L0 heap (as
    /// `remake` makes it). The table of PR #2844, for the adapter this runs on. Asserts nothing.
    #[test]
    fn gpu_memory_home_probe() {
        use super::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        const HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0)"
cbuffer C : register(b0) { uint boxes; uint iters; uint mode; uint stride; };
RWByteAddressBuffer mem : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint acc = 0;
    [loop] for (uint i = 0; i < iters; i++) {
        uint box = (id.x * 7u + i * 131u) % boxes;
        if (mode == 0) {
            uint was;
            mem.InterlockedAdd(box * stride, 1u, was);
            acc += was;
        } else {
            acc += mem.Load(box * stride);
        }
    }
    if (acc == 0x9e3779b9u) mem.Store(4, acc);
}
"#;
        const STRIDE: u32 = 256;
        const SIZE: usize = 65536 * STRIDE as usize;
        if std::env::var_os("XCL_MEMORY_HOME_PROBE").is_none() {
            return crate::gpu::recomp::runner::skip_test("probe: set XCL_MEMORY_HOME_PROBE=1");
        }
        let dev = match crate::gpu::recomp::runner::device() {
            Ok(d) => d,
            Err(e) => return crate::gpu::recomp::runner::skip_test(&e),
        };
        let dxil = match crate::gpu::recomp::dxc::compile_cached(HLSL, "main", "cs_6_6") {
            Ok(d) => d.0,
            Err(e) => return crate::gpu::recomp::runner::skip_test(&e),
        };
        unsafe {
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).unwrap();
            let mut cdesc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&cdesc).unwrap();
            std::mem::ManuallyDrop::drop(&mut cdesc.pRootSignature);
            let buf_desc = |flags: D3D12_RESOURCE_FLAGS| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: SIZE as u64,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let placed = |heap: &ID3D12Heap, flags| -> ID3D12Resource {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreatePlacedResource(heap, 0, &buf_desc(flags), D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                r.unwrap()
            };
            let imported = |extra: u32| -> Option<ID3D12Resource> {
                let p = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE | MEM_COMMIT | extra, PAGE_READWRITE);
                assert!(!p.is_null());
                let dev3: ID3D12Device3 = dev.cast().ok()?;
                match dev3.OpenExistingHeapFromAddress::<ID3D12Heap>(p) {
                    Ok(h) => Some(placed(&h, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER)),
                    Err(e) => {
                        eprintln!("OpenExistingHeapFromAddress: {e}");
                        None
                    }
                }
            };
            let mut homes: Vec<(&str, ID3D12Resource)> = Vec::new();
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &buf_desc(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut r,
            )
            .unwrap();
            homes.push(("video memory", r.unwrap()));
            if let Some(b) = imported(0) {
                homes.push(("imported", b));
            }
            if let Some(b) = imported(MEM_WRITE_WATCH) {
                homes.push(("imported, write watch", b));
            }
            // (CUSTOM L1 write-combine: under vkd3d-proton DEVICE_LOCAL | HOST_VISIBLE, video memory the
            // CPU maps through the BAR)
            for (name, flags, page, pool) in [
                ("CUSTOM L0 write-back", D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0),
                ("CUSTOM L0 write-back, write watch", D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0),
                ("CUSTOM L1 write-combine", D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L1),
            ] {
                let desc = D3D12_HEAP_DESC {
                    SizeInBytes: SIZE as u64,
                    Properties: D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_CUSTOM,
                        CPUPageProperty: page,
                        MemoryPoolPreference: pool,
                        ..Default::default()
                    },
                    Alignment: 0x1_0000,
                    Flags: flags,
                };
                let mut heap: Option<ID3D12Heap> = None;
                match dev.CreateHeap(&desc, &mut heap) {
                    Ok(()) => homes.push((name, placed(&heap.unwrap(), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))),
                    Err(e) => eprintln!("CreateHeap {name}: {e}"),
                }
            }
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let freq = queue.GetTimestampFrequency().unwrap() as f64;
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut qh: Option<ID3D12QueryHeap> = None;
            dev.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut qh).unwrap();
            let qh = qh.unwrap();
            let mut rb: Option<ID3D12Resource> = None;
            let mut rb_desc = buf_desc(D3D12_RESOURCE_FLAG_NONE);
            rb_desc.Width = 16;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &rb_desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
            let rb = rb.unwrap();
            let mut value = 0u64;
            // (groups, iterations): 65,536 threads, as in the PR's 1024x64 groups
            let (groups, iters) = (1024u32, 64u32);
            eprintln!("memory home probe, {}: ns per loop iteration (atomics / loads), {groups}x64 threads, {iters} iterations, {STRIDE} B apart", crate::gpu::recomp::runner::adapter().unwrap_or_default());
            for (name, buf) in &homes {
                let mut row = format!("{name:>34}:");
                for boxes in [1u32, 256, 4096, 65536] {
                    let mut cell = Vec::new();
                    for mode in [0u32, 1] {
                        let mut best = f64::MAX;
                        for _ in 0..3 {
                            alloc.Reset().unwrap();
                            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                            list.SetComputeRootSignature(&rs);
                            list.SetPipelineState(&pso);
                            list.SetComputeRoot32BitConstants(0, 4, [boxes, iters, mode, STRIDE].as_ptr().cast(), 0);
                            list.SetComputeRootUnorderedAccessView(1, buf.GetGPUVirtualAddress());
                            list.EndQuery(&qh, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                            list.Dispatch(groups, 1, 1);
                            list.EndQuery(&qh, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                            list.ResolveQueryData(&qh, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &rb, 0);
                            list.Close().unwrap();
                            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                            value += 1;
                            queue.Signal(&fence, value).unwrap();
                            while fence.GetCompletedValue() < value {
                                std::thread::yield_now();
                            }
                            let mut p: *mut c_void = std::ptr::null_mut();
                            rb.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut p)).unwrap();
                            let t = std::slice::from_raw_parts(p as *const u64, 2);
                            let ns = (t[1].wrapping_sub(t[0])) as f64 / freq * 1e9;
                            rb.Unmap(0, None);
                            best = best.min(ns / iters as f64);
                        }
                        cell.push(format!("{best:.1}"));
                    }
                    row += &format!("  {boxes:>5} boxes {:>8} / {:<8}", cell[0], cell[1]);
                }
                // CPU writes and reads of the whole buffer, where it can be mapped
                let mut p: *mut c_void = std::ptr::null_mut();
                if buf.Map(0, None, Some(&mut p)).is_ok() && !p.is_null() {
                    // where the mapping is, and what VirtualQuery knows of it (under Wine: whether Wine's
                    // allocator placed it)
                    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                    VirtualQuery(p, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>());
                    row += &format!("  at {p:?} (state {:#x}, type {:#x}, allocation {:?})", m.State, m.Type, m.AllocationBase);
                    let t0 = std::time::Instant::now();
                    std::ptr::write_bytes(p as *mut u8, 1, SIZE);
                    let w = t0.elapsed();
                    let t1 = std::time::Instant::now();
                    let sum: u64 = std::slice::from_raw_parts(p as *const u64, SIZE / 8).iter().fold(0u64, |a, &v| a.wrapping_add(v));
                    let r = t1.elapsed();
                    std::hint::black_box(sum);
                    buf.Unmap(0, None);
                    row += &format!("  CPU write {:.2} GB/s, read {:.3} GB/s", SIZE as f64 / w.as_secs_f64() / 1e9, SIZE as f64 / r.as_secs_f64() / 1e9);
                }
                eprintln!("{row}");
            }
        }
    }
}
