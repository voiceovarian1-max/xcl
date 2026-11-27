//! Many small pieces of the title's memory into a window in one dispatch.
//!
//! Between two submits a title writes thousands of scattered bits of its memory (constants,
//! instances, argument buffers): a retail title's world changed ~2 000 runs of 64-byte chunks a
//! submit, ~240 bytes each. A CopyBufferRegion apiece is 0.33 µs in the driver however little it
//! copies – 25 ms of every frame there. Here the runs' bytes go into the staging ring one after
//! another as before, with a table of where each chunk belongs in front of them, and one compute
//! dispatch puts all of them in their places.

use std::ffi::c_void;
use std::mem::transmute_copy;
use std::sync::{Arc, Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;

/// bytes of a chunk: what a thread copies
pub(super) const CHUNK: usize = 64;
/// threads of a group, and the most chunks one dispatch takes (65 535 groups of them are the limit)
const GROUP: usize = 64;
pub(super) const MAX_CHUNKS: usize = 65_535 * GROUP;

/// (the destinations are within 1 GiB of the address bound: `upload_dirty` cuts the runs there)
const PROGRAM: &str = r#"
cbuffer C : register(b0) { uint chunks; };
// where each chunk goes, and the chunks, one after another
ByteAddressBuffer places : register(t0);
ByteAddressBuffer bytes : register(t1);
RWByteAddressBuffer window : register(u0);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    [branch] if (i < chunks) {
        uint to = places.Load(i * 4u);
        uint from = i * 64u;
        window.Store4(to, bytes.Load4(from));
        window.Store4(to + 16u, bytes.Load4(from + 16u));
        window.Store4(to + 32u, bytes.Load4(from + 32u));
        window.Store4(to + 48u, bytes.Load4(from + 48u));
    }
}
"#;

pub(super) struct Scatter {
    rs: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded
unsafe impl Send for Scatter {}
unsafe impl Sync for Scatter {}

/// `XCL_SPARSE_SCATTER=0`: every run goes up with a copy of its own.
pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SPARSE_SCATTER").map_or(true, |v| v.trim() != "0"))
}

/// The pass on `dev`, made on first use (None: it cannot be made there, logged once).
pub(super) unsafe fn of(dev: &ID3D12Device) -> Option<Arc<Scatter>> {
    static MADE: Mutex<Option<(usize, Option<Arc<Scatter>>)>> = Mutex::new(None);
    let key = dev.as_raw() as usize;
    let mut g = MADE.lock().unwrap();
    if let Some((d, s)) = g.as_ref().filter(|(d, _)| *d == key) {
        let _ = d;
        return s.clone();
    }
    // (its own work: the driver's answers as they are, see `dev_create_compute_pso`)
    let made = match super::title_textures::own_work(|| Scatter::new(dev)) {
        Ok(s) => Some(Arc::new(s)),
        Err(e) => {
            crate::xlog_warn!("sparse windows: small uploads go up with a copy each ({e})");
            None
        }
    };
    *g = Some((key, made.clone()));
    made
}

impl Scatter {
    unsafe fn new(dev: &ID3D12Device) -> Result<Scatter, String> {
        let cs = super::real::compile(PROGRAM, "cs_5_0");
        if cs.is_empty() {
            return Err("its program does not compile".into());
        }
        let root = |ty, anonymous| D3D12_ROOT_PARAMETER { ParameterType: ty, Anonymous: anonymous, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL };
        let view = |ty, k| root(ty, D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: k, RegisterSpace: 0 } });
        let params = [
            root(D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 1 } }),
            view(D3D12_ROOT_PARAMETER_TYPE_SRV, 0),
            view(D3D12_ROOT_PARAMETER_TYPE_SRV, 1),
            view(D3D12_ROOT_PARAMETER_TYPE_UAV, 0),
        ];
        let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: params.len() as u32, pParameters: params.as_ptr(), ..Default::default() };
        let mut blob: Option<ID3DBlob> = None;
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).map_err(|e| format!("root signature: {e}"))?;
        let blob = blob.ok_or("root signature")?;
        let rs: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())).map_err(|e| format!("CreateRootSignature: {e}"))?;
        let code = D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as *const c_void, BytecodeLength: cs.len() };
        let pso: ID3D12PipelineState = dev
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&rs), CS: code, ..Default::default() })
            .map_err(|e| format!("CreateComputePipelineState: {e}"))?;
        Ok(Scatter { rs, pso })
    }

    /// The bytes a dispatch of `chunks` chunks takes in the ring: the table, then the chunks (both
    /// from a multiple of `CHUNK`).
    pub fn bytes(chunks: usize) -> usize {
        Self::data_at(chunks) + chunks * CHUNK
    }

    /// Where the chunks begin behind their table.
    pub fn data_at(chunks: usize) -> usize {
        (chunks * 4).next_multiple_of(CHUNK)
    }

    /// Bind the pass on `list`.
    pub unsafe fn bind(&self, list: &ID3D12GraphicsCommandList) {
        list.SetComputeRootSignature(&self.rs);
        list.SetPipelineState(&self.pso);
    }

    /// `chunks` chunks from `ring_va` (the table, then the chunks) to their places from `to_va` on
    /// (bound: `bind`).
    pub unsafe fn dispatch(&self, list: &ID3D12GraphicsCommandList, chunks: usize, ring_va: u64, to_va: u64) {
        debug_assert!(chunks > 0 && chunks <= MAX_CHUNKS);
        list.SetComputeRoot32BitConstant(0, chunks as u32, 0);
        list.SetComputeRootShaderResourceView(1, ring_va);
        list.SetComputeRootShaderResourceView(2, ring_va + Self::data_at(chunks) as u64);
        list.SetComputeRootUnorderedAccessView(3, to_va);
        let timed = super::gpu_profile::begin_own(list.as_raw(), super::gpu_profile::OWN_UPLOAD, chunks as u32);
        list.Dispatch(chunks.div_ceil(GROUP) as u32, 1, 1);
        super::gpu_profile::end(list.as_raw(), timed);
    }
}

/// A piece of a window one dispatch fills: addresses within 1 GiB of `base`, and the runs in it.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Piece {
    pub window: usize,
    /// the address the piece's destinations count from
    pub base: usize,
    pub runs: std::ops::Range<usize>,
    pub chunks: usize,
}

/// Cut `runs` (address, bytes; whole chunks, in the order of their addresses) into pieces of one
/// window and `SPAN` bytes of it and of at most `most` chunks each. `window(a)`: the window `a` is
/// in, with its first address (None: the run is left out).
pub(super) fn pieces(runs: &[(usize, usize)], most: usize, window: impl Fn(usize) -> Option<(usize, usize)>) -> Vec<Piece> {
    /// (far under the 4 GiB a byte offset of a shader reaches)
    const SPAN: usize = 1 << 30;
    let mut out: Vec<Piece> = Vec::new();
    for (k, &(a, n)) in runs.iter().enumerate() {
        let Some((w, start)) = window(a) else { continue };
        let base = start + (a - start) / SPAN * SPAN;
        let chunks = n / CHUNK;
        match out.last_mut() {
            Some(p) if p.window == w && p.base == base && p.runs.end == k && p.chunks + chunks <= most => {
                p.runs.end = k + 1;
                p.chunks += chunks;
            }
            _ => out.push(Piece { window: w, base, runs: k..k + 1, chunks }),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_are_cut_by_window_by_span_and_by_size() {
        const W: usize = 4 << 30;
        let window = |a: usize| (a >= W && a < 4 * W).then(|| (a / W - 1, a / W * W));
        let runs = [
            (W, 64),
            (W + 4096, 192),
            // the last bytes of the first GiB, then the next one
            (W + (1 << 30) - 64, 64),
            (W + (1 << 30), 64),
            // in no window
            (5 * W, 64),
            // another window
            (2 * W + 128, 640),
            (2 * W + 4096, 64),
        ];
        let p = pieces(&runs, 1000, window);
        assert_eq!(
            p,
            vec![
                Piece { window: 0, base: W, runs: 0..3, chunks: 1 + 3 + 1 },
                Piece { window: 0, base: W + (1 << 30), runs: 3..4, chunks: 1 },
                Piece { window: 1, base: 2 * W, runs: 5..7, chunks: 10 + 1 },
            ]
        );
        // a run never falls in two pieces, and a piece holds `most` chunks at most
        let p = pieces(&runs[..3], 4, window);
        assert_eq!(p.iter().map(|p| (p.runs.clone(), p.chunks)).collect::<Vec<_>>(), vec![(0..2, 4), (2..3, 1)]);
        assert_eq!(pieces(&runs[..1], 4, |_| None), vec![]);
        assert_eq!((Scatter::data_at(1), Scatter::data_at(16), Scatter::data_at(17)), (64, 64, 128));
        assert_eq!(Scatter::bytes(17), 128 + 17 * 64);
    }
}
