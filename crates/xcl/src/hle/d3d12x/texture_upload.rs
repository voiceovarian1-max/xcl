//! Uploads of the title's textures: what of them changed, and only that.
//!
//! A title writes its textures piece by piece – a streamed one arrives 64 KiB at a time, a video
//! plane or a glyph cache changes in places – and every submit in between wants what is there.
//! Uploading the whole texture each time sent a retail title's textures up many times over. Here
//! the written pages are compared with what they held (`changed`), the blocks of the levels those
//! bytes are (`tiling::Chain::parts`) are detiled by the standing workers straight into a mapped
//! upload buffer, and one list copies them in. Lists and buffers are kept: making them waits for
//! the video memory manager, and releasing them asked every queue how far it is.

use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};

use crate::gpu::tiling::Rect;
use crate::hle::xg;

const PAGE: usize = 0x1000;
/// what a worker reads at a time
const READ: usize = 64 << 10;
/// uploads of fewer bytes stay on the calling thread
const SHARE_FROM: u64 = 192 << 10;

/// A texture to upload, or parts of it.
pub(super) struct Item {
    pub res: ID3D12Resource,
    /// where its bytes are
    pub src: usize,
    pub tex: Arc<xg::Texture>,
    /// the states it is in and is left in: one for every subresource, or one for each
    pub states: Vec<D3D12_RESOURCE_STATES>,
    /// the bytes of it that changed (from, to; sorted and disjoint); None: all of it
    pub changed: Option<Vec<(usize, usize)>>,
    /// where it is kept what its pages held
    pub known: Option<Known>,
    /// where its pages are read, for a texture in a GPU view whose pages the title mapped
    pub pages: Option<Arc<PageMap>>,
}

// ---------------------------------------------------------------------------------------
// what changed
// ---------------------------------------------------------------------------------------

/// A hash of `b` for "the same bytes as before" (four lanes of multiply-rotate over 8-byte words:
/// gigabytes a second; not for anything adversarial), never 0.
pub(super) fn content_hash(b: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut h = [K, K.rotate_left(17), K.rotate_left(31), K.rotate_left(47)];
    let mut chunks = b.chunks_exact(32);
    for c in &mut chunks {
        for (k, lane) in h.iter_mut().enumerate() {
            let v = u64::from_le_bytes(c[k * 8..k * 8 + 8].try_into().unwrap());
            *lane = (*lane ^ v).wrapping_mul(K).rotate_left(29);
        }
    }
    let mut tail = b.len() as u64;
    for (k, &x) in chunks.remainder().iter().enumerate() {
        tail = (tail ^ ((x as u64) << (8 * (k % 8)))).wrapping_mul(K).rotate_left(23);
    }
    h.iter().fold(tail, |a, &l| (a ^ l).wrapping_mul(K).rotate_left(27)).max(1)
}

/// What a texture's pages held when they were last looked at.
#[derive(Default)]
pub(super) struct Pages {
    /// the hash of each page's bytes that are the texture's (0: not known, the next write there
    /// goes up). Empty: what the PC texture holds is not known at all, its next upload is of all
    /// of it
    hashes: Vec<u64>,
    /// counts the times that was forgotten: a comparison going on then is not kept
    forgotten: u64,
    /// the last upload on each queue that may not have run yet
    last: Vec<Sent>,
    /// what goes up again, and the uploads it has to come behind (`sent`)
    again: Option<(Change, Vec<Sent>)>,
}

/// Where an upload went: the queue, the fence of that queue's uploads and the value behind it.
#[derive(Clone)]
pub(super) struct Sent {
    pub queue: usize,
    pub fence: ID3D12Fence,
    pub value: u64,
}

impl Sent {
    fn done(&self) -> bool {
        unsafe { super::real::fence_completed(&self.fence) >= self.value }
    }
}

pub(super) type Known = Arc<Mutex<Pages>>;

/// The pages `[src, src + size)` lies on.
fn pages_of(src: usize, size: usize) -> usize {
    ((src + size).div_ceil(PAGE) * PAGE - src / PAGE * PAGE) / PAGE
}

/// All of a texture is about to go up from `size` bytes at `src`: from here on writes there are
/// compared page by page.
pub(super) fn all_sent(known: &Known, src: usize, size: usize) {
    let mut k = known.lock().unwrap();
    k.hashes.clear();
    k.hashes.resize(pages_of(src, size), 0);
    k.again = None;
}

/// Nothing is known about what the PC texture holds: its next upload is of all of it.
pub(super) fn forget(known: &Known) {
    let mut k = known.lock().unwrap();
    k.hashes.clear();
    k.forgotten += 1;
    k.again = None;
}

/// What a texture's memory holds from here on is not taken for what the PC texture holds (an
/// upload going on keeps nothing it reads): its next write sends all of it. What is to go up
/// again stays.
pub(super) fn unknown(known: &Known) {
    let mut k = known.lock().unwrap();
    k.hashes.clear();
    k.forgotten += 1;
}

/// `parts` of a texture (from, to) are not what its PC texture holds (a mapping changed under
/// them): they go up again with its next upload, with no comparison (`take_again`).
pub(super) fn stale(known: &Known, parts: Vec<(usize, usize)>) {
    // (sorted and joined)
    let parts = join(Change::Parts(Vec::new()), Change::Parts(parts));
    let mut k = known.lock().unwrap();
    k.again = Some(match k.again.take() {
        Some((c, behind)) => (join(c, parts), behind),
        None => (parts, Vec::new()),
    });
}

/// Whether nothing is known about what the PC texture holds (`forget`).
#[cfg(test)]
pub(super) fn knows_nothing(known: &Known) -> bool {
    known.lock().unwrap().hashes.is_empty()
}

/// `changed` of a texture (None: all of it) went up on `to`. An older upload of the texture on
/// another queue that has not run yet (and that `to`'s queue has not waited for: `waited`) can land
/// over it: it goes up again, behind that one (`take_again`). True when there is such a thing.
pub(super) fn sent(known: &Known, changed: Option<&[(usize, usize)]>, to: &Sent, waited: usize) -> bool {
    let mut k = known.lock().unwrap();
    k.last.retain(|s| !s.done());
    let behind: Vec<Sent> = k.last.iter().filter(|s| s.queue != to.queue && s.queue != waited).cloned().collect();
    k.last.retain(|s| s.queue != to.queue);
    k.last.push(to.clone());
    if !behind.is_empty() {
        let mut change = changed.map_or(Change::All, |c| Change::Parts(c.to_vec()));
        let mut all = behind;
        if let Some((c, b)) = k.again.take() {
            change = join(c, change);
            // (the newest of each queue)
            all.extend(b.into_iter().filter(|o| !all.iter().any(|s| s.queue == o.queue)).collect::<Vec<_>>());
        }
        k.again = Some((change, all));
    }
    k.again.is_some()
}

/// Uploads of `old` that may not have run yet can land over those of `new`, a texture over the same
/// video memory.
pub(super) fn inherit(new: &Known, old: &Known) {
    let unfinished: Vec<Sent> = old.lock().unwrap().last.iter().filter(|s| !s.done()).cloned().collect();
    let mut k = new.lock().unwrap();
    for s in unfinished {
        match k.last.iter_mut().find(|o| o.queue == s.queue) {
            Some(o) => {
                if s.value > o.value {
                    *o = s;
                }
            }
            None => k.last.push(s),
        }
    }
}

/// What a texture has to send again.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Again {
    Nothing,
    /// an upload it has to come behind has not run yet, and is on another queue
    NotYet,
    Now(Change),
}

/// What of a texture goes up again with an upload on `queue`, which has waited for `waited`'s.
pub(super) fn take_again(known: &Known, queue: usize, waited: usize) -> Again {
    let mut k = known.lock().unwrap();
    match k.again.take() {
        None => Again::Nothing,
        Some((c, behind)) if behind.iter().all(|s| s.queue == queue || s.queue == waited || s.done()) => Again::Now(c),
        later => {
            k.again = later;
            Again::NotYet
        }
    }
}

/// Both of two changes.
pub(super) fn join(a: Change, b: Change) -> Change {
    match (a, b) {
        (Change::All, _) | (_, Change::All) => Change::All,
        (Change::None, c) | (c, Change::None) => c,
        (Change::Parts(mut a), Change::Parts(b)) => {
            a.extend(b);
            a.sort_unstable();
            let mut out: Vec<(usize, usize)> = Vec::with_capacity(a.len());
            for (from, to) in a {
                match out.last_mut() {
                    Some(l) if l.1 >= from => l.1 = l.1.max(to),
                    _ => out.push((from, to)),
                }
            }
            Change::Parts(out)
        }
    }
}

/// A texture with written pages, to be compared.
pub(super) struct Written<'a> {
    pub src: usize,
    pub size: usize,
    /// its written pages (sorted addresses)
    pub dirty: &'a [usize],
    pub known: &'a Known,
    /// where its pages are read (`Item::pages`)
    pub pages: Option<&'a PageMap>,
}

/// What became of a texture's written pages.
#[derive(Debug, PartialEq, Eq, Clone)]
pub(super) enum Change {
    /// they hold what they held
    None,
    /// what its PC texture holds is not known
    All,
    /// these bytes of it (from, to; sorted and disjoint) are not what they were
    Parts(Vec<(usize, usize)>),
}

/// The written pages of `textures` against what they held; what they hold now is remembered.
/// Also the bytes compared. `trust` false: a written page has changed (nothing is compared).
pub(super) unsafe fn changed(textures: &[Written], trust: bool) -> (Vec<Change>, usize) {
    struct Page {
        texture: usize,
        at: usize,
        len: usize,
        /// its hash among the texture's
        slot: *mut u64,
        differs: AtomicBool,
    }
    // SAFETY: each page's slot is its own, and the hashes outlive the workers' use of them
    unsafe impl Sync for Page {}
    let mut out: Vec<Change> = textures.iter().map(|_| Change::None).collect();
    // (taken out while they are worked on, put back below)
    let mut taken: Vec<(Vec<u64>, u64)> = textures
        .iter()
        .map(|t| {
            let mut k = t.known.lock().unwrap();
            (std::mem::take(&mut k.hashes), k.forgotten)
        })
        .collect();
    let mut pages: Vec<Page> = Vec::new();
    for (k, (t, (h, _))) in textures.iter().zip(taken.iter_mut()).enumerate() {
        let (first, end) = (t.src / PAGE * PAGE, t.src + t.size);
        if h.len() != pages_of(t.src, t.size) {
            h.clear();
            if t.dirty.iter().any(|&p| p + PAGE > t.src && p < end) {
                out[k] = Change::All;
            }
            continue;
        }
        for &p in t.dirty.iter().filter(|&&p| p >= first && p < end) {
            let (at, to) = (p.max(t.src), (p + PAGE).min(end));
            pages.push(Page { texture: k, at, len: to - at, slot: h.as_mut_ptr().add((p - first) / PAGE), differs: AtomicBool::new(!trust) });
        }
    }
    let bytes = pages.iter().map(|p| p.len).sum();
    if trust {
        // neighbours are read together
        let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
        for (k, p) in pages.iter().enumerate() {
            match groups.last_mut() {
                Some(g) if pages[k - 1].at + pages[k - 1].len == p.at && p.at + p.len - pages[g.start].at <= READ && pages[k - 1].texture == p.texture => g.end = k + 1,
                _ => groups.push(k..k + 1),
            }
        }
        let look = |g: usize| {
            let group = &pages[groups[g].clone()];
            let (first, last) = (&group[0], &group[group.len() - 1]);
            read(textures[first.texture].pages, first.at, last.at + last.len - first.at, |b| {
                for p in group {
                    let h = content_hash(&b[p.at - first.at..][..p.len]);
                    // SAFETY: see `Page`
                    let slot = unsafe { &mut *p.slot };
                    if *slot != h {
                        *slot = h;
                        p.differs.store(true, Ordering::Relaxed);
                    }
                }
            });
        };
        if bytes as u64 >= SHARE_FROM {
            super::worker_pool::run(groups.len(), &look);
        } else {
            (0..groups.len()).for_each(look);
        }
    }
    for p in pages.iter().filter(|p| p.differs.load(Ordering::Relaxed)) {
        let from = p.at - textures[p.texture].src;
        match &mut out[p.texture] {
            Change::Parts(v) => match v.last_mut() {
                Some(l) if l.1 >= from => l.1 = l.1.max(from + p.len),
                _ => v.push((from, from + p.len)),
            },
            c => *c = Change::Parts(vec![(from, from + p.len)]),
        }
    }
    drop(pages);
    for (t, (h, forgotten)) in textures.iter().zip(taken) {
        let mut k = t.known.lock().unwrap();
        if k.forgotten == forgotten {
            k.hashes = h;
        }
    }
    (out, bytes)
}

// ---------------------------------------------------------------------------------------
// reading the title's memory
// ---------------------------------------------------------------------------------------

thread_local! {
    static SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Where the 64 KiB pages of a texture in a GPU view are read, resolved once for an upload
/// (`kernelx::mirrors::resolve`): a read takes no lock and asks Windows nothing but the bytes.
pub(super) struct PageMap {
    /// the 64 KiB page of the texture's first byte
    first: usize,
    /// for each page from `first` on: where the page mapped there is kept, 0 for nothing, or (bit
    /// 0 set) the address read for a page the title never mapped there
    at: Box<[usize]>,
}

impl PageMap {
    const PAGE: usize = crate::hle::kernelx::mirrors::PAGE;

    /// The table of a texture at `va` from `resolve`'s addresses.
    pub fn new(va: usize, at: Vec<usize>) -> Self {
        PageMap { first: va & !(Self::PAGE - 1), at: at.into_boxed_slice() }
    }

    /// The bytes of the texture's address `at` on into `to`, a 64 KiB page at a time: zeros for a
    /// page that shows nothing (or lies past the table), where it is kept for a mapped page (one
    /// read: a physical page is committed whole or not at all), as any title memory otherwise.
    pub unsafe fn copy(&self, at: usize, to: &mut [u8]) {
        let mut done = 0;
        while done < to.len() {
            let a = at + done;
            let (k, off) = (a.wrapping_sub(self.first) / Self::PAGE, a & (Self::PAGE - 1));
            let n = (Self::PAGE - off).min(to.len() - done);
            let piece = &mut to[done..done + n];
            match self.at.get(k).copied().filter(|_| a >= self.first).unwrap_or(0) {
                0 => piece.fill(0),
                p if p & 1 != 0 => super::title_textures::copy_readable((p & !1) + off, piece),
                p => super::title_textures::copy_or_zeros(p + off, piece),
            }
            done += n;
        }
    }
}

/// `f` with the `len` bytes of the title's memory at `at`: where they are in the window that stays
/// committed, else a copy of them (the title may decommit a page meanwhile on another thread), with
/// zeros for pages that cannot be read. With `pages`, they are read where its table says.
unsafe fn read<R>(pages: Option<&PageMap>, at: usize, len: usize, f: impl FnOnce(&[u8]) -> R) -> R {
    if let Some(m) = pages {
        if len > READ {
            let mut v = vec![0u8; len];
            m.copy(at, &mut v);
            return f(&v);
        }
        return SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            if s.len() < len {
                s.resize(READ, 0);
            }
            m.copy(at, &mut s[..len]);
            f(&s[..len])
        });
    }
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(at, len)) {
        return f(std::slice::from_raw_parts(at as *const u8, len));
    }
    if len > READ {
        return f(&super::title_textures::read_committed(at, len));
    }
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        if s.len() < len {
            s.resize(READ.max(len), 0);
        }
        super::title_textures::copy_readable(at, &mut s[..len]);
        f(&s[..len])
    })
}

// ---------------------------------------------------------------------------------------
// the plan of an upload
// ---------------------------------------------------------------------------------------

/// Part of one subresource in an upload buffer.
struct Piece {
    item: usize,
    sub: u32,
    /// array slice, level and the elements of it (3D: the level, all of its depth)
    slice: usize,
    level: usize,
    rect: Rect,
    /// where it goes in the subresource, in texels
    to: (u32, u32),
    /// where it lies in the upload buffer
    at: u64,
    footprint: D3D12_SUBRESOURCE_FOOTPRINT,
}

/// The pieces of item `k`, placed in the upload buffer from `at` on.
unsafe fn plan_item(dev: &ID3D12Device, k: usize, it: &Item, at: &mut u64, out: &mut Vec<Piece>) -> Result<(), String> {
    let s = it.tex.texels().ok_or("a texture without texels")?;
    let c = &s.chain;
    let (mips, array, volume) = (c.levels.len(), s.array as usize, s.depth > 0);
    let rd = it.res.GetDesc();
    let (n, layers) = if volume { (mips, s.depth as usize) } else { (mips * array, array) };
    if rd.MipLevels as usize != mips || rd.DepthOrArraySize as usize != layers || (rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) != volume {
        return Err(format!("{mips} levels of {layers} slices for a resource of {} of {}", rd.MipLevels, rd.DepthOrArraySize));
    }
    if it.states.len() != 1 && it.states.len() != n {
        return Err(format!("{} subresource states for {n} subresources", it.states.len()));
    }
    let bpe = s.bpe as usize;
    let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    if it.changed.is_none() || volume {
        // all of it: the levels have to be the resource's (a part of one is taken to be, after it)
        let (mut fps, mut rows, mut row_bytes) = (vec![fp; n], vec![0u32; n], vec![0u64; n]);
        dev.GetCopyableFootprints(&rd, 0, n as u32, 0, Some(fps.as_mut_ptr()), Some(rows.as_mut_ptr()), Some(row_bytes.as_mut_ptr()), None);
        for i in 0..n {
            let l = &c.levels[i % mips];
            let depth = if volume { l.depth } else { 1 };
            if l.height != rows[i] as usize || l.width * bpe != row_bytes[i] as usize || depth != fps[i].Footprint.Depth as usize {
                return Err(format!("texture subresource {i} is {}x{}x{depth} B, resource expects {}x{}x{} B", l.width * bpe, l.height, row_bytes[i], rows[i], fps[i].Footprint.Depth));
            }
        }
        fp = fps[0];
    } else {
        dev.GetCopyableFootprints(&rd, 0, 1, 0, Some(&mut fp), None, None, None);
    }
    let format: DXGI_FORMAT = fp.Footprint.Format;
    let mut push = |slice: usize, level: usize, rect: Rect, depth: usize| {
        let (w, h) = (rect.x1 - rect.x0, rect.y1 - rect.y0);
        let pitch = (w * bpe).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as usize);
        *at = at.next_multiple_of(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as u64);
        out.push(Piece {
            item: k,
            sub: (slice * mips + level) as u32,
            slice,
            level,
            rect,
            to: ((rect.x0 as u32) * s.side, (rect.y0 as u32) * s.side),
            at: *at,
            footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: format, Width: w as u32 * s.side, Height: h as u32 * s.side, Depth: depth as u32, RowPitch: pitch as u32 },
        });
        *at += (pitch * h * depth) as u64;
    };
    if volume {
        for (i, l) in c.levels.iter().enumerate() {
            push(0, i, Rect { x0: 0, y0: 0, x1: l.width, y1: l.height }, l.depth);
        }
        super::lut3d::wrote(it.res.as_raw() as usize, false, || {
            format!(
                "uploaded from the title's memory at {:#x}: {} bytes ({} tiled, swizzle mode {}), all {} levels and {} depth slices{}",
                it.src,
                s.size(),
                if c.layout.block_d > 1 { "thick" } else { "thin" },
                s.mode,
                mips,
                s.depth,
                if it.changed.is_some() { " (pages of it were written)" } else { "" }
            )
        });
        return Ok(());
    }
    let whole = [(0, c.slice_bytes * array)];
    // cut at the slices
    let mut slice_ranges: Vec<(usize, usize)> = Vec::new();
    let mut slice = 0;
    for &(from, to) in it.changed.as_deref().unwrap_or(&whole) {
        let (mut from, to) = (from, to.min(c.slice_bytes * array));
        while from < to {
            let of = from / c.slice_bytes;
            if of != slice && !slice_ranges.is_empty() {
                c.parts(&slice_ranges).into_iter().for_each(|(i, r)| push(slice, i, r, 1));
                slice_ranges.clear();
            }
            slice = of;
            let end = to.min((of + 1) * c.slice_bytes);
            slice_ranges.push((from - of * c.slice_bytes, end - of * c.slice_bytes));
            from = end;
        }
    }
    if !slice_ranges.is_empty() {
        c.parts(&slice_ranges).into_iter().for_each(|(i, r)| push(slice, i, r, 1));
    }
    Ok(())
}

/// What a worker does at a time.
enum Work {
    /// blocks of one row of them
    Blocks { piece: usize, by: usize, bx: std::ops::Range<usize> },
    /// the levels in a slice's mip tail: one block
    Tail { pieces: std::ops::Range<usize> },
    /// rows of a linear level (of the piece)
    Rows { piece: usize, y: std::ops::Range<usize> },
    /// the levels of a 3D texture
    Volume { pieces: std::ops::Range<usize> },
}

fn work_of(items: &[Item], pieces: &[Piece]) -> Vec<Work> {
    let mut out = Vec::new();
    let mut k = 0;
    while k < pieces.len() {
        let p = &pieces[k];
        // (planned: it has texels)
        let Some(s) = items[p.item].tex.texels() else { break };
        let (c, l) = (&s.chain, &s.chain.levels[p.level]);
        let same = |n: usize, f: &dyn Fn(&Piece) -> bool| pieces[n..].iter().take_while(|q| q.item == p.item && q.slice == p.slice && f(q)).count();
        if s.depth > 0 {
            let n = same(k, &|_| true);
            out.push(Work::Volume { pieces: k..k + n });
            k += n;
        } else if l.tail.is_some() {
            let n = same(k, &|q| c.levels[q.level].tail.is_some());
            out.push(Work::Tail { pieces: k..k + n });
            k += n;
        } else if is_linear(c, p.level) {
            let rows = (READ / (l.pitch << c.layout.elem_log2).max(1)).max(1);
            out.extend((p.rect.y0..p.rect.y1).step_by(rows).map(|y| Work::Rows { piece: k, y: y..(y + rows).min(p.rect.y1) }));
            k += 1;
        } else {
            let (rows, columns) = c.blocks_of(p.level, p.rect);
            let n = (READ / c.layout.block_bytes).max(1);
            for by in rows {
                out.extend(columns.clone().step_by(n).map(|bx| Work::Blocks { piece: k, by, bx: bx..(bx + n).min(columns.end) }));
            }
            k += 1;
        }
    }
    out
}

fn is_linear(c: &crate::gpu::tiling::Chain, level: usize) -> bool {
    c.layout.mode == 0 && c.layout.base_xor == 0 && c.levels[level].origin == (0, 0)
}

/// The page hashes of an item while its bytes are read.
struct Hashes<'a> {
    src: usize,
    end: usize,
    pages: &'a [AtomicU64],
}

impl Hashes<'_> {
    /// `b` was read at `from` and is about to go up: what is known of its pages is that (a page read
    /// in part is not known: its next write goes up). Neighbours of a changed page are read with it,
    /// and a title may write on between the comparison and here; what is kept is what the PC texture
    /// gets, or older than that (a write from here on is compared with it at the next look).
    fn note(&self, from: usize, b: &[u8]) {
        let first = self.src / PAGE * PAGE;
        let mut p = from / PAGE * PAGE;
        while p < from + b.len() {
            let (a, z) = (p.max(self.src), (p + PAGE).min(self.end));
            if let Some(slot) = self.pages.get((p - first) / PAGE).filter(|_| a < z) {
                let whole = a >= from && z <= from + b.len();
                slot.store(if whole { content_hash(&b[a - from..z - from]) } else { 0 }, Ordering::Relaxed);
            }
            p += PAGE;
        }
    }
}

/// The texels of `pieces` into the upload buffer mapped at `ptr`, with what is kept of the items'
/// pages brought to what was read.
unsafe fn fill(items: &[Item], pieces: &[Piece], ptr: *mut u8, bytes: u64) {
    // (taken out while they are worked on)
    let taken: Vec<Option<(Vec<AtomicU64>, u64)>> = items
        .iter()
        .map(|it| {
            let mut k = it.known.as_ref()?.lock().unwrap();
            if k.hashes.len() != pages_of(it.src, it.tex.size() as usize) {
                return None;
            }
            Some((std::mem::take(&mut k.hashes).into_iter().map(AtomicU64::new).collect(), k.forgotten))
        })
        .collect();
    fill_hashed(items, pieces, ptr, bytes, &taken);
    for (it, t) in items.iter().zip(taken) {
        if let (Some(known), Some((h, forgotten))) = (&it.known, t) {
            let mut k = known.lock().unwrap();
            if k.forgotten == forgotten {
                k.hashes = h.into_iter().map(AtomicU64::into_inner).collect();
            }
        }
    }
}

unsafe fn fill_hashed(items: &[Item], pieces: &[Piece], ptr: *mut u8, bytes: u64, hashes: &[Option<(Vec<AtomicU64>, u64)>]) {
    struct Mapped(*mut u8);
    // SAFETY: the workers write disjoint parts of the buffer
    unsafe impl Sync for Mapped {}
    let to = Mapped(ptr);
    let work = work_of(items, pieces);
    let note = |item: usize, from: usize, b: &[u8]| {
        if let Some((pages, _)) = &hashes[item] {
            let it = &items[item];
            Hashes { src: it.src, end: it.src + it.tex.size() as usize, pages }.note(from, b);
        }
    };
    let one = |k: usize| {
        let to = &to;
        // SAFETY: the pieces were placed inside the buffer; the title's memory is only read
        unsafe {
            match &work[k] {
                Work::Blocks { piece, by, bx } => {
                    let p = &pieces[*piece];
                    let (it, c) = (&items[p.item], &items[p.item].tex.texels().unwrap().chain);
                    let from = it.src + p.slice * c.slice_bytes + c.block_at(p.level, bx.start, *by);
                    read(it.pages.as_deref(), from, bx.len() * c.layout.block_bytes, |b| {
                        note(p.item, from, b);
                        c.detile_blocks(b, p.level, p.slice, *by, bx.clone(), p.rect, to.0.add(p.at as usize), p.footprint.RowPitch as usize);
                    });
                }
                Work::Tail { pieces: range } => {
                    let first = &pieces[range.start];
                    let (it, c) = (&items[first.item], &items[first.item].tex.texels().unwrap().chain);
                    let from = it.src + first.slice * c.slice_bytes;
                    read(it.pages.as_deref(), from, c.layout.block_bytes, |b| {
                        note(first.item, from, b);
                        for p in &pieces[range.clone()] {
                            c.detile_blocks(b, p.level, p.slice, 0, 0..1, p.rect, to.0.add(p.at as usize), p.footprint.RowPitch as usize);
                        }
                    });
                }
                Work::Rows { piece, y } => {
                    let p = &pieces[*piece];
                    let (it, c) = (&items[p.item], &items[p.item].tex.texels().unwrap().chain);
                    let (l, e) = (&c.levels[p.level], c.layout.elem_log2);
                    let (stride, row) = (l.pitch << e, (p.rect.x1 - p.rect.x0) << e);
                    let from = it.src + p.slice * c.slice_bytes + l.offset + y.start * stride + (p.rect.x0 << e);
                    read(it.pages.as_deref(), from, y.len() * stride - (p.rect.x0 << e), |b| {
                        note(p.item, from, b);
                        for (n, y) in y.clone().enumerate() {
                            let dst = to.0.add(p.at as usize + (y - p.rect.y0) * p.footprint.RowPitch as usize);
                            std::ptr::copy_nonoverlapping(b[n * stride..][..row].as_ptr(), dst, row);
                        }
                    });
                }
                Work::Volume { pieces: range } => {
                    let first = &pieces[range.start];
                    let it = &items[first.item];
                    let s = it.tex.texels().unwrap();
                    read(it.pages.as_deref(), it.src, s.size() as usize, |b| {
                        note(first.item, it.src, b);
                        for p in &pieces[range.clone()] {
                            let (l, row) = (&s.chain.levels[p.level], p.rect.x1 * s.bpe as usize);
                            let data = s.chain.detile_volume(b, p.level);
                            for (n, r) in data.chunks_exact(row).enumerate().take(l.height * l.depth) {
                                std::ptr::copy_nonoverlapping(r.as_ptr(), to.0.add(p.at as usize + n * p.footprint.RowPitch as usize), row);
                            }
                        }
                    });
                }
            }
        }
    };
    if bytes >= SHARE_FROM {
        super::worker_pool::run(work.len(), &one);
    } else {
        (0..work.len()).for_each(one);
    }
}

fn transition(r: &ID3D12Resource, sub: u32, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: the list keeps no reference; the item holds one until it was submitted
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: sub,
                StateBefore: from,
                StateAfter: to,
            }),
        },
    }
}

/// Record the copies of `pieces` from `buf`, and the transitions of the subresources they touch
/// (of no others: a title's work may be reading those).
unsafe fn record(list: &ID3D12GraphicsCommandList, items: &[Item], pieces: &[Piece], buf: &ID3D12Resource) {
    const COPY: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_COPY_DEST;
    let (mut there, mut back) = (Vec::new(), Vec::new());
    for of_item in pieces.chunk_by(|a, b| a.item == b.item) {
        let it = &items[of_item[0].item];
        let mut subs: Vec<u32> = of_item.iter().map(|p| p.sub).collect();
        subs.sort_unstable();
        subs.dedup();
        let n = it.tex.texels().map_or(0, |s| if s.depth > 0 { s.chain.levels.len() } else { s.chain.levels.len() * s.array as usize });
        if it.states.len() == 1 && subs.len() == n {
            subs = vec![D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES];
        }
        for sub in subs {
            let state = if it.states.len() == 1 { it.states[0] } else { it.states[sub as usize] };
            if state != COPY {
                there.push(transition(&it.res, sub, state, COPY));
                back.push(transition(&it.res, sub, COPY, state));
            }
        }
    }
    if !there.is_empty() {
        list.ResourceBarrier(&there);
    }
    for p in pieces {
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&items[p.item].res),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: p.sub },
        };
        let placed = D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: p.at, Footprint: p.footprint };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(buf), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: placed } };
        list.CopyTextureRegion(&dst, p.to.0, p.to.1, 0, &src, None);
    }
    if !back.is_empty() {
        list.ResourceBarrier(&back);
    }
}

// ---------------------------------------------------------------------------------------
// lists and buffers
// ---------------------------------------------------------------------------------------

/// A list with its allocator and a mapped upload buffer: one batch at a time uses them.
pub(super) struct Job {
    dev: usize,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    buf: ID3D12Resource,
    ptr: *mut u8,
    pub bytes: u64,
}

// D3D12 objects are free-threaded; the mapped pointer is only used by the holder of the job
unsafe impl Send for Job {}

/// jobs the GPU is done with
static FREE: Mutex<Vec<Job>> = Mutex::new(Vec::new());
/// the smallest upload buffer made, and the most bytes of them kept
const BUFFER_MIN: u64 = 1 << 20;

/// The size of the buffer made for `bytes`: up to 64 MiB a power of two (few sizes, so one fits
/// again), then the next 16 MiB.
fn capacity(bytes: u64) -> Option<u64> {
    const LARGE: u64 = 64 << 20;
    const STEP: u64 = 16 << 20;
    if bytes <= LARGE {
        Some(bytes.max(BUFFER_MIN).next_power_of_two())
    } else {
        bytes.checked_next_multiple_of(STEP)
    }
}

const FREE_BYTES: u64 = 512 << 20;

/// A job whose buffer takes `bytes` bytes.
unsafe fn take(dev: &ID3D12Device, bytes: u64) -> Result<Job, String> {
    let key = dev.as_raw() as usize;
    {
        let mut g = FREE.lock().unwrap();
        let best = g.iter().enumerate().filter(|(_, j)| j.dev == key && j.bytes >= bytes).min_by_key(|(_, j)| j.bytes).map(|(i, _)| i);
        if let Some(i) = best {
            return Ok(g.swap_remove(i));
        }
    }
    let e = |x: windows::core::Error| format!("texture upload: {x}");
    let cap = capacity(bytes).ok_or("texture upload buffer size overflow")?;
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
    list.Close().map_err(e)?;
    let _ = list.SetName(windows::core::w!("xcl texture upload list"));
    let mut buf: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: cap,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        },
        D3D12_RESOURCE_STATE_GENERIC_READ,
        None,
        &mut buf,
    )
    .map_err(e)?;
    let buf = buf.ok_or("texture upload buffer")?;
    let mut p: *mut c_void = std::ptr::null_mut();
    buf.Map(0, None, Some(&mut p)).map_err(e)?;
    Ok(Job { dev: key, alloc, list, buf, ptr: p as *mut u8, bytes: cap })
}

/// The GPU is done with `job`.
pub(super) fn give_back(job: Job) {
    let mut g = FREE.lock().unwrap();
    g.push(job);
    // (the biggest goes first: one burst does not keep its buffers for good)
    while g.iter().map(|j| j.bytes).sum::<u64>() > FREE_BYTES {
        let Some(i) = g.iter().enumerate().max_by_key(|(_, j)| j.bytes).map(|(i, _)| i) else { break };
        g.swap_remove(i);
    }
}

/// Upload `items` on `queue` (a DIRECT one): the ones that could not be (with why), and the job to
/// give back once `queue` is past here (None: nothing was submitted).
pub(super) unsafe fn submit(dev: &ID3D12Device, queue: &ID3D12CommandQueue, items: &[Item]) -> Result<(Vec<(usize, String)>, Option<Job>, u64), String> {
    let (mut failed, mut pieces, mut bytes) = (Vec::new(), Vec::new(), 0u64);
    for (k, it) in items.iter().enumerate() {
        let (n, at) = (pieces.len(), bytes);
        if let Err(e) = plan_item(dev, k, it, &mut bytes, &mut pieces) {
            pieces.truncate(n);
            bytes = at;
            failed.push((k, e));
        }
    }
    if pieces.is_empty() {
        return Ok((failed, None, 0));
    }
    let job = take(dev, bytes)?;
    fill(items, &pieces, job.ptr, bytes);
    let e = |x: windows::core::Error| format!("texture upload: {x}");
    let recorded = (|| {
        job.alloc.Reset().map_err(e)?;
        job.list.Reset(&job.alloc, None).map_err(e)?;
        record(&job.list, items, &pieces, &job.buf);
        job.list.Close().map_err(e)
    })();
    if let Err(e) = recorded {
        // (nothing runs it: the job goes, its list may be in any state)
        return Err(e);
    }
    super::sparse::pc_execute(queue, &job.list);
    Ok((failed, Some(job), bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The written pages of a texture are compared one by one: a page never compared has changed,
    /// one that holds what it held has not, the bytes next to the texture on its first and last
    /// page are not the texture's, and a texture nothing is known of changed as a whole.
    #[test]
    fn written_pages_are_compared_with_what_they_held() {
        let mut mem = vec![0u8; 12 * PAGE];
        let base = (mem.as_mut_ptr() as usize).next_multiple_of(PAGE);
        // a texture from the middle of the second page to the middle of the ninth
        let (src, size) = (base + PAGE + 0x800, 7 * PAGE);
        let known: Known = Default::default();
        let dirty: Vec<usize> = (0..11).map(|k| base + k * PAGE).collect();
        let look = |dirty: &[usize]| unsafe { changed(&[Written { src, size, dirty, known: &known, pages: None }], true) };
        assert_eq!(look(&[]).0, vec![Change::None]);
        assert_eq!(look(&dirty).0, vec![Change::All], "nothing is known of it");
        assert_eq!(look(&dirty[9..]).0, vec![Change::None], "pages it does not lie on");
        all_sent(&known, src, size);
        assert_eq!(known.lock().unwrap().hashes.len(), 8);
        // each page is known from its first look on
        assert_eq!(look(&dirty[3..5]).0, vec![Change::Parts(vec![(PAGE + 0x800, 3 * PAGE + 0x800)])]);
        assert_eq!(look(&dirty[3..5]).0, vec![Change::None]);
        assert_eq!(look(&dirty), (vec![Change::Parts(vec![(0, PAGE + 0x800), (3 * PAGE + 0x800, size)])], size), "the others, up to the texture's ends");
        assert_eq!(look(&dirty).0, vec![Change::None]);
        // what lies next to the texture on its pages is another's
        let write = |at: usize, v: u8| unsafe { (at as *mut u8).write_volatile(v) };
        write(src - 1, 1);
        write(src + size, 1);
        assert_eq!(look(&dirty).0, vec![Change::None]);
        write(src, 1);
        write(src + 4 * PAGE, 1);
        write(src + size - 1, 1);
        // (only what was written is looked at)
        assert_eq!(look(&[base + 5 * PAGE]).0, vec![Change::Parts(vec![(3 * PAGE + 0x800, 4 * PAGE + 0x800)])]);
        assert_eq!(look(&dirty).0, vec![Change::Parts(vec![(0, 0x800), (6 * PAGE + 0x800, size)])]);
        // (without trust every written page has)
        assert_eq!(unsafe { changed(&[Written { src, size, dirty: &dirty[2..3], known: &known, pages: None }], false) }.0, vec![Change::Parts(vec![(0x800, PAGE + 0x800)])]);
        forget(&known);
        assert_eq!(look(&dirty[2..3]).0, vec![Change::All]);
        // nothing taken for what the PC texture holds: all of it at the next write, and what was to
        // go up again stays
        all_sent(&known, src, size);
        assert_eq!(look(&dirty[3..5]).0, vec![Change::Parts(vec![(PAGE + 0x800, 3 * PAGE + 0x800)])]);
        known.lock().unwrap().again = Some((Change::Parts(vec![(0, 64)]), Vec::new()));
        unknown(&known);
        assert!(known.lock().unwrap().again.is_some());
        assert_eq!(look(&dirty[3..5]).0, vec![Change::All]);
        known.lock().unwrap().again = None;
        // several textures at once, many pages: the same as one by one
        let big = vec![7u8; 600 * PAGE];
        let at = (big.as_ptr() as usize).next_multiple_of(PAGE);
        let (a, b): (Known, Known) = (Default::default(), Default::default());
        all_sent(&a, at, 300 * PAGE);
        all_sent(&b, at + 300 * PAGE, 290 * PAGE);
        let pages: Vec<usize> = (0..590).map(|k| at + k * PAGE).collect();
        let both = || unsafe { changed(&[Written { src: at, size: 300 * PAGE, dirty: &pages, known: &a, pages: None }, Written { src: at + 300 * PAGE, size: 290 * PAGE, dirty: &pages, known: &b, pages: None }], true).0 };
        assert_eq!(both(), vec![Change::Parts(vec![(0, 300 * PAGE)]), Change::Parts(vec![(0, 290 * PAGE)])]);
        assert_eq!(both(), vec![Change::None, Change::None]);
    }

    /// The default adapter's device, else WARP's (`XCL_ADAPTER=warp`: WARP's); None, the test
    /// skipped, where there is neither (a DXGI that refuses the factory, #2238).
    unsafe fn test_device() -> Option<ID3D12Device> {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        let warp_only = crate::gpu::adapter::Selector::from_env_or("auto") == crate::gpu::adapter::Selector::Warp;
        let mut dev: Option<ID3D12Device> = None;
        if warp_only || D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            dev = None;
            if let Ok(warp) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)).and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>()) {
                let _ = D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev);
            }
        }
        if dev.is_none() {
            crate::gpu::recomp::runner::skip_test("no D3D12 device");
        }
        dev
    }

    /// Textures of every kind: (dimension, width, height, slices or depth, levels, format, layout).
    const KINDS: [(u32, u64, u32, u16, u16, u32, u32); 10] = [
        (3, 64, 64, 2, 3, 28, 0x105),
        (3, 200, 72, 1, 5, 28, 0x105),
        (4, 40, 24, 6, 3, 28, 0x105),
        (3, 256, 256, 1, 9, 98, 0x10A),
        (3, 4, 4, 1, 1, 28, 0x105),
        (3, 1030, 516, 3, 8, 71, 0x109),
        (3, 777, 555, 2, 1, 61, 1),
        (3, 512, 300, 6, 10, 10, 0x11B),
        (2, 300, 1, 1, 1, 28, 1),
        (3, 2048, 1024, 1, 12, 98, 0x11B),
    ];

    struct Made {
        res: ID3D12Resource,
        tex: Arc<xg::Texture>,
    }

    unsafe fn make(dev: &ID3D12Device, k: (u32, u64, u32, u16, u16, u32, u32)) -> Made {
        let console = super::super::title_textures::tests::console_desc(k.0, k.1, k.2, k.3, k.4, k.5, k.6);
        let pc: D3D12_RESOURCE_DESC = std::ptr::read_unaligned(super::super::device_x::pc_resource_desc(&console).as_ptr() as *const _);
        let mut res: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &pc, D3D12_RESOURCE_STATE_COMMON, None, &mut res)
            .unwrap_or_else(|e| panic!("{k:?}: {e}"));
        Made { res: res.unwrap(), tex: Arc::new(xg::Texture::new(&console).unwrap_or_else(|e| panic!("{k:?}: {e:#x}"))) }
    }

    fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..n.div_ceil(8))
            .flat_map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x.to_le_bytes()
            })
            .take(n)
            .collect()
    }

    /// Some pages of a texture, as ranges of its bytes.
    fn some_ranges(size: usize, seed: usize) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = Vec::new();
        let mut at = seed * 37 % size.div_ceil(PAGE) * PAGE;
        let mut n = seed;
        while at < size && out.len() < 6 {
            n = n * 1103515245 + 12345;
            let to = (at + PAGE * (1 + n / 65536 % 4)).min(size);
            out.push((at, to));
            at = to + PAGE * (1 + n / 16 % (size / PAGE / 5 + 2));
        }
        out
    }

    /// What the pieces of an upload hold is what the texture's levels detile to, every kind of
    /// texture, all of it and some of its bytes, with the workers and without.
    #[test]
    fn pieces_hold_the_texels_of_their_levels() {
        unsafe {
            let Some(dev) = test_device() else { return };
            for (n, kind) in KINDS.into_iter().enumerate() {
                let m = make(&dev, kind);
                let s = m.tex.texels().unwrap();
                let size = m.tex.size() as usize;
                let mem = random_bytes(size, n as u64 * 977 + 5);
                let want = super::super::title_textures::subresources(mem.as_ptr() as usize, &m.tex);
                for round in 0..5 {
                    let changed = (round > 0).then(|| some_ranges(size, n * 11 + round));
                    let items = [Item { res: m.res.clone(), src: mem.as_ptr() as usize, tex: Arc::clone(&m.tex), states: vec![D3D12_RESOURCE_STATE_COMMON], changed: changed.clone(), known: None, pages: None }];
                    let (mut pieces, mut bytes) = (Vec::new(), 0u64);
                    plan_item(&dev, 0, &items[0], &mut bytes, &mut pieces).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
                    if round == 0 {
                        assert_eq!(pieces.len(), want.len(), "{kind:?}: all subresources");
                    }
                    for share in [0, u64::MAX] {
                        let mut buf = vec![0xA5u8; bytes as usize + 1];
                        fill(&items, &pieces, buf.as_mut_ptr(), share);
                        assert_eq!(buf[bytes as usize], 0xA5, "{kind:?}: past the buffer");
                        for p in &pieces {
                            let (w, f) = (&want[p.sub as usize], p.footprint);
                            let (row, rows) = ((p.rect.x1 - p.rect.x0) * s.bpe as usize, (p.rect.y1 - p.rect.y0) * f.Depth as usize);
                            assert_eq!(p.at % 512, 0);
                            assert_eq!((f.RowPitch % 256, f.Width, p.to.0), (0, (p.rect.x1 - p.rect.x0) as u32 * s.side, p.rect.x0 as u32 * s.side));
                            for y in 0..rows {
                                let got = &buf[p.at as usize + y * f.RowPitch as usize..][..row];
                                let from = (p.rect.y0 + y) * w.row_bytes + p.rect.x0 * s.bpe as usize;
                                assert!(got == &w.data[from..from + row], "{kind:?} {changed:x?}: subresource {} row {y} of {:?}", p.sub, p.rect);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Parts of a texture that go up change those texels and no others, on the GPU: every kind of
    /// texture, its levels left in different states, and the lists and buffers used again.
    #[test]
    fn gpu_parts_of_a_texture_go_up_and_the_rest_stays() {
        unsafe {
            let Some(dev) = test_device() else { return };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut value = 0;
            let mut run = |items: &[Item]| {
                let (failed, job, bytes) = submit(&dev, &queue, items).unwrap();
                assert!(failed.is_empty(), "{failed:?}");
                value += 1;
                queue.Signal(&fence, value).unwrap();
                crate::gpu::fence::wait(&fence, value, &dev, None, 20_000).unwrap();
                let job = job.expect("a job");
                let buffer = job.buf.as_raw() as usize;
                give_back(job);
                (buffer, bytes)
            };
            let mut buffers = Vec::new();
            for (n, kind) in KINDS.into_iter().enumerate() {
                let m = make(&dev, kind);
                let s = m.tex.texels().unwrap();
                let size = m.tex.size() as usize;
                let first = random_bytes(size, n as u64 * 31 + 3);
                let item = |mem: &[u8], changed| Item { res: m.res.clone(), src: mem.as_ptr() as usize, tex: Arc::clone(&m.tex), states: vec![D3D12_RESOURCE_STATE_COMMON], changed, known: None, pages: None };
                let (_, all) = run(&[item(&first, None)]);
                let mut want = super::super::title_textures::subresources(first.as_ptr() as usize, &m.tex);
                assert!(read_back(&dev, &queue, &m.res, want.len()) == want.iter().map(|w| w.data.clone()).collect::<Vec<_>>(), "{kind:?}: all of it");
                for round in 1..4 {
                    // all of the memory is written again, but only these bytes of it are taken to be
                    let mem = random_bytes(size, n as u64 * 31 + 100 * round as u64);
                    let changed = some_ranges(size, n * 13 + round);
                    let it = item(&mem, Some(changed.clone()));
                    let (mut pieces, mut bytes) = (Vec::new(), 0u64);
                    plan_item(&dev, 0, &it, &mut bytes, &mut pieces).unwrap();
                    if pieces.is_empty() {
                        continue;
                    }
                    let new = super::super::title_textures::subresources(mem.as_ptr() as usize, &m.tex);
                    for p in &pieces {
                        let w = &mut want[p.sub as usize];
                        for y in p.rect.y0..p.rect.y1 * p.footprint.Depth as usize {
                            let (from, n) = (y * w.row_bytes + p.rect.x0 * s.bpe as usize, (p.rect.x1 - p.rect.x0) * s.bpe as usize);
                            w.data[from..from + n].copy_from_slice(&new[p.sub as usize].data[from..from + n]);
                        }
                    }
                    let (buffer, sent) = run(&[it]);
                    buffers.push(buffer);
                    assert!(sent <= all, "{kind:?}: {sent} of {all} bytes");
                    assert!(read_back(&dev, &queue, &m.res, want.len()) == want.iter().map(|w| w.data.clone()).collect::<Vec<_>>(), "{kind:?}: {changed:x?}");
                }
            }
            buffers.sort_unstable();
            let made = buffers.len();
            buffers.dedup();
            assert!(buffers.len() < made, "an upload buffer is used again");
            dev.GetDeviceRemovedReason().unwrap();
        }
    }

    /// The rows of each subresource of `r` (in the common state).
    unsafe fn read_back(dev: &ID3D12Device, queue: &ID3D12CommandQueue, r: &ID3D12Resource, n: usize) -> Vec<Vec<u8>> {
        let rd = r.GetDesc();
        let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
        let (mut rows, mut row_bytes, mut total) = (vec![0u32; n], vec![0u64; n], 0u64);
        dev.GetCopyableFootprints(&rd, 0, n as u32, 0, Some(fps.as_mut_ptr()), Some(rows.as_mut_ptr()), Some(row_bytes.as_mut_ptr()), Some(&mut total));
        let mut rb: Option<ID3D12Resource> = None;
        let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
        let rb = rb.unwrap();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        list.ResourceBarrier(&[transition(r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        for (k, fp) in fps.iter().enumerate() {
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k as u32 } };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        }
        list.ResourceBarrier(&[transition(r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        crate::gpu::texture::wait(dev, queue).unwrap();
        let mut p = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).unwrap();
        (0..n)
            .map(|k| {
                let f = &fps[k];
                (0..rows[k] as usize * f.Footprint.Depth as usize)
                    .flat_map(|y| std::slice::from_raw_parts((p as *const u8).add(f.Offset as usize + y * f.Footprint.RowPitch as usize), row_bytes[k] as usize).to_vec())
                    .collect()
            })
            .collect()
    }

    /// What is kept of a texture's pages is what went up: after an upload of all of it, or of a part
    /// (the pages read along with the changed ones too), the same bytes written again are nothing to
    /// send; bytes that were others, or not there, while they were read go up when they come back.
    #[test]
    fn what_is_kept_is_what_went_up() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        unsafe {
            let Some(dev) = test_device() else { return };
            // 64 KiB blocks: 256x256 RGBA8 in 64KB_S, 4 of them
            let m = make(&dev, (3, 256, 256, 1, 1, 28, 0x109));
            let size = m.tex.size() as usize;
            assert_eq!(size, 4 * 0x1_0000);
            let src = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            let content = random_bytes(size, 77);
            let put = |from: usize, to: usize| std::ptr::copy_nonoverlapping(content.as_ptr().add(from), (src + from) as *mut u8, to - from);
            put(0, size);
            let known: Known = Default::default();
            let pages: Vec<usize> = (0..size / PAGE).map(|k| src + k * PAGE).collect();
            let look = |of: &[usize]| changed(&[Written { src, size, dirty: of, known: &known, pages: None }], true).0.remove(0);
            let upload = |changed: Option<Vec<(usize, usize)>>| {
                let it = [Item { res: m.res.clone(), src, tex: Arc::clone(&m.tex), states: vec![D3D12_RESOURCE_STATE_COMMON], changed, known: Some(Arc::clone(&known)), pages: None }];
                let (mut p, mut bytes) = (Vec::new(), 0u64);
                plan_item(&dev, 0, &it[0], &mut bytes, &mut p).unwrap();
                let mut buf = vec![0u8; bytes as usize];
                fill(&it, &p, buf.as_mut_ptr(), bytes);
            };
            all_sent(&known, src, size);
            upload(None);
            assert_eq!(look(&pages), Change::None, "all of it went up as it is");
            // a page changes; while its block is read its neighbours are not there
            *((src + 0x1_0000) as *mut u8) ^= 1;
            assert_eq!(look(&pages), Change::Parts(vec![(0x1_0000, 0x1_1000)]));
            assert_ne!(VirtualFree((src + 0x1_8000) as *mut c_void, 0x8000, MEM_DECOMMIT), 0);
            upload(Some(vec![(0x1_0000, 0x1_1000)]));
            assert!(!VirtualAlloc((src + 0x1_8000) as *const c_void, 0x8000, MEM_COMMIT, PAGE_READWRITE).is_null());
            put(0x1_8000, 0x2_0000);
            assert_eq!(look(&pages[0x18..0x20]), Change::Parts(vec![(0x1_8000, 0x2_0000)]), "they went up as zeros");
            upload(Some(vec![(0x1_8000, 0x2_0000)]));
            assert_eq!(look(&pages), Change::None);
            // written between the comparison and the upload, and back again
            *((src + 5) as *mut u8) ^= 1;
            assert_eq!(look(&pages), Change::Parts(vec![(0, PAGE)]));
            std::ptr::write_bytes(src as *mut u8, 0, PAGE);
            upload(Some(vec![(0, PAGE)]));
            put(0, PAGE);
            *((src + 5) as *mut u8) ^= 1;
            assert_eq!(look(&pages), Change::Parts(vec![(0, PAGE)]), "the cleared page is what went up");
            VirtualFree(src as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// An upload on another queue than an older one of the same texture that has not run yet goes
    /// up again behind it: on that queue, or anywhere once it has run. A queue that waited for the
    /// other one's uploads needs none.
    #[test]
    fn an_upload_that_may_be_overtaken_goes_up_again() {
        unsafe {
            let Some(dev) = test_device() else { return };
            let (direct, own, other) = (0x1000usize, 0x2000usize, 0x3000usize);
            let fence = || -> ID3D12Fence { dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap() };
            let (fd, fo, fx) = (fence(), fence(), fence());
            let at = |queue: usize, f: &ID3D12Fence, value: u64| Sent { queue, fence: f.clone(), value };
            let known: Known = Default::default();
            // all of it on the direct queue (which waits for XCL's own), then parts on XCL's own
            assert!(!sent(&known, None, &at(direct, &fd, 1), own));
            assert!(sent(&known, Some(&[(0, 64), (4096, 8192)]), &at(own, &fo, 1), 0), "the direct queue's is older and may run later");
            assert_eq!(take_again(&known, own, 0), Again::NotYet);
            assert_eq!(take_again(&known, other, own), Again::NotYet, "another queue has no order with it");
            assert!(sent(&known, Some(&[(100, 5000)]), &at(own, &fo, 2), 0));
            assert_eq!(take_again(&known, direct, own), Again::Now(Change::Parts(vec![(0, 64), (100, 8192)])), "both, behind it on its queue");
            assert_eq!(take_again(&known, direct, own), Again::Nothing);
            // sent again there: the direct queue waited for the own one, nothing is overtaken
            assert!(!sent(&known, Some(&[(0, 64), (100, 8192)]), &at(direct, &fd, 2), own));
            // once the older one has run, it goes up wherever
            assert!(sent(&known, None, &at(own, &fo, 3), 0));
            assert!(sent(&known, None, &at(other, &fx, 1), own), "two of the title's queues");
            assert_eq!(take_again(&known, own, 0), Again::NotYet);
            fd.Signal(2).unwrap();
            assert_eq!(take_again(&known, own, 0), Again::Now(Change::All));
            // an upload of all of it, or nothing known of it: what was to go up again is moot
            assert!(sent(&known, Some(&[(0, 64)]), &at(own, &fo, 4), 0), "(the other queue's is still to run)");
            all_sent(&known, 0x10_0000, 4096);
            assert_eq!(take_again(&known, other, own), Again::Nothing);
            assert!(sent(&known, Some(&[(0, 64)]), &at(own, &fo, 5), 0));
            forget(&known);
            assert_eq!(take_again(&known, other, own), Again::Nothing);
            // another texture over the same video memory takes them over
            assert!(!sent(&Known::default(), None, &at(own, &fo, 6), 0));
            let over: Known = Default::default();
            inherit(&over, &known);
            assert!(sent(&over, None, &at(own, &fo, 7), 0), "behind the other queue's, which is still to run");
            assert_eq!(take_again(&over, own, 0), Again::NotYet);
            assert_eq!(take_again(&over, other, own), Again::Now(Change::All));
            // nothing pending: nothing overtaken
            for (f, v) in [(&fd, 9), (&fo, 9), (&fx, 9)] {
                f.Signal(v).unwrap();
            }
            assert!(!sent(&known, None, &at(own, &fo, 10), 0));
        }
    }

    #[test]
    fn large_buffers_grow_in_steps_of_sixteen_mib() {
        let mib = 1 << 20;
        assert_eq!(capacity(1), Some(mib));
        assert_eq!(capacity(33 * mib), Some(64 * mib));
        assert_eq!(capacity(64 * mib + 1), Some(80 * mib));
        assert_eq!(capacity(138_713_408), Some(144 * mib));
        assert_eq!(capacity(256 * mib + 1), Some(272 * mib));
        assert_eq!(capacity(u64::MAX), None);
    }

    /// A page the title decommits while it is read (or that never was committed) reads as zeros.
    #[test]
    fn memory_that_is_not_there_reads_as_zeros() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            VirtualAlloc((p + PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE);
            std::ptr::write_bytes((p + PAGE) as *mut u8, 0x77, PAGE);
            // (what an earlier read left in the scratch memory does not show)
            read(None, p + PAGE, PAGE, |b| assert!(b.iter().all(|&x| x == 0x77)));
            read(None, p, 4 * PAGE, |b| {
                assert_eq!(b.len(), 4 * PAGE);
                assert!(b[..PAGE].iter().all(|&x| x == 0) && b[PAGE..2 * PAGE].iter().all(|&x| x == 0x77) && b[2 * PAGE..].iter().all(|&x| x == 0));
            });
            VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A texture whose GPU-view pages are kept apart is read through its table, in one read across
    /// them: two pages where they are kept, a page that shows nothing, a page never mapped (read as
    /// any memory) and a page whose memory is gone; and `changed` compares through the table.
    #[test]
    fn an_upload_reads_pages_kept_apart() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const VIEW: usize = 0x1_0000;
        unsafe {
            let page = |v: u8| {
                let p = VirtualAlloc(std::ptr::null(), VIEW, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
                assert_ne!(p, 0);
                std::ptr::write_bytes(p as *mut u8, v, VIEW);
                p
            };
            let (a, b, c, gone) = (page(0x11), page(0x22), page(0x33), page(0x44));
            VirtualFree(gone as *mut c_void, 0, MEM_RELEASE);
            // (the texture's own address: a GPU view's, never touched here)
            let first = 0x7E5A_0000_0000usize;
            let m = PageMap::new(first + 0x8000, vec![a, b, 0, c | 1, gone]);
            let want = |off: usize| match off / VIEW {
                0 => 0x11,
                1 => 0x22,
                3 => 0x33,
                _ => 0,
            };
            read(Some(&m), first + 0x8000, 5 * VIEW - 0x8000, |got| {
                assert_eq!(got.len(), 5 * VIEW - 0x8000);
                assert!(got.iter().enumerate().all(|(k, &x)| x == want(k + 0x8000)), "each page from where it is kept");
            });
            read(Some(&m), first + VIEW - 0x1000, 0x2000, |got| assert!(got[..0x1000].iter().all(|&x| x == 0x11) && got[0x1000..].iter().all(|&x| x == 0x22)));
            // past the table: nothing
            read(Some(&m), first + 5 * VIEW, PAGE, |got| assert!(got.iter().all(|&x| x == 0)));
            // the written pages compared through the table
            let (src, size) = (first, 4 * VIEW);
            let known: Known = Default::default();
            let dirty: Vec<usize> = (0..size / PAGE).map(|k| src + k * PAGE).collect();
            let look = || changed(&[Written { src, size, dirty: &dirty, known: &known, pages: Some(&m) }], true).0.remove(0);
            all_sent(&known, src, size);
            assert_eq!(look(), Change::Parts(vec![(0, size)]), "nothing was known of them");
            assert_eq!(look(), Change::None);
            *((b + 0x1005) as *mut u8) ^= 1;
            assert_eq!(look(), Change::Parts(vec![(VIEW + 0x1000, VIEW + 0x2000)]), "the page that differs where it is kept");
            for p in [a, b, c] {
                VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
            }
        }
    }
}
