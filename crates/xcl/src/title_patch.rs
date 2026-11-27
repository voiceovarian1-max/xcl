//! (local) Opt-in load-time patches of one retail title build (1.0.2.1 of `game_scarlett_master.exe`):
//! render scale, output size and its watchdog. Off unless asked for; every site's bytes are compared
//! and a lever is written all or not at all, so another build is left untouched.
//!
//! Provenance (GPL-3.0-or-later):
//! - watchdog, render scale (own upscaler and FSR), own upscaler: miaohaixiang1-netizen/cool-fox-12gb,
//!   https://github.com/miaohaixiang1-netizen/cool-fox-12gb, commit 6d28e9bf,
//!   crates/victory-loader/src/hooks.rs (ported there from ZigZagBoi1/ScarlettPort `runtime/victory`
//!   via vitutu1/victory-perf);
//! - output size and the back buffer pool check: avatarmanz/VIctory-7900XTX,
//!   https://github.com/avatarmanz/VIctory-7900XTX, commits e151e71f / 45cad4c5 / f63b4b12,
//!   crates/victory-loader/src/resolution.rs.
//! Only image offsets and short byte signatures live here, nothing of the title. Not checked against
//! the binary here (no game files were read); a mismatch writes nothing and says so.
//!
//! Switches: `XCL_NO_WATCHDOG=1`, `XCL_RENDER_SCALE=0.5..1.0`, `XCL_OWN_UPSCALER=1`,
//! `XCL_OUTPUT_SIZE=<w>x<h>` (one of the title's modes). They change the picture (render scale,
//! output size) or what a stall does (the watchdog: a real deadlock then hangs instead of ending).

/// One place to change: the bytes at `rva` must be `expect`; `new` is as long.
#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    pub rva: usize,
    pub what: &'static str,
    pub expect: Vec<u8>,
    pub new: Vec<u8>,
}

/// Writes every site of one lever over `image` (the mapped image, before its code runs), or none:
/// how many were written, or why none was.
pub fn apply(image: &mut [u8], sites: &[Site]) -> Result<usize, String> {
    for s in sites {
        if s.new.len() != s.expect.len() {
            return Err(format!("{}: replacement of another length", s.what));
        }
        let end = s.rva.checked_add(s.expect.len()).filter(|&e| e <= image.len()).ok_or_else(|| format!("{}: past the image", s.what))?;
        if image[s.rva..end] != s.expect[..] {
            return Err(format!("{}: the bytes at {:#x} are not the expected ones", s.what, s.rva));
        }
    }
    for s in sites {
        image[s.rva..s.rva + s.new.len()].copy_from_slice(&s.new);
    }
    Ok(sites.len())
}

// ---------------------------------------------------------------------------------------------
// the watchdog (cool-fox-12gb hooks.rs WATCHDOG)
// ---------------------------------------------------------------------------------------------

/// The title's watchdog ends the process on purpose when its main thread has not advanced for
/// ~30 one-second ticks: the handler it calls (`mov dword ptr [0], imm32`) and a fallback after the
/// call (`mov dword ptr [0], 0; int3`). The jump skips both; the NOP disarms the handler.
pub fn watchdog_sites() -> Vec<Site> {
    vec![
        Site {
            rva: 0x2215c27,
            what: "watchdog: skip the handler call and the fallback fault",
            expect: vec![0x48, 0x8B, 0x05, 0x1A, 0x07, 0xFA, 0x04], // mov rax, [rip + disp32]
            new: vec![0xEB, 0x18, 0x90, 0x90, 0x90, 0x90, 0x90],   // jmp +0x18; NOPs
        },
        Site {
            rva: 0x23cf683,
            what: "watchdog: the handler's deliberate fault",
            expect: vec![0xC7, 0x04, 0x25, 0, 0, 0, 0, 0x21, 0xD4, 0x8B, 0x4D],
            new: vec![0x66, 0x66, 0x2E, 0x0F, 0x1F, 0x84, 0, 0, 0, 0, 0], // an 11-byte NOP
        },
    ]
}

// ---------------------------------------------------------------------------------------------
// render scale (cool-fox-12gb hooks.rs render_scale_patches; VIctory-7900XTX resolution.rs OWN_SCALE)
// ---------------------------------------------------------------------------------------------

/// `XCL_RENDER_SCALE`: 0.5 to 1.0 of the output size (VIctory's range; below 2/3 the title's
/// largest render size follows only with the max patch).
pub fn parse_scale(text: &str) -> Option<f32> {
    let v: f32 = text.trim().parse().ok()?;
    (0.5..=1.0).contains(&v).then_some(v)
}

/// Read-only float constants of this build that FSR's scale load can point at: value, RVA.
const FSR_SCALES: [(f32, usize); 4] = [(0.5, 0x5fc6edc), (0.4, 0x5fc6f90), (1.0 / 3.0, 0x615a240), (0.75, 0x6159b04)];

/// The render scale sites for `scale` with the image at `base`: the title's own upscaler's minimum
/// (an immediate, 2/3 by default) and its maximum made equal to it (`vmaxss` -> `vmovss`); with
/// `fsr` and a constant of that value in this build, FSR's scale load (2/3 on Series S) pointed at
/// it and FSR's maximum size made the scaled size (vtable slot 11 -> slot 10's function).
pub fn render_scale_sites(base: usize, scale: f32, fsr: bool) -> Vec<Site> {
    let mut v = vec![
        Site {
            rva: 0x28f1e7e,
            what: "render scale: the own upscaler's minimum",
            expect: (2.0f32 / 3.0).to_le_bytes().to_vec(), // AB AA 2A 3F
            new: scale.to_le_bytes().to_vec(),
        },
        Site {
            rva: 0x28f1e87,
            what: "render scale: the own upscaler's maximum = its minimum",
            expect: vec![0xC5, 0xFA, 0x5F, 0x86, 0x90, 0, 0, 0], // vmaxss xmm0, xmm0, [rsi+90h]
            new: vec![0xC5, 0xFA, 0x10, 0x86, 0x90, 0, 0, 0],    // vmovss xmm0, [rsi+90h]
        },
    ];
    if let Some(&(_, constant)) = FSR_SCALES.iter().find(|(c, _)| fsr && (c - scale).abs() < 1e-3) {
        // vmovss xmm0, [rip + disp32]: the next instruction is at +8
        const LOAD: usize = 0x28ef6b2;
        let disp = (constant as i64 - (LOAD as i64 + 8)) as i32;
        let mut new = vec![0xC5, 0xFA, 0x10, 0x05];
        new.extend_from_slice(&disp.to_le_bytes());
        v.push(Site { rva: LOAD, what: "render scale: FSR's scale", expect: vec![0xC5, 0xFA, 0x10, 0x05, 0xA2, 0xE2, 0x86, 0x03], new });
        v.push(Site {
            rva: 0x6300ba8,
            what: "render scale: FSR's maximum = the scaled size",
            expect: ((base + 0x28ef6f0) as u64).to_le_bytes().to_vec(),
            new: ((base + 0x28ef680) as u64).to_le_bytes().to_vec(),
        });
    }
    v
}

/// The upscaler choice: `mov eax, 6` (own) then `cmovne eax, ecx` (FSR when the profile asks):
/// NOPing the cmovne keeps the title's own upscaler on every profile.
pub fn own_upscaler_sites() -> Vec<Site> {
    vec![Site { rva: 0x7e9426, what: "upscaler: always the title's own", expect: vec![0x0F, 0x45, 0xC1], new: vec![0x0F, 0x1F, 0x00] }]
}

// ---------------------------------------------------------------------------------------------
// output size (VIctory-7900XTX resolution.rs OUTPUT, POOL_CHECK, output_code)
// ---------------------------------------------------------------------------------------------

/// The title's own display modes (its table, not DXGI's).
pub const MODES: [(u32, u32); 5] = [(1280, 720), (1664, 936), (1920, 1080), (2560, 1440), (3840, 2160)];

/// `XCL_OUTPUT_SIZE=<w>x<h>`: one of `MODES`.
pub fn parse_output(text: &str) -> Option<(u32, u32)> {
    let (w, h) = text.trim().split_once(['x', 'X'])?;
    let size = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    MODES.contains(&size).then_some(size)
}

/// The screen size setter, its screen height, and what the block calls when the height changed.
pub const SET_SCREEN_SIZE: usize = 0x2478130;
pub const SCREEN_HEIGHT: usize = 0xa9438d0;
pub const HEIGHT_CHANGED: usize = 0xd8c3a0;
/// The block that hands the settings' output size to the screen size (61 bytes).
pub const OUTPUT_AT: usize = 0x224352b;
pub const OUTPUT_BYTES: [u8; 61] = [
    0x8b, 0x0d, 0x9b, 0x03, 0x70, 0x08, // mov ecx, [+0xa9438cc]
    0x8b, 0x86, 0x04, 0x02, 0x00, 0x00, // mov eax, [rsi+204h]
    0x39, 0xc1, 0x75, 0x0d, // cmp ecx, eax ; jne
    0x8b, 0x96, 0x08, 0x02, 0x00, 0x00, // mov edx, [rsi+208h]
    0xe8, 0xea, 0x4b, 0x23, 0x00, 0xeb, 0x20, // call +0x2478130 ; jmp end
    0x8b, 0xbe, 0x08, 0x02, 0x00, 0x00, // mov edi, [rsi+208h]
    0x8b, 0x2d, 0x7c, 0x03, 0x70, 0x08, // mov ebp, [+0xa9438d0]
    0x89, 0xc1, 0x89, 0xfa, // mov ecx, eax ; mov edx, edi
    0xe8, 0xd3, 0x4b, 0x23, 0x00, // call +0x2478130
    0x39, 0xfd, 0x74, 0x07, 0xb1, 0x01, // cmp ebp, edi ; je end ; mov cl, 1
    0xe8, 0x38, 0x8e, 0xb4, 0xfe, // call +0xd8c3a0
];
/// After the back buffers are made in their pool: `int3` unless under 2 MB of it is left (sized for
/// 3840x2160); for another size the `jbe` over the `int3` becomes a `jmp`.
pub const POOL_AT: usize = 0x247af25;
pub const POOL_BYTES: [u8; 8] = [0x48, 0x3d, 0xff, 0xff, 0x1f, 0x00, 0x76, 0x01]; // cmp rax, 1FFFFFh ; jbe +1

/// `rel32` from the end of an instruction at image offset `next` to `to`.
fn rel(to: usize, next: usize) -> [u8; 4] {
    ((to as i64 - next as i64) as i32).to_le_bytes()
}

/// What goes at `OUTPUT_AT` for `width` x `height`: the size written into the live settings, then
/// the title's own path to the screen size (same length).
pub fn output_code(width: u32, height: u32) -> Vec<u8> {
    let at = OUTPUT_AT;
    let mut c = Vec::with_capacity(OUTPUT_BYTES.len());
    c.push(0xb8); // mov eax, width
    c.extend(width.to_le_bytes());
    c.extend([0x89, 0x86, 0x04, 0x02, 0x00, 0x00]); // mov [rsi+204h], eax
    c.push(0xbf); // mov edi, height
    c.extend(height.to_le_bytes());
    c.extend([0x89, 0xbe, 0x08, 0x02, 0x00, 0x00]); // mov [rsi+208h], edi
    c.extend([0x8b, 0x2d]); // mov ebp, [screen height]
    c.extend(rel(SCREEN_HEIGHT, at + c.len() + 4));
    c.extend([0x89, 0xc1, 0x89, 0xfa]); // mov ecx, eax ; mov edx, edi
    c.push(0xe8); // call the setter
    c.extend(rel(SET_SCREEN_SIZE, at + c.len() + 4));
    let end = OUTPUT_BYTES.len();
    c.extend([0x39, 0xfd, 0x74, (end - (c.len() + 4)) as u8]); // cmp ebp, edi ; je end
    c.extend([0xb1, 0x01, 0xe8]); // mov cl, 1 ; call
    c.extend(rel(HEIGHT_CHANGED, at + c.len() + 4));
    c.extend([0xeb, (end - (c.len() + 2)) as u8]); // jmp end
    c.resize(end, 0xcc);
    c
}

/// The output size sites for `width` x `height` (no pool change at 3840x2160).
pub fn output_sites(width: u32, height: u32) -> Vec<Site> {
    let mut v = vec![Site { rva: OUTPUT_AT, what: "output size: the settings' size applied", expect: OUTPUT_BYTES.to_vec(), new: output_code(width, height) }];
    if (width, height) != (3840, 2160) {
        let mut new = POOL_BYTES.to_vec();
        new[6] = 0xeb;
        v.push(Site { rva: POOL_AT, what: "output size: the back buffer pool's left-over check", expect: POOL_BYTES.to_vec(), new });
    }
    v
}

// ---------------------------------------------------------------------------------------------
// what the environment asks for
// ---------------------------------------------------------------------------------------------

/// The levers the environment asks for with the image at `base`: (name, sites). An unparsable value
/// is logged and left out.
pub fn asked(base: usize, get: impl Fn(&str) -> Option<String>) -> Vec<(&'static str, Vec<Site>)> {
    let mut v = Vec::new();
    if get("XCL_NO_WATCHDOG").is_some_and(|s| s.trim() == "1") {
        v.push(("XCL_NO_WATCHDOG", watchdog_sites()));
    }
    let own = get("XCL_OWN_UPSCALER").is_some_and(|s| s.trim() == "1");
    if own {
        v.push(("XCL_OWN_UPSCALER", own_upscaler_sites()));
    }
    if let Some(text) = get("XCL_RENDER_SCALE") {
        match parse_scale(&text) {
            // (with the own upscaler forced, FSR is left as it is)
            Some(s) => v.push(("XCL_RENDER_SCALE", render_scale_sites(base, s, !own))),
            None => crate::xlog_warn!("title patch: XCL_RENDER_SCALE={text}: a number from 0.5 to 1.0 is expected"),
        }
    }
    if let Some(text) = get("XCL_OUTPUT_SIZE") {
        match parse_output(&text) {
            Some((w, h)) => v.push(("XCL_OUTPUT_SIZE", output_sites(w, h))),
            None => crate::xlog_warn!("title patch: XCL_OUTPUT_SIZE={text}: one of 1280x720, 1664x936, 1920x1080, 2560x1440, 3840x2160 is expected"),
        }
    }
    v
}

/// The levers asked for, written over the main image (mapped at `base`, its bytes `image`, before
/// its code runs and while it is writable): one log line per lever. How many sites were written.
pub fn apply_asked(image: &mut [u8], base: usize) -> usize {
    let mut written = 0;
    for (name, sites) in asked(base, |k| std::env::var(k).ok()) {
        match apply(image, &sites) {
            Ok(n) => {
                written += n;
                crate::xlog_info!("title patch: {name}: {n} site(s) written ({})", sites.iter().map(|s| s.what).collect::<Vec<_>>().join("; "));
            }
            Err(e) => crate::xlog_warn!("title patch: {name}: nothing written – {e} (another build?)"),
        }
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An image holding every site's expected bytes, with `base`.
    fn image_with(sites: &[Site]) -> Vec<u8> {
        let len = sites.iter().map(|s| s.rva + s.expect.len()).max().unwrap_or(0) + 16;
        let mut img = vec![0u8; len];
        for s in sites {
            img[s.rva..s.rva + s.expect.len()].copy_from_slice(&s.expect);
        }
        img
    }

    #[test]
    fn scales_and_sizes_are_parsed_strictly() {
        assert_eq!(parse_scale("0.5"), Some(0.5));
        assert_eq!(parse_scale(" 1 "), Some(1.0));
        for bad in ["0.4", "1.5", "half", "nan", ""] {
            assert_eq!(parse_scale(bad), None, "{bad:?}");
        }
        assert_eq!(parse_output("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_output(" 2560X1440 "), Some((2560, 1440)));
        for bad in ["1600x900", "1080p", "1920*1080", ""] {
            assert_eq!(parse_output(bad), None, "{bad:?}");
        }
    }

    /// A site is written only over its expected bytes; past the image, a byte off, or written
    /// already: refused, and nothing changes.
    #[test]
    fn a_site_is_written_only_over_its_bytes() {
        let sites = watchdog_sites();
        let mut img = image_with(&sites);
        let before = img.clone();
        let mut off = img.clone();
        off[sites[0].rva + 3] ^= 1;
        assert!(apply(&mut off, &sites).is_err());
        assert_eq!(off[sites[1].rva..sites[1].rva + 11], sites[1].expect[..], "the other site is not written either");
        assert_eq!(apply(&mut img, &sites), Ok(2));
        assert_eq!(img[sites[0].rva..sites[0].rva + 7], sites[0].new[..]);
        assert_ne!(img, before);
        let once = img.clone();
        assert!(apply(&mut img, &sites).is_err(), "a second time: the bytes are the new ones");
        assert_eq!(img, once);
        let mut short = vec![0u8; sites[0].rva + 3];
        assert!(apply(&mut short, &sites).is_err(), "past the image");
    }

    /// Every site keeps its length; the watchdog's jump lands where the original's block ends.
    #[test]
    fn replacements_keep_their_lengths() {
        let mut all = watchdog_sites();
        all.extend(render_scale_sites(0x1_4000_0000, 0.5, true));
        all.extend(own_upscaler_sites());
        all.extend(output_sites(1920, 1080));
        assert!(all.iter().all(|s| s.new.len() == s.expect.len()), "{all:?}");
        let w = &watchdog_sites()[0];
        assert_eq!(w.new[..2], [0xEB, 0x18]);
        assert_eq!(w.rva + 2 + 0x18, 0x2215c27 + 0x1a);
    }

    /// The render scale at 0.5 changes both upscalers; another value only the title's own.
    #[test]
    fn render_scale_sites_point_fsr_at_the_constant() {
        let base = 0x7ff0_0000_0000;
        let p = render_scale_sites(base, 0.5, true);
        assert_eq!(p.len(), 4);
        assert_eq!(p[0].expect, [0xAB, 0xAA, 0x2A, 0x3F], "2/3, as the title has it");
        assert_eq!(p[0].new, 0.5f32.to_le_bytes());
        assert_eq!(p[2].new, [0xC5, 0xFA, 0x10, 0x05, 0x22, 0x78, 0x6D, 0x03], "the load reads 0.5 at 0x5fc6edc");
        assert_eq!(p[3].new, ((base + 0x28ef680) as u64).to_le_bytes());
        assert_eq!(render_scale_sites(0, 0.6, true).len(), 2);
        assert_eq!(render_scale_sites(0, 0.5, false).len(), 2);
    }

    /// The calls and loads in the bytes compared are the ones the replacement keeps.
    #[test]
    fn the_output_block_is_the_titles_and_its_code_keeps_its_path() {
        let b = OUTPUT_BYTES;
        let target = |c: &[u8], op_end: usize| (OUTPUT_AT + op_end) as i64 + i64::from(i32::from_le_bytes(c[op_end - 4..op_end].try_into().unwrap()));
        assert_eq!(target(&b, 27), SET_SCREEN_SIZE as i64);
        assert_eq!(target(&b, 50), SET_SCREEN_SIZE as i64);
        assert_eq!(target(&b, 61), HEIGHT_CHANGED as i64);
        assert_eq!(target(&b, 41), SCREEN_HEIGHT as i64);
        assert_eq!(target(&b, 6), (SCREEN_HEIGHT - 4) as i64, "the width before it");
        assert_eq!((b[27], 29 + b[28] as usize), (0xeb, 61));
        assert_eq!((b[52], 54 + b[53] as usize), (0x74, 61));
        let c = output_code(1920, 1080);
        assert_eq!(c.len(), 61);
        assert_eq!((c[0], &c[1..5]), (0xb8, &1920u32.to_le_bytes()[..]));
        assert_eq!(&c[12..16], &1080u32.to_le_bytes()[..]);
        assert_eq!(target(&c, 28), SCREEN_HEIGHT as i64);
        assert_eq!(target(&c, 37), SET_SCREEN_SIZE as i64);
        assert_eq!(41 + c[40] as usize, 61, "je lands at the end");
        assert_eq!(target(&c, 48), HEIGHT_CHANGED as i64);
        assert_eq!(50 + c[49] as usize, 61, "jmp lands at the end");
        assert!(c[50..].iter().all(|&x| x == 0xcc));
        assert_eq!(output_sites(3840, 2160).len(), 1, "no pool change at the size it is made for");
        assert_eq!(output_sites(1920, 1080)[1].new[6], 0xeb);
    }

    /// What the environment asks for: nothing by default; an unparsable value leaves its lever out;
    /// the own upscaler forced leaves FSR alone.
    #[test]
    fn only_the_levers_asked_for_are_built() {
        let none = |_: &str| None;
        assert!(asked(0, none).is_empty());
        let env = |pairs: &'static [(&'static str, &'static str)]| move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string());
        let v = asked(0, env(&[("XCL_RENDER_SCALE", "0.5"), ("XCL_OUTPUT_SIZE", "1920x1080"), ("XCL_NO_WATCHDOG", "1")]));
        let names: Vec<_> = v.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, ["XCL_NO_WATCHDOG", "XCL_RENDER_SCALE", "XCL_OUTPUT_SIZE"]);
        assert_eq!(v[1].1.len(), 4);
        let own = asked(0, env(&[("XCL_RENDER_SCALE", "0.5"), ("XCL_OWN_UPSCALER", "1")]));
        assert_eq!(own.iter().find(|(n, _)| *n == "XCL_RENDER_SCALE").unwrap().1.len(), 2);
        assert!(asked(0, env(&[("XCL_RENDER_SCALE", "2")])).is_empty());
        // a whole image of another build: every lever refused, nothing written
        let mut other = vec![0x90u8; 0x2500_000];
        let before = other[0x2215c27..0x2215c27 + 7].to_vec();
        for (_, sites) in asked(0, env(&[("XCL_NO_WATCHDOG", "1"), ("XCL_OUTPUT_SIZE", "1920x1080")])) {
            assert!(apply(&mut other, &sites).is_err());
        }
        assert_eq!(other[0x2215c27..0x2215c27 + 7], before[..]);
    }
}
