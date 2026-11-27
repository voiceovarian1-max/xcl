//! Synthetic Xbox shader containers shared by the parser and integration fixtures.

pub(crate) fn container(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let header = 32 + parts.len() * 4;
    let mut v = vec![0u8; header];
    v[0..4].copy_from_slice(b"DXBC");
    v[20..24].copy_from_slice(&1u32.to_le_bytes());
    v[28..32].copy_from_slice(&(parts.len() as u32).to_le_bytes());
    for (i, (cc, data)) in parts.iter().enumerate() {
        let off = v.len() as u32;
        v[32 + i * 4..36 + i * 4].copy_from_slice(&off.to_le_bytes());
        v.extend_from_slice(*cc);
        v.extend_from_slice(&(data.len() as u32).to_le_bytes());
        v.extend_from_slice(data);
    }
    let total = v.len() as u32;
    v[24..28].copy_from_slice(&total.to_le_bytes());
    v
}

pub(crate) fn le32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

/// FSSM with the observed layout: code pointer at 0xC0, stage descriptors at 0x100 (+0x88).
pub(crate) fn fssm(code: &[u8], stages: &[(u8, u8)]) -> Vec<u8> {
    let code_at = 0x308usize;
    let mut v = vec![0u8; code_at];
    v[0..4].copy_from_slice(b"FSSM");
    v[4..8].copy_from_slice(&0x101u32.to_le_bytes());
    for q in (0x18..0xC0).step_by(8) {
        v[q..q + 8].copy_from_slice(&0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
    }
    v[0xC0..0xC8].copy_from_slice(&((code_at - 0x10) as u64).to_le_bytes());
    v[0xC8..0xD0].copy_from_slice(&(code.len() as u64 + 8).to_le_bytes());
    for (i, &(hw, api)) in stages.iter().enumerate() {
        let o = 0x100 + i * 0x88;
        let q = 0x0000_00C0_0002_0000u64 | (api as u64) << 8 | hw as u64;
        v[o..o + 8].copy_from_slice(&q.to_le_bytes());
    }
    v.extend_from_slice(code);
    v.extend_from_slice(&[0x00, 0x00, 0x9f, 0xbf, 0x00, 0x00, 0x9f, 0xbf]); // s_code_end padding
    v.extend_from_slice(b"\x03\0\0\0metadata\0\0\0\0");
    let size = (v.len() - 0x10) as u32;
    v[8..12].copy_from_slice(&size.to_le_bytes());
    v
}

fn xbch(fssm: &[u8]) -> Vec<u8> {
    let z = miniz_oxide::deflate::compress_to_vec_zlib(fssm, 6);
    let mut v = b"XBCH".to_vec();
    le32(&mut v, 3);
    le32(&mut v, z.len() as u32);
    le32(&mut v, 0);
    le32(&mut v, fssm.len() as u32);
    le32(&mut v, 0);
    v.extend_from_slice(&z);
    v
}

pub(crate) fn program_header(kind: u32, size_dw: u32, bitcode_off: u32) -> Vec<u8> {
    let mut v = Vec::new();
    le32(&mut v, kind << 16 | 0x66);
    le32(&mut v, size_dw);
    v.extend_from_slice(b"DXIL");
    le32(&mut v, 0x106);
    le32(&mut v, bitcode_off);
    le32(&mut v, 4);
    v
}

/// DXIL part exactly as found in retail data: header, XDAT (hash, XBCH…, nested header), NOBC.
pub(crate) fn dxil_part(kind: u32, fssms: &[Vec<u8>]) -> Vec<u8> {
    let mut blobs: Vec<(u32, Vec<u8>)> = vec![(0x12, (0u8..16).collect())];
    for f in fssms {
        blobs.push((0x0D, xbch(f)));
    }
    let mut nested = program_header(kind, 0x8000_0007, 0x10);
    nested.extend_from_slice(b"NOBC");
    blobs.push((0x07, nested));
    let table = 0x18 + 12;
    let mut data_at = table + blobs.len() * 12;
    let mut xdat = b"XDAT".to_vec();
    le32(&mut xdat, blobs.len() as u32);
    le32(&mut xdat, table as u32);
    for (id, b) in &blobs {
        le32(&mut xdat, *id);
        le32(&mut xdat, data_at as u32);
        le32(&mut xdat, b.len() as u32);
        data_at += b.len();
    }
    for (_, b) in &blobs {
        xdat.extend_from_slice(b);
    }
    while (0x18 + xdat.len()) % 4 != 0 {
        xdat.push(0);
    }
    let nobc_at = 0x18 + xdat.len();
    let mut v = program_header(kind, ((nobc_at + 4) / 4) as u32, (nobc_at - 8) as u32);
    v.extend_from_slice(&xdat);
    v.extend_from_slice(b"NOBC");
    v
}

// Some integration crates only need the container builders.
#[allow(dead_code)]
pub(crate) fn mesh_psv(max_v: u16, max_p: u16, threads: [u32; 3], wave_lanes: [u32; 2]) -> Vec<u8> {
    let mut r = vec![0u8; 48];
    r[12..14].copy_from_slice(&max_v.to_le_bytes());
    r[14..16].copy_from_slice(&max_p.to_le_bytes());
    r[16..20].copy_from_slice(&wave_lanes[0].to_le_bytes());
    r[20..24].copy_from_slice(&wave_lanes[1].to_le_bytes());
    r[24] = 13; // MS
    r[27] = 2; // triangles
    r[36..40].copy_from_slice(&threads[0].to_le_bytes());
    r[40..44].copy_from_slice(&threads[1].to_le_bytes());
    r[44..48].copy_from_slice(&threads[2].to_le_bytes());
    let mut p = 48u32.to_le_bytes().to_vec();
    p.extend_from_slice(&r);
    le32(&mut p, 0);
    p
}
