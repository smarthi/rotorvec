//! Read/write rotorvec index files.
//!
//! ## File format (v2)
//!
//! Both `.rv` (positional) and `.rvim` (id-mapped) files share the same
//! prefix: 4-byte magic + 1-byte version + core header. The core header is
//! 10 bytes: `[bits:u8][rotation_kind:u8][dim:u32 LE][n_vectors:u32 LE]`.
//! Then `packed_codes`, `norms`, and an 8-byte little-endian `seed`. The
//! `.rvim` variant additionally appends a `slot_to_id` table of `u64` LE
//! values (one per vector).

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::rotor::Rotation;

const RV_MAGIC: &[u8; 4] = b"RVEC";
const RVIM_MAGIC: &[u8; 4] = b"RVIM";
const VERSION: u8 = 2;
const CORE_HEADER_SIZE: usize = 10;

#[allow(clippy::too_many_arguments)]
pub fn write(
    path: impl AsRef<Path>,
    bits: usize,
    dim: usize,
    n_vectors: usize,
    rotation: Rotation,
    seed: u64,
    packed_codes: &[u8],
    norms: &[f32],
) -> io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(RV_MAGIC)?;
    f.write_all(&[VERSION])?;
    write_core(&mut f, bits, rotation, dim, n_vectors, packed_codes, norms)?;
    f.write_all(&seed.to_le_bytes())?;
    f.flush()
}

pub fn load(
    path: impl AsRef<Path>,
) -> io::Result<(usize, usize, usize, Rotation, u64, Vec<u8>, Vec<f32>)> {
    let mut f = BufReader::new(File::open(path)?);
    expect_magic(&mut f, RV_MAGIC, "RVEC")?;
    let (bits, rotation, dim, n_vectors, packed_codes, norms) = read_core(&mut f)?;
    let mut seed_bytes = [0u8; 8];
    f.read_exact(&mut seed_bytes)?;
    let seed = u64::from_le_bytes(seed_bytes);
    Ok((bits, dim, n_vectors, rotation, seed, packed_codes, norms))
}

#[allow(clippy::too_many_arguments)]
pub fn write_id_map(
    path: impl AsRef<Path>,
    bits: usize,
    dim: usize,
    n_vectors: usize,
    rotation: Rotation,
    seed: u64,
    packed_codes: &[u8],
    norms: &[f32],
    slot_to_id: &[u64],
) -> io::Result<()> {
    assert_eq!(slot_to_id.len(), n_vectors);
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(RVIM_MAGIC)?;
    f.write_all(&[VERSION])?;
    write_core(&mut f, bits, rotation, dim, n_vectors, packed_codes, norms)?;
    f.write_all(&seed.to_le_bytes())?;
    for &id in slot_to_id {
        f.write_all(&id.to_le_bytes())?;
    }
    f.flush()
}

pub fn load_id_map(
    path: impl AsRef<Path>,
) -> io::Result<(
    usize,
    usize,
    usize,
    Rotation,
    u64,
    Vec<u8>,
    Vec<f32>,
    Vec<u64>,
)> {
    let mut f = BufReader::new(File::open(path)?);
    expect_magic(&mut f, RVIM_MAGIC, "RVIM")?;
    let (bits, rotation, dim, n_vectors, packed_codes, norms) = read_core(&mut f)?;
    let mut seed_bytes = [0u8; 8];
    f.read_exact(&mut seed_bytes)?;
    let seed = u64::from_le_bytes(seed_bytes);
    let mut slot_to_id = Vec::with_capacity(n_vectors);
    let mut buf = [0u8; 8];
    for _ in 0..n_vectors {
        f.read_exact(&mut buf)?;
        slot_to_id.push(u64::from_le_bytes(buf));
    }
    Ok((
        bits,
        dim,
        n_vectors,
        rotation,
        seed,
        packed_codes,
        norms,
        slot_to_id,
    ))
}

fn expect_magic<R: Read>(r: &mut R, want: &[u8; 4], name: &str) -> io::Result<()> {
    let mut magic = [0u8; 4];
    let mut version = [0u8; 1];
    r.read_exact(&mut magic)?;
    if &magic != want {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("not a {name} file: wrong magic"),
        ));
    }
    r.read_exact(&mut version)?;
    if version[0] != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported {name} version: {}", version[0]),
        ));
    }
    Ok(())
}

fn write_core<W: Write>(
    w: &mut W,
    bits: usize,
    rotation: Rotation,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    norms: &[f32],
) -> io::Result<()> {
    w.write_all(&[bits as u8])?;
    w.write_all(&[rotation.as_byte()])?;
    w.write_all(&(dim as u32).to_le_bytes())?;
    w.write_all(&(n_vectors as u32).to_le_bytes())?;
    w.write_all(packed_codes)?;
    for &n in norms {
        w.write_all(&n.to_le_bytes())?;
    }
    Ok(())
}

fn read_core<R: Read>(r: &mut R) -> io::Result<(usize, Rotation, usize, usize, Vec<u8>, Vec<f32>)> {
    let mut header = [0u8; CORE_HEADER_SIZE];
    r.read_exact(&mut header)?;
    let bits = header[0] as usize;
    let rotation = Rotation::from_byte(header[1]).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown rotation kind tag: {}", header[1]),
        )
    })?;
    let dim = u32::from_le_bytes([header[2], header[3], header[4], header[5]]) as usize;
    let n_vectors = u32::from_le_bytes([header[6], header[7], header[8], header[9]]) as usize;

    let packed_bytes = (dim / 8) * bits * n_vectors;
    let mut packed_codes = vec![0u8; packed_bytes];
    r.read_exact(&mut packed_codes)?;

    let mut norms_bytes = vec![0u8; n_vectors * 4];
    r.read_exact(&mut norms_bytes)?;
    let norms: Vec<f32> = norms_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok((bits, rotation, dim, n_vectors, packed_codes, norms))
}
