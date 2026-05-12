//! Bit-plane packing of quantized codes.
//!
//! Each coordinate is quantized to `bits` bits (2/3/4). For each vector we
//! store `bits` planes; each plane is a bit-vector of length `dim` packed MSB-
//! first into `dim/8` bytes. Plane `p` holds bit `p` of every coordinate.
//!
//! Bit-plane format makes per-coordinate dot-product accumulation natural for
//! both scalar and SIMD code paths (matches turbovec's on-disk layout).

/// Pack quantized codes (one byte per coordinate, value in `0..2^bits`) into
/// bit-plane format. Returns a flat `Vec<u8>` of length `n * bits * dim/8`.
pub fn pack_codes(codes: &[u8], n: usize, dim: usize, bits: usize) -> Vec<u8> {
    debug_assert_eq!(codes.len(), n * dim);
    debug_assert_eq!(dim % 8, 0, "dim must be a multiple of 8");

    let bytes_per_plane = dim / 8;
    let bytes_per_row = bits * bytes_per_plane;
    let mut packed = vec![0u8; n * bytes_per_row];

    for i in 0..n {
        for j in 0..dim {
            let code = codes[i * dim + j];
            let byte_pos = j / 8;
            let bit_pos = 7 - (j % 8);
            for p in 0..bits {
                if code & (1 << p) != 0 {
                    packed[i * bytes_per_row + p * bytes_per_plane + byte_pos] |= 1 << bit_pos;
                }
            }
        }
    }

    packed
}

/// Unpack bit-plane codes for a single vector back into per-coordinate codes.
/// Returns a `Vec<u8>` of length `dim`.
pub fn unpack_vector(packed: &[u8], vec_idx: usize, dim: usize, bits: usize) -> Vec<u8> {
    let bytes_per_plane = dim / 8;
    let bytes_per_row = bits * bytes_per_plane;
    let row = &packed[vec_idx * bytes_per_row..(vec_idx + 1) * bytes_per_row];

    let mut codes = vec![0u8; dim];
    for j in 0..dim {
        let byte_pos = j / 8;
        let bit_pos = 7 - (j % 8);
        let mask = 1u8 << bit_pos;
        let mut code = 0u8;
        for p in 0..bits {
            if row[p * bytes_per_plane + byte_pos] & mask != 0 {
                code |= 1 << p;
            }
        }
        codes[j] = code;
    }
    codes
}

// ─── SIMD-blocked layout (4-bit only) ────────────────────────────────────────
//
// Re-organizes bit-plane codes into a layout that lets one NEON `vqtbl1q_u8`
// instruction fetch the same coordinate-pair from 32 vectors in parallel.
//
// Layout (matches turbovec's NEON sequential layout exactly):
//
//   blocked[block_idx * n_byte_groups * BLOCK
//         + group_idx * BLOCK
//         + lane] = byte for vector (block_idx * BLOCK + lane), coord pair g
//
// where:
//   BLOCK         = 32
//   n_blocks      = ceil(n_vectors / BLOCK)
//   n_byte_groups = dim / 2     (two 4-bit codes per byte)
//   total_bytes   = n_blocks * n_byte_groups * BLOCK
//
// Each byte packs:
//   high nibble (bits 4..7) = code for coordinate (2 * g)
//   low  nibble (bits 0..3) = code for coordinate (2 * g + 1)
//
// Lanes past `n_vectors - 1` in the final block are padded with zeros; the
// scoring kernel masks them out at write-back via the `n_vectors` bound.

use crate::BLOCK;

/// Repack 4-bit bit-plane codes into the NEON-friendly blocked layout.
///
/// Returns `(blocked_bytes, n_blocks)`. Caller is responsible for matching
/// the layout assumptions in `search_neon::score_4bit_block_neon`.
pub fn repack_4bit(packed_codes: &[u8], n_vectors: usize, dim: usize) -> (Vec<u8>, usize) {
    debug_assert_eq!(dim % 8, 0, "dim must be a multiple of 8");
    let bits = 4usize;
    let bytes_per_plane = dim / 8;
    let bytes_per_row = bits * bytes_per_plane;
    let n_byte_groups = dim / 2;
    let n_blocks = n_vectors.div_ceil(BLOCK);
    let mut blocked = vec![0u8; n_blocks * n_byte_groups * BLOCK];

    // First decode each vector's bit-planes into per-coord codes, then pack
    // pairs into nibbles and write into the blocked slot. Doing the decode
    // up front (rather than re-deriving codes nibble by nibble) keeps this
    // straightforward at the cost of `dim` extra bytes per vector.
    let mut tmp_codes = vec![0u8; dim];

    for vec_idx in 0..n_vectors {
        // Bit-plane decode into tmp_codes.
        for j in 0..dim {
            let byte_pos = j / 8;
            let bit_pos = 7 - (j % 8);
            let mask = 1u8 << bit_pos;
            let mut code = 0u8;
            for p in 0..bits {
                if packed_codes[vec_idx * bytes_per_row + p * bytes_per_plane + byte_pos] & mask
                    != 0
                {
                    code |= 1 << p;
                }
            }
            tmp_codes[j] = code;
        }

        let block_idx = vec_idx / BLOCK;
        let lane = vec_idx % BLOCK;

        // Pack pairs of 4-bit codes into nibbles.
        for g in 0..n_byte_groups {
            let hi = tmp_codes[2 * g] & 0x0F;
            let lo = tmp_codes[2 * g + 1] & 0x0F;
            let byte_val = (hi << 4) | lo;
            let dst = block_idx * n_byte_groups * BLOCK + g * BLOCK + lane;
            blocked[dst] = byte_val;
        }
    }

    (blocked, n_blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for &bits in &[2, 3, 4] {
            let dim = 64;
            let n = 5;
            let max_code = (1u8 << bits) - 1;
            let codes: Vec<u8> = (0..n * dim).map(|i| (i as u8) & max_code).collect();
            let packed = pack_codes(&codes, n, dim, bits);
            assert_eq!(packed.len(), n * bits * dim / 8);
            for i in 0..n {
                let recovered = unpack_vector(&packed, i, dim, bits);
                assert_eq!(&recovered[..], &codes[i * dim..(i + 1) * dim]);
            }
        }
    }

    #[test]
    fn repack_4bit_layout() {
        // Build a known set of codes, pack them, repack, and verify the
        // blocked layout has the right nibble-packed bytes at the right
        // (block, group, lane) offsets.
        let dim = 64;
        let n = 35; // forces 2 blocks (32 + 3 padded)
        let codes: Vec<u8> = (0..n * dim).map(|i| (i as u8) & 0x0F).collect();
        let packed = pack_codes(&codes, n, dim, 4);
        let (blocked, n_blocks) = repack_4bit(&packed, n, dim);

        assert_eq!(n_blocks, 2);
        let n_byte_groups = dim / 2;
        assert_eq!(blocked.len(), n_blocks * n_byte_groups * BLOCK);

        for vec_idx in 0..n {
            let block = vec_idx / BLOCK;
            let lane = vec_idx % BLOCK;
            for g in 0..n_byte_groups {
                let want_hi = codes[vec_idx * dim + 2 * g] & 0x0F;
                let want_lo = codes[vec_idx * dim + 2 * g + 1] & 0x0F;
                let want = (want_hi << 4) | want_lo;
                let got = blocked[block * n_byte_groups * BLOCK + g * BLOCK + lane];
                assert_eq!(
                    got, want,
                    "vec {vec_idx} group {g}: got {got:#04x}, want {want:#04x}"
                );
            }
        }

        // Padded lanes (vec_idx >= n in the last block) should be zero.
        for lane in (n % BLOCK)..BLOCK {
            for g in 0..n_byte_groups {
                let got = blocked[n_byte_groups * BLOCK + g * BLOCK + lane];
                assert_eq!(got, 0, "padded lane {lane} group {g} should be zero");
            }
        }
    }
}
