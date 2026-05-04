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
}
