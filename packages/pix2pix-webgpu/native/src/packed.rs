//! Packed FP16 convolution weights interleaved with FP32 biases and folded normalization.
//! Pure Rust so the format can be unit tested on native targets independently of web-sys.

pub const EXPECTED_PACKED_FP16_BYTES: usize = 108_869_644;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WeightEncoding {
    Fp32,
    PackedFp16,
}

pub fn stored_bytes_for(
    label: &str,
    count_floats: usize,
    encoding: WeightEncoding,
) -> Option<usize> {
    let width = if encoding == WeightEncoding::PackedFp16 && label.ends_with("_weights") {
        2
    } else {
        4
    };
    count_floats.checked_mul(width)
}

pub fn is_packed_convolution(label: &str, encoding: WeightEncoding) -> bool {
    encoding == WeightEncoding::PackedFp16 && label.ends_with("_weights")
}

fn half_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mut fraction = (bits & 0x03ff) as u32;
    let mapped = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut new_exp = -14i32;
            while (fraction & 0x0400) == 0 {
                fraction <<= 1;
                new_exp -= 1;
            }
            sign | (((new_exp + 127) as u32) << 23) | ((fraction & 0x03ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | (((exponent - 15 + 127) as u32) << 23) | (fraction << 13),
    };
    f32::from_bits(mapped)
}

pub fn decode_fp16_chunk(source: &[u8], destination: &mut [u8]) -> Result<(), &'static str> {
    if source.len() % 2 != 0 || source.len().checked_mul(2) != Some(destination.len()) {
        return Err("FP16 chunk must expand to exactly two FP32 bytes per source byte");
    }
    for (input, output) in source.chunks_exact(2).zip(destination.chunks_exact_mut(4)) {
        let bits = u16::from_le_bytes([input[0], input[1]]);
        output.copy_from_slice(&half_bits_to_f32(bits).to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_layout_uses_half_precision_only_for_convolution_weights() {
        assert_eq!(EXPECTED_PACKED_FP16_BYTES, 108_869_644);
        assert_eq!(
            stored_bytes_for("d2_weights", 1024, WeightEncoding::PackedFp16),
            Some(2048)
        );
        assert_eq!(
            stored_bytes_for("d2_bias", 512, WeightEncoding::PackedFp16),
            Some(2048)
        );
        assert_eq!(
            stored_bytes_for("d2_norm_beta", 512, WeightEncoding::PackedFp16),
            Some(2048)
        );
        assert_eq!(
            stored_bytes_for("e1_weights", 3072, WeightEncoding::Fp32),
            Some(12288)
        );
    }

    #[test]
    fn fp16_decode_expands_le_bytes_and_rejects_invalid_sizes() {
        let src = [0x00, 0x3c, 0x00, 0xbc, 0x00, 0x38, 0x00, 0x00];
        let mut dst = [0u8; 16];
        decode_fp16_chunk(&src, &mut dst).unwrap();
        let floats: Vec<f32> = dst
            .chunks_exact(4)
            .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
            .collect();
        assert_eq!(floats, vec![1.0, -1.0, 0.5, 0.0]);
        assert!(decode_fp16_chunk(&src[..3], &mut dst).is_err());
        assert!(decode_fp16_chunk(&src, &mut dst[..8]).is_err());
    }

    #[test]
    fn converts_subnormals_and_infinity_exactly() {
        assert_eq!(half_bits_to_f32(0x0001).to_bits(), 0x3380_0000);
        assert_eq!(half_bits_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(half_bits_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(half_bits_to_f32(0x7e00).is_nan());
    }
}
