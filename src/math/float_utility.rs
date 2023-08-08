use crate::Float;

// Returns the least number greater than `v`.
#[inline]
pub fn next_float_up(v: Float) -> Float {
    // TODO: Available in nightly. Impl is near identical.
    // Return same for +Infinity or NaN
    if v.is_infinite() && v > 0.0 || v.is_nan() {
        return v;
    }

    let bits = v.to_bits();
    // Return 0x1 for -0.0 or +0.0.
    // Otherwise, bump the bits
    let next_bits = if v == 0.0 {
        0x1
    } else if v.is_sign_positive() {
        bits + 1
    } else {
        bits - 1
    };

    Float::from_bits(next_bits)
}

// Returns the greatest number less than `v`.
#[inline]
pub fn next_float_down(mut v: Float) -> Float {
    // TODO: Available in nightly. Impl is near identical.
    // Return same for -Infinity or NaN
    if v.is_infinite() && v < 0.0 || v.is_nan() {
        return v;
    }
    // Change -0.0 to just 0.0
    if v == -0.0 {
        v = 0.0;
    }

    let bits = v.to_bits();
    // Bump the bits
    let next_bits = if v.is_sign_positive() {
        bits - 1
    } else {
        bits + 1
    };

    Float::from_bits(next_bits)
}
