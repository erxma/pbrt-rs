use std::ops::{Add, AddAssign};

use crate::Float;

pub const ONE_MINUS_EPSILON: Float = 1.0 - Float::EPSILON;

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

#[inline]
pub fn exponent(value: f32) -> i16 {
    const EXPONENT_MASK: u32 = 0x7F800000;
    ((value.to_bits() & EXPONENT_MASK) - 127) as i16
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CompensatedFloat {
    pub val: Float,
    pub err: Float,
}

impl CompensatedFloat {
    pub const ZERO: Self = Self::new(0.0, 0.0);

    pub const fn new(val: Float, err: Float) -> Self {
        Self { val, err }
    }

    pub fn from_mul(a: Float, b: Float) -> Self {
        let product = a * b;
        let err = a.mul_add(b, -product);
        Self::new(product, err)
    }

    pub fn from_add(a: Float, b: Float) -> Self {
        let sum = a + b;
        let delta = sum - a;
        let err = a - (sum - delta) + (b - delta);
        Self::new(sum, err)
    }

    pub fn compensated_val(self) -> Float {
        self.val + self.err
    }
}

impl Add<Float> for CompensatedFloat {
    type Output = Self;

    fn add(mut self, rhs: Float) -> Self::Output {
        self += rhs;
        self
    }
}

impl AddAssign<Float> for CompensatedFloat {
    fn add_assign(&mut self, rhs: Float) {
        let delta = rhs - self.err;
        let sum = self.val + delta;
        self.err = sum - self.val - delta;
        self.val = sum;
    }
}
