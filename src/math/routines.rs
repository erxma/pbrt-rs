use crate::{Float, MACHINE_EPSILON};

#[inline]
pub fn lerp(v1: Float, v2: Float, t: Float) -> Float {
    (1.0 - t) * v1 + t * v2
}

pub fn gamma(n: i32) -> Float {
    let n = n as Float;
    (MACHINE_EPSILON * n) / (1.0 - n * MACHINE_EPSILON)
}

#[inline]
pub fn safe_asin(x: Float) -> Float {
    assert!(x >= -1.0001 && x <= 1.0001);
    x.clamp(-1.0, 1.0).asin()
}

#[inline]
pub fn safe_acos(x: Float) -> Float {
    assert!(x >= -1.0001 && x <= 1.0001);
    x.clamp(-1.0, 1.0).acos()
}
