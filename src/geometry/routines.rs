use crate::Float;

pub fn lerp(v1: Float, v2: Float, t: Float) -> Float {
    (1.0 - t) * v1 + t * v2
}
