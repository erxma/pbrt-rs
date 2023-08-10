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

#[inline]
pub fn safe_sqrt(x: Float) -> Float {
    assert!(x >= -1e-3);
    x.max(0.0).sqrt()
}

/// Evaluate a polynomial with variable `t` and `coefficients`,
/// where the first coefficient is the 0th degree (constant),
/// the second is for the 1st degree, and so on.
pub fn evaluate_polynomial(t: Float, coefficients: &[Float]) -> Float {
    coefficients.iter().rfold(0.0, |sum, &c| sum.mul_add(t, c))
}

#[cfg(test)]
mod test {
    use super::*;

    use approx::assert_relative_eq;

    use crate::Float;

    #[test]
    fn evaluate_polynomial_test() {
        let t: Float = 7.5;
        let coefficients = [0.4, 1.1, -0.22, 0.033];
        let expected = 0.033 * t.powi(3) - 0.22 * t.powi(2) + 1.1 * t + 0.4;
        assert_relative_eq!(expected, evaluate_polynomial(t, &coefficients));
    }
}
