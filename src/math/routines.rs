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
    assert!((-1.0001..=1.0001).contains(&x));
    x.clamp(-1.0, 1.0).asin()
}

#[inline]
pub fn safe_acos(x: Float) -> Float {
    assert!((-1.0001..=1.0001).contains(&x));
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

#[inline]
pub fn difference_of_products<T>(a: T, b: T, c: T, d: T) -> T
where
    T: num_traits::Float,
{
    let cd = c * d;
    let diff_of_prods = a.mul_add(b, -cd);
    let error = (-c).mul_add(d, cd);
    diff_of_prods + error
}

macro_rules! inner_product {
    ($a: expr $(,)?) => {
        compile_error!("inner_product must receive an even number of arguments")
    };
    ($a:expr, $b:expr $(,)?) => {
        crate::math::CompensatedFloat::from_mul($a, $b)
    };
    ($a: expr, $b:expr, $($rest:expr),+ $(,)?) => {
        {
            let ab = crate::math::CompensatedFloat::from_mul($a, $b);
            let tp = inner_product!($($rest),+);
            let sum = crate::math::CompensatedFloat::from_add(ab.val, tp.val);
            crate::math::CompensatedFloat::new(sum.val, tp.err + sum.err)
        }
    }
}

pub(crate) use inner_product;

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
