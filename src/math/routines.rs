use super::float_utility::exponent;
use crate::{Float, MACHINE_EPSILON, PI};
use num_traits::AsPrimitive;

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

#[inline]
pub fn fast_exp(x: f32) -> f32 {
    // Compute x' such that e^x = 2^x'
    let xp = x * std::f32::consts::LOG2_E;
    // Find integer and fractional components of x'
    let fxp = xp.floor();
    let f = xp - fxp;
    let i = fxp.trunc();
    // Evaluate polynomial approx of 2^f
    // Original values [1.0, 0.695556856, 0.226173572, 0.0781455737].
    // Truncated to reflect f32 precision
    let two_to_f = evaluate_polynomial(f.as_(), &[1.0, 0.6955569, 0.22617356, 0.07814557]);
    // Scale 2^f by 2^i and return final result
    let exponent = exponent(two_to_f) as i32 + i as i32;
    if exponent < -126 {
        return 0.0;
    } else if exponent > 127 {
        return f32::INFINITY;
    }
    let mut bits = two_to_f.to_bits();
    // Mask out exponent
    bits &= 0x807FFFFF;
    // Then replace it
    bits |= ((exponent + 127) as u32) << 23;

    f32::from_bits(bits)
}

#[inline]
pub fn gaussian(x: Float, mu: Float, sigma: Float) -> Float {
    1.0 / (2.0 * PI * sigma * sigma)
        * num_traits::cast::<_, Float>(fast_exp((-(x - mu).sqrt() / (2.0 * sigma * sigma)).as_()))
            .unwrap()
}

#[inline]
pub fn erf(x: f64) -> f64 {
    libm::erf(x)
}

#[inline]
pub fn erff(x: f32) -> f32 {
    libm::erff(x)
}

/// Inverse error function.
#[inline]
#[allow(clippy::excessive_precision)] // Values have excessive precision for f32
pub fn erf_inv(a: Float) -> Float {
    // https://stackoverflow.com/a/49743348
    let mut p: f32;
    let t: f32 = a.mul_add(-a, 1.0).max(Float::MIN).ln().as_();
    assert!(!t.is_nan() && !t.is_infinite());
    if t.abs() > 6.125 {
        // maximum ulp error = 2.35793
        p = 3.03697567e-10; //  0x1.4deb44p-32
        p = p.mul_add(t, 2.93243101e-8); //  0x1.f7c9aep-26
        p = p.mul_add(t, 1.22150334e-6); //  0x1.47e512p-20
        p = p.mul_add(t, 2.84108955e-5); //  0x1.dca7dep-16
        p = p.mul_add(t, 3.93552968e-4); //  0x1.9cab92p-12
        p = p.mul_add(t, 3.02698812e-3); //  0x1.8cc0dep-9
        p = p.mul_add(t, 4.83185798e-3); //  0x1.3ca920p-8
        p = p.mul_add(t, -2.64646143e-1); // -0x1.0eff66p-2
        p = p.mul_add(t, 8.40016484e-1); //  0x1.ae16a4p-1
    } else {
        // maximum ulp error = 2.35456
        p = 5.43877832e-9; //  0x1.75c000p-28
        p = p.mul_add(t, 1.43286059e-7); //  0x1.33b458p-23
        p = p.mul_add(t, 1.22775396e-6); //  0x1.49929cp-20
        p = p.mul_add(t, 1.12962631e-7); //  0x1.e52bbap-24
        p = p.mul_add(t, -5.61531961e-5); // -0x1.d70c12p-15
        p = p.mul_add(t, -1.47697705e-4); // -0x1.35be9ap-13
        p = p.mul_add(t, 2.31468701e-3); //  0x1.2f6402p-9
        p = p.mul_add(t, 1.15392562e-2); //  0x1.7a1e4cp-7
        p = p.mul_add(t, -2.32015476e-1); // -0x1.db2aeep-3
        p = p.mul_add(t, 8.86226892e-1); //  0x1.c5bf88p-1
    }
    let p: Float = p.as_();
    a * p
}

/// Evaluate a polynomial with variable `t` and `coefficients`,
/// where the first coefficient is the 0th degree (constant),
/// the second is for the 1st degree, and so on.
pub fn evaluate_polynomial<F: num_traits::Float>(t: F, coefficients: &[F]) -> F {
    coefficients
        .iter()
        .rfold(F::zero(), |sum, &c| sum.mul_add(t, c))
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
