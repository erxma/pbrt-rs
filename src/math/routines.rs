use std::ops::{Add, Mul};

use super::float_utility::exponent;
use crate::{
    float::{MACHINE_EPSILON, PI},
    Float,
};
use num_traits::{AsPrimitive, Pow};

#[inline]
pub fn lerp<T>(v1: T, v2: T, t: Float) -> T
where
    T: Mul<Float, Output = T> + Add<T, Output = T>,
{
    v1 * (1.0 - t) + v2 * t
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
    1.0 / (2.0 * PI * sigma * sigma).sqrt()
        * num_traits::cast::<_, Float>(fast_exp(
            (-(x - mu).pow(2i32) / (2.0 * sigma * sigma)).as_(),
        ))
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

#[inline]
pub fn encode_morton_3(x: f32, y: f32, z: f32) -> u32 {
    (left_shift_3(z.to_bits()) << 2) | (left_shift_3(y.to_bits()) << 1) | left_shift_3(x.to_bits())
}

#[inline]
fn left_shift_3(mut x: u32) -> u32 {
    if x == (1 << 10) {
        x -= 1;
    }
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    x = (x | (x << 8)) & 0b00000011000000001111000000001111;
    x = (x | (x << 4)) & 0b00000011000011000011000011000011;
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;

    x
}

pub(crate) use inner_product;

#[cfg(test)]
mod test {
    use super::*;

    use approx::assert_relative_eq;

    use crate::Float;

    #[test]
    fn test_fast_exp() {
        assert_relative_eq!(2.71828, fast_exp(1.0), max_relative = 1e-4);
        assert_relative_eq!(1.0, fast_exp(0.0));
        assert_relative_eq!(15.02927, fast_exp(2.71), max_relative = 1e-4);
        assert_relative_eq!(1.70067e-12, fast_exp(-27.1));
        assert_relative_eq!(2.14644e14, fast_exp(33.0), max_relative = 1e-4);
    }

    #[test]
    fn test_gaussian() {
        assert_relative_eq!(0.39894228, gaussian(0.0, 0.0, 1.0), max_relative = 1e-8);
        assert_relative_eq!(0.19418605, gaussian(-1.2, 0.0, 1.0), max_relative = 1e-6);
        assert_relative_eq!(
            0.00827709,
            gaussian(460.6, 444.4, 45.2),
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_erf_inv() {
        assert_relative_eq!(0.47693628, erf_inv(0.5), max_relative = 1e-5);
        assert_relative_eq!(0.0, erf_inv(0.0));
        assert_relative_eq!(-2.3267537655, erf_inv(-0.999), max_relative = 1e-5);
    }

    #[test]
    #[should_panic]
    fn test_erf_inv_panic_on_inf() {
        erf_inv(1.0);
    }

    #[test]
    #[should_panic]
    fn test_erf_inv_panic_on_nan() {
        erf_inv(Float::NAN);
    }

    #[test]
    fn test_evaluate_polynomial() {
        let t: Float = 7.5;
        let coefficients = [0.4, 1.1, -0.22, 0.033];
        let expected = 0.033 * t.powi(3) - 0.22 * t.powi(2) + 1.1 * t + 0.4;
        assert_relative_eq!(expected, evaluate_polynomial(t, &coefficients));
    }
}
