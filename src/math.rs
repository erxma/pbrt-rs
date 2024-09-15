use num_traits::{AsPrimitive, Pow};

use crate::core::{
    constants::{PI, SQRT_2},
    erf, fast_exp, Float,
};

/// The [Gaussian function](https://en.wikipedia.org/wiki/Gaussian_function).
#[inline]
pub fn gaussian(x: Float, mean: Float, std: Float) -> Float {
    let exp_factor: Float = fast_exp((-(x - mean).pow(2i32) / (2.0 * std * std)).as_()).as_();
    1.0 / (2.0 * PI * std * std).sqrt() * exp_factor
}

/// Integral of the Gaussian function over a range `[x_from, x_to]`.
#[inline]
pub fn gaussian_integral(x_from: Float, x_to: Float, mean: Float, std: Float) -> Float {
    let std_root2 = std * SQRT_2;
    0.5 * erf((mean - x_from) / std_root2) - erf((mean - x_to) / std_root2)
}

/// Evaluate a polynomial with variable `t` and `coefficients`,
/// where the first coefficient is the 0th degree (constant),
/// the second is for the 1st degree, and so on.
pub fn evaluate_polynomial<F: num_traits::Float>(t: F, coefficients: &[F]) -> F {
    coefficients
        .iter()
        .rfold(F::zero(), |sum, &c| sum.mul_add(t, c))
}

/// Solve a quadratic equation `at^2 + bt + c = 0` for values of `t`.
///
/// - If two solutions, they are returned as `(lower, higher)`.
///
/// - If one solution, it is returned for both values.
///
/// - If no solutions, returns `None`.
///
/// # Panics
/// If all coefficients are zero, as there are infinite solutions.
#[inline]
pub fn solve_quadratic(a: Float, b: Float, c: Float) -> Option<(Float, Float)> {
    // Panic on all zero, would have infinite solutions
    assert!(
        a != 0.0 || b != 0.0 || c != 0.0,
        "solve_quadratic shouldn't be used would all-zero coefficients, as there are infinite solutions"
    );

    // Handle case of a = 0 (actually linear)
    if a == 0.0 {
        if b == 0.0 {
            return None;
        } else {
            let t = -c / b;
            return Some((t, t));
        }
    }

    let discrim = difference_of_products(b, b, 4.0 * a, c);
    // If discrim is neg, no real roots
    if discrim < 0.0 {
        return None;
    }
    let root_discrim = discrim.sqrt();

    // Compute ts
    let q = -0.5 * (b + root_discrim.copysign(b));
    let t0 = q / a;
    let t1 = c / q;

    if t0 <= t1 {
        Some((t0, t1))
    } else {
        Some((t1, t0))
    }
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
        crate::core::CompensatedFloat::from_mul($a, $b)
    };
    ($a: expr, $b:expr, $($rest:expr),+ $(,)?) => {
        {
            let ab = crate::core::CompensatedFloat::from_mul($a, $b);
            let tp = crate::math::inner_product!($($rest),+);
            let sum = crate::core::CompensatedFloat::from_add(ab.val, tp.val);
            crate::core::CompensatedFloat::new(sum.val, tp.err + sum.err)
        }
    }
}
pub(crate) use inner_product;

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

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
}
