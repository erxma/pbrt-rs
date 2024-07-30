use crate::{
    math::{lerp, next_float_down, Point2f, ONE_MINUS_EPSILON},
    Float,
};

#[inline]
pub fn balance_heuristic(nf: i32, f_pdf: Float, ng: i32, g_pdf: Float) -> Float {
    let nf = nf as Float;
    let ng = ng as Float;
    (nf * f_pdf) / (nf * f_pdf + ng * g_pdf)
}

#[inline]
pub fn power_heuristic(nf: i32, f_pdf: Float, ng: i32, g_pdf: Float) -> Float {
    let f = nf as Float * f_pdf;
    let g = ng as Float * g_pdf;
    // Hardcoded beta = 2
    f.powi(2) / (f + g).powi(2)
}

/// Samples a discrete distribution with a given uniform random sample `u`.
///
/// If `weights` is empty, returns None.
///
/// Otherwise, returns:
///
/// - The index of one of the `weights` with probability proportional to its weight,
///   using `u` for the sample.
/// - The value of the PMF for the sample.
/// - A new uniform random sample derived from `u`.
#[inline]
pub fn sample_discrete(weights: &[Float], u: Float) -> Option<(usize, Float, Float)> {
    // Handle empty weights for discrete sampling
    if weights.is_empty() {
        return None;
    }

    // Compute sum of weights
    let sum_weights: Float = weights.iter().sum();

    // Compute rescaled u sample
    let mut up = u * sum_weights;
    if up == sum_weights {
        up = next_float_down(up);
    }

    // Find offset in weights corresponding to u'
    let mut offset = 0;
    let mut sum = 0.0;
    while sum + weights[offset] <= up {
        sum += weights[offset];
        offset += 1;
    }

    // Compute PMF and remapped u value
    let pmf = weights[offset] / sum_weights;
    let u_remapped = ((up - sum) / weights[offset]).min(ONE_MINUS_EPSILON);

    Some((offset, pmf, u_remapped))
}

#[inline]
pub fn linear_pdf(x: Float, a: Float, b: Float) -> Float {
    if !(0.0..=1.0).contains(&x) {
        0.0
    } else {
        2.0 * lerp(x, a, b) / (a + b)
    }
}

#[inline]
pub fn sample_linear(u: Float, a: Float, b: Float) -> Float {
    if u == 0.0 || a == 0.0 {
        return 0.0;
    }

    let x = u * (a + b) / (a + lerp(u, a * a, b * b).sqrt());
    // Ensure < 1.0
    x.min(ONE_MINUS_EPSILON)
}

/// For a 1D function, return the random sample that corresponds to the value `x`.
#[inline]
pub fn invert_linear_sample(x: Float, a: Float, b: Float) -> Float {
    x * (a * (2.0 - x) + b * x) / (a + b)
}

#[inline]
pub fn bilinear_pdf(p: Point2f, w: &[Float]) -> Float {
    let w_sum: Float = w.iter().sum();

    if p.x() < 0.0 || p.x() > 1.0 || p.y() < 0.0 || p.y() > 1.0 {
        0.0
    } else if w_sum == 0.0 {
        1.0
    } else {
        4.0 * ((1.0 - p.x()) * (1.0 - p.y()) * w[0]
            + p.x() * (1.0 - p.y()) * w[1]
            + p.y() * (1.0 - p.x()) * w[2]
            + p.x() * p.y() * w[3])
            / w_sum
    }
}

#[inline]
pub fn sample_bilinear(u: Point2f, w: &[Float]) -> Point2f {
    // Sample y for bilnear marginal distribution
    let y = sample_linear(u.y(), w[0] + w[1], w[2] + w[3]);
    let x = sample_linear(u.x(), lerp(y, w[0], w[2]), lerp(y, w[1], w[3]));

    Point2f::new(x, y)
}

#[inline]
pub fn invert_bilinear_sample(p: Point2f, w: &[Float]) -> Point2f {
    let x = invert_linear_sample(p.x(), lerp(p.y(), w[0], w[2]), lerp(p.y(), w[1], w[3]));
    let y = invert_linear_sample(p.y(), w[0] + w[1], w[2] + w[3]);

    Point2f::new(x, y)
}
