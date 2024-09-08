use crate::{
    float::{FRAC_1_PI, FRAC_PI_2, FRAC_PI_4, INV_2_PI, INV_4_PI, PI, SQRT_2},
    geometry::Bounds2f,
    math::{
        self, find_interval, gaussian, lerp, next_float_down, safe_sqrt, Array2D, Point2f, Point2i,
        Point3f, Vec2f, Vec3f, ONE_MINUS_EPSILON,
    },
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
/// If `weights` is non-empty, returns:
///
/// - The index of one of the `weights` with probability proportional to its weight,
///   using `u` for the sample.
/// - The value of the PMF for the sample.
/// - A new uniform random sample derived from `u`.
///
/// If `weights` is empty, returns `None`.
///
/// The set of weights does not need to be normalized.
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

/// Compute the probability density function (PDF) for a linear distribution,
/// defined with `[a, b]` as the endpoint values, normalized so that the total probability is 1,
/// where `x` is the input value in `[0.0, 1.0]` to evaluate at.
///
/// If `x` is not in `[0.0, 1.0]`, returns 0.0.
#[inline]
pub fn linear_pdf(x: Float, a: Float, b: Float) -> Float {
    if !(0.0..=1.0).contains(&x) {
        0.0
    } else {
        2.0 * lerp(a, b, x) / (a + b)
    }
}

/// Sample the probability density function (PDF) for a linear distribution,
/// defined with `[a, b]` as the endpoint values, normalized so that the total probability is 1,
/// using random variable `u` in `[0, 1)`.
///
/// Returns value in `[0.0, 1.0)`.
#[inline]
pub fn sample_linear(u: Float, a: Float, b: Float) -> Float {
    if u == 0.0 || a == 0.0 {
        return 0.0;
    }

    let x = u * (a + b) / (a + lerp(a * a, b * b, u).sqrt());
    // Ensure < 1.0
    x.min(ONE_MINUS_EPSILON)
}

/// For a 1D linear function between `a` and `b`, return the random sample
/// that corresponds to the value `x` (equivalent to evaluating the CDF).
#[inline]
pub fn invert_linear_sample(x: Float, a: Float, b: Float) -> Float {
    x * (a * (2.0 - x) + b * x) / (a + b)
}

#[inline]
pub fn tent_pdf(x: Float, r: Float) -> Float {
    if x.abs() < r {
        1.0 / r - x.abs() / r.sqrt()
    } else {
        0.0
    }
}

/// Samples the [tent/trianglular function](https://en.wikipedia.org/wiki/Triangular_function)
/// with a given uniform random sample `u`, within the interval `[-r, r]`.
#[inline]
pub fn sample_tent(u: Float, r: Float) -> Float {
    // Pick negative (idx 0) or positive (idx 1) half with probs 0.5.
    // "This property is helpful for preserving well-distributed sample points
    // (e.g., if they have low discrepancy)."
    let (idx, _, u) = sample_discrete(&[0.5, 0.5], u).unwrap();
    if idx == 0 {
        // Negative
        -r + r * sample_linear(u, 0.0, 1.0)
    } else {
        // Positive
        r * sample_linear(u, 1.0, 0.0)
    }
}

pub fn invert_tent_sample(x: Float, r: Float) -> Float {
    if x <= 0.0 {
        (1.0 - invert_linear_sample(-x / r, 1.0, 0.0)) / 2.0
    } else {
        0.5 + invert_linear_sample(x / r, 1.0, 0.0) / 2.0
    }
}

#[inline]
pub fn exponential_pdf(x: Float, a: Float) -> Float {
    a * (-a * x).exp()
}

#[inline]
pub fn sample_exponential(u: Float, a: Float) -> Float {
    -(1.0 - u).ln() / a
}

pub fn invert_exponential_sample(x: Float, a: Float) -> Float {
    1.0 - (-a * x).exp()
}

#[inline]
pub fn normal_pdf(x: Float, mu: Float, sigma: Float) -> Float {
    gaussian(x, mu, sigma)
}

#[inline]
pub fn std_normal_pdf(x: Float) -> Float {
    normal_pdf(x, 0.0, 1.0)
}

#[inline]
pub fn sample_normal(u: Float, mu: Float, sigma: Float) -> Float {
    mu + SQRT_2 * sigma * math::erf_inv(2.0 * u - 1.0)
}

#[inline]
pub fn sample_std_normal(u: Float) -> Float {
    sample_normal(u, 0.0, 1.0)
}

#[inline]
pub fn invert_normal_sample(x: Float, mu: Float, sigma: Float) -> Float {
    #[cfg(not(feature = "use-f64"))]
    return 0.5 * (1.0 + math::erff((x - mu) / (sigma * SQRT_2)));
    #[cfg(feature = "use-f64")]
    return 0.5 * (1.0 + math::erf((x - mu) / (sigma * SQRT_2)));
}

#[inline]
pub fn invert_std_normal_sample(x: Float) -> Float {
    invert_normal_sample(x, 0.0, 1.0)
}

#[inline]
pub fn sample_two_normal(u: Point2f, mu: Float, sigma: Float) -> Point2f {
    let r2 = -2.0 * (1.0 - u.x()).ln();
    Point2f::new(
        mu + sigma * (r2 * (2.0 * PI * u.y()).cos()).sqrt(),
        mu + sigma * (r2 * (2.0 * PI * u.y()).sin()).sqrt(),
    )
}

#[inline]
pub fn sample_two_std_normal(u: Point2f) -> Point2f {
    sample_two_normal(u, 0.0, 1.0)
}

#[inline]
pub fn logistic_pdf(mut x: Float, s: Float) -> Float {
    x = x.abs();
    (-x / s).exp() / (s * (1.0 + (-x / s).exp()).sqrt())
}

#[inline]
pub fn sample_logistic(u: Float, s: Float) -> Float {
    -s * (1.0 / u - 1.0).ln()
}

#[inline]
pub fn invert_logistic_sample(x: Float, s: Float) -> Float {
    1.0 / (1.0 + (-x / s).exp())
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

#[inline]
pub fn invert_bilinear(_p: Point2f, _verts: &[Point2f]) -> Point2f {
    todo!()
}

pub const UNIFORM_SPHERE_PDF: Float = INV_4_PI;

#[inline]
pub fn sample_uniform_sphere(u: Point2f) -> Vec3f {
    let z = 1.0 - 2.0 * u.x();
    let r = safe_sqrt(1.0 - z * z);
    let phi = 2.0 * PI * u.y();
    Vec3f::new(r * phi.cos(), r * phi.sin(), z)
}

pub const UNIFORM_HEMISPHERE_PDF: Float = INV_2_PI;

#[inline]
pub fn sample_uniform_hemisphere(u: Point2f) -> Vec3f {
    let z = u[0];
    let r = safe_sqrt(1.0 - z * z);
    let phi = 2.0 * PI * u[1];
    Vec3f::new(r * phi.cos(), r * phi.sin(), z)
}

#[inline]
pub fn sample_uniform_disk_concentric(u: Point2f) -> Point2f {
    // Map u to [-1, 1]^2 and handle degeneracy at origin
    let u_offset = 2.0 * u - Vec2f::new(1.0, 1.0);
    if u_offset == Point2f::ZERO {
        return Point2f::ZERO;
    }

    // Apply concentric mapping to point
    let r;
    let theta;
    if u_offset.x().abs() > u_offset.y().abs() {
        r = u_offset.x();
        theta = FRAC_PI_4 * (u_offset.y() / u_offset.x());
    } else {
        r = u_offset.y();
        theta = FRAC_PI_2 - FRAC_PI_4 * (u_offset.x() / u_offset.y());
    }

    r * Point2f::new(theta.cos(), theta.sin())
}

#[inline]
pub fn sample_uniform_disk_polar(u: Point2f) -> Point2f {
    let r = u[0].sqrt();
    let theta = 2.0 * PI * u[1];
    Point2f::new(r * theta.cos(), r * theta.sin())
}

#[inline]
pub fn sample_spherical_rectangle(
    _p_ref: Point3f,
    _s: Point3f,
    _ex: Vec3f,
    _ey: Vec3f,
    _u: Point2f,
) -> (Point3f, Float) {
    todo!()
}

#[inline]
pub fn invert_spherical_rectangle_sample(
    _p_ref: Point3f,
    _s: Point3f,
    _ex: Vec3f,
    _ey: Vec3f,
    _p_rect: Point3f,
) -> Point2f {
    todo!()
}

#[inline]
pub fn sample_cosine_hemisphere(u: Point2f) -> Vec3f {
    let d = sample_uniform_disk_concentric(u);
    let z = safe_sqrt(1.0 - d.x() * d.x() - d.y() * d.y());
    Vec3f::new(d.x(), d.y(), z)
}

#[inline]
pub fn cosine_hemisphere_pdf(cos_theta: Float) -> Float {
    cos_theta * FRAC_1_PI
}

#[derive(Clone, Debug)]
pub struct PiecewiseConstant1D {
    func_vals: Vec<Float>,
    /// Cumulative distribution function.
    cdf: Vec<Float>,
    min: Float,
    max: Float,
    integral: Float,
}

impl PiecewiseConstant1D {
    pub fn new(func_vals: impl Into<Vec<Float>>, min: Float, max: Float) -> Self {
        let mut func_vals = func_vals.into();

        // Take abs vals of func vals
        for val in func_vals.iter_mut() {
            *val = val.abs();
        }

        // Compute integral of step function at x_i
        let n = func_vals.len();
        let mut cdf = Vec::with_capacity(n + 1);
        cdf.push(0.0);
        for i in 1..=n {
            cdf.push(cdf[i - 1] + func_vals[i - 1] * (max - min) / n as Float);
        }

        // Then transform step function integral into CDF
        let integral = cdf[n];
        if integral == 0.0 {
            for (i, val) in cdf.iter_mut().enumerate().take(n + 1).skip(1) {
                *val = i as Float / n as Float;
            }
        } else {
            for val in cdf.iter_mut() {
                *val /= integral;
            }
        }

        Self {
            func_vals,
            cdf,
            min,
            max,
            integral,
        }
    }

    pub fn sample(&self, u: Float) -> PiecewiseConstant1DSample {
        // Find surrounding CDF segments and offset
        let offset = find_interval(self.cdf.len(), |i| self.cdf[i] <= u).unwrap();

        // Compute offset along CDF segment
        let mut du = u - self.cdf[offset];
        if self.cdf[offset + 1] - self.cdf[offset] > 0.0 {
            du /= self.cdf[offset + 1] - self.cdf[offset];
        }

        // Compute PDF for sampled offset
        let pdf = if self.integral > 0.0 {
            self.func_vals[offset] / self.integral
        } else {
            0.0
        };

        // Return x corresponding to sample
        let value = lerp(
            self.min,
            self.max,
            (offset as Float + du) / self.size() as Float,
        );

        PiecewiseConstant1DSample { value, pdf, offset }
    }

    pub fn invert(&self, _x: Float) -> Option<Float> {
        todo!()
    }

    pub fn integral(&self) -> Float {
        self.integral
    }

    pub fn size(&self) -> usize {
        self.func_vals.len()
    }
}

#[derive(Clone, Debug)]
pub struct PiecewiseConstant2D {
    _domain: Bounds2f,
    prob_conditional_v: Vec<PiecewiseConstant1D>,
    prob_marginal: PiecewiseConstant1D,
}

impl PiecewiseConstant2D {
    pub fn new(func_vals: &[Float], num_u: usize, num_v: usize, domain: Bounds2f) -> Self {
        // Compute conditional sampling distribution for v
        let prob_conditional_v: Vec<_> = (0..num_v)
            .map(|v| {
                PiecewiseConstant1D::new(
                    &func_vals[v * num_u..(v + 1) * num_u],
                    domain.p_min[0],
                    domain.p_max[0],
                )
            })
            .collect();

        // Compute marginal sampling distribution p[v]
        let marginal_func: Vec<_> = prob_conditional_v
            .iter()
            .map(|distrib| distrib.integral())
            .collect();
        let prob_marginal =
            PiecewiseConstant1D::new(marginal_func, domain.p_min[1], domain.p_max[1]);

        Self {
            _domain: domain,
            prob_conditional_v,
            prob_marginal,
        }
    }

    pub fn from_array2d(func_vals: &Array2D<Float>, domain: Bounds2f) -> Self {
        Self::new(
            func_vals.as_slice(),
            func_vals.x_size(),
            func_vals.y_size(),
            domain,
        )
    }

    pub fn sample(&self, u: Point2f) -> PiecewiseConstant2DSample {
        let PiecewiseConstant1DSample {
            value: d1,
            pdf: pdf1,
            offset: v,
        } = self.prob_marginal.sample(u[1]);
        let PiecewiseConstant1DSample {
            value: d0,
            pdf: pdf0,
            offset: u,
        } = self.prob_conditional_v[v].sample(u[0]);

        let value = Point2f::new(d0, d1);
        let pdf = pdf0 * pdf1;
        let offset = Point2i::new(u as i32, v as i32);

        PiecewiseConstant2DSample { value, pdf, offset }
    }

    pub fn pdf(&self, _pr: Point2f) -> Float {
        todo!()
    }

    pub fn invert(&self, _p: Point2f) -> Option<Point2f> {
        todo!()
    }

    pub fn integral(&self) -> Float {
        self.prob_marginal.integral()
    }
}

#[derive(Clone, Debug)]
pub struct PiecewiseConstant1DSample {
    pub value: Float,
    pub pdf: Float,
    pub offset: usize,
}

#[derive(Clone, Debug)]
pub struct PiecewiseConstant2DSample {
    pub value: Point2f,
    pub pdf: Float,
    pub offset: Point2i,
}
