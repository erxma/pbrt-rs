use delegate::delegate;
use enum_dispatch::enum_dispatch;

use crate::{
    core::{
        gaussian, gaussian_integral, lerp, Array2D, Bounds2f, Bounds2i, Float, Point2f, Point2i,
        Vec2f,
    },
    sampling::routines::{sample_tent, PiecewiseConstant2D, PiecewiseConstant2DSample},
};

#[enum_dispatch]
#[derive(Clone, Debug)]
pub enum FilterEnum {
    Box(BoxFilter),
    Triangle(TriangleFilter),
    Gaussian(GaussianFilter),
}

impl FilterEnum {
    delegate! {
        #[through(Filter)]
        to self {
            pub fn radius(&self) -> Vec2f;
            pub fn integral(&self) -> Float;
            pub fn eval(&self, p: Point2f) -> Float;
            pub fn sample(&self, u: Point2f) -> FilterSample;
        }
    }
}

#[enum_dispatch(FilterEnum)]
pub trait Filter {
    fn radius(&self) -> Vec2f;
    fn eval(&self, p: Point2f) -> Float;
    fn integral(&self) -> Float;
    fn sample(&self, u: Point2f) -> FilterSample;
}

pub struct FilterSample {
    pub p: Point2f,
    pub weight: Float,
}

#[derive(Clone, Debug)]
pub struct BoxFilter {
    radius: Vec2f,
}

impl BoxFilter {
    pub fn new(radius: Vec2f) -> Self {
        Self { radius }
    }
}

impl Default for BoxFilter {
    fn default() -> Self {
        Self {
            radius: Vec2f::new(0.5, 0.5),
        }
    }
}

impl Filter for BoxFilter {
    fn radius(&self) -> Vec2f {
        self.radius
    }

    fn integral(&self) -> Float {
        2.0 * self.radius.x() * 2.0 * self.radius.y()
    }

    fn eval(&self, p: Point2f) -> Float {
        if p.x().abs() <= self.radius.x() && p.y().abs() <= self.radius.y() {
            1.0
        } else {
            0.0
        }
    }

    fn sample(&self, u: Point2f) -> FilterSample {
        let p = Point2f::new(
            lerp(-self.radius.x(), self.radius.x(), u[0]),
            lerp(-self.radius.y(), self.radius.y(), u[1]),
        );
        FilterSample { p, weight: 1.0 }
    }
}

#[derive(Clone, Debug)]
pub struct TriangleFilter {
    radius: Vec2f,
}

impl TriangleFilter {
    pub fn new(radius: Vec2f) -> Self {
        Self { radius }
    }
}

impl Filter for TriangleFilter {
    fn radius(&self) -> Vec2f {
        self.radius
    }

    fn eval(&self, p: Point2f) -> Float {
        (self.radius.x() - p.x().abs()).max(0.0) * (self.radius.y() - p.y().abs()).max(0.0)
    }

    fn integral(&self) -> Float {
        self.radius.x().powi(2) * self.radius.y().powi(2)
    }

    fn sample(&self, u: Point2f) -> FilterSample {
        let p = Point2f::new(
            sample_tent(u[0], self.radius.x()),
            sample_tent(u[1], self.radius.y()),
        );
        FilterSample { p, weight: 1.0 }
    }
}

/// Filter that applies a Gaussian bump centered at
/// the pixel and radially symmetric around it,
/// parameterized by a standard deviation and implicitly
/// centered at `0.0`.
#[derive(Clone, Debug)]
pub struct GaussianFilter {
    radius: Vec2f,
    /// Standard deviation.
    std: Float,
    /// Precomputed constant term for gauss(radius.x, 0, std)
    exp_x: Float,
    /// Precomputed constant term for gauss(radius.y, 0, std)
    exp_y: Float,
    sampler: FilterSampler,
}

impl GaussianFilter {
    pub fn new(radius: Vec2f, std: Float) -> Self {
        let exp_x = gaussian(radius.x(), 0.0, std);
        let exp_y = gaussian(radius.y(), 0.0, std);

        let mut result = Self {
            radius,
            std,
            exp_x,
            exp_y,
            sampler: FilterSampler::default(),
        };
        result.sampler = FilterSampler::new(&result);

        result
    }
}

impl Filter for GaussianFilter {
    fn radius(&self) -> Vec2f {
        self.radius
    }

    fn eval(&self, p: Point2f) -> Float {
        (gaussian(p.x(), 0.0, self.std) - self.exp_x).max(0.0)
            * (gaussian(p.y(), 0.0, self.std) - self.exp_y).max(0.0)
    }

    fn integral(&self) -> Float {
        let rad_x = self.radius.x();
        let rad_y = self.radius.y();
        (gaussian_integral(-rad_x, rad_x, 0.0, self.std) - 2.0 * rad_x * self.exp_x)
            * (gaussian_integral(-rad_y, rad_y, 0.0, self.std) - 2.0 * rad_y * self.exp_y)
    }

    fn sample(&self, u: Point2f) -> FilterSample {
        self.sampler.sample(u)
    }
}

#[derive(Debug)]
struct FilterSampler {
    domain: Bounds2f,
    vals: Array2D<Float>,
    distrib: PiecewiseConstant2D,
}

impl FilterSampler {
    const SAMPLING_RATE: usize = 32;

    pub fn new(filter: &impl Filter) -> Self {
        let domain = Bounds2f::new((-filter.radius()).into(), filter.radius().into());

        let val_bounds = Bounds2i::new(
            Point2i::ZERO,
            Point2i::new(
                (Self::SAMPLING_RATE as Float * filter.radius().x()) as i32,
                (Self::SAMPLING_RATE as Float * filter.radius().y()) as i32,
            ),
        );

        // Tabularize unnormalized filter function
        let mut vals = Array2D::fill_default(val_bounds);
        for idx in val_bounds {
            let p = domain.lerp(Point2f::new(
                (idx.x() as Float + 0.5) / vals.x_size() as Float,
                (idx.y() as Float + 0.5) / vals.y_size() as Float,
            ));
            vals[idx] = filter.eval(p);
        }

        // Compute sampling distribution
        let distrib = PiecewiseConstant2D::from_array2d(&vals, domain);

        Self {
            domain,
            vals,
            distrib,
        }
    }

    pub fn sample(&self, u: Point2f) -> FilterSample {
        let PiecewiseConstant2DSample { value, pdf, offset } = self.distrib.sample(u);
        FilterSample {
            p: value,
            weight: self.vals[offset] / pdf,
        }
    }
}

impl Clone for FilterSampler {
    fn clone(&self) -> Self {
        Self {
            domain: self.domain,
            // Safety: vals should never be mutated after construction anyway,
            // so reading for clone should always be safe
            vals: unsafe { self.vals.clone_unchecked() },
            distrib: self.distrib.clone(),
        }
    }
}

// Implemented because `new` requires the filter that needs it
// to itself be constructed first.
impl Default for FilterSampler {
    fn default() -> Self {
        Self {
            domain: Default::default(),
            vals: Array2D::fill_default(Bounds2i::default()),
            distrib: PiecewiseConstant2D::new(&[], 0, 0, Bounds2f::empty()),
        }
    }
}
