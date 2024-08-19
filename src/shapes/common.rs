use super::Sphere;
use crate::{
    geometry::{Bounds3f, DirectionCone, Ray, SampleInteraction, SurfaceInteraction},
    math::{next_float_down, next_float_up, Normal3f, Point2f, Point3f, Point3fi, Tuple, Vec3f},
    Float,
};
use enum_dispatch::enum_dispatch;

#[enum_dispatch]
#[derive(Clone, Debug)]
pub enum ShapeEnum {
    Sphere,
}

#[enum_dispatch(ShapeEnum)]
pub trait Shape {
    /// Axis-aligned bounding box (AABB) of this shape in rendering space.
    fn bounds(&self) -> Bounds3f;

    /// The range of this shape's surface normals.
    fn normal_bounds(&self) -> DirectionCone;

    /// Perform a ray-shape intersection test, returning the parametric distance
    /// along `ray` (within its range) and geometric info about the hit, if any.
    fn intersect(&self, ray: &Ray, t_max: Option<Float>) -> Option<ShapeIntersection>;

    /// Perform a ray-shape intersection test, only determining whether an intersection occurs.
    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool {
        // By default, use the full test.
        // This is likely wasteful, so a better method should be provided if possible.
        self.intersect(ray, t_max).is_some()
    }

    /// Surface area of this shape in rendering space.
    fn area(&self) -> Float;

    /// Sample a point on `self`'s surface using a distribution with respect to surface area.
    ///
    /// Returns the local geometric info about the point.
    fn sample(&self, u: Point2f) -> Option<ShapeSample>;

    /// Sample a point on `self`'s surface using a distribution
    /// with respect to solid angle from a reference `ctx` point.
    fn sample_with_context(&self, ctx: &ShapeSampleContext, u: Point2f) -> Option<ShapeSample>;

    /// Returns the probability density for sampling the specified point on the shape
    /// corresponding to the given `Interaction`.
    ///
    /// The provided point is assumed to actually be on the surface.
    fn pdf(&self, interaction: &SampleInteraction) -> Float;

    /// Returns the probability density for sampling the specified point on the shape
    /// such that the incident direction at a reference `ctx` point is `wi`.
    ///
    /// The provided point is assumed to actually be on the surface.
    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vec3f) -> Float;
}

#[derive(Debug)]
pub struct ShapeIntersection {
    pub intr: SurfaceInteraction,
    pub t_hit: Float,
}

#[derive(Debug)]
pub struct ShapeSample {
    pub intr: SampleInteraction,
    pub pdf: Float,
}

#[derive(Clone, Debug)]
pub struct ShapeSampleContext {
    pub pi: Point3fi,
    pub n: Option<Normal3f>,
    pub ns: Option<Normal3f>,
    pub time: Float,
}

impl ShapeSampleContext {
    pub fn new(pi: Point3fi, n: Option<Normal3f>, ns: Option<Normal3f>, time: Float) -> Self {
        Self { pi, n, ns, time }
    }

    pub fn from_surface_interaction(si: &SurfaceInteraction) -> Self {
        Self {
            pi: si.pi,
            n: Some(si.n),
            ns: Some(si.shading.n),
            time: si.time,
        }
    }

    pub fn offset_ray_origin(&self, w: Vec3f) -> Point3f {
        let n_as_v = Vec3f::from(self.n.unwrap());
        // Find vector offset to corner of error bounds, compute initial po
        let d = n_as_v.abs().dot(self.pi.error());
        let mut offset = d * n_as_v;
        if w.dot(n_as_v) < 0.0 {
            offset *= -1.0;
        }
        let mut po = self.pi.midpoints_only() + offset;

        // Round offset point po away from p
        for i in 0..3 {
            if offset[i] > 0.0 {
                po[i] = next_float_up(po[i]);
            } else if offset[i] < 0.0 {
                po[i] = next_float_down(po[i]);
            }
        }

        po
    }

    pub fn spawn_ray(&self, w: Vec3f) -> Ray {
        Ray::new(self.offset_ray_origin(w), w, self.time, None)
    }
}

/// Information about an intersection on a quadric surface.
pub struct QuadricIntersection {
    /// The parametric t along the ray where the intersection occurred.
    pub t_hit: Float,
    /// The object space intersection point.
    pub p_obj: Point3f,
    /// The phi value of the quadric at the intersection.
    pub phi: Float,
}
