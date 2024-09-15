use super::{BilinearPatch, Sphere};
use crate::core::{
    Bounds3f, DirectionCone, Float, Normal3f, Point2f, Point3f, Point3fi, Ray, SampleInteraction,
    SurfaceInteraction, Vec3f,
};
use enum_dispatch::enum_dispatch;

#[enum_dispatch]
#[derive(Clone, Debug)]
pub enum ShapeEnum {
    Sphere,
    BilinearPatch,
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
pub struct ShapeIntersection<'a> {
    pub intr: SurfaceInteraction<'a>,
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

    pub fn offset_ray_origin_with_dir(&self, dir: Vec3f) -> Point3f {
        Ray::offset_ray_origin(self.pi, self.n.unwrap(), dir)
    }

    pub fn offset_ray_origin_towards(&self, to_point: Point3f) -> Point3f {
        self.offset_ray_origin_with_dir(to_point - self.pi.midpoints())
    }

    pub fn spawn_ray_with_dir(&self, dir: Vec3f) -> Ray {
        Ray::spawn_with_dir(self.pi, self.n.unwrap(), self.time, dir)
    }
}

/// Information about an intersection on a quadric surface.
#[derive(Clone, Debug)]
pub struct QuadricIntersection {
    /// The parametric t along the ray where the intersection occurred.
    pub t_hit: Float,
    /// The object space intersection point.
    pub p_obj: Point3f,
    /// The phi value of the quadric at the intersection.
    pub phi: Float,
}
