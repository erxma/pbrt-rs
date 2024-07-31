use crate::{
    geometry::{
        bounds::Bounds3f,
        direction_cone::DirectionCone,
        interaction::{Interaction, SurfaceInteraction},
        ray::Ray,
        transform::Transform,
    },
    math::{Normal3f, Point2f, Point3f, Point3fi, Vec3f},
    Float,
};
use enum_dispatch::enum_dispatch;
use std::fmt::Debug;

#[enum_dispatch]
#[derive(Clone, Debug)]
pub enum ShapeEnum {
    Sphere,
}

#[enum_dispatch(ShapeEnum)]
pub trait Shape {
    fn bounds(&self) -> Bounds3f;

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

    fn area(&self) -> Float;

    fn sample(&self, u: Point2f) -> Option<ShapeSample>;

    fn sample_with_context(&self, ctx: &ShapeSampleContext, u: Point2f) -> Option<ShapeSample>;

    /// Returns the probability density for sampling the specified point on the shape
    /// corresponding to the given `Intersection`.
    ///
    /// The provided point is assumed to be on the surface.
    fn pdf(&self, interaction: &Interaction) -> Float;

    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vec3f) -> Float;
}

#[derive(Debug)]
pub struct ShapeIntersection {
    pub intr: SurfaceInteraction,
    pub t_hit: Float,
}

#[derive(Debug)]
pub struct ShapeSample<'a> {
    pub intr: Interaction<'a>,
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
            pi: si.common.pi,
            n: Some(si.n),
            ns: Some(si.shading.n),
            time: si.common.time,
        }
    }
}

pub struct QuadricIntersection {
    pub t_hit: Float,
    pub p_obj: Point3f,
    pub phi: Float,
}

#[derive(Clone, Debug)]
pub struct Sphere {
    radius: Float,
    z_min: Float,
    z_max: Float,
    theta_z_min: Float,
    theta_z_max: Float,
    phi_max: Float,
    render_from_object: Transform,
    object_from_render: Transform,
    reverse_orientation: bool,
    transform_swaps_handedness: bool,
}

impl Shape for Sphere {
    fn bounds(&self) -> Bounds3f {
        &self.render_from_object
            * Bounds3f::new(
                Point3f::new(-self.radius, -self.radius, self.z_min),
                Point3f::new(self.radius, self.radius, self.z_max),
            )
    }

    fn normal_bounds(&self) -> DirectionCone {
        DirectionCone::ENTIRE_SPHERE
    }

    fn intersect(&self, ray: &Ray, t_max: Option<Float>) -> Option<ShapeIntersection> {
        let isect = self.basic_intersect(ray, t_max);

        isect.map(|isect| {
            let intr = self.interaction_from_intersection(&isect, -ray.dir, ray.time);
            ShapeIntersection {
                intr,
                t_hit: isect.t_hit,
            }
        })
    }

    fn area(&self) -> Float {
        self.phi_max * self.radius * (self.z_max - self.z_min)
    }

    fn sample(&self, u: Point2f) -> Option<ShapeSample> {
        todo!()
    }

    fn sample_with_context(&self, ctx: &ShapeSampleContext, u: Point2f) -> Option<ShapeSample> {
        todo!()
    }

    fn pdf(&self, interaction: &Interaction) -> Float {
        todo!()
    }

    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vec3f) -> Float {
        todo!()
    }
}

impl Sphere {
    pub fn basic_intersect(&self, ray: &Ray, t_max: Option<Float>) -> Option<QuadricIntersection> {
        todo!()
    }

    pub fn interaction_from_intersection(
        &self,
        isect: &QuadricIntersection,
        wo: Vec3f,
        time: Float,
    ) -> SurfaceInteraction {
        todo!()
    }
}
