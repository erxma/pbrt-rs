use std::fmt::Debug;

use crate::{
    geometry::{
        bounds3::Bounds3f,
        direction_cone::DirectionCone,
        interaction::{Interaction, SurfaceInteraction},
        ray::Ray,
        transform::Transform,
    },
    math::{
        normal3::Normal3f,
        point::{Point2f, Point3fi},
        vec3::Vec3f,
    },
    Float,
};

pub trait Shape: Debug {
    fn object_bound(&self) -> Bounds3f;
    fn world_bound(&self) -> Bounds3f {
        self.object_to_world() * self.object_bound()
    }
    fn normal_bounds(&self) -> DirectionCone;

    /// Perform a ray-shape intersection test, returning the parametric distance
    /// along `ray` (within its range) and geometric info about the hit, if any.
    fn intersect(&self, ray: Ray, test_alpha_texture: bool) -> Option<ShapeIntersection>;

    /// Perform a ray-shape intersection test, only determining whether an intersection occurs.
    fn intersect_p(&self, ray: Ray, test_alpha_texture: bool) -> bool {
        // By default, use the full test.
        // This is likely wasteful, so a better method should be provided if possible.
        self.intersect(ray, test_alpha_texture).is_some()
    }

    fn area(&self) -> Float;

    fn object_to_world(&self) -> &Transform;
    fn world_to_object(&self) -> &Transform;
    fn reverse_orientation(&self) -> bool;
    fn transform_swaps_handedness(&self) -> bool {
        self.object_to_world().swaps_handedness()
    }

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
pub struct ShapeIntersection<'a> {
    pub intr: SurfaceInteraction<'a>,
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
