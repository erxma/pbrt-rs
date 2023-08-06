use crate as pbrt;

use super::{
    bounds3::Bounds3f, ray::Ray, surface_interaction::SurfaceInteraction, transform::Transform,
};

pub trait Shape<'a> {
    fn object_bound(&self) -> Bounds3f;
    fn world_bound(&self) -> Bounds3f {
        self.object_to_world() * self.object_bound()
    }

    /// Perform a ray-shape intersection test, returning the parametric distance
    /// along `ray` (within its range) and geometric info about the hit, if any.
    fn intersect(
        &self,
        ray: Ray,
        test_alpha_texture: bool,
    ) -> Option<(pbrt::Float, SurfaceInteraction)>;

    /// Perform a ray-shape intersection test, only determining whether an intersection occurs.
    fn intersect_p(&self, ray: Ray, test_alpha_texture: bool) -> bool {
        // By default, use the full test.
        // This is likely wasteful, so a better method should be provided if possible.
        self.intersect(ray, test_alpha_texture).is_some()
    }

    fn area(&self) -> pbrt::Float;

    fn object_to_world(&self) -> &'a Transform;
    fn world_to_object(&self) -> &'a Transform;
    fn reverse_orientation(&self) -> bool;
    fn transform_swaps_handedness(&self) -> bool {
        self.object_to_world().swaps_handedness()
    }
}
