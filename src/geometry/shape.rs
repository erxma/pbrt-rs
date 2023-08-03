use crate as pbrt;

use super::{
    bounds3::Bounds3f, ray::Ray, surface_interaction::SurfaceInteraction, transform::Transform,
};

pub trait Shape {
    fn object_bound(self) -> Bounds3f;
    fn world_bound(self) -> Bounds3f;

    fn intersect(
        self,
        ray: impl Ray,
        test_alpha_texture: bool,
    ) -> Option<(pbrt::Float, SurfaceInteraction)>;
    fn intersect_p(self, ray: impl Ray, test_alpha_texture: bool) -> bool;

    fn area(self) -> pbrt::Float;

    fn object_to_world<'a>(self) -> &'a Transform;
    fn world_to_object<'a>(self) -> &'a Transform;
    fn reverse_orientation(self) -> bool;
    fn transform_swaps_handedness(self) -> bool;
}
