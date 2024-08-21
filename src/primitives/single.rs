use std::sync::Arc;

use crate::{
    geometry::{Bounds3f, Ray},
    materials::Material,
    shapes::{Shape, ShapeEnum, ShapeIntersection},
    Float,
};

use super::Primitive;

#[derive(Clone)]
pub struct SimplePrimitive {
    // TODO: Cost of Arcs acceptable?
    shape: Arc<ShapeEnum>,
    material: Arc<Material>,
}

impl SimplePrimitive {
    pub fn new(shape: Arc<ShapeEnum>, material: Arc<Material>) -> Self {
        Self { shape, material }
    }
}

impl Primitive for SimplePrimitive {
    fn bounds(&self) -> Bounds3f {
        self.shape.bounds()
    }

    fn intersect<'a>(
        &'a self,
        ray: &'a Ray,
        t_max: Option<Float>,
    ) -> Option<ShapeIntersection<'a>> {
        // Intersect with shape
        let mut shape_intersection = self.shape.intersect(ray, t_max)?;

        // Initialize SurfaceInteraction
        shape_intersection
            .intr
            .set_properties(Some(&self.material), None, None, ray.medium);

        Some(shape_intersection)
    }

    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool {
        self.shape.intersect_p(ray, t_max)
    }
}
