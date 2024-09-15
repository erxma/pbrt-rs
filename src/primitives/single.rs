use std::sync::Arc;

use crate::{
    core::{Bounds3f, Float, Ray},
    lights::LightEnum,
    materials::MaterialEnum,
    media::MediumInterface,
    shapes::{Shape, ShapeEnum, ShapeIntersection},
};

use super::Primitive;

pub struct GeometricPrimitive {
    shape: Arc<ShapeEnum>,
    material: Arc<MaterialEnum>,
    area_light: Option<Arc<LightEnum>>,
    medium_interface: MediumInterface,
    // alpha: Option<Arc<FloatTextureEnum>>,
}

impl GeometricPrimitive {
    pub fn new(
        shape: Arc<ShapeEnum>,
        material: Arc<MaterialEnum>,
        area_light: Option<Arc<LightEnum>>,
        medium_interface: MediumInterface,
        // alpha: Option<Arc<FloatTextureEnum>>,
    ) -> Self {
        Self {
            shape,
            material,
            area_light,
            medium_interface,
            // alpha,
        }
    }
}

impl Primitive for GeometricPrimitive {
    fn bounds(&self) -> Bounds3f {
        self.shape.bounds()
    }

    fn intersect<'a>(
        &'a self,
        ray: &'a Ray,
        t_max: Option<Float>,
    ) -> Option<ShapeIntersection<'a>> {
        // Have the shape perform the intersection test
        let mut si = self.shape.intersect(ray, t_max)?;

        // TODO: Test intersection against alpha texture, if present

        // Initialize SurfaceInteraction of intersection
        si.intr.set_properties(
            Some(&*self.material),
            self.area_light.as_deref(),
            Some(self.medium_interface.clone()),
            ray.medium.clone(),
        );

        Some(si)
    }

    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool {
        // TODO: Handle alpha
        self.shape.intersect_p(ray, t_max)
    }
}

#[derive(Clone)]
pub struct SimplePrimitive {
    shape: Arc<ShapeEnum>,
    material: Arc<MaterialEnum>,
}

impl SimplePrimitive {
    pub fn new(shape: Arc<ShapeEnum>, material: Arc<MaterialEnum>) -> Self {
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
        shape_intersection.intr.set_properties(
            Some(&self.material),
            None,
            None,
            ray.medium.clone(),
        );

        Some(shape_intersection)
    }

    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool {
        self.shape.intersect_p(ray, t_max)
    }
}
