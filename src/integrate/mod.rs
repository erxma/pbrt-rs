use std::sync::Arc;

use crate::{
    camera::Camera,
    geometry::Ray,
    lights::LightEnum,
    primitives::{Primitive, PrimitiveEnum},
    sampling::SamplerEnum,
    shapes::ShapeIntersection,
    Float,
};

pub trait Integrate {
    fn render(&mut self);
    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection>;
    fn intersect_p(&self, ray: &Ray, t_max: Float) -> bool;
}

struct Integrator {
    aggregate: PrimitiveEnum,
    lights: Vec<Arc<LightEnum>>,
    infinite_lights: Vec<Arc<LightEnum>>,
}

impl Integrator {
    fn new(aggregate: PrimitiveEnum, lights: Vec<Arc<LightEnum>>) -> Self {
        let scene_bounds = aggregate.bounds();

        let mut infinite_lights = Vec::new();
        for light in &lights {
            light.preprocess(scene_bounds);
            infinite_lights.push(light.clone());
        }

        Self {
            aggregate,
            lights,
            infinite_lights,
        }
    }
}

pub struct ImageTileIntegrator {
    integrator: Integrator,
    camera: Camera,
    sampler: SamplerEnum,
}

impl ImageTileIntegrator {
    pub fn new(
        camera: Camera,
        sampler: SamplerEnum,
        aggregate: PrimitiveEnum,
        lights: Vec<Arc<LightEnum>>,
    ) -> Self {
        Self {
            integrator: Integrator::new(aggregate, lights),
            camera,
            sampler,
        }
    }
}

impl Integrate for ImageTileIntegrator {
    fn render(&mut self) {
        todo!()
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        todo!()
    }

    fn intersect_p(&self, ray: &Ray, t_max: Float) -> bool {
        todo!()
    }
}
