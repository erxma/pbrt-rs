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

struct Integrator<'a> {
    aggregate: PrimitiveEnum,
    lights: Vec<&'a LightEnum>,
    infinite_lights: Vec<&'a LightEnum>,
}

impl<'a> Integrator<'a> {
    fn new(aggregate: PrimitiveEnum, lights: Vec<&'a LightEnum>) -> Self {
        let scene_bounds = aggregate.bounds();

        let mut infinite_lights = Vec::new();
        for &light in &lights {
            light.preprocess(scene_bounds);
            infinite_lights.push(light);
        }

        Self {
            aggregate,
            lights,
            infinite_lights,
        }
    }
}

pub struct ImageTileIntegrator<'a> {
    integrator: Integrator<'a>,
    camera: Camera,
    sampler: SamplerEnum,
}

impl<'a> ImageTileIntegrator<'a> {
    pub fn new(
        camera: Camera,
        sampler: SamplerEnum,
        aggregate: PrimitiveEnum,
        lights: Vec<&'a LightEnum>,
    ) -> Self {
        Self {
            integrator: Integrator::new(aggregate, lights),
            camera,
            sampler,
        }
    }
}

impl<'a> Integrate for ImageTileIntegrator<'a> {
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
