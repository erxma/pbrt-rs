use std::{cell::RefCell, sync::Arc};

use crate::{
    camera::Camera,
    geometry::Ray,
    lights::LightEnum,
    math::Point2i,
    memory::ScratchBuffer,
    parallel::parallel_for_2d_with,
    primitives::{Primitive, PrimitiveEnum},
    sampling::Sampler,
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

fn image_tile_render<S: Sampler + Send + Sync + Clone>(
    camera: &impl Camera,
    sampler: &S,
    eval_pixel_sample: impl Fn(Point2i, usize, &S, &ScratchBuffer) + Send + Sync,
) {
    // Declare common vars for rendering iamge in tiles

    let pixel_bounds = camera.film().pixel_bounds();

    // TODO: Progress reporter

    // Render image in waves
    let mut wave_start = 0;
    let mut wave_end = 1;
    let mut next_wave_size = 1;

    while wave_start < sampler.samples_per_pixel() {
        // Render current wave's image tiles in parallel
        parallel_for_2d_with(pixel_bounds, sampler.clone(), |sampler, tile_bounds| {
            thread_local! {
                static SCRATCH_BUFFER: RefCell<ScratchBuffer> = RefCell::new(ScratchBuffer::new());
            }

            for p_pixel in tile_bounds {
                // Render samples in pixel p_pixel
                for sample_idx in wave_start..wave_end {
                    sampler.start_pixel_sample(p_pixel, sample_idx, 0).unwrap();
                    SCRATCH_BUFFER.with_borrow_mut(|scratch_buffer| {
                        eval_pixel_sample(p_pixel, sample_idx, &sampler, scratch_buffer);
                        scratch_buffer.reset();
                    });
                }
            }
        });

        wave_start = wave_end;
        wave_end = sampler.samples_per_pixel().min(wave_end + next_wave_size);
        next_wave_size = (2 * next_wave_size).min(64);
    }
}
