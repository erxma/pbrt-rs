use std::{cell::RefCell, sync::Arc};

use indicatif::ProgressBar;

use crate::{
    camera::{Camera, CameraEnum},
    geometry::Ray,
    lights::LightEnum,
    math::Point2i,
    memory::ScratchBuffer,
    parallel::parallel_for_2d_with,
    primitives::{Primitive, PrimitiveEnum},
    sampling::{Sampler, SamplerEnum},
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

trait ImageTileIntegrate: Integrate + Send + Sync {
    fn eval_pixel_sample(
        &self,
        p_pixel: Point2i,
        sample_idx: usize,
        sampler: &mut impl Sampler,
        scratch_buffer: &mut ScratchBuffer,
    );

    fn image_tile_render(&self) {
        // Declare common vars for rendering iamge in tiles

        let pixel_bounds = self.camera().film().pixel_bounds();

        // TODO: Progress reporter

        // Render image in waves
        let mut wave_start = 0;
        let mut wave_end = 1;
        let mut next_wave_size = 1;

        // Progress bar with units of work = num samples = num pixels * samples per pixel
        let progress_bar = ProgressBar::new(
            (pixel_bounds.area() as usize * self.sampler().samples_per_pixel()) as u64,
        );

        while wave_start < self.sampler().samples_per_pixel() {
            // Render current wave's image tiles in parallel
            parallel_for_2d_with(
                pixel_bounds,
                (self.sampler().to_owned(), progress_bar.clone()),
                |(sampler, progress_bar), tile_bounds| {
                    thread_local! {
                        static SCRATCH_BUFFER: RefCell<ScratchBuffer> = RefCell::new(ScratchBuffer::new());
                    }

                    for p_pixel in tile_bounds {
                        // Render samples in pixel p_pixel
                        for sample_idx in wave_start..wave_end {
                            sampler.start_pixel_sample(p_pixel, sample_idx, 0).unwrap();
                            SCRATCH_BUFFER.with_borrow_mut(|scratch_buffer| {
                                self.eval_pixel_sample(
                                    p_pixel,
                                    sample_idx,
                                    sampler,
                                    scratch_buffer,
                                );
                                scratch_buffer.reset();
                            });
                        }
                    }
                    // Advance progress bar by num samples completed = num waves completed * num pixels in bounds
                    progress_bar
                        .inc(((wave_end - wave_start) * tile_bounds.area() as usize) as u64);
                },
            );

            wave_start = wave_end;
            wave_end = self
                .sampler()
                .samples_per_pixel()
                .min(wave_end + next_wave_size);
            next_wave_size = (2 * next_wave_size).min(64);
        }
    }

    fn camera(&self) -> &CameraEnum;
    fn sampler(&self) -> &SamplerEnum;
}
