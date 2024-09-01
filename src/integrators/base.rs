use std::{cell::RefCell, sync::Arc};

use indicatif::ProgressBar;

use crate::{
    camera::{CameraEnum, VisibleSurface},
    geometry::{Ray, RayDifferential},
    image::ImageMetadata,
    lights::{LightEnum, LightType},
    math::Point2i,
    memory::ScratchBuffer,
    parallel::parallel_for_2d_tiled_with,
    primitives::{Primitive, PrimitiveEnum},
    sampling::{
        spectrum::{SampledSpectrum, SampledWavelengths},
        Sampler, SamplerEnum,
    },
    shapes::ShapeIntersection,
    Float,
};

pub trait Integrate {
    fn render(&mut self);
    fn intersect<'a>(&'a self, ray: &'a Ray, t_max: Option<Float>)
        -> Option<ShapeIntersection<'a>>;
    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool;
}

pub(super) struct SceneData {
    pub aggregate: PrimitiveEnum,
    pub lights: Vec<Arc<LightEnum>>,
    pub infinite_lights: Vec<Arc<LightEnum>>,
}

impl SceneData {
    pub fn new(aggregate: PrimitiveEnum, lights: Vec<Arc<LightEnum>>) -> Self {
        let scene_bounds = aggregate.bounds();

        let mut infinite_lights = Vec::new();
        for light in &lights {
            light.preprocess(scene_bounds);
            if light.light_type() == LightType::Infinite {
                infinite_lights.push(light.clone());
            }
        }

        Self {
            aggregate,
            lights,
            infinite_lights,
        }
    }
}

pub(super) trait ImageTileIntegrate: Send + Sync {
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
            // For the sake of simple safety, operation will not directly write.
            // Instead, collect all the results and do all adding after
            parallel_for_2d_tiled_with(
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

        self.camera().film().write_image(
            &ImageMetadata {},
            1.0 / self.sampler().samples_per_pixel() as Float,
        );
    }

    fn camera(&self) -> &CameraEnum;
    fn sampler(&self) -> &SamplerEnum;
}

pub(super) trait RayIntegrate: ImageTileIntegrate {
    fn incident_radiance(
        &self,
        ray: RayDifferential,
        lambda: &SampledWavelengths,
        sampler: &mut impl Sampler,
        scratch_buffer: &mut ScratchBuffer,
        initialize_visible_surface: bool,
    ) -> (SampledSpectrum, Option<VisibleSurface>);

    fn eval_pixel_sample(
        &self,
        p_pixel: Point2i,
        _sample_idx: usize,
        sampler: &mut impl Sampler,
        scratch_buffer: &mut ScratchBuffer,
    ) {
        let film = self.camera().film();

        // Sample wavelengths for the ray
        let lu = sampler.get_1d();
        let lambda = film.sample_wavelengths(lu);

        // Initialize CameraSample for current sampple
        let filter = film.filter();
        let camera_sample = sampler.get_camera_sample(p_pixel, filter);

        // Generate camera ray for current sample
        let camera_ray = self
            .camera()
            .generate_ray_differential(camera_sample, &lambda);

        let l;
        let visible_surface;
        match camera_ray {
            // Trace camera ray if valid
            Some(mut camera_ray) => {
                // Scale camera ray differentials based on image sampling rate
                let ray_diff_scale = (1.0 / (sampler.samples_per_pixel() as Float))
                    .sqrt()
                    .max(0.125);
                camera_ray.ray.scale_differentials(ray_diff_scale);
                // Evaluate radiance along camera ray
                let initialize_visible_surface = film.uses_visible_surface();
                (l, visible_surface) = self.incident_radiance(
                    camera_ray.ray,
                    &lambda,
                    sampler,
                    scratch_buffer,
                    initialize_visible_surface,
                );
            }
            None => {
                l = SampledSpectrum::with_single_value(0.0);
                visible_surface = None;
            }
        }

        // TODO: Issue warning if unexpected radiance value is returned

        // Add camera ray's contribution to image
        unsafe {
            film.add_sample_unchecked(
                p_pixel,
                &l,
                &lambda,
                visible_surface.as_ref(),
                camera_sample.filter_weight,
            );
        }
    }
}
