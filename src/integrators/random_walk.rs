use delegate::delegate;

use crate::{
    bxdf::TransportMode,
    camera::{CameraEnum, VisibleSurface},
    float::PI,
    geometry::{Ray, RayDifferential},
    math::Point2i,
    memory::ScratchBuffer,
    sampling::{
        spectrum::{SampledSpectrum, SampledWavelengths},
        Sampler, SamplerEnum,
    },
    shapes::ShapeIntersection,
    util::sampling::sample_uniform_sphere,
    Float,
};

use super::{
    base::{ImageTileIntegrate, RayIntegrate, SceneData},
    Integrate,
};

pub struct RandomWalkIntegrator {
    scene_data: SceneData,
    camera: CameraEnum,
    sampler: SamplerEnum,
    max_depth: usize,
}

impl ImageTileIntegrate for RandomWalkIntegrator {
    delegate! {
        #[through(RayIntegrate)]
        to self {
            fn eval_pixel_sample(
                &self,
                p_pixel: Point2i,
                sample_idx: usize,
                sampler: &mut impl Sampler,
                scratch_buffer: &mut ScratchBuffer,
            );
        }
    }

    fn camera(&self) -> &CameraEnum {
        &self.camera
    }

    fn sampler(&self) -> &SamplerEnum {
        &self.sampler
    }
}

impl Integrate for RandomWalkIntegrator {
    fn render(&mut self) {
        self.image_tile_render();
    }

    fn intersect(&self, ray: &Ray, t_max: Option<Float>) -> Option<ShapeIntersection> {
        todo!()
    }

    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool {
        todo!()
    }
}

impl RayIntegrate for RandomWalkIntegrator {
    fn incident_radiance(
        &self,
        ray: RayDifferential,
        lambda: &SampledWavelengths,
        sampler: &mut impl Sampler,
        scratch_buffer: &mut ScratchBuffer,
        _initialize_visible_surface: bool,
    ) -> (SampledSpectrum, Option<VisibleSurface>) {
        let sampled_spectrum =
            self.incident_radiance_random_walk(ray, lambda, sampler, scratch_buffer, 0);
        (sampled_spectrum, None)
    }
}

impl RandomWalkIntegrator {
    fn incident_radiance_random_walk(
        &self,
        ray_diff: RayDifferential,
        wavelengths: &SampledWavelengths,
        sampler: &mut impl Sampler,
        scratch_buffer: &mut ScratchBuffer,
        depth: usize,
    ) -> SampledSpectrum {
        // Intersect ray with scene and return if no intersection
        let si = self.intersect(&ray_diff.ray, None);

        if si.is_none() {
            // Return emitted light from infinite light sources
            let le = self
                .scene_data
                .infinite_lights
                .iter()
                .map(|light| light.radiance_infinite(&ray_diff.ray, wavelengths))
                .sum();
            return le;
        }

        let mut isect = si.unwrap().intr;

        // Get emitted radiance at surface intersection
        let wo = -ray_diff.ray.dir;
        let le = isect.emitted_radiance(wo, wavelengths);

        // Terminate random walk if max depth reached
        if depth == self.max_depth {
            return le;
        }

        // Compute BSDF at random walk intersection point
        let bsdf = isect
            .get_bsdf(
                &ray_diff,
                wavelengths,
                self.camera(),
                scratch_buffer,
                sampler,
            )
            .unwrap();

        // Randomly sample direction leaving surface for random walk
        let u = sampler.get_2d();
        let wp = sample_uniform_sphere(u);

        let eval = bsdf.eval(wo, wp, TransportMode::Radiance);
        if eval.is_none() {
            return le;
        }

        let f_cos = eval.unwrap() * wp.absdot(isect.shading.n.into());

        // Recursively trace ray to estimate incident radiance at surface
        let ray_diff = isect.spawn_ray(wp);

        le + f_cos
            * self.incident_radiance_random_walk(
                ray_diff,
                wavelengths,
                sampler,
                scratch_buffer,
                depth + 1,
            )
            / (1.0 / (4.0 * PI))
    }
}
