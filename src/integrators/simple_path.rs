use std::sync::Arc;

use delegate::delegate;

use crate::{
    camera::{CameraEnum, VisibleSurface},
    geometry::{Ray, RayDifferential},
    lights::{LightEnum, LightSampleContext},
    math::Point2i,
    memory::ScratchBuffer,
    primitives::{Primitive, PrimitiveEnum},
    reflection::{BxDFFlags, BxDFReflTransFlags, TransportMode},
    sampling::{
        routines::{
            sample_uniform_hemisphere, sample_uniform_sphere, UNIFORM_HEMISPHERE_PDF,
            UNIFORM_SPHERE_PDF,
        },
        spectrum::{SampledSpectrum, SampledWavelengths},
        LightSampler, Sampler, SamplerEnum, UniformLightSampler,
    },
    shapes::ShapeIntersection,
    Float,
};

use super::{
    base::{ImageTileIntegrate, RayIntegrate, SceneData},
    Integrate,
};

pub struct SimplePathIntegrator {
    scene_data: SceneData,
    camera: CameraEnum,
    sampler_prototype: SamplerEnum,
    max_depth: usize,
    sample_lights: bool,
    sample_bsdf: bool,
    light_sampler: UniformLightSampler,
}

impl SimplePathIntegrator {
    pub fn new(
        max_depth: usize,
        sample_lights: bool,
        sample_bsdf: bool,
        camera: CameraEnum,
        sampler: SamplerEnum,
        aggregate: PrimitiveEnum,
        lights: Vec<Arc<LightEnum>>,
    ) -> Self {
        let light_sampler = UniformLightSampler::new(&lights);
        Self {
            scene_data: SceneData::new(aggregate, lights),
            camera,
            sampler_prototype: sampler,
            max_depth,
            sample_lights,
            sample_bsdf,
            light_sampler,
        }
    }
}

impl Integrate for SimplePathIntegrator {
    fn render(&mut self) {
        self.image_tile_render();
    }

    fn intersect<'a>(
        &'a self,
        ray: &'a Ray,
        t_max: Option<Float>,
    ) -> Option<ShapeIntersection<'a>> {
        self.scene_data.aggregate.intersect(ray, t_max)
    }

    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool {
        self.scene_data.aggregate.intersect_p(ray, t_max)
    }
}

impl ImageTileIntegrate for SimplePathIntegrator {
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
        &self.sampler_prototype
    }
}

impl RayIntegrate for SimplePathIntegrator {
    fn incident_radiance(
        &self,
        mut ray_diff: RayDifferential,
        lambda: &SampledWavelengths,
        sampler: &mut impl Sampler,
        scratch_buffer: &mut ScratchBuffer,
        _initialize_visible_surface: bool,
    ) -> (SampledSpectrum, Option<VisibleSurface>) {
        // Estimate radiance along ray using simple path tracing
        let mut radiance = SampledSpectrum::with_single_value(0.0);
        let mut beta = SampledSpectrum::with_single_value(1.0);
        let mut specular_bounce = true;
        let mut depth = 0;

        while !beta.is_all_zero() {
            // Find next vertex and accumulate contribution:

            // Intersect ray with scene
            let si = self.intersect(&ray_diff.ray, None);

            //Account for infinite lights if ray has no intersection
            if si.is_none() {
                if !self.sample_lights || specular_bounce {
                    for light in &self.scene_data.infinite_lights {
                        radiance += &beta * light.radiance_infinite(&ray_diff.ray, lambda);
                    }
                    break;
                }
            }
            let si = si.unwrap();

            // Account for emissive surface if light was not sampled
            let mut isect = si.intr;
            if !self.sample_lights || specular_bounce {
                radiance += &beta * isect.emitted_radiance(-ray_diff.ray.dir, lambda);
            }

            // Get BSDF and skip over medium boundaries
            let bsdf = isect.get_bsdf(&ray_diff, lambda, &self.camera, scratch_buffer, sampler);
            if bsdf.is_none() {
                ray_diff = isect.skip_intersection(&ray_diff, si.t_hit);
                continue;
            }
            let bsdf = bsdf.unwrap();

            // End path if maximum depth reached
            depth += 1;
            if depth == self.max_depth {
                break;
            }

            // Sample direct illumination if sample_lights is true
            let outgoing = -ray_diff.ray.dir;
            if self.sample_lights {
                if let Some(sampled_light) = self.light_sampler.sample(sampler.get_1d()) {
                    // Sample point on sampled light to estimate direct illumination
                    let u_light = sampler.get_2d();
                    if let Some(light_sample) = sampled_light.light.sample_li(
                        LightSampleContext::with_surface_interaction(&isect),
                        u_light,
                        lambda,
                        false,
                    ) {
                        if !light_sample.l.is_all_zero() && light_sample.pdf > 0.0 {
                            // Evaluate BSDF for light and possibly add scattered radiance
                            let incident = light_sample.wi;
                            let bsdf_val = bsdf
                                .eval(outgoing, incident, TransportMode::Radiance)
                                // TODO: Confirm unwrap is okay
                                .unwrap()
                                * incident.absdot(isect.shading.n.into());

                            {
                                if !bsdf_val.is_all_zero()
                                    && self.unoccluded(&isect, light_sample.p_light)
                                {
                                    radiance += &beta * bsdf_val * light_sample.l
                                        / (sampled_light.prob * light_sample.pdf);
                                }
                            }
                        }
                    }
                }
            }

            // Sample outgoing direction at intersection to continue path
            if self.sample_bsdf {
                // Sample BSDF for new path dir
                let u = sampler.get_1d();
                let u2 = sampler.get_2d();
                if let Some(bs) = bsdf.sample_func(
                    outgoing,
                    u,
                    u2,
                    TransportMode::Radiance,
                    BxDFReflTransFlags::all(),
                ) {
                    beta *= bs.value * bs.incident.absdot(isect.shading.n.into()) / bs.pdf;
                    specular_bounce = bs.flags.contains(BxDFFlags::SPECULAR);
                } else {
                    break;
                }
            } else {
                // Uniformly sample sphere or hemisphere to get new path dir
                let mut incident;
                let pdf;
                let flags = bsdf.flags();
                if flags.contains(BxDFFlags::REFLECTION) && flags.contains(BxDFFlags::TRANSMISSION)
                {
                    incident = sample_uniform_sphere(sampler.get_2d());
                    pdf = UNIFORM_SPHERE_PDF;
                } else {
                    incident = sample_uniform_hemisphere(sampler.get_2d());
                    pdf = UNIFORM_HEMISPHERE_PDF;
                    let need_flip_towards_reflection = flags.contains(BxDFFlags::REFLECTION)
                        && outgoing.dot(isect.n.into()) * incident.dot(isect.n.into()) < 0.0;
                    let need_flip_towards_transmission = flags.contains(BxDFFlags::TRANSMISSION)
                        && outgoing.dot(isect.n.into()) * incident.dot(isect.n.into()) > 0.0;
                    if need_flip_towards_reflection || need_flip_towards_transmission {
                        incident = -incident;
                    }
                }

                beta *= bsdf
                    .eval(outgoing, incident, TransportMode::Radiance)
                    // TODO: Confirm unwrap is okay
                    .unwrap()
                    * incident.absdot(isect.shading.n.into())
                    / pdf;
                specular_bounce = false;
                ray_diff = isect.spawn_ray(incident);
            }
        }

        (radiance, None)
    }
}
