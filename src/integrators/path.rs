use std::sync::Arc;

use delegate::delegate;

use crate::{
    camera::{CameraEnum, VisibleSurface},
    core::{Float, Point2f, Point2i, Ray, RayDifferential, SurfaceInteraction},
    lights::{LightEnum, LightSampleContext},
    memory::ScratchBuffer,
    primitives::{Primitive, PrimitiveEnum},
    reflection::{BxDFEnum, BxDFFlags, BxDFReflTransFlags, TransportMode, BSDF},
    sampling::{
        routines::power_heuristic,
        spectrum::{SampledSpectrum, SampledWavelengths},
        LightSampler, Sampler, SamplerEnum, UniformLightSampler,
    },
    shapes::ShapeIntersection,
};

use super::{
    base::{ImageTileIntegrate, RayIntegrate, SceneData},
    Integrate,
};

pub struct PathIntegrator {
    scene_data: SceneData,
    camera: CameraEnum,
    sampler_prototype: SamplerEnum,
    max_depth: usize,
    regularize: bool,
    light_sampler: UniformLightSampler,
}

impl PathIntegrator {
    pub fn new(
        max_depth: usize,
        regularize: bool,
        camera: CameraEnum,
        sampler: SamplerEnum,
        aggregate: PrimitiveEnum,
        lights: Vec<Arc<LightEnum>>,
    ) -> Self {
        let scene_data = SceneData::new(aggregate, lights);
        let light_sampler = UniformLightSampler::new(&scene_data.lights);
        Self {
            scene_data,
            camera,
            sampler_prototype: sampler,
            max_depth,
            regularize,
            light_sampler,
        }
    }

    fn sample_ld(
        &self,
        intr: &SurfaceInteraction,
        bsdf: &BSDF<'_, BxDFEnum>,
        lambda: &SampledWavelengths,
        sampler: &impl Sampler,
    ) -> SampledSpectrum {
        todo!()
    }
}

impl Integrate for PathIntegrator {
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

impl ImageTileIntegrate for PathIntegrator {
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

impl RayIntegrate for PathIntegrator {
    fn incident_radiance(
        &self,
        mut ray_diff: RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut impl Sampler,
        scratch_buffer: &mut ScratchBuffer,
        initialize_visible_surface: bool,
    ) -> (SampledSpectrum, Option<VisibleSurface>) {
        // Estimate radiance along ray using path tracing:

        // Current estimated scattered radiance running total
        let mut radiance = SampledSpectrum::with_single_value(0.0);

        // Path throughput weight (factors of the throughput function,
        // i.e. the product of the BSDF values and cosine terms for verts
        // generated so far, divided by the respective sampling PDFs)
        // The product of beta with scattered light from direct lighting
        // from the final vertex of the path gives the contribution for a path.
        let mut beta = SampledSpectrum::with_single_value(1.0);
        // The PDF for sampling the next direction,
        // needed for the MIS-based direct lighting estimate.
        let mut bsdf_pdf = 1.0;
        // Accumulated product of scaling factors that have been applied to beta
        // due to rays being transmitted between media of different indices of
        // refraction. Used in the Russian roulette computation.
        let mut eta_scale = 1.0;
        // Whether the last outgoing path direction sampled was due to
        // perfect specular reflection...
        let mut specular_bounce = true;
        // ...and whether any so far have NOT been,
        // used for path regularization if enabled.
        let mut any_non_specular_bounces = true;
        // Geometric info about the intersection point the sampled ray is leaving.
        // Used in MIS-computation for direct lighting.
        let mut prev_intr_ctx = None;

        // Number of reflections so far in the path.
        let mut depth = 0;

        // To be set if requested with `initialize_visible_surface`
        let mut visible_surf = None;

        // Sample path from camera and accumulate radiance estimate
        // Each loop accounts for an additional segment of a path
        // Loop continues until max depth reached, or path is terminated via
        // Russian roulette.
        loop {
            // Each loop, find next vertex and accumulate contribution.

            // Trace ray and find closest path vertex and its BSDF
            let si = self.intersect(&ray_diff.ray, None);

            // Add emitted light at intersection point or from environment:

            // If no intersection, ray path ends.
            if si.is_none() {
                // Before finishing, incorporate emission from infinite lights
                // for escaped ray
                for light in &self.scene_data.infinite_lights {
                    let emission = light.radiance_infinite(&ray_diff.ray, lambda);
                    if depth == 0 || specular_bounce {
                        // For initial ray or after a perfect specular scattering event,
                        // don't use MIS weighting, since light sampling wasn't performed
                        // at the previous vertex
                        radiance += &beta * emission;
                    } else {
                        // Compute MIS weight for infinite light
                        // Since depth is > 0, ctx must be Some
                        let prev_intr_ctx = prev_intr_ctx.take().unwrap();
                        let light_pdf = self
                            .light_sampler
                            .pmf_with_context(&prev_intr_ctx, light.as_ref())
                            * light.pdf_li(prev_intr_ctx, ray_diff.ray.dir, true);
                        let mis_weight = power_heuristic(1, bsdf_pdf, 1, light_pdf);
                        radiance += &beta * mis_weight * emission;
                    }
                }
                break;
            }
            let si = si.unwrap();
            let mut isect = si.intr;

            // Incorporate emission from emissive surface hit by ray
            // (almost the same as for infinite light case)
            let emission = isect.emitted_radiance(-ray_diff.ray.dir, lambda);
            if !emission.is_all_zero() {
                if depth == 0 || specular_bounce {
                    radiance += &beta * emission;
                } else {
                    // Compute MIS weight for infinite light

                    // Since depth is > 0, ctx must be Some
                    let prev_intr_ctx = prev_intr_ctx.take().unwrap();

                    // Get PDF for the ray's direction from sampling the light
                    // (prob of sampling the light) * (prob the light returns for sampling the dir)
                    let area_light = isect.area_light.unwrap();
                    let light_pdf = self
                        .light_sampler
                        .pmf_with_context(&prev_intr_ctx, area_light)
                        * area_light.pdf_li(prev_intr_ctx, ray_diff.ray.dir, true);
                    let mis_weight = power_heuristic(1, bsdf_pdf, 1, light_pdf);
                    radiance += &beta * mis_weight * emission;
                }
            }

            // Get BSDF at the intersection point
            let bsdf = isect.get_bsdf(&ray_diff, lambda, &self.camera, scratch_buffer, sampler);
            // If no BSDF is returned, the current surface should have no effect on light,
            // skip it
            // Such surfaces exist at transitions between participating media,
            // whose boundaries are optically inactive (have same IOR on both sides)
            if bsdf.is_none() {
                ray_diff = isect.skip_intersection(&ray_diff, si.t_hit);
                continue;
            }
            let mut bsdf = bsdf.unwrap();

            // If needed and at first intersection, initialize visible_surf
            visible_surf = if depth == 0 && initialize_visible_surface {
                // Estimate BSDF's albedo,
                // as the hemispherical-directional reflectance
                let albedo = bsdf.reflectance_hemispherical_directional(isect.wo, &UC_RHO, &U_RHO);
                Some(VisibleSurface::new(&isect, albedo))
            } else {
                None
            };

            // Possibly regularize the BSDF,
            // if enabled and any non-specular scattering has occurred.
            // No need if only perfect specular has occurred.
            if self.regularize && any_non_specular_bounces {
                bsdf.regularize();
            }

            // Increment depth, end path if max reached
            depth += 1;
            if depth == self.max_depth {
                break;
            }

            // Sample direct illumination from the light sources,
            // unless BSDF is purely specular (in which case BSDF for sampled point
            // on a light will be 0)
            if bsdf
                .flags()
                .intersects(BxDFFlags::DIFFUSE | BxDFFlags::GLOSSY)
            {
                let ld = self.sample_ld(&isect, &bsdf, &lambda, sampler);
                radiance += &beta * ld;
            }

            // Sample BSDF to get new path direction
            let outgoing = -ray_diff.ray.dir;
            let u = sampler.get_1d();
            let u2 = sampler.get_2d();
            // Proceed if a result was returned. Otherwise, break.
            if let Some(bs) = bsdf.sample_func(
                outgoing,
                u,
                u2,
                TransportMode::Radiance,
                BxDFReflTransFlags::all(),
            ) {
                // Update path state variables after surface scattering
                // see their definitions above

                beta *= bs.value * bs.incident.absdot(isect.shading.n.into()) / bs.pdf;

                bsdf_pdf = if bs.pdf_is_proportional {
                    bsdf.pdf(
                        outgoing,
                        bs.incident,
                        TransportMode::Radiance,
                        BxDFReflTransFlags::all(),
                    )
                } else {
                    bs.pdf
                };

                specular_bounce = bs.flags.contains(BxDFFlags::SPECULAR);
                any_non_specular_bounces |= !specular_bounce;

                if bs.flags.contains(BxDFFlags::TRANSMISSION) {
                    eta_scale *= bs.eta * bs.eta;
                }

                prev_intr_ctx = Some(LightSampleContext::with_surface_interaction(&isect));

                ray_diff = isect.spawn_ray(&ray_diff, &bsdf, bs.incident, bs.flags, bs.eta);

                // Possibly terminate the path with Russian roulette.
                // beta is corrected with eta_scale to exclude radiance scaling
                // due to refraction
                let rr_beta = &beta * eta_scale;

                // Set prob of termination to max component value of adjusted beta.
                // Gives better results when surface reflectances are highly saturated
                // and some samples have much lower betas than others,
                // since it prevents any beta components from going above 1 due to Russian roulette.
                if rr_beta.max_component_value().unwrap() < 1.0 && depth > 1 {
                    let terminate_prob = (1.0 - rr_beta.max_component_value().unwrap()).max(0.0);
                    // If 1D sample hits termination range, do so
                    if sampler.get_1d() < terminate_prob {
                        break;
                    }
                    // Scale beta according to prob
                    beta /= 1.0 - terminate_prob;
                    debug_assert!(!beta.y(lambda).is_finite());
                }
            } else {
                break;
            }
        }

        (radiance, visible_surf)
    }
}

// Sample arrays precomputed Owen-scrambled Halton points
// for reflectance estimate
const N_RHO_SAMPLES: usize = 16;
const UC_RHO: [Float; N_RHO_SAMPLES] = [
    0.75741637,
    0.37870818,
    0.7083487,
    0.18935409,
    0.9149363,
    0.35417435,
    0.5990858,
    0.09467703,
    0.8578725,
    0.45746812,
    0.686759,
    0.17708716,
    0.9674518,
    0.2995429,
    0.5083201,
    0.047338516,
];
const U_RHO: [Point2f; N_RHO_SAMPLES] = [
    Point2f::new(0.855985, 0.570367),
    Point2f::new(0.381823, 0.851844),
    Point2f::new(0.285328, 0.764262),
    Point2f::new(0.733380, 0.114073),
    Point2f::new(0.542663, 0.344465),
    Point2f::new(0.127274, 0.414848),
    Point2f::new(0.964700, 0.947162),
    Point2f::new(0.594089, 0.643463),
    Point2f::new(0.095109, 0.170369),
    Point2f::new(0.825444, 0.263359),
    Point2f::new(0.429467, 0.454469),
    Point2f::new(0.244460, 0.816459),
    Point2f::new(0.756135, 0.731258),
    Point2f::new(0.516165, 0.152852),
    Point2f::new(0.180888, 0.214174),
    Point2f::new(0.898579, 0.503897),
];
