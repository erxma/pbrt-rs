use std::sync::OnceLock;

use crate::{
    float::PI,
    geometry::{Bounds3f, Ray},
    lights::LightType,
    math::{Normal3f, Point2f, Point3f, Vec3f},
    memory::ArcIntern,
    sampling::{
        routines::{sample_uniform_sphere, UNIFORM_SPHERE_PDF},
        spectrum::{DenselySampledSpectrum, SampledSpectrum, SampledWavelengths, Spectrum},
    },
    Float,
};

use super::{base::SpectrumCache, Light, LightLiSample, LightSampleContext};

#[derive(Debug)]
pub struct UniformInfiniteLight {
    emitted_radiance: ArcIntern<DenselySampledSpectrum>,
    scale: Float,
    // To be set late via preprocess()
    scene_center: OnceLock<Point3f>,
    scene_radius: OnceLock<Float>,
}

impl UniformInfiniteLight {
    pub fn new(emitted_radiance: &impl Spectrum, scale: Float) -> Self {
        Self {
            emitted_radiance: SpectrumCache::lookup_spectrum(emitted_radiance),
            scale,
            scene_center: OnceLock::new(),
            scene_radius: OnceLock::new(),
        }
    }
}

impl Light for UniformInfiniteLight {
    fn phi(&self, wavelengths: &SampledWavelengths) -> SampledSpectrum {
        let scene_radius = self
            .scene_radius
            .get()
            .expect("Must call preprocess() with scene bounds info for UniformInfiniteLight first");

        4.0 * PI.powi(2)
            * scene_radius.powi(2)
            * self.scale
            * self.emitted_radiance.sample(wavelengths)
    }

    fn light_type(&self) -> LightType {
        LightType::Infinite
    }

    fn sample_li(
        &self,
        ctx: LightSampleContext,
        u: Point2f,
        wavelengths: &SampledWavelengths,
        allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        let scene_radius = self
            .scene_radius
            .get()
            .expect("Must call preprocess() with scene bounds info for UniformInfiniteLight first");

        if allow_incomplete_pdf {
            return None;
        }

        // Return uniform spherical sample for uniform infinite light
        let incident = sample_uniform_sphere(u);
        let pdf = UNIFORM_SPHERE_PDF;
        Some(LightLiSample {
            l: self.scale * self.emitted_radiance.sample(wavelengths),
            wi: incident,
            pdf,
            p_light: ctx.pi_mids() + incident * (2.0 * scene_radius),
            medium_interface: None,
        })
    }

    fn pdf_li(&self, _ctx: LightSampleContext, _wi: Vec3f, allow_incomplete_pdf: bool) -> Float {
        if allow_incomplete_pdf {
            0.0
        } else {
            UNIFORM_SPHERE_PDF
        }
    }

    fn radiance(
        &self,
        _p: Point3f,
        _n: Normal3f,
        _uv: Point2f,
        _w: Vec3f,
        _wavelengths: &SampledWavelengths,
    ) -> SampledSpectrum {
        SampledSpectrum::with_single_value(0.0)
    }

    fn radiance_infinite(&self, _ray: &Ray, wavelengths: &SampledWavelengths) -> SampledSpectrum {
        self.scale * self.emitted_radiance.sample(wavelengths)
    }

    fn preprocess(&self, scene_bounds: Bounds3f) {
        let (scene_center, scene_radius) = scene_bounds.bounding_sphere();
        self.scene_center
            .set(scene_center)
            .expect("Preprocess should only be called once");
        self.scene_radius.set(scene_radius).unwrap();
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;

    use crate::{
        color::{RGB, SRGB},
        sampling::spectrum::RgbIlluminantSpectrum,
    };

    use super::*;

    #[test]
    fn radiance_infinite() {
        let radiance = RgbIlluminantSpectrum::new(&SRGB, RGB::new(0.4, 0.45, 0.5));
        let light = UniformInfiniteLight::new(&radiance, 1.0 / radiance.to_photometric());

        let dummy_ray = Ray::new(Point3f::ZERO, Vec3f::new(0.0, 0.0, 1.0), 0.0, None);

        let lambda1 = SampledWavelengths::from_parts(
            [474.55463, 542.08417, 611.77325, 783.2514],
            [0.0032198546, 0.0039363992, 0.0030081926, 0.00043523454],
        );
        let result1 = light.radiance_infinite(&dummy_ray, &lambda1);
        assert_relative_eq!(
            result1,
            SampledSpectrum::new([0.0052793995, 0.0044468315, 0.0035051198, 0.001987834])
        );
    }
}
