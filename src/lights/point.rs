use crate::{
    float::PI,
    geometry::{Bounds3f, Ray, Transform},
    math::{Normal3f, Point2f, Point3f, Vec3f},
    media::MediumInterface,
    memory::ArcIntern,
    sampling::spectrum::{DenselySampledSpectrum, SampledSpectrum, SampledWavelengths, Spectrum},
    Float,
};

use super::base::{Light, LightLiSample, LightSampleContext, LightType, SpectrumCache};

#[derive(Debug)]
pub struct PointLight {
    render_from_light: Transform,
    medium_interface: MediumInterface,

    intensity: ArcIntern<DenselySampledSpectrum>,
    scale: Float,
}

impl PointLight {
    pub fn new(
        render_from_light: Transform,
        medium_interface: MediumInterface,
        i: &impl Spectrum,
        scale: Float,
    ) -> Self {
        Self {
            render_from_light,
            medium_interface,
            intensity: SpectrumCache::lookup_spectrum(i),
            scale,
        }
    }
}

impl Light for PointLight {
    fn phi(&self, wavelengths: &SampledWavelengths) -> SampledSpectrum {
        4.0 * PI * self.scale * self.intensity.sample(wavelengths)
    }

    fn light_type(&self) -> LightType {
        LightType::DeltaPosition
    }

    fn sample_li(
        &self,
        ctx: LightSampleContext,
        _u: Point2f,
        wavelengths: &SampledWavelengths,
        _allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        let p = &self.render_from_light * Point3f::ZERO;
        let wi = (p - ctx.pi_mids()).normalized();
        let li =
            self.scale * self.intensity.sample(wavelengths) / p.distance_squared(ctx.pi_mids());

        Some(LightLiSample {
            l: li,
            wi,
            pdf: 1.0,
            p_light: p,
            medium_interface: Some(&self.medium_interface),
        })
    }

    fn pdf_li(&self, _ctx: LightSampleContext, _wi: Vec3f, _allow_incomplete_pdf: bool) -> Float {
        0.0
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

    fn radiance_infinite(&self, _ray: &Ray, _wavelengths: &SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::with_single_value(0.0)
    }

    fn preprocess(&self, _scene_bounds: Bounds3f) {
        // Nothing to do
    }
}
