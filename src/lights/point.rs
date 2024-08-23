use crate::{
    float::PI,
    geometry::{Bounds3f, Ray, Transform},
    math::{Normal3f, Point2f, Point3f, Vec3f},
    media::MediumInterface,
    memory::cache::ArcIntern,
    sampling::spectrum::{
        DenselySampledSpectrum, SampledSpectrum, SampledWavelengths, Spectrum, SpectrumEnum,
    },
    Float,
};
use derive_builder::Builder;

use super::base::{Light, LightLiSample, LightSampleContext, LightType, SpectrumCache};

pub struct PointLight {
    render_from_light: Transform,
    medium_interface: MediumInterface,

    i: ArcIntern<DenselySampledSpectrum>,
    scale: Float,
}

#[derive(Builder)]
#[builder(
    public,
    name = "PointLightBuilder",
    build_fn(private, name = "build_params")
)]
struct PointLightParams<'a> {
    render_from_light: Transform,
    medium_interface: MediumInterface,

    i: &'a SpectrumEnum<'a>,
    scale: Float,
}

impl<'a> PointLightBuilder<'a> {
    pub fn build(&self) -> Result<PointLight, PointLightBuilderError> {
        let params = self.build_params()?;

        Ok(PointLight {
            render_from_light: params.render_from_light,
            medium_interface: params.medium_interface,
            i: SpectrumCache::lookup_spectrum(params.i),
            scale: params.scale,
        })
    }
}

impl PointLight {
    pub fn builder<'a>() -> PointLightBuilder<'a> {
        PointLightBuilder::default()
    }
}

impl Light for PointLight {
    fn phi(&self, wavelengths: &SampledWavelengths) -> SampledSpectrum {
        4.0 * PI * self.scale * self.i.sample(wavelengths)
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
        let wi = (p - ctx.p()).normalized();
        let li = self.scale * self.i.sample(wavelengths) / p.distance_squared(ctx.p());

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
