use std::sync::RwLock;

use crate::{
    float::PI,
    geometry::{Bounds3f, Ray, Transform},
    math::{Normal3f, Point2f, Point3f, Vec3f},
    memory::cache::ArcIntern,
    sampling::spectrum::{DenselySampledSpectrum, SampledSpectrum, SampledWavelengths, Spectrum},
    Float,
};

use super::{base::SpectrumCache, Light, LightLiSample, LightSampleContext, LightType};

pub struct DirectionalLight {
    render_from_light: Transform,

    emitted_radiance: ArcIntern<DenselySampledSpectrum>,
    scale: Float,
    // Likely to be set late via preprocess()
    // Behind lock to take the burden off Light users; they won't be changed
    // during actual work anyway.
    scene_center: RwLock<Option<Point3f>>,
    scene_radius: RwLock<Option<Float>>,
}

impl DirectionalLight {
    pub fn new(
        render_from_light: Transform,
        emitted_radiance: &impl Spectrum,
        scale: Float,
    ) -> Self {
        Self {
            render_from_light,
            emitted_radiance: SpectrumCache::lookup_spectrum(emitted_radiance),
            scale,
            scene_center: RwLock::new(None),
            scene_radius: RwLock::new(None),
        }
    }
}

impl Light for DirectionalLight {
    fn phi(&self, wavelengths: &SampledWavelengths) -> SampledSpectrum {
        let scene_radius = self
            .scene_radius
            .read()
            .unwrap()
            .expect("Must call preprocess() with scene bounds info for DirectionalLight first");

        self.scale * self.emitted_radiance.sample(wavelengths) * PI * scene_radius * scene_radius
    }

    fn light_type(&self) -> LightType {
        LightType::DeltaDirection
    }

    fn sample_li(
        &self,
        ctx: LightSampleContext,
        _u: Point2f,
        wavelengths: &SampledWavelengths,
        _allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        let scene_radius = self
            .scene_radius
            .read()
            .unwrap()
            .expect("Must call preprocess() with scene bounds info for DirectionalLight first");
        let incident = (&self.render_from_light * Vec3f::UP).normalized();
        let p_outside = ctx.p() + incident * (2.0 * scene_radius);

        Some(LightLiSample {
            l: self.scale * self.emitted_radiance.sample(wavelengths),
            wi: incident,
            pdf: 1.0,
            p_light: p_outside,
            medium_interface: None,
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

    fn preprocess(&self, scene_bounds: Bounds3f) {
        let (scene_center, scene_radius) = scene_bounds.bounding_sphere();
        *self.scene_center.write().unwrap() = Some(scene_center);
        *self.scene_radius.write().unwrap() = Some(scene_radius);
    }
}
