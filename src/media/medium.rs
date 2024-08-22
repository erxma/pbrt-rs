use std::iter::{once, Once};

use crate::{
    geometry::Ray,
    math::Point3f,
    sampling::spectrum::{DenselySampledSpectrum, SampledSpectrum, SampledWavelengths, Spectrum},
    Float,
};

use super::{HGPhaseFunction, PhaseFunctionEnum};

#[derive(Clone, Debug, PartialEq)]
pub enum MediumEnum {}

pub trait Medium {
    type MajorantIter: Iterator<Item = RayMajorantSegment>;

    fn is_emissive(&self) -> bool;
    fn sample_point(&self, p: Point3f, wavelengths: &SampledWavelengths) -> MediumProperties;
    fn sample_ray(
        &self,
        ray: &Ray,
        t_max: Float,
        wavelengths: &SampledWavelengths,
    ) -> Self::MajorantIter;
}

pub struct MediumProperties<'a> {
    pub sigma_a: SampledSpectrum,
    pub sigma_s: SampledSpectrum,
    pub phase: &'a PhaseFunctionEnum,
    pub emission: SampledSpectrum,
}

pub struct RayMajorantSegment {
    t_min: Float,
    t_max: Float,
    sigma_maj: SampledSpectrum,
}

pub struct HomogeneousMedium {
    sigma_a_spec: DenselySampledSpectrum,
    sigma_s_spec: DenselySampledSpectrum,
    emission_spec: DenselySampledSpectrum,
    phase: PhaseFunctionEnum,
}

impl HomogeneousMedium {
    pub fn new(
        sigma_a: &impl Spectrum,
        sigma_s: &impl Spectrum,
        sigma_scale: Float,
        emission: &impl Spectrum,
        g: Float,
    ) -> Self {
        let sigma_a_spec = DenselySampledSpectrum::new(sigma_a, None, None).scaled(sigma_scale);
        let sigma_s_spec = DenselySampledSpectrum::new(sigma_s, None, None).scaled(sigma_scale);
        let emission_spec = DenselySampledSpectrum::new(emission, None, None);
        let phase = HGPhaseFunction::new(g).into();

        Self {
            sigma_a_spec,
            sigma_s_spec,
            emission_spec,
            phase,
        }
    }
}

impl Medium for HomogeneousMedium {
    type MajorantIter = HomegeneousMajorantIter;

    fn is_emissive(&self) -> bool {
        self.emission_spec.max_value() > 0.0
    }

    fn sample_point(&self, p: Point3f, wavelengths: &SampledWavelengths) -> MediumProperties {
        MediumProperties {
            sigma_a: self.sigma_a_spec.sample(wavelengths),
            sigma_s: self.sigma_s_spec.sample(wavelengths),
            phase: &self.phase,
            emission: self.emission_spec.sample(wavelengths),
        }
    }

    fn sample_ray(
        &self,
        _ray: &Ray,
        t_max: Float,
        wavelengths: &SampledWavelengths,
    ) -> Self::MajorantIter {
        let sigma_a = self.sigma_a_spec.sample(wavelengths);
        let sigma_s = self.sigma_s_spec.sample(wavelengths);
        HomegeneousMajorantIter::new(0.0, t_max, sigma_a + sigma_s)
    }
}

pub struct HomegeneousMajorantIter(Once<RayMajorantSegment>);

impl HomegeneousMajorantIter {
    fn new(t_min: Float, t_max: Float, sigma_maj: SampledSpectrum) -> Self {
        let segment = RayMajorantSegment {
            t_min,
            t_max,
            sigma_maj,
        };
        Self(once(segment))
    }
}

impl Iterator for HomegeneousMajorantIter {
    type Item = RayMajorantSegment;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
