use crate::{
    geometry::Ray,
    math::Point3f,
    sampling::spectrum::{SampledSpectrum, SampledWavelengths},
    Float,
};

#[derive(Clone, Debug, PartialEq)]
pub enum MediumEnum {}

pub trait Medium {
    type MajorantIter: Iterator<Item = RayMajorantSegment>;

    fn is_emissive(&self) -> bool;
    fn sample_point(&self, p: Point3f, wavelengths: &SampledWavelengths) -> MediumProperties;
    fn sample_ray(
        &self,
        ray: &Ray,
        t_max: Option<Float>,
        wavelengths: &SampledWavelengths,
    ) -> Self::MajorantIter;
}

pub struct MediumProperties {}

pub struct RayMajorantSegment {
    t_min: Float,
    t_max: Float,
    sigma_maj: SampledSpectrum,
}

#[derive(Clone, Debug)]
pub struct PhaseFunction {}
