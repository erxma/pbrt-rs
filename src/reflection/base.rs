use bitflags::bitflags;
use enum_dispatch::enum_dispatch;

use crate::{
    geometry::Frame,
    math::{Normal3f, Point2f, Vec3f},
    sampling::spectrum::SampledSpectrum,
    Float,
};

use super::{DielectricBxDF, DiffuseBxDF};

#[enum_dispatch]
pub enum BxDFEnum {
    Diffuse(DiffuseBxDF),
    Dielectric(DielectricBxDF),
}

#[enum_dispatch(BxDFEnum)]
pub trait BxDF {
    fn flags(&self) -> BxDFFlags;

    fn func(&self, outgoing: Vec3f, incident: Vec3f, mode: TransportMode) -> SampledSpectrum;
    fn sample_func(
        &self,
        outgoing: Vec3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample>;
    fn pdf(
        &self,
        outgoing: Vec3f,
        incident: Vec3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float;
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BxDFFlags: u32 {
        const REFLECTION = 1;
        const TRANSMISSION = 1 << 1;
        const DIFFUSE = 1 << 2;
        const GLOSSY = 1 << 3;
        const SPECULAR = 1 << 4;

        const DIFFUSE_REFLECTION = Self::DIFFUSE.bits() | Self::REFLECTION.bits();
        const DIFFUSE_TRANSMISSION = Self::DIFFUSE.bits() | Self::TRANSMISSION.bits();
        const GLOSSY_REFLECTION = Self::GLOSSY.bits() | Self::REFLECTION.bits();
        const GLOSSY_TRANSMISSION = Self::GLOSSY.bits() | Self::TRANSMISSION.bits();
        const SPECULAR_REFLECTION = Self::SPECULAR.bits() | Self::REFLECTION.bits();
        const SPECULAR_TRANSMISSION = Self::SPECULAR.bits() | Self::TRANSMISSION.bits();
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BxDFReflTransFlags: u32 {
        const REFLECTION = 1;
        const TRANSMISSION = 1 << 1;
    }
}

pub struct BSDF<'a, BxDF> {
    bxdf: &'a BxDF,
    shading_frame: Frame,
}

impl<'a, BxDF: super::BxDF> BSDF<'a, BxDF> {
    pub fn new(shading_normal: Normal3f, shading_dpdu: Vec3f, bxdf: &'a BxDF) -> Self {
        Self {
            bxdf,
            shading_frame: Frame::from_xz(shading_dpdu.normalized(), shading_normal.into()),
        }
    }

    pub fn eval(
        &self,
        _wo_render: Vec3f,
        _wi_render: Vec3f,
        _mode: TransportMode,
    ) -> Option<SampledSpectrum> {
        todo!()
    }
}

pub struct BSDFSample {
    pub value: SampledSpectrum,
    pub incident: Vec3f,
    pub pdf: Float,
    pub pdf_is_proportional: bool,
    pub flags: BxDFFlags,
    pub eta: Float,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportMode {
    Radiance,
    Importance,
}

pub(super) fn same_hemisphere(w: Vec3f, wp: Vec3f) -> bool {
    w.z() * wp.z() > 0.0
}
