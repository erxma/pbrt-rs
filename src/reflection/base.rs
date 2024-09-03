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

impl From<BxDFFlags> for BxDFReflTransFlags {
    fn from(value: BxDFFlags) -> Self {
        Self::from_bits_truncate(value.bits())
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
        outgoing_render: Vec3f,
        incident_render: Vec3f,
        mode: TransportMode,
    ) -> Option<SampledSpectrum> {
        let incident = self.render_to_local(incident_render);
        let outgoing = self.render_to_local(outgoing_render);

        if outgoing.z() != 0.0 {
            Some(self.bxdf.func(outgoing, incident, mode))
        } else {
            None
        }
    }

    pub fn sample_func(
        &self,
        outgoing_render: Vec3f,
        u: Float,
        u2: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        let outgoing = self.render_to_local(outgoing_render);
        if outgoing.z() == 0.0 || !sample_flags.intersects(self.flags().into()) {
            return None;
        }

        // Sample BxDF and return sample
        let mut bs = self.bxdf.sample_func(outgoing, u, u2, mode, sample_flags)?;
        if bs.value.is_all_zero() || bs.pdf == 0.0 || bs.incident.z() == 0.0 {
            None
        } else {
            bs.incident = self.local_to_render(bs.incident);
            Some(bs)
        }
    }

    pub fn pdf(
        &self,
        outgoing_render: Vec3f,
        incident_render: Vec3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        let outgoing = self.render_to_local(outgoing_render);
        let incident = self.render_to_local(incident_render);
        if outgoing.z() != 0.0 {
            self.bxdf.pdf(outgoing, incident, mode, sample_flags)
        } else {
            0.0
        }
    }

    pub fn flags(&self) -> BxDFFlags {
        self.bxdf.flags()
    }

    pub fn render_to_local(&self, v: Vec3f) -> Vec3f {
        self.shading_frame.to_local(v)
    }

    pub fn local_to_render(&self, v: Vec3f) -> Vec3f {
        self.shading_frame.from_local(v)
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
