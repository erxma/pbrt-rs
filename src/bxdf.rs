use crate::{math::Vec3f, sampling::spectrum::SampledSpectrum};

pub struct BSDF {}

impl BSDF {
    pub fn eval(
        &self,
        _wo_render: Vec3f,
        _wi_render: Vec3f,
        _mode: TransportMode,
    ) -> Option<SampledSpectrum> {
        todo!()
    }
}

pub enum TransportMode {
    Radiance,
    Importance,
}
