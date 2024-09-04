use crate::{
    float::FRAC_1_PI,
    math::{Point2f, Vec3f},
    sampling::{
        routines::{cosine_hemisphere_pdf, sample_cosine_hemisphere},
        spectrum::SampledSpectrum,
    },
    Float,
};

use super::{
    base::same_hemisphere, BSDFSample, BxDF, BxDFFlags, BxDFReflTransFlags, TransportMode,
};

#[derive(Debug)]
pub struct DiffuseBxDF {
    reflectance: SampledSpectrum,
}

impl DiffuseBxDF {
    pub fn new(reflectance: SampledSpectrum) -> Self {
        Self { reflectance }
    }
}

impl BxDF for DiffuseBxDF {
    fn flags(&self) -> BxDFFlags {
        BxDFFlags::DIFFUSE_REFLECTION
    }

    fn func(&self, outgoing: Vec3f, incident: Vec3f, _mode: TransportMode) -> SampledSpectrum {
        if same_hemisphere(outgoing, incident) {
            &self.reflectance * FRAC_1_PI
        } else {
            SampledSpectrum::with_single_value(0.0)
        }
    }

    fn sample_func(
        &self,
        outgoing: Vec3f,
        _uc: Float,
        u: Point2f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if sample_flags.contains(BxDFReflTransFlags::REFLECTION) {
            // Sample cosine-weighted hemisphere to compute wi and pdf
            let mut incident = sample_cosine_hemisphere(u);
            if outgoing.z() < 0.0 {
                *incident.z_mut() *= -1.0;
            }
            let pdf = cosine_hemisphere_pdf(incident.cos_theta().abs());

            Some(BSDFSample {
                value: &self.reflectance * FRAC_1_PI,
                incident,
                pdf,
                pdf_is_proportional: false,
                flags: BxDFFlags::DIFFUSE_REFLECTION,
                eta: 1.0,
            })
        } else {
            None
        }
    }

    fn pdf(
        &self,
        outgoing: Vec3f,
        incident: Vec3f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if sample_flags.contains(BxDFReflTransFlags::REFLECTION)
            && same_hemisphere(outgoing, incident)
        {
            cosine_hemisphere_pdf(incident.cos_theta().abs())
        } else {
            0.0
        }
    }
}
