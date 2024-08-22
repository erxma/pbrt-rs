use crate::{
    math::{safe_sqrt, Normal3f, Point2f, Vec3f},
    sampling::spectrum::SampledSpectrum,
    Float,
};

use super::base::{BSDFSample, BxDF, BxDFFlags, BxDFReflTransFlags, TransportMode};

pub struct DielectricBxDF {
    eta: Float,
    microfacet_distrib: TrowbridgeReitz,
}

impl DielectricBxDF {
    pub fn new(eta: Float, microfacet_distrib: TrowbridgeReitz) -> Self {
        Self {
            eta,
            microfacet_distrib,
        }
    }
}

impl BxDF for DielectricBxDF {
    fn flags(&self) -> BxDFFlags {
        let mut flags = if self.eta == 1.0 {
            BxDFFlags::TRANSMISSION
        } else {
            BxDFFlags::REFLECTION | BxDFFlags::TRANSMISSION
        };

        flags |= if self.microfacet_distrib.effectively_smooth() {
            BxDFFlags::SPECULAR
        } else {
            BxDFFlags::GLOSSY
        };

        flags
    }

    fn sample_f(
        &self,
        outgoing_dir: Vec3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if self.eta == 1.0 || self.microfacet_distrib.effectively_smooth() {
            // Sample perfect specular dielectric BSDF
            let reflectance = fresnel_dielectric(outgoing_dir.cos_theta(), self.eta);
            let transmittance = 1.0 - reflectance;

            // Compute probs for sampling reflection and transmission
            let prob_reflect = if sample_flags.contains(BxDFReflTransFlags::REFLECTION) {
                reflectance
            } else {
                0.0
            };
            let prob_transmit = if sample_flags.contains(BxDFReflTransFlags::TRANSMISSION) {
                transmittance
            } else {
                0.0
            };
            if prob_reflect == 0.0 && prob_transmit == 0.0 {
                return None;
            }

            if uc < prob_reflect / (prob_reflect + prob_transmit) {
                let incident_dir =
                    Vec3f::new(-outgoing_dir.x(), -outgoing_dir.y(), outgoing_dir.z());
                let fresnel = SampledSpectrum::with_single_value(
                    reflectance / incident_dir.cos_theta().abs(),
                );
                let sample = BSDFSample {
                    value: fresnel,
                    incident_dir,
                    pdf: prob_reflect / (prob_reflect + prob_transmit),
                    pdf_is_proportional: false,
                    flags: BxDFFlags::SPECULAR_REFLECTION,
                    eta: 1.0,
                };
                Some(sample)
            } else {
                // Compute ray dir for specular transmission
                let (incident_dir, etap) =
                    refract(outgoing_dir, Normal3f::new(0.0, 0.0, 1.0), self.eta)?;
                let mut ft = SampledSpectrum::with_single_value(
                    transmittance / incident_dir.cos_theta().abs(),
                );
                // Account for non-symmetry with transmission to different medium
                if mode == TransportMode::Radiance {
                    ft /= etap * etap;
                }
                let sample = BSDFSample {
                    value: ft,
                    incident_dir,
                    pdf: prob_transmit / (prob_reflect + prob_transmit),
                    pdf_is_proportional: false,
                    flags: BxDFFlags::SPECULAR_TRANSMISSION,
                    eta: etap,
                };
                Some(sample)
            }
        } else {
            // Sample rough dielectric BSDF

            todo!()
        }
    }
}

pub struct TrowbridgeReitz {}

impl TrowbridgeReitz {
    pub fn effectively_smooth(&self) -> bool {
        todo!()
    }
}

fn fresnel_dielectric(mut cos_theta_i: Float, mut eta: Float) -> Float {
    cos_theta_i = cos_theta_i.clamp(-1.0, 1.0);

    // Potentially flipinterface orientation for Fresnel equations
    if cos_theta_i < 0.0 {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
    }

    // Compute cos theta for Fresnel equations using Snell's law
    let sin2_theta_i = 1.0 - cos_theta_i * cos_theta_i;
    let sin2_theta_t = sin2_theta_i / (eta * eta);

    // Handle total internal reflection case
    if sin2_theta_t >= 1.0 {
        // Indicates all scattering takes place via reflection component
        return 1.0;
    }

    let cos_theta_t = safe_sqrt(1.0 - sin2_theta_t);

    let r_parallel = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    let r_perpendicular = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

    (r_parallel.powi(2) + r_perpendicular.powi(2)) / 2.0
}

fn refract(incident_dir: Vec3f, mut n: Normal3f, mut eta: Float) -> Option<(Vec3f, Float)> {
    let mut cos_theta_i = n.dot(incident_dir.into());

    // Potentially flip interface orientation for Snell's law
    if cos_theta_i < 0.0 {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
        n = -n;
    }

    // Compute cos theta using Snell's law
    let sin2_theta_i = (1.0 - cos_theta_i * cos_theta_i).max(0.0);
    let sin2_theta_t = sin2_theta_i / (eta * eta);

    // Handle total internal reflection case
    if sin2_theta_t >= 1.0 {
        return None;
    }

    let cos_theta_t = safe_sqrt(1.0 - sin2_theta_t);
    let refracted_dir = -incident_dir / eta + (cos_theta_i / eta - cos_theta_t) * Vec3f::from(n);

    // Provide relative IOR along ray
    let etap = eta;

    Some((refracted_dir, etap))
}
