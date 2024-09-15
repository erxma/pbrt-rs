use crate::{
    core::{constants::PI, lerp, safe_sqrt, Float, Normal3f, Point2f, Vec2f, Vec3f},
    reflection::base::same_hemisphere,
    sampling::{routines::sample_uniform_disk_polar, spectrum::SampledSpectrum},
};

use super::base::{BSDFSample, BxDF, BxDFFlags, BxDFReflTransFlags, TransportMode};

#[derive(Debug)]
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

    // Sample perfect specular dielectric BSDF
    fn sample_perfect_specular(
        &self,
        outgoing: Vec3f,
        uc: Float,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        let reflectance = fresnel_dielectric(outgoing.cos_theta(), self.eta);
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
            Some(self.sample_perfect_specular_reflect(
                outgoing,
                reflectance,
                prob_reflect / (prob_reflect + prob_transmit),
            ))
        } else {
            self.sample_perfect_specular_transmit(
                outgoing,
                mode,
                transmittance,
                prob_transmit / (prob_reflect + prob_transmit),
            )
        }
    }

    fn sample_perfect_specular_reflect(
        &self,
        outgoing: Vec3f,
        reflectance: Float,
        normalized_prob_reflect: Float,
    ) -> BSDFSample {
        let incident = Vec3f::new(-outgoing.x(), -outgoing.y(), outgoing.z());
        let fresnel = SampledSpectrum::with_single_value(reflectance / incident.cos_theta().abs());

        BSDFSample {
            value: fresnel,
            incident,
            pdf: normalized_prob_reflect,
            pdf_is_proportional: false,
            flags: BxDFFlags::SPECULAR_REFLECTION,
            eta: 1.0,
        }
    }

    fn sample_perfect_specular_transmit(
        &self,
        outgoing: Vec3f,
        mode: TransportMode,
        transmittance: Float,
        normalized_prob_transmit: Float,
    ) -> Option<BSDFSample> {
        // Compute ray dir for specular transmission
        let (incident, etap) = refract(outgoing, Normal3f::new(0.0, 0.0, 1.0), self.eta)?;
        let mut ft = SampledSpectrum::with_single_value(transmittance / incident.cos_theta().abs());
        // Account for non-symmetry with transmission to different medium
        if mode == TransportMode::Radiance {
            ft /= etap * etap;
        }

        Some(BSDFSample {
            value: ft,
            incident,
            pdf: normalized_prob_transmit,
            pdf_is_proportional: false,
            flags: BxDFFlags::SPECULAR_TRANSMISSION,
            eta: etap,
        })
    }

    // Sample rough dielectric BSDF
    fn sample_rough(
        &self,
        outgoing: Vec3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        let microfacet_n = self.microfacet_distrib.sample_microfacet_n(outgoing, u);
        let reflectance = fresnel_dielectric(outgoing.dot(microfacet_n), self.eta);
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
            self.sample_rough_reflect(
                outgoing,
                microfacet_n,
                reflectance,
                prob_reflect / (prob_reflect + prob_transmit),
            )
        } else {
            self.sample_rough_transmit(
                outgoing,
                mode,
                microfacet_n,
                transmittance,
                prob_transmit / (prob_reflect + prob_transmit),
            )
        }
    }

    fn sample_rough_reflect(
        &self,
        outgoing: Vec3f,
        microfacet_n: Vec3f,
        reflectance: Float,
        normalized_prob_reflect: Float,
    ) -> Option<BSDFSample> {
        // Sample reflection at rough dielectric interface
        let incident = reflect(outgoing, microfacet_n);

        if !same_hemisphere(outgoing, incident) {
            return None;
        }

        // Compute PDF of rough dielectric reflection
        let pdf = self.microfacet_distrib.pdf(outgoing, microfacet_n)
            / (4.0 * outgoing.absdot(incident))
            * normalized_prob_reflect;

        let fresnel = SampledSpectrum::with_single_value(
            self.microfacet_distrib.density(microfacet_n)
                * self
                    .microfacet_distrib
                    .masking_shadowing(outgoing, incident)
                * reflectance
                / (4.0 * incident.cos_theta() * outgoing.cos_theta()),
        );

        Some(BSDFSample {
            value: fresnel,
            incident,
            pdf,
            pdf_is_proportional: false,
            flags: BxDFFlags::GLOSSY_REFLECTION,
            eta: 1.0,
        })
    }

    fn sample_rough_transmit(
        &self,
        outgoing: Vec3f,
        mode: TransportMode,
        microfacet_n: Vec3f,
        transmittance: Float,
        normalized_prob_transmit: Float,
    ) -> Option<BSDFSample> {
        // Compute ray dir for transmission
        // Due to floating point, refract may indicate a total internal reflection
        // even though transmission wouldn't have been sampled in that case
        let (incident, etap) = refract(outgoing, microfacet_n.into(), self.eta)?;

        // Similar possible inconsistencies
        if same_hemisphere(outgoing, incident) || incident.z() == 0.0 {
            return None;
        }

        let in_dot_m = incident.dot(microfacet_n);
        let out_dot_m = outgoing.dot(microfacet_n);

        let denom = (in_dot_m + out_dot_m / etap).powi(2);
        let dwm_dwi = in_dot_m.abs() / denom;
        let pdf = self.microfacet_distrib.pdf(outgoing, microfacet_n)
            * dwm_dwi
            * normalized_prob_transmit;

        // Evaluate BRDF and return sample
        let mut ft = SampledSpectrum::with_single_value(
            transmittance
                * self.microfacet_distrib.density(microfacet_n)
                * self
                    .microfacet_distrib
                    .masking_shadowing(outgoing, incident)
                * (in_dot_m * out_dot_m / (incident.cos_theta() * outgoing.cos_theta() * denom))
                    .abs(),
        );
        // Account for non-symmetry with transmission to different medium
        if mode == TransportMode::Radiance {
            ft /= etap * etap;
        }

        Some(BSDFSample {
            value: ft,
            incident,
            pdf,
            pdf_is_proportional: false,
            flags: BxDFFlags::GLOSSY_TRANSMISSION,
            eta: etap,
        })
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

    fn eval(&self, outgoing: Vec3f, incident: Vec3f, mode: TransportMode) -> SampledSpectrum {
        if self.eta == 1.0 || self.microfacet_distrib.effectively_smooth() {
            SampledSpectrum::with_single_value(0.0)
        } else {
            // Evaluate sampling PDF of rough dielectric BSDF

            // Compute generalized half vector w_m
            let cos_theta_o = outgoing.cos_theta();
            let cos_theta_i = incident.cos_theta();
            let reflect = cos_theta_o * cos_theta_i > 0.0;
            let etap = if reflect {
                1.0
            } else if cos_theta_o > 0.0 {
                self.eta
            } else {
                1.0 / self.eta
            };
            let mut microfacet_n = incident * etap + outgoing;
            if cos_theta_i == 0.0 || cos_theta_o == 0.0 || microfacet_n.length_squared() == 0.0 {
                return SampledSpectrum::with_single_value(0.0);
            }
            microfacet_n = Normal3f::from(microfacet_n.normalized())
                .face_forward(Vec3f::new(0.0, 0.0, 1.0))
                .into();

            // Discard backfacing microfacets
            let in_dot_m = incident.dot(microfacet_n);
            let out_dot_m = outgoing.dot(microfacet_n);

            if in_dot_m * cos_theta_i < 0.0 || out_dot_m * cos_theta_o < 0.0 {
                return SampledSpectrum::with_single_value(0.0);
            }

            let fresnel = fresnel_dielectric(out_dot_m, self.eta);
            if reflect {
                // Compute reflection
                SampledSpectrum::with_single_value(
                    self.microfacet_distrib.density(microfacet_n)
                        * self
                            .microfacet_distrib
                            .masking_shadowing(outgoing, incident)
                        * fresnel
                        / (4.0 * cos_theta_i * cos_theta_o).abs(),
                )
            } else {
                // Compute transmission
                let denom = (in_dot_m + out_dot_m / etap).powi(2) * cos_theta_i * cos_theta_o;
                let mut ft = self.microfacet_distrib.density(microfacet_n)
                    * (1.0 - fresnel)
                    * self
                        .microfacet_distrib
                        .masking_shadowing(outgoing, incident)
                    * (in_dot_m * out_dot_m / denom).abs();
                // Account for non-symmetry with transmission to different medium
                if mode == TransportMode::Radiance {
                    ft /= etap * etap;
                }

                SampledSpectrum::with_single_value(ft)
            }
        }
    }

    fn sample_func(
        &self,
        outgoing: Vec3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if self.eta == 1.0 || self.microfacet_distrib.effectively_smooth() {
            self.sample_perfect_specular(outgoing, uc, mode, sample_flags)
        } else {
            self.sample_rough(outgoing, uc, u, mode, sample_flags)
        }
    }

    fn pdf(
        &self,
        outgoing: Vec3f,
        incident: Vec3f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if self.eta == 1.0 || self.microfacet_distrib.effectively_smooth() {
            return 0.0;
        }

        // Evaluate PDF of rough dielectric BSDF
        // Compute generalized half vector w_m
        let cos_theta_o = outgoing.cos_theta();
        let cos_theta_i = incident.cos_theta();
        let reflect = cos_theta_o * cos_theta_i > 0.0;
        let etap = if reflect {
            1.0
        } else if cos_theta_o > 0.0 {
            self.eta
        } else {
            1.0 / self.eta
        };
        let mut microfacet_n = incident * etap + outgoing;
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 || microfacet_n.length_squared() == 0.0 {
            return 0.0;
        }
        microfacet_n = Normal3f::from(microfacet_n.normalized())
            .face_forward(Vec3f::new(0.0, 0.0, 1.0))
            .into();

        // Discard backfacing microfacets
        let in_dot_m = incident.dot(microfacet_n);
        let out_dot_m = outgoing.dot(microfacet_n);

        if in_dot_m * cos_theta_i < 0.0 || out_dot_m * cos_theta_o < 0.0 {
            return 0.0;
        }

        // Determine Fresnel reflectance of rough dielectric boundary
        let reflectance = fresnel_dielectric(out_dot_m, self.eta);
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
            return 0.0;
        }

        // Return PDF for rough dielectric
        if reflect {
            // Compute reflection
            self.microfacet_distrib.pdf(outgoing, microfacet_n) / (4.0 * out_dot_m.abs())
                * prob_reflect
                / (prob_reflect + prob_transmit)
        } else {
            // Compute transmission
            let denom = (in_dot_m + out_dot_m / etap).powi(2) / etap;
            let dwm_dwi = in_dot_m.abs() / denom;
            self.microfacet_distrib.pdf(outgoing, microfacet_n) * dwm_dwi * prob_transmit
                / (prob_reflect + prob_transmit)
        }
    }
}

#[derive(Debug)]
pub struct TrowbridgeReitz {
    alpha_x: Float,
    alpha_y: Float,
}

impl TrowbridgeReitz {
    pub fn new(alpha_x: Float, alpha_y: Float) -> Self {
        Self { alpha_x, alpha_y }
    }
    pub fn effectively_smooth(&self) -> bool {
        self.alpha_x.max(self.alpha_y) < 0.001
    }

    pub fn sample_microfacet_n(&self, incident: Vec3f, u: Point2f) -> Vec3f {
        // Transform incident to hemispherical config
        let mut in_hemi = Vec3f::new(
            self.alpha_x * incident.x(),
            self.alpha_y * incident.y(),
            incident.z(),
        )
        .normalized();
        if in_hemi.z() < 0.0 {
            in_hemi = -in_hemi;
        }

        // Find orthonormal basis for visible normal sampling
        let t1 = if in_hemi.z() < 0.99999 {
            Vec3f::new(0.0, 0.0, 1.0).cross(in_hemi).normalized()
        } else {
            Vec3f::new(1.0, 0.0, 0.0)
        };
        let t2 = in_hemi.cross(t1);

        // Generate uniformly distributed point on the unit disk
        let mut p = sample_uniform_disk_polar(u);

        // Warp hemispherical projection for visible normal sampling
        let height = (1.0 - p.x() * p.x()).sqrt();
        *p.y_mut() = lerp(height, p.y(), (1.0 + in_hemi.z()) / 2.0);

        // Reproject to hemisphere and transform normal to ellipsoid config
        let pz = (1.0 - Vec2f::from(p).length_squared()).max(0.0).sqrt();
        let nh = p.x() * t1 + p.y() * t2 + pz * in_hemi;

        Vec3f::new(
            self.alpha_x * nh.x(),
            self.alpha_y * nh.y(),
            nh.z().max(1e-6),
        )
        .normalized()
    }

    pub fn masking_shadowing(&self, outgoing: Vec3f, incident: Vec3f) -> Float {
        1.0 / (1.0 + self.lambda(outgoing) + self.lambda(incident))
    }

    // i.e. D
    pub fn density(&self, microfacet_n: Vec3f) -> Float {
        let tan2_theta = microfacet_n.tan2_theta();
        if tan2_theta.is_finite() {
            let cos4_theta = microfacet_n.cos2_theta().powi(2);
            let e = tan2_theta
                * ((microfacet_n.cos_phi() / self.alpha_x).powi(2)
                    + (microfacet_n.sin_phi() / self.alpha_y).powi(2));
            1.0 / (PI * self.alpha_x * self.alpha_y * cos4_theta * (1.0 + e).powi(2))
        } else {
            0.0
        }
    }

    pub fn density_visible(&self, w: Vec3f, microfacet_n: Vec3f) -> Float {
        self.masking(w) / w.cos_theta().abs() * self.density(microfacet_n) * w.absdot(microfacet_n)
    }

    pub fn pdf(&self, w: Vec3f, microfacet_n: Vec3f) -> Float {
        self.density_visible(w, microfacet_n)
    }

    pub fn masking(&self, w: Vec3f) -> Float {
        1.0 / (1.0 + self.lambda(w))
    }

    pub fn lambda(&self, w: Vec3f) -> Float {
        let tan2_theta = w.tan2_theta();
        if tan2_theta.is_finite() {
            let alpha2 =
                (w.cos_phi() * self.alpha_x).powi(2) + (w.sin_phi() * self.alpha_y).powi(2);
            ((1.0 + alpha2 * tan2_theta).sqrt() - 1.0) / 2.0
        } else {
            0.0
        }
    }

    pub fn roughness_to_alpha(roughness: Float) -> Float {
        roughness.sqrt()
    }
}

fn fresnel_dielectric(mut cos_theta_i: Float, mut eta: Float) -> Float {
    cos_theta_i = cos_theta_i.clamp(-1.0, 1.0);

    // Potentially flip interface orientation for Fresnel equations
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
    let r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

    (r_parallel.powi(2) + r_perp.powi(2)) / 2.0
}

fn refract(incident_dir: Vec3f, mut n: Normal3f, mut eta: Float) -> Option<(Vec3f, Float)> {
    let mut cos_theta_i = n.dot_v(incident_dir);

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

fn reflect(outgoing: Vec3f, n: Vec3f) -> Vec3f {
    -outgoing + 2.0 * outgoing.dot(n) * n
}
