use std::{borrow::Cow, sync::Arc};

use crate::{
    camera::Camera,
    lights::LightEnum,
    materials::{Material, MaterialEnum, MaterialEvalContext, UniversalTextureEvaluator},
    math::{
        difference_of_products, next_float_down, next_float_up, Normal3f, Point2f, Point3f,
        Point3fi, Tuple, Vec3f,
    },
    media::{MediumEnum, MediumInterface, PhaseFunctionEnum},
    memory::ScratchBuffer,
    reflection::{BxDFEnum, BSDF},
    sampling::{
        spectrum::{SampledSpectrum, SampledWavelengths},
        Sampler,
    },
    Float,
};
use derive_builder::Builder;

use super::{Ray, RayDifferential};

#[derive(Clone, Debug)]
pub struct SampleInteraction {
    pub pi: Point3fi,
    pub time: Float,

    pub n: Normal3f,
    pub uv: Point2f,
}

impl SampleInteraction {
    pub fn new(pi: Point3fi, time: Option<Float>, n: Normal3f, uv: Point2f) -> Self {
        let time = time.unwrap_or(0.0);
        Self { pi, time, n, uv }
    }

    pub fn spawn_ray(&self, dir: Vec3f) -> RayDifferential {
        RayDifferential::new_without_diff(Ray::new(
            self.offset_ray_origin(dir),
            dir,
            self.time,
            None,
        ))
    }

    fn offset_ray_origin(&self, w: Vec3f) -> Point3f {
        let n_as_v = Vec3f::from(self.n);
        // Find vector offset to corner of error bounds, compute initial po
        let d = n_as_v.abs().dot(self.pi.error());
        let mut offset = d * n_as_v;
        if w.dot(n_as_v) < 0.0 {
            offset *= -1.0;
        }
        let mut po = self.pi.midpoints_only() + offset;

        // Round offset point po away from p
        for i in 0..3 {
            if offset[i] > 0.0 {
                po[i] = next_float_up(po[i]);
            } else if offset[i] < 0.0 {
                po[i] = next_float_down(po[i]);
            }
        }

        po
    }
}

#[derive(Clone)]
pub struct SurfaceInteraction<'a> {
    pub pi: Point3fi,
    pub time: Float,
    pub wo: Option<Vec3f>,

    pub n: Normal3f,
    pub uv: Point2f,
    pub dpdu: Vec3f,
    pub dpdv: Vec3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
    pub shading: Shading,
    // TODO: Restructure these?
    pub material: Option<&'a MaterialEnum>,
    pub area_light: Option<&'a LightEnum>,
    pub medium_interface: Option<MediumInterface>,
    pub medium: Option<Arc<MediumEnum>>,
    pub dpdx: Vec3f,
    pub dpdy: Vec3f,
    pub dudx: Float,
    pub dvdx: Float,
    pub dudy: Float,
    pub dvdy: Float,
}

#[derive(Clone, Debug)]
pub struct Shading {
    pub n: Normal3f,
    pub dpdu: Vec3f,
    pub dpdv: Vec3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
}

#[derive(Builder)]
#[builder(
    name = "SurfaceInteractionBuilder",
    public,
    setter(strip_option),
    build_fn(private, name = "build_params")
)]
struct SurfaceInteractionParams {
    pi: Point3fi,
    uv: Point2f,
    wo: Option<Vec3f>,
    dpdu: Vec3f,
    dpdv: Vec3f,
    dndu: Normal3f,
    dndv: Normal3f,
    time: Float,
    flip_normal: bool,
    #[builder(default)]
    dpdx: Vec3f,
    #[builder(default)]
    dpdy: Vec3f,
    #[builder(default)]
    dudx: Float,
    #[builder(default)]
    dvdx: Float,
    #[builder(default)]
    dudy: Float,
    #[builder(default)]
    dvdy: Float,
}

impl<'a> SurfaceInteraction<'a> {
    pub fn builder() -> SurfaceInteractionBuilder {
        SurfaceInteractionBuilder::default()
    }

    // Update the shading geometry info
    pub fn set_shading_geometry(
        &mut self,
        mut n_s: Normal3f,
        dpdu_s: Vec3f,
        dpdv_s: Vec3f,
        dndu_s: Normal3f,
        dndv_s: Normal3f,
        orientation_is_authoritative: bool,
    ) {
        // Compute shading normal
        if orientation_is_authoritative {
            self.n = self.n.face_forward(n_s.into());
        } else {
            n_s = n_s.face_forward(self.n.into());
        }

        // Set shading values
        self.shading = Shading {
            n: n_s,
            dpdu: dpdu_s,
            dpdv: dpdv_s,
            dndu: dndu_s,
            dndv: dndv_s,
        };
    }

    pub fn set_properties(
        &mut self,
        material: Option<&'a MaterialEnum>,
        area_light: Option<&'a LightEnum>,
        prim_medium_interface: Option<MediumInterface>,
        ray_medium: Option<Arc<MediumEnum>>,
    ) {
        self.material = material;
        self.area_light = area_light;
        // FIXME
        if let Some(ref mi) = prim_medium_interface {
            if mi.is_transition() {
                self.medium_interface = prim_medium_interface;
            }
        } else {
            self.medium = ray_medium;
        }
    }

    pub fn emitted_radiance(&self, w: Vec3f, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self.area_light {
            Some(light) => light.radiance(self.pi.midpoints_only(), self.n, self.uv, w, lambda),
            None => SampledSpectrum::with_single_value(0.0),
        }
    }

    pub fn get_bsdf<'b>(
        &mut self,
        ray: &RayDifferential,
        wavelengths: &SampledWavelengths,
        camera: &impl Camera,
        scratch_buffer: &'b mut ScratchBuffer,
        sampler: &mut impl Sampler,
    ) -> Option<BSDF<'b, BxDFEnum>> {
        // Estimate (u, v) and pos differentials at intersection point
        self.compute_differentials(ray, camera, sampler.samples_per_pixel());

        // TODO: Resolve MixMaterial if necessary

        // Return unset BSDF if surface has no material
        if self.material.is_none() {
            return None;
        }

        // TODO: Eval normal or bump map, if present

        // Get shading dp/du and dp/dv using normal or bump map

        // Return BSDF
        let bsdf = self.material.unwrap().bsdf(
            &UniversalTextureEvaluator::new(),
            &MaterialEvalContext::from_surface_interaction(self),
            Cow::Borrowed(wavelengths),
            scratch_buffer,
        );

        Some(bsdf)
    }

    pub fn spawn_ray(&self, dir: Vec3f) -> RayDifferential {
        let ray = Ray::new(
            self.offset_ray_origin(dir),
            dir,
            self.time,
            self.medium.clone(),
        );
        RayDifferential::new_without_diff(ray)
    }

    pub fn compute_differentials(
        &mut self,
        ray_diffs: &RayDifferential,
        camera: &impl Camera,
        samples_per_pixel: usize,
    ) {
        match &ray_diffs.differentials {
            Some(diffs)
                if self.n.dot_v(diffs.rx_dir) != 0.0 && self.n.dot_v(diffs.ry_dir) != 0.0 =>
            {
                // Estimate screen-space change in p using differentials:
                // Compute auxillary intersection points with plane, px and py
                let d = -self.n.dot(self.pi.midpoints_only().into());
                let tx = (-self.n.dot(diffs.rx_origin.into()) - d) / self.n.dot_v(diffs.rx_dir);
                let px = diffs.rx_origin + tx * diffs.rx_dir;
                let ty = (-self.n.dot(diffs.ry_origin.into()) - d) / self.n.dot_v(diffs.ry_dir);
                let py = diffs.ry_origin + ty * diffs.ry_dir;

                self.dpdx = px - self.pi.midpoints_only();
                self.dpdy = py - self.pi.midpoints_only();
            }

            Some(_) | None => {
                // Estiamte screen-space change in p based on cam projection
                (self.dpdx, self.dpdy) = camera.approximate_dp_dxy(
                    self.pi.midpoints_only(),
                    self.n,
                    self.time,
                    samples_per_pixel,
                );
            }
        }

        // Estimate screen-space change in (u, v):

        // Compute A^T*A and its determinant
        // TODO: Move into seperate func?
        let ata00 = self.dpdu.dot(self.dpdu);
        let ata01 = self.dpdu.dot(self.dpdv);
        let ata11 = self.dpdv.dot(self.dpdv);
        let inv_det = {
            let inv_det = 1.0 / difference_of_products(ata00, ata11, ata01, ata01);
            if inv_det.is_finite() {
                inv_det
            } else {
                0.0
            }
        };

        // Compute A^T*b for x and y
        let atb0x = self.dpdu.dot(self.dpdx);
        let atb1x = self.dpdv.dot(self.dpdx);
        let atb0y = self.dpdu.dot(self.dpdy);
        let atb1y = self.dpdv.dot(self.dpdy);

        // Compute u, v deriatives with respect to x, y
        self.dudx = difference_of_products(ata11, atb0x, ata01, atb1x) * inv_det;
        self.dvdx = difference_of_products(ata00, atb1x, ata01, atb0x) * inv_det;
        self.dudy = difference_of_products(ata11, atb0y, ata01, atb1y) * inv_det;
        self.dvdy = difference_of_products(ata00, atb1y, ata01, atb0y) * inv_det;

        // Clamp derivatives to reasonable values, in case of very large values or inf,
        // e.g. with highly distorted (u, v) or at silhouette edges
        self.dudx = if self.dudx.is_finite() {
            self.dudx.clamp(-1e8, 1e8)
        } else {
            0.0
        };
        self.dvdx = if self.dvdx.is_finite() {
            self.dvdx.clamp(-1e8, 1e8)
        } else {
            0.0
        };
        self.dudy = if self.dudy.is_finite() {
            self.dudy.clamp(-1e8, 1e8)
        } else {
            0.0
        };
        self.dvdy = if self.dvdy.is_finite() {
            self.dvdy.clamp(-1e8, 1e8)
        } else {
            0.0
        };
    }

    pub fn offset_ray_origin(&self, w: Vec3f) -> Point3f {
        Ray::offset_ray_origin(self.pi, self.n, w)
    }

    pub fn medium(&self) -> Option<&MediumEnum> {
        self.medium_interface
            .as_ref()
            .map(|mi| &*mi.inside)
            .or(self.medium.as_deref())
    }

    pub fn skip_intersection(&self, ray_diff: &mut RayDifferential, t: Float) {
        *ray_diff = self.spawn_ray(ray_diff.ray.dir);
        if let Some(ref mut diffs) = ray_diff.differentials {
            diffs.rx_origin += t * diffs.rx_dir;
            diffs.ry_origin += t * diffs.ry_dir;
        }
    }
}

impl SurfaceInteractionBuilder {
    pub fn build<'a, 'b>(&self) -> Result<SurfaceInteraction<'a>, SurfaceInteractionBuilderError> {
        let params = self.build_params()?;

        let mut n = params.dpdu.cross(params.dpdv).normalized().into();
        // Adjust normal based on orientation and handedness
        if params.flip_normal {
            n *= -1.0;
        }
        Ok(SurfaceInteraction {
            pi: params.pi,
            time: params.time,
            wo: params.wo,

            n,
            uv: params.uv,
            dpdu: params.dpdu,
            dpdv: params.dpdv,
            dndu: params.dndu,
            dndv: params.dndv,
            shading: Shading {
                n,
                dpdu: params.dpdu,
                dpdv: params.dpdv,
                dndu: params.dndu,
                dndv: params.dndv,
            },
            material: None,
            area_light: None,
            medium_interface: None,
            medium: None,
            dpdx: params.dpdx,
            dpdy: params.dpdy,
            dudx: params.dudx,
            dvdx: params.dvdx,
            dudy: params.dudy,
            dvdy: params.dvdy,
        })
    }
}

pub enum MediumInteraction<'a> {
    Interface(MediumInterfaceInteraction),
    IntraMedium(IntraMediumInteraction<'a>),
}

impl<'a> MediumInteraction<'a> {
    pub fn pi(&self) -> Point3fi {
        match self {
            MediumInteraction::Interface(intr) => intr.pi,
            MediumInteraction::IntraMedium(intr) => intr.pi,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MediumInterfaceInteraction {
    pub pi: Point3fi,
    pub time: Float,
    pub wo: Vec3f,

    pub n: Normal3f,
    pub medium_interface: MediumInterface,
    pub phase: PhaseFunctionEnum,
}

impl MediumInterfaceInteraction {
    pub fn new(
        p: Point3f,
        wo: Vec3f,
        time: Float,
        n: Normal3f,
        medium_interface: MediumInterface,
        phase: PhaseFunctionEnum,
    ) -> Self {
        Self {
            pi: Point3fi::from(p),
            time,
            wo,

            n,
            medium_interface,
            phase,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IntraMediumInteraction<'a> {
    pub pi: Point3fi,
    pub time: Float,
    pub wo: Vec3f,

    pub medium: &'a MediumEnum,
    pub phase: PhaseFunctionEnum,
}

impl<'a> IntraMediumInteraction<'a> {
    pub fn new(
        p: Point3f,
        wo: Vec3f,
        time: Float,
        medium: &'a MediumEnum,
        phase: PhaseFunctionEnum,
    ) -> Self {
        Self {
            pi: Point3fi::from(p),
            time,
            wo,

            medium,
            phase,
        }
    }
}
