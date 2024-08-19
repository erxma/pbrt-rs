use crate::{
    bxdf::BSDF,
    camera::Camera,
    lights::LightEnum,
    materials::Material,
    math::{next_float_down, next_float_up, Normal3f, Point2f, Point3f, Point3fi, Tuple, Vec3f},
    media::{Medium, MediumInterface, PhaseFunction},
    memory::ScratchBuffer,
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
    pub material: Option<&'a Material>,
    pub area_light: Option<&'a LightEnum>,
    pub medium_interface: Option<&'a MediumInterface>,
    pub medium: Option<&'a Medium>,
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
        material: Option<&'a Material>,
        area_light: Option<&'a LightEnum>,
        prim_medium_interface: Option<&'a MediumInterface>,
        ray_medium: Option<&'a Medium>,
    ) {
        self.material = material;
        self.area_light = area_light;
        // FIXME
        if let Some(mi) = prim_medium_interface {
            if mi.is_transition() {
                self.medium_interface = prim_medium_interface;
            }
        } else {
            self.medium = ray_medium;
        }
    }

    pub fn emitted_radiance(&self, _w: Vec3f, _lambda: &SampledWavelengths) -> SampledSpectrum {
        todo!()
    }

    pub fn get_bsdf(
        &mut self,
        _ray: &RayDifferential,
        _wavelengths: &SampledWavelengths,
        _camera: &impl Camera,
        _scratch_buffer: &mut ScratchBuffer,
        _sampler: &mut impl Sampler,
    ) -> Option<&BSDF> {
        todo!()
    }

    pub fn spawn_ray(&self, _dir: Vec3f) -> RayDifferential {
        todo!()
    }
}

impl SurfaceInteractionBuilder {
    pub fn build<'a>(&self) -> Result<SurfaceInteraction<'a>, SurfaceInteractionBuilderError> {
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
    pub phase: PhaseFunction,
}

impl MediumInterfaceInteraction {
    pub fn new(
        p: Point3f,
        wo: Vec3f,
        time: Float,
        n: Normal3f,
        medium_interface: MediumInterface,
        phase: PhaseFunction,
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

    pub medium: &'a Medium,
    pub phase: PhaseFunction,
}

impl<'a> IntraMediumInteraction<'a> {
    pub fn new(
        p: Point3f,
        wo: Vec3f,
        time: Float,
        medium: &'a Medium,
        phase: PhaseFunction,
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
