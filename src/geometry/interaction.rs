use crate::{
    bxdf::BSDF,
    camera::Camera,
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
use enum_as_inner::EnumAsInner;

use super::{Ray, RayDifferential};

// TODO: This is getting messy with the fields, should refactor
#[derive(Clone, Debug, EnumAsInner)]
pub enum Interaction<'a> {
    Sample(SampleInteraction),
    Surface(Box<SurfaceInteraction>),
    MediumInterface(MediumInterfaceInteraction),
    IntraMedium(IntraMediumInteraction<'a>),
}

impl<'a> Interaction<'a> {
    pub fn pi(&self) -> Point3fi {
        self.common().pi
    }

    pub fn time(&self) -> Float {
        self.common().time
    }

    pub fn wo(&self) -> Option<Vec3f> {
        self.common().wo
    }

    fn common(&self) -> &InteractionCommon {
        match self {
            Interaction::Sample(i) => &i.common,
            Interaction::Surface(i) => &i.common,
            Interaction::MediumInterface(i) => &i.common,
            Interaction::IntraMedium(i) => &i.common,
        }
    }

    pub fn offset_ray_origin(&self, w: Vec3f) -> Point3f {
        let n_as_v = Vec3f::from(self.as_sample().unwrap().n);
        // Find vector offset to corner of error bounds, compute initial po
        let d = n_as_v.abs().dot(self.pi().error());
        let mut offset = d * n_as_v;
        if w.dot(n_as_v) < 0.0 {
            offset *= -1.0;
        }
        let mut po = self.pi().midpoints_only() + offset;

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

    pub fn spawn_ray(&self, dir: Vec3f) -> RayDifferential {
        RayDifferential::new_without_diff(Ray::new(
            self.offset_ray_origin(dir),
            dir,
            self.time(),
            self.medium_at(dir),
        ))
    }

    pub fn medium_at(&self, w: Vec3f) -> Option<&Medium> {
        match self {
            Interaction::MediumInterface(inter) => {
                if w.dot(inter.n.into()) > 0.0 {
                    Some(&inter.medium_interface.outside)
                } else {
                    Some(&inter.medium_interface.inside)
                }
            }
            Interaction::IntraMedium(inter) => Some(inter.medium),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct InteractionCommon {
    pub pi: Point3fi,
    pub time: Float,
    pub wo: Option<Vec3f>,
}

#[derive(Clone, Debug)]
pub struct SampleInteraction {
    pub common: InteractionCommon,
    pub n: Normal3f,
    pub uv: Point2f,
}

impl SampleInteraction {
    pub fn new(pi: Point3fi, time: Option<Float>, n: Normal3f, uv: Point2f) -> Self {
        let time = time.unwrap_or(0.0);
        Self {
            common: InteractionCommon { pi, time, wo: None },
            n,
            uv,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Shading {
    pub n: Normal3f,
    pub dpdu: Vec3f,
    pub dpdv: Vec3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
}

#[derive(Clone, Debug)]
pub struct SurfaceInteraction {
    pub common: InteractionCommon,
    pub n: Normal3f,
    pub uv: Point2f,
    pub dpdu: Vec3f,
    pub dpdv: Vec3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
    pub shading: Shading,
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
    #[builder(setter(strip_option))]
    wo: Option<Vec3f>,
    dpdu: Vec3f,
    dpdv: Vec3f,
    dndu: Normal3f,
    dndv: Normal3f,
    time: Float,
    flip_normal: bool,
}

impl SurfaceInteraction {
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
    pub fn build(&self) -> Result<SurfaceInteraction, SurfaceInteractionBuilderError> {
        let params = self.build_params()?;

        let mut n = params.dpdu.cross(params.dpdv).normalized().into();
        // Adjust normal based on orientation and handedness
        if params.flip_normal {
            n *= -1.0;
        }
        Ok(SurfaceInteraction {
            common: InteractionCommon {
                pi: params.pi,
                time: params.time,
                wo: params.wo,
            },
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
        })
    }
}

#[derive(Clone, Debug)]
pub struct MediumInterfaceInteraction {
    pub common: InteractionCommon,
    pub n: Normal3f,
    pub medium_interface: MediumInterface,
    pub phase: PhaseFunction,
}

impl MediumInterfaceInteraction {
    pub fn new(
        p: Point3f,
        wo: Option<Vec3f>,
        time: Float,
        n: Normal3f,
        medium_interface: MediumInterface,
        phase: PhaseFunction,
    ) -> Self {
        Self {
            common: InteractionCommon {
                pi: Point3fi::from(p),
                time,
                wo,
            },
            n,
            medium_interface,
            phase,
        }
    }

    pub fn with_point_and_interface(p: Point3f, medium_interface: MediumInterface) -> Self {
        todo!()
    }
}

#[derive(Clone, Debug)]
pub struct IntraMediumInteraction<'a> {
    pub common: InteractionCommon,
    pub medium: &'a Medium,
    pub phase: PhaseFunction,
}

impl<'a> IntraMediumInteraction<'a> {
    pub fn new(
        p: Point3f,
        wo: Option<Vec3f>,
        time: Float,
        medium: &'a Medium,
        phase: PhaseFunction,
    ) -> Self {
        Self {
            common: InteractionCommon {
                pi: Point3fi::from(p),
                time,
                wo,
            },
            medium,
            phase,
        }
    }
}
