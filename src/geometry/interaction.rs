use crate::{
    math::{Normal3f, Point2f, Point3f, Point3fi, Vec3f},
    medium::{Medium, MediumInterface, PhaseFunction},
    Float,
};
use derive_builder::Builder;

#[derive(Clone, Debug)]
pub enum Interaction<'a> {
    Surface(Box<SurfaceInteraction>),
    MediumInterface(MediumInterfaceInteraction<'a>),
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
            Interaction::Surface(i) => &i.common,
            Interaction::MediumInterface(i) => &i.common,
            Interaction::IntraMedium(i) => &i.common,
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
pub struct MediumInterfaceInteraction<'a> {
    pub common: InteractionCommon,
    pub medium_interface: MediumInterface<'a>,
    pub phase: PhaseFunction,
}

impl<'a> MediumInterfaceInteraction<'a> {
    pub fn new(
        p: Point3f,
        wo: Option<Vec3f>,
        time: Float,
        medium_interface: MediumInterface<'a>,
        phase: PhaseFunction,
    ) -> Self {
        Self {
            common: InteractionCommon {
                pi: Point3fi::from(p),
                time,
                wo,
            },
            medium_interface,
            phase,
        }
    }

    pub fn with_point_and_interface(p: Point3f, medium_interface: MediumInterface<'a>) -> Self {
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
