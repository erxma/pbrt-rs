use crate::{
    math::{
        normal3::Normal3f,
        point::{Point2f, Point3f, Point3fi},
        vec::Vec3f,
    },
    media::{
        medium::{Medium, PhaseFunction},
        medium_interface::MediumInterface,
    },
    shapes::Shape,
    Float,
};

#[derive(Clone, Debug)]
pub enum Interaction<'a> {
    Surface(SurfaceInteraction<'a>),
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
pub struct SurfaceInteraction<'a> {
    pub common: InteractionCommon,
    pub n: Normal3f,
    pub uv: Point2f,
    pub dpdu: Vec3f,
    pub dpdv: Vec3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
    shape: &'a dyn Shape,
    pub shading: Shading,
}

#[derive(Clone)]
pub struct SurfaceInteractionParams<'a> {
    pub pi: Point3fi,
    pub uv: Point2f,
    pub wo: Option<Vec3f>,
    pub dpdu: Vec3f,
    pub dpdv: Vec3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
    pub time: Float,
    pub shape: &'a dyn Shape,
}

impl<'a> SurfaceInteraction<'a> {
    pub fn new(params: &SurfaceInteractionParams<'a>) -> Self {
        let &SurfaceInteractionParams {
            pi,
            uv,
            wo,
            dpdu,
            dpdv,
            dndu,
            dndv,
            time,
            shape,
        } = params;

        let mut n = dpdu.cross(dpdv).normalized().into();
        // Adjust normal based on orientation and handedness
        if shape.reverse_orientation() ^ shape.transform_swaps_handedness() {
            n *= -1.0;
        }
        Self {
            common: InteractionCommon { pi, time, wo },
            n,
            uv,
            dpdu,
            dpdv,
            dndu,
            dndv,
            shape,
            shading: Shading {
                n,
                dpdu,
                dpdv,
                dndu,
                dndv,
            },
        }
    }

    // Update the shading geometry info
    pub fn set_shading_geometry(
        &mut self,
        dpdu: Vec3f,
        dpdv: Vec3f,
        dndu: Normal3f,
        dndv: Normal3f,
        orientation_is_authoritative: bool,
    ) {
        // Compute shading normal, flip if needed
        let mut shading_n = Normal3f::from(dpdu.cross(dpdv).normalized());
        if self.shape.reverse_orientation() ^ self.shape.transform_swaps_handedness() {
            shading_n *= -1.0;
        }

        // Align geometric normal to shading, or vice versa
        if orientation_is_authoritative {
            self.n = self.n.face_forward(shading_n.into());
        } else {
            shading_n = shading_n.face_forward(self.n.into());
        }

        // Set shading values
        self.shading = Shading {
            n: shading_n,
            dpdu,
            dpdv,
            dndu,
            dndv,
        }
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
