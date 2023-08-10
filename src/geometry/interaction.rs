use crate::{media::medium_interface::MediumInterface, Float};

use super::{normal3::Normal3f, point2::Point2f, point3fi::Point3fi, shape::Shape, vec3::Vec3f};

#[derive(Debug)]
pub struct Interaction {
    pub pi: Point3fi,
    pub time: Float,
    pub p_error: Vec3f,
    pub wo: Option<Vec3f>,
    pub n: Option<Normal3f>,
    pub medium_interface: Option<MediumInterface>,
}

#[derive(Clone, Debug)]
pub struct Shading {
    pub n: Option<Normal3f>,
    pub dpdu: Vec3f,
    pub dpdv: Vec3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
}

#[derive(Debug)]
pub struct SurfaceInteraction<'a> {
    pub common: Interaction,
    pub uv: Point2f,
    pub dpdu: Vec3f,
    pub dpdv: Vec3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
    shape: &'a dyn Shape<'a>,
    pub shading: Shading,
}

impl<'a> SurfaceInteraction<'a> {
    pub fn new(
        pi: Point3fi,
        p_error: Vec3f,
        uv: Point2f,
        wo: Vec3f,
        dpdu: Vec3f,
        dpdv: Vec3f,
        dndu: Normal3f,
        dndv: Normal3f,
        time: Float,
        shape: &'a dyn Shape<'a>,
    ) -> Self {
        let mut n = dpdu.cross(dpdv).normalized().into();
        // Adjust normal based on orientation and handedness
        if shape.reverse_orientation() ^ shape.transform_swaps_handedness() {
            n *= -1.0;
        }
        Self {
            common: Interaction {
                pi,
                time,
                p_error,
                wo: Some(wo),
                n: Some(n),
                medium_interface: None,
            },
            uv,
            dpdu,
            dpdv,
            dndu,
            dndv,
            shape,
            shading: Shading {
                n: Some(n),
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
        let mut geometric_n = self.common.n.expect(
            "Interaction should have a geometric normal set before setting shading geometry",
        );

        // Compute shading normal, flip if needed
        let mut n = Normal3f::from(dpdu.cross(dpdv).normalized());
        if self.shape.reverse_orientation() ^ self.shape.transform_swaps_handedness() {
            n *= -1.0;
        }

        // Align geometric normal to shading, or vice versa
        if orientation_is_authoritative {
            geometric_n = geometric_n.face_forward(n.into());
            self.common.n = Some(geometric_n);
        } else {
            n = n.face_forward(geometric_n.into());
        }

        // Set shading values
        self.shading = Shading {
            n: Some(n),
            dpdu,
            dpdv,
            dndu,
            dndv,
        }
    }
}
