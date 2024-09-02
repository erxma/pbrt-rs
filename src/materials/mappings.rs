use crate::{
    geometry::SurfaceInteraction,
    math::{Normal3f, Point2f, Point3f, Vec3f},
    Float,
};

pub trait TextureMapping2D {
    fn map(&self, ctx: TextureEvalContext) -> TexCoord2D;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexCoord2D {
    pub st: Point2f,
    pub dsdx: Float,
    pub dsdy: Float,
    pub dtdx: Float,
    pub dtdy: Float,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexCoord3D {
    pub p: Point3f,
    dpdx: Vec3f,
    dpdy: Vec3f,
}

#[derive(Clone, Debug)]
pub struct TextureEvalContext {
    pub p: Point3f,
    pub dpdx: Vec3f,
    pub dpdy: Vec3f,
    pub n: Normal3f,
    pub uv: Point2f,
    pub dudx: Float,
    pub dudy: Float,
    pub dvdx: Float,
    pub dvdy: Float,
}

impl TextureEvalContext {
    pub fn from_surface_interaction(si: &SurfaceInteraction) -> Self {
        Self {
            p: si.pi.midpoints_only(),
            dpdx: si.dpdx,
            dpdy: si.dpdy,
            n: si.n,
            uv: si.uv,
            dudx: si.dudx,
            dudy: si.dudy,
            dvdx: si.dvdx,
            dvdy: si.dvdy,
        }
    }
}

#[derive(Clone, Debug)]
pub struct UvMapping {
    pub su: Float,
    pub sv: Float,
    pub du: Float,
    pub dv: Float,
}

impl UvMapping {
    pub fn new(su: Float, sv: Float, du: Float, dv: Float) -> Self {
        Self { su, sv, du, dv }
    }
}

impl TextureMapping2D for UvMapping {
    fn map(&self, ctx: TextureEvalContext) -> TexCoord2D {
        // Compute texture differentials for 2D UV mapping
        let dsdx = self.su * ctx.dudx;
        let dsdy = self.su * ctx.dudy;
        let dtdx = self.sv * ctx.dvdx;
        let dtdy = self.sv * ctx.dvdy;

        let st = Point2f::new(self.su * ctx.uv[0] + self.du, self.sv * ctx.uv[1] + self.dv);
        TexCoord2D {
            st,
            dsdx,
            dsdy,
            dtdx,
            dtdy,
        }
    }
}

pub trait TextureMapping3D {
    fn map(&self, ctx: TextureEvalContext) -> TexCoord3D;
}

pub struct PointTransformMapping {}

impl TextureMapping3D for PointTransformMapping {
    fn map(&self, _ctx: TextureEvalContext) -> TexCoord3D {
        todo!()
    }
}
