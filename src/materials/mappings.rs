use enum_dispatch::enum_dispatch;

use crate::{
    math::{Point2f, Point3f, Vec3f},
    Float,
};

#[enum_dispatch]
pub enum TextureMapping2DEnum {
    Uv(UvMapping),
}

#[enum_dispatch(TextureMapping2DEnum)]
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

pub struct TextureEvalContext {}

pub struct UvMapping {}

impl TextureMapping2D for UvMapping {
    fn map(&self, _ctx: TextureEvalContext) -> TexCoord2D {
        todo!()
    }
}

#[enum_dispatch]
pub enum TextureMapping3DEnum {
    PointTransform(PointTransformMapping),
}

#[enum_dispatch(TextureMapping3DEnum)]
pub trait TextureMapping3D {
    fn map(&self, ctx: TextureEvalContext) -> TexCoord3D;
}

pub struct PointTransformMapping {}

impl TextureMapping3D for PointTransformMapping {
    fn map(&self, _ctx: TextureEvalContext) -> TexCoord3D {
        todo!()
    }
}
