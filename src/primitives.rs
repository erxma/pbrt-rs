use enum_dispatch::enum_dispatch;

use crate::{
    geometry::{Bounds3f, Ray},
    shapes::ShapeIntersection,
    Float,
};

#[enum_dispatch]
pub enum PrimitiveEnum {
    Simple(SimplePrimitive),
}

#[enum_dispatch(PrimitiveEnum)]
pub trait Primitive {
    fn bounds(&self) -> Bounds3f;
    fn intersect(&self, ray: &Ray, t_max: Option<Float>) -> Option<ShapeIntersection>;
    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool;
}

pub struct SimplePrimitive {}

impl Primitive for SimplePrimitive {
    fn bounds(&self) -> Bounds3f {
        todo!()
    }

    fn intersect(&self, ray: &Ray, t_max: Option<Float>) -> Option<ShapeIntersection> {
        todo!()
    }

    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool {
        todo!()
    }
}
