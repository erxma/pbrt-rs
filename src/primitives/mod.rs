use enum_dispatch::enum_dispatch;

use crate::{
    core::{Bounds3f, Float, Ray},
    shapes::ShapeIntersection,
};

mod aggregates;
mod single;

pub use aggregates::{BVHAggregate, BVHSplitMethod};
pub use single::{GeometricPrimitive, SimplePrimitive};

#[enum_dispatch]
pub enum PrimitiveEnum {
    Geometric(GeometricPrimitive),
    Simple(SimplePrimitive),
    BVH(BVHAggregate),
}

#[enum_dispatch(PrimitiveEnum)]
pub trait Primitive {
    fn bounds(&self) -> Bounds3f;
    fn intersect<'a>(&'a self, ray: &'a Ray, t_max: Option<Float>)
        -> Option<ShapeIntersection<'a>>;
    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool;
}
