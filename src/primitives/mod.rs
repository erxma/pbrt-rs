use enum_dispatch::enum_dispatch;

use crate::{
    geometry::{Bounds3f, Ray},
    shapes::ShapeIntersection,
    Float,
};

mod aggregates;
mod single;

pub use aggregates::{BVHAggregate, BVHSplitMethod};
pub use single::SimplePrimitive;

#[enum_dispatch]
pub enum PrimitiveEnum {
    Simple(SimplePrimitive),
}

#[enum_dispatch(PrimitiveEnum)]
pub trait Primitive {
    fn bounds(&self) -> Bounds3f;
    fn intersect<'a>(&'a self, ray: &'a Ray, t_max: Option<Float>)
        -> Option<ShapeIntersection<'a>>;
    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool;
}
