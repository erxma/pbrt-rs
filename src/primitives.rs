use crate::geometry::Bounds3f;

pub enum PrimitiveEnum {
    SimplePrimitive,
}

pub trait Primitive {
    fn bounds(&self) -> Bounds3f;
}

impl Primitive for PrimitiveEnum {
    fn bounds(&self) -> Bounds3f {
        todo!()
    }
}
