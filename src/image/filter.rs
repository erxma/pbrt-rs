use delegate::delegate;
use enum_dispatch::enum_dispatch;

use crate::{
    math::{Point2f, Vec2f},
    Float,
};

#[enum_dispatch]
#[derive(Clone, Debug)]
pub enum Filter {
    BoxFilter(BoxFilter),
}

impl Filter {
    delegate! {
        #[through(FilterTrait)]
        to self {
            pub fn radius(&self) -> Vec2f;
            pub fn integral(&self) -> Float;
            pub fn eval(&self, p: Point2f) -> Float;
        }
    }
}

#[enum_dispatch(Filter)]
trait FilterTrait {
    fn radius(&self) -> Vec2f;
    fn eval(&self, p: Point2f) -> Float;
    fn integral(&self) -> Float;
}

#[derive(Clone, Debug)]
pub struct BoxFilter {}

impl FilterTrait for BoxFilter {
    fn radius(&self) -> Vec2f {
        todo!()
    }

    fn integral(&self) -> Float {
        todo!()
    }

    fn eval(&self, p: Point2f) -> Float {
        todo!()
    }
}
