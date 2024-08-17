use delegate::delegate;
use enum_dispatch::enum_dispatch;

use crate::{
    math::{Point2f, Vec2f},
    Float,
};

#[enum_dispatch]
#[derive(Clone, Debug)]
pub enum FilterEnum {
    BoxFilter(BoxFilter),
}

impl FilterEnum {
    delegate! {
        #[through(Filter)]
        to self {
            pub fn radius(&self) -> Vec2f;
            pub fn integral(&self) -> Float;
            pub fn eval(&self, p: Point2f) -> Float;
            pub fn sample(&self, u: Point2f) -> FilterSample;
        }
    }
}

#[enum_dispatch(FilterEnum)]
pub trait Filter {
    fn radius(&self) -> Vec2f;
    fn eval(&self, p: Point2f) -> Float;
    fn integral(&self) -> Float;
    fn sample(&self, u: Point2f) -> FilterSample;
}

pub struct FilterSample {
    pub p: Point2f,
    pub weight: Float,
}

#[derive(Clone, Debug)]
pub struct BoxFilter {}

impl Filter for BoxFilter {
    fn radius(&self) -> Vec2f {
        todo!()
    }

    fn integral(&self) -> Float {
        todo!()
    }

    fn eval(&self, p: Point2f) -> Float {
        todo!()
    }

    fn sample(&self, u: Point2f) -> FilterSample {
        todo!()
    }
}
