use delegate::delegate;
use enum_dispatch::enum_dispatch;

use crate::{
    math::{lerp, Point2f, Vec2f},
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
pub struct BoxFilter {
    radius: Vec2f,
}

impl BoxFilter {
    pub fn new(radius: Vec2f) -> Self {
        Self { radius }
    }
}

impl Default for BoxFilter {
    fn default() -> Self {
        Self {
            radius: Vec2f::new(0.5, 0.5),
        }
    }
}

impl Filter for BoxFilter {
    fn radius(&self) -> Vec2f {
        self.radius
    }

    fn integral(&self) -> Float {
        2.0 * self.radius.x() * 2.0 * self.radius.y()
    }

    fn eval(&self, p: Point2f) -> Float {
        if p.x().abs() <= self.radius.x() && p.y().abs() <= self.radius.y() {
            1.0
        } else {
            0.0
        }
    }

    fn sample(&self, u: Point2f) -> FilterSample {
        let p = Point2f::new(
            lerp(-self.radius.x(), self.radius.x(), u[0]),
            lerp(-self.radius.y(), self.radius.y(), u[1]),
        );
        FilterSample { p, weight: 1.0 }
    }
}
