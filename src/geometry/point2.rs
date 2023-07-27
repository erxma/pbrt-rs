use crate as pbrt;

pub struct Point2<T> {
    pub x: T,
    pub y: T,
}

pub type Point2i = Point2<i32>;
pub type Point2f = Point2<pbrt::Float>;
