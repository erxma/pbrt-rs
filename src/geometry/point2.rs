use crate as pbrt;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point2<T> {
    pub x: T,
    pub y: T,
}

pub type Point2i = Point2<i32>;
pub type Point2f = Point2<pbrt::Float>;

impl<T> Point2<T> {
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}
