use crate as pbrt;

use super::point3::Point3;

/// A 3D axis-aligned bounding box (AABB).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds3<T> {
    pub p_min: Point3<T>,
    pub p_max: Point3<T>,
}

pub type Bounds3i = Bounds3<i32>;
pub type Bounds3f = Bounds3<pbrt::Float>;
