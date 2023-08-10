use crate::{math::interval::Interval, Float};

use super::point3::{Point3, Point3f};

pub type Point3fi = Point3<Interval>;

impl Point3fi {
    pub fn new_fi(values: Point3f, errors: Point3f) -> Self {
        Self::new(
            Interval::new_with_err(values.x, errors.x),
            Interval::new_with_err(values.y, errors.y),
            Interval::new_with_err(values.z, errors.z),
        )
    }

    pub fn new_fi_exact(x: Float, y: Float, z: Float) -> Self {
        Self::new(Interval::new(x), Interval::new(y), Interval::new(z))
    }

    pub fn error(&self) -> Point3f {
        Point3f::new(
            self.x.width() / 2.0,
            self.y.width() / 2.0,
            self.z.width() / 2.0,
        )
    }

    pub fn is_exact(&self) -> bool {
        self.x.width() == 0.0 && self.y.width() == 0.0 && self.z.width() == 0.0
    }
}

impl From<Point3f> for Point3fi {
    fn from(p: Point3f) -> Self {
        Self::new_fi_exact(p.x, p.y, p.z)
    }
}
