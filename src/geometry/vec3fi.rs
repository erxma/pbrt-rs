use crate::{math::interval::Interval, Float};

use super::vec3::{Vec3, Vec3f};

pub type Vec3fi = Vec3<Interval>;

impl Vec3fi {
    pub fn new_fi(values: Vec3f, errors: Vec3f) -> Self {
        Self::new(
            Interval::new_with_err(values.x, errors.x),
            Interval::new_with_err(values.y, errors.y),
            Interval::new_with_err(values.z, errors.z),
        )
    }

    pub fn new_fi_exact(x: Float, y: Float, z: Float) -> Self {
        Self::new(Interval::new(x), Interval::new(y), Interval::new(z))
    }

    pub fn with_intervals(x: Interval, y: Interval, z: Interval) -> Self {
        Self::new(x, y, z)
    }

    pub fn error(&self) -> Vec3f {
        Vec3f::new(
            self.x.width() / 2.0,
            self.y.width() / 2.0,
            self.z.width() / 2.0,
        )
    }

    pub fn is_exact(&self) -> bool {
        self.x.width() == 0.0 && self.y.width() == 0.0 && self.z.width() == 0.0
    }
}

impl From<Vec3f> for Vec3fi {
    fn from(v: Vec3f) -> Self {
        Self::new_fi_exact(v.x, v.y, v.z)
    }
}
