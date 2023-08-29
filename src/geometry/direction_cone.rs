use crate::{
    geometry::{bounds3::Bounds3f, transform::Transform},
    math::{
        point::Point3f,
        routines::{safe_acos, safe_sqrt},
        vec3::Vec3f,
    },
    Float, PI,
};

#[derive(Clone, Copy, Debug)]
pub struct DirectionCone {
    /// Central direction/axis of the cone.
    pub w: Vec3f,
    /// The cosine of the spread angle of the cone.
    pub cos_theta: Float,
}

impl DirectionCone {
    pub const fn new(w: Vec3f, cos_theta: Float) -> Self {
        Self { w, cos_theta }
    }

    pub fn from_dir(v: Vec3f) -> Self {
        Self {
            w: v,
            cos_theta: 1.0,
        }
    }

    pub const EMPTY: Self = Self {
        w: Vec3f::new(0.0, 0.0, 0.0),
        cos_theta: Float::INFINITY,
    };

    pub const ENTIRE_SPHERE: Self = Self {
        w: Vec3f::new(0.0, 0.0, 1.0),
        cos_theta: -1.0,
    };

    pub fn is_empty(&self) -> bool {
        self.cos_theta.is_infinite()
    }

    pub fn contains(&self, w: Vec3f) -> bool {
        !self.is_empty() && self.w.dot(w.normalized()) >= self.cos_theta
    }

    /// Returns a cone that bounds the directions subtended by bounding box `b`
    /// with respect to point `p`.
    pub fn bound_subtended_directions(bounds: &Bounds3f, p: Point3f) -> Self {
        // Compute bounding sphere for bounding box
        let (center, radius) = bounds.bounding_sphere();

        // If p is inside sphere, return cone bounding all directions.
        if p.distance_squared(center) < radius * radius {
            return Self::ENTIRE_SPHERE;
        }

        // Compute cone for bounding sphere
        let w = (center - p).normalized();
        let sin2_theta_max = radius * radius / center.distance_squared(p);
        let cos_theta_max = safe_sqrt(1.0 - sin2_theta_max);

        Self {
            w,
            cos_theta: cos_theta_max,
        }
    }

    pub fn union(self, other: Self) -> Self {
        // One or both are empty
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }

        // Determine if one cone is inside the other,
        // in which case, their union is the larger one
        let theta_self = safe_acos(self.cos_theta);
        let theta_other = safe_acos(other.cos_theta);
        // Angle between the two cones' centers
        let theta_diff = Vec3f::angle_between(self.w, other.w);
        if (theta_diff + theta_other).min(PI) <= theta_self {
            return self;
        }
        if (theta_diff + theta_self).min(PI) <= theta_other {
            return other;
        }

        // Other cases

        // Compute merged spread angle
        let theta_o = (theta_self + theta_diff + theta_other) / 2.0;
        if theta_o > PI {
            return Self::ENTIRE_SPHERE;
        }

        // Find merged cone axis
        let theta_r = theta_o - theta_self;
        let wr = Vec3f::cross(self.w, other.w);
        if wr.length_squared() == 0.0 {
            return Self::ENTIRE_SPHERE;
        }
        let w = &Transform::rotate(theta_r.to_degrees(), wr) * self.w;

        Self {
            w,
            cos_theta: theta_o.cos(),
        }
    }
}
