use crate::{math::vec3::Vec3f, Float, PI};

use super::routines::safe_acos;

#[inline]
pub fn spherical_triangle_area(a: Vec3f, b: Vec3f, c: Vec3f) -> Float {
    (2.0 * a
        .dot(b.cross(c))
        .atan2(1.0 + a.dot(b) + a.dot(c) + b.dot(c)))
    .abs()
}

#[inline]
pub fn spherical_quad_area(a: Vec3f, b: Vec3f, c: Vec3f, d: Vec3f) -> Float {
    let mut axb = a.cross(b);
    let mut bxc = b.cross(c);
    let mut cxd = c.cross(d);
    let mut dxa = d.cross(a);

    let vecs = [&mut axb, &mut bxc, &mut cxd, &mut dxa];

    if vecs.iter().any(|v| v.length_squared() == 0.0) {
        return 0.0;
    }

    for v in vecs {
        *v = v.normalized();
    }

    let alpha = dxa.angle_between(-axb);
    let beta = axb.angle_between(-bxc);
    let gamma = bxc.angle_between(-cxd);
    let delta = cxd.angle_between(-dxa);

    (alpha + beta + gamma + delta - 2.0 * PI).abs()
}

/// Converts a spherical coordinate pair `theta` and `phi` to a 3D unit cartesian coordinate.
#[inline]
pub fn spherical_direction(sin_theta: Float, cos_theta: Float, phi: Float) -> Vec3f {
    Vec3f::new(
        sin_theta.clamp(-1.0, 1.0) * phi.cos(),
        sin_theta.clamp(-1.0, 1.0) * phi.sin(),
        cos_theta.clamp(-1.0, 1.0),
    )
}

impl Vec3f {
    /// Returns the `theta` of the spherical coordinates corresponding to vector `v`.
    #[inline]
    pub fn spherical_theta(self) -> Float {
        safe_acos(self.z())
    }

    /// Returns the `phi` of the spherical coordinates corresponding to vector `v`.
    #[inline]
    pub fn spherical_phi(self) -> Float {
        let p = self.y().atan2(self.x());

        if p < 0.0 {
            p + 2.0 * PI
        } else {
            p
        }
    }

    #[inline]
    pub fn cos_theta(self) -> Float {
        self.z()
    }

    #[inline]
    pub fn cos2_theta(self) -> Float {
        self.z() * self.z()
    }

    #[inline]
    pub fn abs_cos_theta(w: Vec3f) -> Float {
        w.z().abs()
    }

    #[inline]
    pub fn sin_theta(self) -> Float {
        self.sin2_theta().sqrt()
    }

    #[inline]
    pub fn sin2_theta(self) -> Float {
        (1.0 - self.cos2_theta()).max(0.0)
    }

    #[inline]
    pub fn tan_theta(self) -> Float {
        self.sin_theta() / self.cos_theta()
    }

    #[inline]
    pub fn tan2_theta(self) -> Float {
        self.sin2_theta() / self.cos2_theta()
    }

    #[inline]
    pub fn cos_phi(self) -> Float {
        let sin_theta = self.sin_theta();
        if sin_theta == 0.0 {
            1.0
        } else {
            (self.x() / sin_theta).clamp(-1.0, 1.0)
        }
    }

    #[inline]
    pub fn sin_phi(self) -> Float {
        let sin_theta = self.sin_theta();
        if sin_theta == 0.0 {
            0.0
        } else {
            (self.y() / sin_theta).clamp(-1.0, 1.0)
        }
    }

    #[inline]
    pub fn cos_delta_phi(self: Vec3f, other: Vec3f) -> Float {
        let xy = self.x() * self.x() + self.y() * self.y();
        let other_xy = other.x() * other.x() + other.y() * other.y();
        if xy == 0.0 || other_xy == 0.0 {
            1.0
        } else {
            ((self.x() * other.x() + self.y() * other.y()) / (xy * other_xy).sqrt())
                .clamp(-1.0, 1.0)
        }
    }
}
