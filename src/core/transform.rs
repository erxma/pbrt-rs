use std::{
    fmt,
    ops::{self, Mul},
};

use approx::abs_diff_ne;
use itertools::iproduct;
use overload::overload;

use super::{
    bounds::Bounds3f, gamma, ray::Ray, Differentials, Normal3f, Point3f, Point3fi, RayDifferential,
    SquareMatrix, Tuple, Vec3f, Vec3fi,
};
use crate::core::Float;

/// Represents a 3D transformation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transform {
    m: SquareMatrix<4>,
    m_inv: SquareMatrix<4>,
}

impl Transform {
    pub const IDENTITY: Self = Self {
        m: SquareMatrix::IDENTITY,
        m_inv: SquareMatrix::IDENTITY,
    };

    /// Construct a new transform with the given matrix and inverse.
    ///
    /// The given inverse `m_inv` is assumed to be correct.
    pub fn new(m: SquareMatrix<4>, m_inv: SquareMatrix<4>) -> Self {
        Self { m, m_inv }
    }

    /// Construct a new transform from the given matrix.
    ///
    /// The inverse is calculated from the matrix.
    pub fn from_mat(m: SquareMatrix<4>) -> Self {
        let m_inv = m
            .inverse()
            .expect("Supplied matrix should have an inverse (not singular)");

        Self { m, m_inv }
    }

    /// Construct a new transform from a matrix given as a 4x4 array.
    ///
    /// The inverse is calculated from the matrix.
    pub fn from_arr(mat: [[Float; 4]; 4]) -> Self {
        let m = SquareMatrix::new(mat);
        let m_inv = m
            .inverse()
            .expect("Supplied matrix should have an inverse (not singular)");

        Self { m, m_inv }
    }

    /// Construct a transform representing a translation.
    pub fn translate(delta: Vec3f) -> Self {
        let m = SquareMatrix::new([
            [1.0, 0.0, 0.0, delta.x()],
            [0.0, 1.0, 0.0, delta.y()],
            [0.0, 0.0, 1.0, delta.z()],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let m_inv = SquareMatrix::new([
            [1.0, 0.0, 0.0, -delta.x()],
            [0.0, 1.0, 0.0, -delta.y()],
            [0.0, 0.0, 1.0, -delta.z()],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        Self { m, m_inv }
    }

    /// Construct a transform representing a scale.
    pub fn scale(x: Float, y: Float, z: Float) -> Self {
        let m = SquareMatrix::new([
            [x, 0.0, 0.0, 0.0],
            [0.0, y, 0.0, 0.0],
            [0.0, 0.0, z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let m_inv = SquareMatrix::new([
            [1.0 / x, 0.0, 0.0, 0.0],
            [0.0, 1.0 / y, 0.0, 0.0],
            [0.0, 0.0, 1.0 / z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        Self { m, m_inv }
    }

    /// Construct a transform representing a rotation about an axis.
    ///
    /// `theta` should be given in degrees.
    pub fn rotate(theta: Float, axis: Vec3f) -> Self {
        let a = axis.normalized();
        let (sin_theta, cos_theta) = theta.to_radians().sin_cos();
        let m = SquareMatrix::new([
            [
                a.x() * a.x() + (1.0 - a.x() * a.x()) * cos_theta,
                a.x() * a.y() * (1.0 - cos_theta) - a.z() * sin_theta,
                a.x() * a.z() * (1.0 - cos_theta) + a.y() * sin_theta,
                0.0,
            ],
            [
                a.x() * a.y() * (1.0 - cos_theta) + a.z() * sin_theta,
                a.y() * a.y() * (1.0 - a.y() * a.y()) * cos_theta,
                a.y() * a.z() * (1.0 - cos_theta) - a.x() * sin_theta,
                0.0,
            ],
            [
                a.x() * a.z() * (1.0 - cos_theta) - a.y() * sin_theta,
                a.y() * a.z() * (1.0 - cos_theta) + a.x() * sin_theta,
                a.z() * a.z() + (1.0 - a.z() * a.z()) * cos_theta,
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        let m_inv = m.transpose();

        Self { m, m_inv }
    }

    /// Construct a transform representing a rotation about the x axis.
    ///
    /// `theta` should be given in degrees.
    pub fn rotate_x(theta: Float) -> Self {
        let (sin_theta, cos_theta) = theta.to_radians().sin_cos();
        let m = SquareMatrix::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta, 0.0],
            [0.0, sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        let m_inv = m.transpose();

        Self { m, m_inv }
    }

    /// Construct a transform representing a rotation about the y axis.
    ///
    /// `theta` should be given in degrees.
    pub fn rotate_y(theta: Float) -> Self {
        let (sin_theta, cos_theta) = theta.to_radians().sin_cos();
        let m = SquareMatrix::new([
            [cos_theta, 0.0, sin_theta, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_theta, 0.0, cos_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        let m_inv = m.transpose();

        Self { m, m_inv }
    }

    /// Construct a transform representing a rotation about the z axis.
    ///
    /// `theta` should be given in degrees.
    pub fn rotate_z(theta: Float) -> Self {
        let (sin_theta, cos_theta) = theta.to_radians().sin_cos();
        let m = SquareMatrix::new([
            [cos_theta, -sin_theta, 0.0, 0.0],
            [sin_theta, cos_theta, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        let m_inv = m.transpose();

        Self { m, m_inv }
    }

    /// Transformation representing a rotation from one vector to another.
    pub fn rotate_from_to(from: Vec3f, to: Vec3f) -> Self {
        // Compute intermediate vec for vec reflection
        let refl = if from.x().abs() < 0.72 && to.x().abs() < 0.72 {
            Vec3f::RIGHT
        } else if from.y().abs() < 0.72 && to.y().abs() < 0.72 {
            Vec3f::UP
        } else {
            Vec3f::FORWARD
        };

        let u = refl - from;
        let v = refl - to;
        let mut mat: SquareMatrix<4> = Default::default();
        for (i, j) in iproduct!(0..3, 0..3) {
            mat[i][j] = if i == j { 1.0 } else { 0.0 }
                - 2.0 / u.dot(u) * u[i] * u[j]
                - 2.0 / v.dot(v) * v[i] * v[j]
                + 4.0 * u.dot(v) / (u.dot(u) * v.dot(v)) * v[i] * u[j];
        }

        Self::from_mat(mat)
    }

    /// Construct a look-at transformation, given the position of the viewer,
    /// the point to look at, and the up vector of the view.
    ///
    /// All parameters should be in world space.
    pub fn look_at(cam_pos: Point3f, look_pos: Point3f, up: Vec3f) -> Self {
        let dir = (look_pos - cam_pos).normalized();
        let right = up.normalized().cross(dir).normalized();
        let new_up = dir.cross(right);

        let camera_to_world = SquareMatrix::new([
            [right.x(), new_up.x(), dir.x(), cam_pos.x()],
            [right.y(), new_up.y(), dir.y(), cam_pos.y()],
            [right.z(), new_up.z(), dir.z(), cam_pos.z()],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        Self {
            m: camera_to_world.inverse().unwrap(),
            m_inv: camera_to_world,
        }
    }

    /// Transformation representing a projection of camera-space points
    /// onto an orthographic viewing plane.
    pub fn orthographic(near_z: Float, far_z: Float) -> Self {
        Self::scale(1.0, 1.0, 1.0 / (near_z - far_z))
            * Self::translate(Vec3f::new(0.0, 0.0, -near_z))
    }

    /// Transformation representing a projection of camera-space points
    /// onto a perspective viewing plane.
    ///
    /// FOV should be given in degrees.
    pub fn perspective(fov: Float, near_z: Float, far_z: Float) -> Self {
        // Perform projective divide for projection
        let persp = Transform::from_arr([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                far_z / (far_z - near_z),
                -far_z * near_z / (far_z - near_z),
            ],
            [0.0, 0.0, 1.0, 0.0],
        ]);

        // Scale canonical perspective view to specified FOV
        let inv_tag_ang = 1.0 / (fov.to_radians() / 2.0).tan();
        let scale = Self::scale(inv_tag_ang, inv_tag_ang, 1.0);

        scale * persp
    }

    /// Construct the inverse of a transform.
    pub fn inverse(&self) -> Self {
        Self {
            m: self.m_inv.clone(),
            m_inv: self.m.clone(),
        }
    }

    /// Construct the transposition of a transform.
    pub fn transpose(&self) -> Self {
        Self {
            m: self.m.transpose(),
            m_inv: self.m_inv.transpose(),
        }
    }

    /// Returns `true` if `self` is the identity transformation.
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    /// Returns `true` if `self` has a scaling term.
    pub fn has_scale(&self) -> bool {
        let la2 = (self * Vec3f::new(1.0, 0.0, 0.0)).length_squared();
        let lb2 = (self * Vec3f::new(0.0, 1.0, 0.0)).length_squared();
        let lc2 = (self * Vec3f::new(0.0, 0.0, 1.0)).length_squared();

        abs_diff_ne!(la2, 1.0) || abs_diff_ne!(lb2, 1.0) || abs_diff_ne!(lc2, 1.0)
    }

    /// Returns `true` if the `self` changes a left-handed coordinate system
    /// into a right-handed one, or vice versa.
    pub fn swaps_handedness(&self) -> bool {
        let m = &self.m;

        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

        det < 0.0
    }

    /// Get the transform's matrix.
    pub fn matrix(&self) -> &SquareMatrix<4> {
        &self.m
    }

    /// Get the transform's inverse matrix.
    pub fn inverse_matrix(&self) -> &SquareMatrix<4> {
        &self.m_inv
    }
}

// Compute the composite of two transformations,
// equivalent to applying `rhs` then `self`.
overload!((t1: ?Transform) * (t2: ?Transform) -> Transform {
    let m = &t1.m * &t2.m;
    let m_inv = &t2.m_inv * &t1.m_inv;

    Transform { m, m_inv }
});

// Apply transform to a point.
overload!((t: ?Transform) * (p: Point3f) -> Point3f {
    let m = &t.m;

    let mut x = p.x() * m[0][0] + p.y() * m[0][1] + p.z() * m[0][2] + m[0][3];
    let mut y = p.x() * m[1][0] + p.y() * m[1][1] + p.z() * m[1][2] + m[1][3];
    let mut z = p.x() * m[2][0] + p.y() * m[2][1] + p.z() * m[2][2] + m[2][3];
    let w = p.x() * m[3][0] + p.y() * m[3][1] + p.z() * m[3][2] + m[3][3];

    if w == 0.0 {
        x /= w;
        y /= w;
        z /= w;
    }

    Point3f::new(x, y, z)
});

// Apply transform to a point with intervals.
overload!((t: ?Transform) * (p: Point3fi) -> Point3fi {
    let m = &t.m;

    let x = p.x().midpoint();
    let y = p.y().midpoint();
    let z = p.z().midpoint();

    // Compute transformed coordinates from point (x, y, z)_
    let xp = (m[0][0] * x + m[0][1] * y) + (m[0][2] * z + m[0][3]);
    let yp = (m[1][0] * x + m[1][1] * y) + (m[1][2] * z + m[1][3]);
    let zp = (m[2][0] * x + m[2][1] * y) + (m[2][2] * z + m[2][3]);
    let wp = (m[3][0] * x + m[3][1] * y) + (m[3][2] * z + m[3][3]);

    // Compute absolute error for transformed point,_p_error_
    let p_error = if p.is_exact() {
        // Compute error for transformed exact p
        Vec3f::new(
            gamma(3)
                * ((m[0][0] * x).abs()
                    + (m[0][1] * y).abs()
                    + (m[0][2] * z).abs()
                    + m[0][3].abs()),
            gamma(3)
                * ((m[1][0] * x).abs()
                    + (m[1][1] * y).abs()
                    + (m[1][2] * z).abs()
                    + m[1][3].abs()),
            gamma(3)
                * ((m[2][0] * x).abs()
                    + (m[2][1] * y).abs()
                    + (m[2][2] * z).abs()
                    + m[2][3].abs()),
        )
    } else {
        // Compute error for transformed approximate _p_
        let p_in_error = p.error();

        Vec3f::new(
            (gamma(3) + 1.0)
                * (m[0][0].abs() * p_in_error.x()
                    + m[0][1].abs() * p_in_error.y()
                    + m[0][2].abs() * p_in_error.z())
                + gamma(3)
                    * ((m[0][0] * x).abs()
                        + (m[0][1] * y).abs()
                        + (m[0][2] * z).abs()
                        + m[0][3].abs()),
            (gamma(3) + 1.0)
                * (m[1][0].abs() * p_in_error.x()
                    + m[1][1].abs() * p_in_error.y()
                    + m[1][2].abs() * p_in_error.z())
                + gamma(3)
                    * ((m[1][0] * x).abs()
                        + (m[1][1] * y).abs()
                        + (m[1][2] * z).abs()
                        + m[1][3].abs()),
            (gamma(3) + 1.0)
                * (m[2][0].abs() * p_in_error.x()
                    + m[2][1].abs() * p_in_error.y()
                    + m[2][2].abs() * p_in_error.z())
                + gamma(3)
                    * ((m[2][0] * x).abs()
                        + (m[2][1] * y).abs()
                        + (m[2][2] * z).abs()
                        + (m[2][3]).abs()),
        )
    };

    if wp == 1.0 {
        Point3fi::new_fi(Point3f::new(xp, yp, zp), p_error)
    } else {
        Point3fi::new_fi(Point3f::new(xp, yp, zp), p_error) / wp
    }
});

// Apply transform to a vector.
overload!((t: ?Transform) * (v: Vec3f) -> Vec3f {
    let m = &t.m;

    Vec3f::new(
        v.x() * m[0][0] + v.y() * m[0][1] + v.z() * m[0][2],
        v.x() * m[1][0] + v.y() * m[1][1] + v.z() * m[1][2],
        v.x() * m[2][0] + v.y() * m[2][1] + v.z() * m[2][2],
    )
});

// Apply transform to a vector of intervals.
overload!((t: ?Transform) * (v: Vec3fi) -> Vec3fi {
    let m = &t.m;

    Vec3fi::new(
        v.x() * m[0][0] + v.y() * m[0][1] + v.z() * m[0][2],
        v.x() * m[1][0] + v.y() * m[1][1] + v.z() * m[1][2],
        v.x() * m[2][0] + v.y() * m[2][1] + v.z() * m[2][2],
    )
});

// Apply transform to a normal.
overload!((t: ?Transform) * (n: Normal3f) -> Normal3f {
    let m_inv = &t.m_inv;
    Normal3f::new(
        n.x() * m_inv[0][0] + n.y() * m_inv[1][0] + n.z() * m_inv[2][0],
        n.x() * m_inv[0][1] + n.y() * m_inv[1][1] + n.z() * m_inv[2][1],
        n.x() * m_inv[0][2] + n.y() * m_inv[1][2] + n.z() * m_inv[2][2],
    )
});

impl Transform {
    pub fn mul_ray(&self, ray: Ray, mut t_max: Option<Float>) -> (Ray, Option<Float>) {
        let mut o = self * Point3fi::from(ray.o);
        let dir = self * ray.dir;

        // Offset ray origin to edge of error bounds and compute t_max
        let len_sq = dir.length_squared();
        if len_sq > 0.0 {
            let dt = dir.abs().dot(o.error()) / len_sq;
            o += Vec3fi::from(dir) * dt;
            t_max = t_max.map(|t| t - dt);
        }

        (
            Ray {
                o: o.midpoints(),
                dir,
                ..ray
            },
            t_max,
        )
    }

    pub fn mul_ray_diff(
        &self,
        ray_diff: RayDifferential,
        t_max: Option<Float>,
    ) -> (RayDifferential, Option<Float>) {
        // Apply to ray
        let (ray, t_max) = self.mul_ray(ray_diff.ray, t_max);
        // Apply to diffs if they've been set
        let differentials = ray_diff.differentials.map(|diff| {
            let rx_origin = self * diff.rx_origin;
            let ry_origin = self * diff.ry_origin;
            let rx_dir = self * diff.rx_dir;
            let ry_dir = self * diff.ry_dir;
            Differentials {
                rx_origin,
                ry_origin,
                rx_dir,
                ry_dir,
            }
        });

        (RayDifferential { ray, differentials }, t_max)
    }
}

// Cases where there is definitely no t_max
impl Mul<Ray> for Transform {
    type Output = Ray;

    /// Apply `self` to a ray.
    fn mul(self, r: Ray) -> Self::Output {
        &self * r
    }
}

impl Mul<Ray> for &Transform {
    type Output = Ray;

    /// Apply `self` to a ray.
    fn mul(self, r: Ray) -> Self::Output {
        let (ray, _) = self.mul_ray(r, None);
        ray
    }
}

impl Mul<RayDifferential> for Transform {
    type Output = RayDifferential;

    fn mul(self, rd: RayDifferential) -> Self::Output {
        &self * rd
    }
}

impl Mul<RayDifferential> for &Transform {
    type Output = RayDifferential;

    /// Apply `self` to a ray.
    fn mul(self, rd: RayDifferential) -> Self::Output {
        let (rd, _) = self.mul_ray_diff(rd, None);
        rd
    }
}

// Apply transform to a bounding box.
overload!((t: ?Transform) * (b: Bounds3f) -> Bounds3f {
    let m = &t.m;

    // Each transformation can be split into a translation and rotation--

    // Extract translation from matrix (3rd column),
    // which is the center of the new box - it was originally 0, 0, 0.
    let translation = Point3f::new(m[0][3], m[1][3], m[2][3]);
    let mut res = Bounds3f::new_with_point(translation);

    // The 3x3 rotation portion of the matrix remains.
    // Now find the extremes of the transformed points.

    // Consider that all the other 6 points are some combination of
    // the x, y, z values from the min & max points.

    // Now consider the usual multiplication (cross) of a point.
    // Each axis in the new point is a linear combination of the original x, y, z.
    // Specifically, the factors are the corresponding row in the matrix.

    // So, for each axis, find the choices of x, y, z from the original
    // min, max that leads to the (new) min/max linear combinations.
    for i in 0..=2 {
        for j in 0..=2 {
            let a = m[i][j] * b.p_min[j];
            let b = m[i][j] * b.p_max[j];
            // Can directly add, since, again, the starting point is
            // the translated origin/center.
            res.p_min[i] += a.min(b);
            res.p_max[i] += a.max(b);
        }
    }

    res
});

impl fmt::Display for Transform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match f.precision() {
            Some(p) => {
                if f.alternate() {
                    write!(f, "Transform:\n{:#.*}", p, self.m)
                } else {
                    write!(f, "Transform:\n{:.*}", p, self.m)
                }
            }
            None => {
                if f.alternate() {
                    write!(f, "Transform:\n{:#}", self.m)
                } else {
                    write!(f, "Transform:\n{}", self.m)
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use approx::{assert_relative_eq, AbsDiffEq, RelativeEq};

    use super::*;

    impl AbsDiffEq for Bounds3f {
        type Epsilon = <Float as AbsDiffEq>::Epsilon;

        fn default_epsilon() -> Self::Epsilon {
            Float::default_epsilon()
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            for p in 0..2 {
                for i in 0..3 {
                    if Float::abs_diff_ne(&self[p][i], &other[p][i], epsilon) {
                        return false;
                    }
                }
            }

            true
        }
    }

    impl RelativeEq for Bounds3f {
        fn default_max_relative() -> Self::Epsilon {
            Float::default_max_relative()
        }

        fn relative_eq(
            &self,
            other: &Self,
            epsilon: Self::Epsilon,
            max_relative: Self::Epsilon,
        ) -> bool {
            for p in 0..2 {
                for i in 0..3 {
                    if Float::relative_ne(&self[p][i], &other[p][i], epsilon, max_relative) {
                        return false;
                    }
                }
            }

            true
        }
    }

    #[test]
    fn efficient_bounds_transform() {
        fn naive_transform(t: &Transform, b: Bounds3f) -> Bounds3f {
            let mut ret = Bounds3f::new_with_point(t * b.corner(0));

            for i in 1..8 {
                ret = ret.union_point(t * b.corner(i));
            }

            ret
        }

        let b = Bounds3f::new(Point3f::new(-1.0, -1.0, -1.0), Point3f::new(2.0, 2.0, 2.0));

        let translate = Transform::translate(Vec3f::new(-1.0, 2.0, 2.0));
        let rotate = Transform::rotate_x(90.0);
        let scale = Transform::scale(1.1, 3.3, 4.3);
        let composite = translate.clone() * rotate.clone() * scale.clone();

        assert_relative_eq!(naive_transform(&translate, b), &translate * b);
        assert_relative_eq!(naive_transform(&rotate, b), &rotate * b);
        assert_relative_eq!(naive_transform(&scale, b), &scale * b);
        assert_relative_eq!(naive_transform(&composite, b), &composite * b);
    }
}
