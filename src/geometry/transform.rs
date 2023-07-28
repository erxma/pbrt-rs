use std::ops::{Add, Div, Mul};

use approx::abs_diff_ne;

use num_traits::One;

use crate::Float;

use super::{
    bounds3::Bounds3f,
    matrix4x4::Matrix4x4,
    normal3::Normal3,
    point3::{Point3, Point3f},
    ray::Ray,
    vec3::{Vec3, Vec3f},
};

/// Represents a 3D transformation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transform {
    m: Matrix4x4,
    m_inv: Matrix4x4,
}

impl Transform {
    pub const IDENTITY: Self = Self {
        m: Matrix4x4::IDENTITY,
        m_inv: Matrix4x4::IDENTITY,
    };

    /// Construct a new transform with the given matrix and inverse.
    ///
    /// The given inverse `m_inv` is assumed to be correct.
    pub fn new(m: Matrix4x4, m_inv: Matrix4x4) -> Self {
        Self { m, m_inv }
    }

    /// Construct a new transform from the given matrix.
    ///
    /// The inverse is calculated from the matrix.
    pub fn new_from_mat(mat: [[Float; 4]; 4]) -> Self {
        let m = Matrix4x4::new(mat);
        let m_inv = m
            .inverse()
            .expect("Supplied matrix should have an inverse (not singular)");

        Self { m, m_inv }
    }

    /// Construct a transform representing a translation.
    pub fn translate(delta: Vec3f) -> Self {
        let m = Matrix4x4::new([
            [1.0, 0.0, 0.0, delta.x],
            [0.0, 1.0, 0.0, delta.y],
            [0.0, 0.0, 1.0, delta.z],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let m_inv = Matrix4x4::new([
            [1.0, 0.0, 0.0, -delta.x],
            [0.0, 1.0, 0.0, -delta.y],
            [0.0, 0.0, 1.0, -delta.z],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        Self { m, m_inv }
    }

    /// Construct a transform representing a scale.
    pub fn scale(x: Float, y: Float, z: Float) -> Self {
        let m = Matrix4x4::new([
            [x, 0.0, 0.0, 0.0],
            [0.0, y, 0.0, 0.0],
            [0.0, 0.0, z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let m_inv = Matrix4x4::new([
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
        let m = Matrix4x4::new([
            [
                a.x * a.x + (1.0 - a.x * a.x) * cos_theta,
                a.x * a.y * (1.0 - cos_theta) - a.z * sin_theta,
                a.x * a.z * (1.0 - cos_theta) + a.y * sin_theta,
                0.0,
            ],
            [
                a.x * a.y * (1.0 - cos_theta) + a.z * sin_theta,
                a.y * a.y * (1.0 - a.y * a.y) * cos_theta,
                a.y * a.z * (1.0 - cos_theta) - a.x * sin_theta,
                0.0,
            ],
            [
                a.x * a.z * (1.0 - cos_theta) - a.y * sin_theta,
                a.y * a.z * (1.0 - cos_theta) + a.x * sin_theta,
                a.z * a.z + (1.0 - a.z * a.z) * cos_theta,
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.0],
        ]);

        let m_inv = m.transpose();

        Self { m, m_inv }
    }

    /// Construct a transform representing a rotation about the x axis.
    ///
    /// `theta` should be given in degrees.
    pub fn rotate_x(theta: Float) -> Self {
        let (sin_theta, cos_theta) = theta.to_radians().sin_cos();
        let m = Matrix4x4::new([
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
        let m = Matrix4x4::new([
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
        let m = Matrix4x4::new([
            [cos_theta, -sin_theta, 0.0, 0.0],
            [sin_theta, cos_theta, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        let m_inv = m.transpose();

        Self { m, m_inv }
    }

    /// Construct a look-at transformation, given the position of the viewer,
    /// the point to look at, and the up vector of the view.
    ///
    /// All parameters should be in world space.
    pub fn look_at(cam_pos: Point3f, look_pos: Point3f, up: Vec3f) -> Self {
        let dir = (look_pos - cam_pos).normalized();
        let right = up.normalized().cross(dir).normalized();
        let new_up = dir.cross(right);

        let camera_to_world = Matrix4x4::new([
            [right.x, new_up.x, dir.x, cam_pos.x],
            [right.y, new_up.y, dir.y, cam_pos.y],
            [right.z, new_up.z, dir.z, cam_pos.z],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        Self {
            m: camera_to_world.inverse().unwrap(),
            m_inv: camera_to_world,
        }
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
        let m = &self.m.m;

        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

        det < 0.0
    }

    /// Get the transform's matrix.
    pub fn matrix(&self) -> &Matrix4x4 {
        &self.m
    }

    /// Get the transform's inverse matrix.
    pub fn inverse_matrix(&self) -> &Matrix4x4 {
        &self.m_inv
    }
}

impl Mul for Transform {
    type Output = Self;

    /// Compute the composite of two transformations,
    /// equivalent to applying `rhs` then `self`.
    fn mul(self, rhs: Self) -> Self {
        let m = self.m * rhs.m;
        let m_inv = rhs.m_inv * self.m_inv;

        Self { m, m_inv }
    }
}

impl<T> Mul<Point3<T>> for Transform
where
    T: Mul<Float, Output = T> + Add<Output = T> + One + PartialEq + Copy,
    Point3<T>: Div<T, Output = Point3<T>>,
{
    type Output = Point3<T>;

    /// Apply `self` to a point.
    #[inline]
    fn mul(self, p: Point3<T>) -> Self::Output {
        let m = &self.m.m;

        let x = p.x * m[0][0] + p.y * m[0][1] + p.z * m[0][2];
        let y = p.x * m[1][0] + p.y * m[1][1] + p.z * m[1][2];
        let z = p.x * m[2][0] + p.y * m[2][1] + p.z * m[2][2];
        let w = p.x * m[3][0] + p.y * m[3][1] + p.z * m[3][2];

        if w.is_one() {
            Self::Output { x, y, z }
        } else {
            Self::Output { x, y, z } / w
        }
    }
}

impl<T> Mul<Vec3<T>> for &Transform
where
    T: Mul<Float, Output = T> + Add<Output = T> + Copy,
{
    type Output = Vec3<T>;

    /// Apply `self` to a vector.
    #[inline]
    fn mul(self, v: Vec3<T>) -> Self::Output {
        let m = &self.m.m;

        Self::Output {
            x: v.x * m[0][0] + v.y * m[0][1] + v.z * m[0][2],
            y: v.x * m[1][0] + v.y * m[1][1] + v.z * m[1][2],
            z: v.x * m[2][0] + v.y * m[2][1] + v.z * m[2][2],
        }
    }
}

impl<T> Mul<Normal3<T>> for &Transform {
    type Output = Normal3<T>;

    /// Apply `self` to a normal.
    fn mul(self, rhs: Normal3<T>) -> Self::Output {
        todo!()
    }
}

impl Mul<Ray> for Transform {
    type Output = Ray;

    /// Apply `self` to a ray.
    fn mul(self, rhs: Ray) -> Self::Output {
        todo!()
    }
}

impl Mul<Bounds3f> for Transform {
    type Output = Bounds3f;

    /// Apply `self` to a bounding box.
    fn mul(self, rhs: Bounds3f) -> Self::Output {
        todo!()
    }
}
