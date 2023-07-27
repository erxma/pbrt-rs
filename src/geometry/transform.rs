use std::ops::Mul;

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
    /// Construct a new transform with the given matrix and inverse.
    ///
    /// The given inverse `m_inv` is assumed to be correct.
    pub fn new(m: Matrix4x4, m_inv: Matrix4x4) -> Self {
        todo!()
    }

    /// Construct a new transform from the given matrix.
    ///
    /// The inverse is calculated from the matrix.
    pub fn new_from_mat(mat: [[Float; 4]; 4]) -> Self {
        todo!()
    }

    /// Construct a transform representing a translation.
    pub fn translate(delta: Vec3f) -> Self {
        todo!()
    }

    /// Construct a transform representing a scale.
    pub fn scale(x: Float, y: Float, z: Float) -> Self {
        todo!()
    }

    /// Construct a transform representing a rotation about an axis.
    pub fn rotate(theta: Float, axis: Vec3f) -> Self {
        todo!()
    }

    /// Construct a transform representing a rotation about the x axis.
    pub fn rotate_x(theta: Float) -> Self {
        todo!()
    }

    /// Construct a transform representing a rotation about the y axis.
    pub fn rotate_y(theta: Float) -> Self {
        todo!()
    }

    /// Construct a transform representing a rotation about the z axis.
    pub fn rotate_z(theta: Float) -> Self {
        todo!()
    }

    /// Construct a look-at transformation, given the position of the viewer,
    /// the point to look at, and the up vector of the view.
    ///
    /// All parameters should be in world space.
    pub fn look_at(cam_pos: Point3f, look_pos: Point3f, up: Vec3f) -> Self {
        todo!()
    }

    /// Construct the inverse of a transform.
    pub fn inverse(&self) -> Self {
        todo!()
    }

    /// Construct the transposition of a transform.
    pub fn transpose(&self) -> Self {
        todo!()
    }

    /// Returns `true` if `self` is the identity transformation.
    pub fn is_identity(&self) -> bool {
        todo!()
    }

    /// Returns `true` if `self` has a scaling term.
    pub fn has_scale(&self) -> bool {
        todo!()
    }

    /// Returns `true` if the `self` changes a left-handed coordinate system
    /// into a right-handed one, or vice versa.
    pub fn swaps_handedness(&self) -> bool {
        todo!()
    }

    /// Get the transform's matrix.
    pub fn matrix(&self) -> &Matrix4x4 {
        todo!()
    }

    /// Get the transform's inverse matrix.
    pub fn inverse_matrix(&self) -> &Matrix4x4 {
        todo!()
    }
}

impl Mul for Transform {
    type Output = Self;

    /// Compute the composite of two transformations,
    /// equivalent to applying `rhs` then `self`.
    fn mul(self, rhs: Self) -> Self {
        todo!()
    }
}

impl<T> Mul<Point3<T>> for Transform {
    type Output = Point3<T>;

    /// Apply `self` to a point.
    fn mul(self, rhs: Point3<T>) -> Self::Output {
        todo!()
    }
}

impl<T> Mul<Vec3<T>> for Transform {
    type Output = Vec3<T>;

    /// Apply `self` to a vector.
    fn mul(self, rhs: Vec3<T>) -> Self::Output {
        todo!()
    }
}

impl<T> Mul<Normal3<T>> for Transform {
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
