use std::sync::LazyLock;

use crate::{
    math::{Normal3f, Tuple, Vec3f},
    Float,
};

/// Represents a transformation that aligns three orthonormal vectors
/// in a coordinate system with the x, y, z axes.
///
/// This allows simpler computations than `Transform`.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    x: Vec3f,
    y: Vec3f,
    z: Vec3f,
}

impl Frame {
    pub const IDENTITY: LazyLock<Self> = LazyLock::new(|| {
        Self::new(
            Vec3f::new(1.0, 0.0, 0.0),
            Vec3f::new(0.0, 1.0, 0.0),
            Vec3f::new(0.0, 0.0, 1.0),
        )
    });

    /// Construct a new frame from the three basis vectors.
    /// The vectors must be orthonormal.
    pub fn new(x: Vec3f, y: Vec3f, z: Vec3f) -> Self {
        // Debug check that vecs are orthonormal
        debug_assert!((x.length_squared() - 1.0).abs() < 1e-4);
        debug_assert!((y.length_squared() - 1.0).abs() < 1e-4);
        debug_assert!((z.length_squared() - 1.0).abs() < 1e-4);
        debug_assert!(x.dot(y).abs() < 1e-4);
        debug_assert!(y.dot(z).abs() < 1e-4);
        debug_assert!(z.dot(x).abs() < 1e-4);
        Self { x, y, z }
    }

    /// Construct a frame with given x, y vectors, and z from their cross product.
    pub fn from_xy(x: Vec3f, y: Vec3f) -> Self {
        Self::new(x, y, x.cross(y))
    }

    /// Construct a frame with given x, z vectors, and y from their cross product.
    pub fn from_xz(x: Vec3f, z: Vec3f) -> Self {
        Self::new(x, z.cross(x), z)
    }

    /// Construct a frame with the given z vector, and the others set to arbitrary valid values.
    pub fn from_z(z: Vec3f) -> Self {
        let (z, x, y) = z.coordinate_system();
        Self::new(x, y, z)
    }

    /// Transform a vector/normal out of this frame's local coordinate space.
    pub fn from_local<T: FrameTransform>(&self, val: T) -> T {
        let v = val[0] * self.x + val[1] * self.y + val[2] * self.z;
        T::from([v.x(), v.y(), v.z()])
    }

    /// Transform a vector/normal into this frame's local coordinate space.
    pub fn to_local<T: FrameTransform>(&self, val: T) -> T {
        let val_as_v = Vec3f::from(val.into());
        T::from([
            val_as_v.dot(self.x),
            val_as_v.dot(self.y),
            val_as_v.dot(self.z),
        ])
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::IDENTITY.clone()
    }
}

pub trait FrameTransform: Tuple<3, Float> {}
impl FrameTransform for Vec3f {}
impl FrameTransform for Normal3f {}
