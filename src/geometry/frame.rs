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
    /// Frame representing an identity transformation.
    pub const IDENTITY: Self = Self::identity();

    /// Construct a new frame from the three basis vectors.
    /// The vectors must be orthonormal.
    pub fn new(x: Vec3f, y: Vec3f, z: Vec3f) -> Self {
        // Debug check that vecs are orthonormal
        debug_assert!(
            (x.length_squared() - 1.0).abs() < 1e-4,
            "Basis vector x must be normalized, but got sqlen {}",
            x.length_squared()
        );
        debug_assert!(
            (y.length_squared() - 1.0).abs() < 1e-4,
            "Basis vector y must be normalized, but got sqlen {}",
            y.length_squared()
        );
        debug_assert!(
            (z.length_squared() - 1.0).abs() < 1e-4,
            "Basis vector z must be normalized, but got sqlen {}",
            z.length_squared()
        );
        debug_assert!(x.absdot(y) < 1e-4);
        debug_assert!(y.absdot(z) < 1e-4);
        debug_assert!(z.absdot(x) < 1e-4);
        Self { x, y, z }
    }

    const fn identity() -> Self {
        Self {
            x: Vec3f::new(1.0, 0.0, 0.0),
            y: Vec3f::new(0.0, 1.0, 0.0),
            z: Vec3f::new(0.0, 0.0, 1.0),
        }
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
        Self::identity()
    }
}

pub trait FrameTransform: Tuple<3, Float> {}
impl FrameTransform for Vec3f {}
impl FrameTransform for Normal3f {}
