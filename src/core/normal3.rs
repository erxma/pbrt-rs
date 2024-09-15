use std::{
    fmt,
    ops::{Index, IndexMut},
};

use bytemuck::{Pod, Zeroable};
use delegate::delegate;

use super::{impl_tuple_math_ops, Float, Point3f, Tuple, Vec3f};

/// A 3-element normal of `f32`, or `f64` if feature `use-f64` is enabled.
// Wrapper around the vector equivalent.
#[derive(
    Clone,
    Copy,
    Default,
    PartialEq,
    derive_more::Neg,
    derive_more::Add,
    derive_more::Sub,
    Zeroable,
    Pod,
)]
#[repr(transparent)]
pub struct Normal3f(Vec3f);

impl Tuple<3, Float> for Normal3f {}
impl_tuple_math_ops!(Normal3f; 3; Float);

impl From<[Float; 3]> for Normal3f {
    fn from(arr: [Float; 3]) -> Self {
        let [x, y, z] = arr;
        Self::new(x, y, z)
    }
}

impl Normal3f {
    /// Construct a new normal with given elements.
    pub const fn new(x: Float, y: Float, z: Float) -> Self {
        Self(Vec3f::new(x, y, z))
    }

    delegate! {
        to self.0 {
            #[inline(always)] pub fn x(&self) -> Float;
            #[inline(always)] pub fn y(&self) -> Float;
            #[inline(always)] pub fn z(&self) -> Float;
            #[inline(always)] pub fn x_mut(&mut self) -> &mut Float;
            #[inline(always)] pub fn y_mut(&mut self) -> &mut Float;
            #[inline(always)] pub fn z_mut(&mut self) -> &mut Float;

            pub fn dot(self, #[newtype] rhs: Self) -> Float;

            /// The absolute value of the dot product of two normals.
            pub fn absdot(self, #[newtype] rhs: Self) -> Float;

            /// Returns the dot product of `self` and a vector.
            #[call(dot)]
            pub fn dot_v(self, rhs: Vec3f) -> Float;

            /// The absolute value of the dot product of `self` and a vector.
            #[call(absdot)]
            pub fn absdot_v(self, rhs: Vec3f) -> Float;

            /// The squared length of `self`.
            pub fn length_squared(self) -> Float;

            /// The length of `self`.
            pub fn length(self) -> Float;

            /// Returns the normalization of `self`.
            #[into]
            pub fn normalized(self) -> Self;

            /// Returns true if any component is NaN.
            pub fn has_nan(self) -> bool;
        }
    }

    /// If needed, returns `self` flipped so that it lies in the same hemisphere
    /// as the given vector. If already satisfied, returns `self`.
    pub fn face_forward(self, v: Vec3f) -> Self {
        if self.dot_v(v).is_sign_positive() {
            self
        } else {
            -self
        }
    }
}

impl Index<usize> for Normal3f {
    type Output = Float;

    /// Index `self`'s elements by 0, 1, 2.
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0..=2 => &self.0[index],
            _ => panic!("Index out of bounds for Normal3"),
        }
    }
}

impl IndexMut<usize> for Normal3f {
    /// Index `self`'s elements by 0, 1, 2, mutably.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0..=2 => &mut self.0[index],
            _ => panic!("Index out of bounds for Normal3"),
        }
    }
}

impl From<Vec3f> for Normal3f {
    #[inline]
    fn from(value: Vec3f) -> Self {
        Self(value)
    }
}

impl From<Point3f> for Normal3f {
    #[inline]
    fn from(value: Point3f) -> Self {
        Self(Vec3f::from(value))
    }
}

impl fmt::Debug for Normal3f {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Normal3f")
            .field("x", &self.x())
            .field("y", &self.y())
            .field("z", &self.z())
            .finish()
    }
}

impl fmt::Display for Normal3f {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(p) = f.precision() {
            write!(
                f,
                "[{:.*}, {:.*}, {:.*}]",
                p,
                self.x(),
                p,
                self.y(),
                p,
                self.z()
            )
        } else {
            write!(f, "[{}, {}, {}]", self.x(), self.y(), self.z())
        }
    }
}
