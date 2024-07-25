use std::ops::{Index, IndexMut};

use delegate::delegate;
use derive_more::{Add, Neg, Sub};

use crate::{self as pbrt, impl_tuple_math_ops};

use super::{tuple::Tuple, vec3::Vec3f};

/// A 3-element normal of `f32`, or `f64` if feature `use-f64` is enabled.
// Wrapper around the vector equivalent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Neg, Add, Sub)]
#[repr(transparent)]
pub struct Normal3f(Vec3f);

impl Tuple<3, pbrt::Float> for Normal3f {}
impl_tuple_math_ops!(Normal3f; 3; pbrt::Float);

impl From<[pbrt::Float; 3]> for Normal3f {
    fn from(arr: [pbrt::Float; 3]) -> Self {
        let [x, y, z] = arr;
        Self::new(x, y, z)
    }
}

impl Normal3f {
    /// Construct a new normal with given elements.
    pub const fn new(x: pbrt::Float, y: pbrt::Float, z: pbrt::Float) -> Self {
        Self(Vec3f::new(x, y, z))
    }

    delegate! {
        to self.0 {
            #[inline(always)] pub fn x(&self) -> pbrt::Float;
            #[inline(always)] pub fn y(&self) -> pbrt::Float;
            #[inline(always)] pub fn z(&self) -> pbrt::Float;
            #[inline(always)] pub fn x_mut(&mut self) -> &mut pbrt::Float;
            #[inline(always)] pub fn y_mut(&mut self) -> &mut pbrt::Float;
            #[inline(always)] pub fn z_mut(&mut self) -> &mut pbrt::Float;

            pub fn dot(self, #[newtype] rhs: Self) -> pbrt::Float;

            /// The absolute value of the dot product of two normals.
            pub fn absdot(self, #[newtype] rhs: Self) -> pbrt::Float;

            /// Returns the dot product of `self` and a vector.
            #[call(dot)]
            pub fn dot_v(self, rhs: Vec3f) -> pbrt::Float;

            /// The absolute value of the dot product of `self` and a vector.
            #[call(absdot)]
            pub fn absdot_v(self, rhs: Vec3f) -> pbrt::Float;

            /// The squared length of `self`.
            pub fn length_squared(self) -> pbrt::Float;

            /// The length of `self`.
            pub fn length(self) -> pbrt::Float;

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
    type Output = pbrt::Float;

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
