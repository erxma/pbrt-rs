use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::Float;

/// A 3D vector of floats (precision depending on config).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub type Vec3i = Vec3<i32>;
pub type Vec3f = Vec3<Float>;

impl<T> Vec3<T> {
    /// Construct a new vector with given elements.
    pub fn new(x: T, y: T, z: T) -> Self {
        todo!()
    }

    /// Returns true if any component is NaN.
    pub fn has_nans(self) -> bool {
        todo!()
    }

    /// Returns a vector with the absolute values of the components.
    pub fn abs(self) -> Self {
        todo!()
    }

    /// Returns the dot product of two vectors.
    pub fn dot(self, rhs: Self) -> T {
        todo!()
    }

    /// The absolute value of the dot product of two vectors.
    pub fn absdot(self, rhs: Self) -> T {
        todo!()
    }

    /// The cross product of two vectors.
    pub fn cross(self, rhs: Self) -> Self {
        todo!()
    }

    /// The squared length of a vector.
    pub fn length_squared(self) -> Float {
        todo!()
    }

    /// The length of a vector.
    pub fn length(self) -> Float {
        todo!()
    }

    /// Returns the normalization of a vector.
    pub fn normalized(self) -> Self {
        todo!()
    }

    /// Returns the element with the smallest value.
    pub fn min_component(self) -> T {
        todo!()
    }

    /// Returns the element with the largest value.
    pub fn max_component(self) -> T {
        todo!()
    }

    /// Returns the index of the component with the largest value.
    pub fn max_dimension(self) -> usize {
        todo!()
    }

    /// Returns a vector containing the min values for each
    /// component of `self` and `other` (the component-wise min).
    pub fn min(self, other: Self) -> Self {
        todo!()
    }

    /// Returns a vector containing the max values for each
    /// component of `self` and `other` (the component-wise max).
    pub fn max(self, other: Self) -> Self {
        todo!()
    }

    /// Permute a vector's elements according to the index values
    /// given.
    pub fn permute(self, x: usize, y: usize, z: usize) -> Self {
        todo!()
    }

    /// Construct a local coordinate system given a vector.
    ///
    /// Returns the set of three orthogonal vectors representing
    /// the system.
    pub fn coordinate_system(self) -> (Self, Self, Self) {
        todo!()
    }
}

impl<T> Index<usize> for Vec3<T> {
    type Output = T;

    /// Index a vector's elements by 0, 1, 2.
    fn index(&self, index: usize) -> &Self::Output {
        todo!()
    }
}

impl<T> IndexMut<usize> for Vec3<T> {
    /// Index a vector's elements by 0, 1, 2, mutably.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        todo!()
    }
}

impl<T> Add for Vec3<T> {
    type Output = Self;

    /// Add two vectors to get new vector of same type.
    fn add(self, rhs: Self) -> Self {
        todo!();
    }
}

impl<T> AddAssign for Vec3<T> {
    /// Add assign a vector with another of same type.
    fn add_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl<T> Sub for Vec3<T> {
    type Output = Self;

    /// Subtract a vector from another of same type.
    fn sub(self, rhs: Self) -> Self {
        todo!()
    }
}

impl<T> SubAssign for Vec3<T> {
    /// Subtract assign a vector with another of same type.
    fn sub_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl<T> Mul<T> for Vec3<T> {
    type Output = Self;

    /// Multiply a vector by a scalar of same type,
    /// returning a vector.
    fn mul(self, rhs: T) -> Self {
        todo!()
    }
}

impl<T> MulAssign<T> for Vec3<T> {
    /// Multiply assign a vector by a scalar of same type.
    fn mul_assign(&mut self, rhs: T) {
        todo!();
    }
}

impl Mul<Vec3i> for i32 {
    type Output = Vec3i;

    /// Multiply a vector by an i32 scalar, returning a new
    /// vector of same type.
    fn mul(self, rhs: Vec3i) -> Vec3i {
        todo!()
    }
}

impl Mul<Vec3f> for Float {
    type Output = Vec3f;

    /// Multiply a vector by a float scalar, returning a new
    /// vector of same type.
    fn mul(self, rhs: Vec3f) -> Vec3f {
        todo!()
    }
}

impl<T> Div<T> for Vec3<T> {
    type Output = Self;

    /// Divide a vector by a scalar of same type,
    /// returning a new vector.
    fn div(self, rhs: T) -> Self {
        todo!()
    }
}

impl<T> DivAssign<T> for Vec3<T> {
    /// Divide assign a vector by a scalar of same type.
    fn div_assign(&mut self, rhs: T) {
        todo!()
    }
}

impl<T> Neg for Vec3<T> {
    type Output = Self;

    /// Negate a vector's elements.
    fn neg(self) -> Self {
        todo!()
    }
}
