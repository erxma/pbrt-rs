use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub};

use num_traits::{Float, Inv, Num, Signed};

use crate as pbrt;

use super::vec3::Vec3;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Normal3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub type Normal3f = Normal3<pbrt::Float>;

impl<T> Normal3<T> {
    /// Construct a new normal with given elements.
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl<T: Num + Signed + Copy> Normal3<T> {
    /// If needed, returns `self` flipped so that it lies in the same hemisphere
    /// as the given vector. If already satisfied, returns `self`.
    #[inline]
    pub fn face_forward(self, v: Vec3<T>) -> Self
    where
        Self: Neg<Output = Self>,
    {
        if self.dot_v(v).is_positive() {
            self
        } else {
            -self
        }
    }

    /// Returns the dot product of two normals.
    pub fn dot(self, rhs: Self) -> T {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// The absolute value of the dot product of two normals.
    pub fn absdot(self, rhs: Self) -> T {
        self.dot(rhs).abs()
    }

    /// Returns the dot product of `self` and a vector.
    pub fn dot_v(self, rhs: Vec3<T>) -> T {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// The absolute value of the dot product of `self` and a vector.
    pub fn absdot_v(self, rhs: Vec3<T>) -> T {
        self.dot_v(rhs).abs()
    }

    /// Convert `self`'s elements into another type.
    #[inline]
    pub fn into_<U>(self) -> Normal3<U>
    where
        T: Into<U>,
    {
        Normal3 {
            x: self.x.into(),
            y: self.y.into(),
            z: self.z.into(),
        }
    }
}

impl<T> Normal3<T>
where
    T: Num + Signed + Into<pbrt::Float> + Copy,
{
    /// The squared length of `self`.
    pub fn length_squared(self) -> pbrt::Float {
        (self.x * self.x + self.y * self.y + self.z * self.z).into()
    }

    /// The length of `self`.
    pub fn length(self) -> pbrt::Float {
        self.length_squared().sqrt()
    }

    /// Returns the normalization of `self`.
    pub fn normalized(self) -> Normal3<pbrt::Float> {
        self.into_() / self.length()
    }
}

impl<T: Float> Normal3<T> {
    /// Returns true if any component is NaN.
    pub fn has_nans(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
}

impl<T> Index<usize> for Normal3<T> {
    type Output = T;

    /// Index `self`'s elements by 0, 1, 2.
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds for Normal3"),
        }
    }
}

impl<T> IndexMut<usize> for Normal3<T> {
    /// Index `self`'s elements by 0, 1, 2, mutably.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index out of bounds for Normal3"),
        }
    }
}

impl<T: Add<Output = T>> Add for Normal3<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T> AddAssign for Normal3<T>
where
    Self: Add<Output = Self> + Copy,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl<T: Sub<Output = T>> Sub for Normal3<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T, U, V> Mul<U> for Normal3<T>
where
    T: Mul<U, Output = V>,
    U: Copy,
{
    type Output = Normal3<V>;

    /// Multiply `self` by a scalar of same type,
    /// returning a new normal.
    fn mul(self, rhs: U) -> Self::Output {
        Self::Output {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T, U> MulAssign<U> for Normal3<T>
where
    Self: Mul<U, Output = Self> + Copy,
{
    /// Multiply assign `self` by a scalar of same type.
    fn mul_assign(&mut self, rhs: U) {
        *self = *self * rhs
    }
}

impl Mul<Normal3f> for pbrt::Float {
    type Output = Normal3f;

    /// Multiply `self` by a float scalar, returning a new
    /// normal of same type.
    fn mul(self, rhs: Normal3f) -> Normal3f {
        rhs * self
    }
}

impl<T, U> Div<U> for Normal3<T>
where
    T: Mul<<U as Inv>::Output, Output = T>,
    U: Num + Inv + Copy,
{
    type Output = Self;

    /// Divide `self` by a scalar of same type,
    /// returning a new normal.
    fn div(self, rhs: U) -> Self {
        assert!(!rhs.is_zero());

        Self {
            x: self.x * rhs.inv(),
            y: self.y * rhs.inv(),
            z: self.z * rhs.inv(),
        }
    }
}

impl<T, U> DivAssign<U> for Normal3<T>
where
    Self: Div<U, Output = Self> + Copy,
    U: Num,
{
    /// Divide assign `self` by a scalar of same type.
    fn div_assign(&mut self, rhs: U) {
        assert!(!rhs.is_zero());

        *self = *self / rhs
    }
}

impl<T: Neg<Output = T>> Neg for Normal3<T> {
    type Output = Self;

    /// Negate `self`'s elements.
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl<T> From<Vec3<T>> for Normal3<T> {
    #[inline]
    fn from(value: Vec3<T>) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}
