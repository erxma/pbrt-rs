use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use num_traits::{Float, Inv, Num, NumCast, Signed};

use crate as pbrt;

use super::normal3::Normal3;

/// A 3D vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub type Vec3i = Vec3<i32>;
pub type Vec3f = Vec3<pbrt::Float>;

impl<T> Vec3<T> {
    /// Construct a new vector with given elements.
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    /// Convert vector elements into another type.
    pub fn into_<U>(self) -> Vec3<U>
    where
        T: Into<U>,
    {
        Vec3 {
            x: self.x.into(),
            y: self.y.into(),
            z: self.z.into(),
        }
    }
}

impl<T: Num + Copy> Vec3<T> {
    /// Returns a vector with the absolute values of the components.
    pub fn abs(self) -> Self
    where
        T: Signed,
    {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    /// The squared length of a vector.
    pub fn length_squared(self) -> T {
        self.dot(self)
    }

    /// The cross product of two vectors.
    ///
    /// May have precision loss or truncation if required to fit the result in `T`.
    /// No indication of this case will be given.
    pub fn cross(self, rhs: Self) -> Self
    where
        T: Into<f64> + NumCast,
    {
        let (v1x, v1y, v1z): (f64, f64, f64) = (self.x.into(), self.y.into(), self.z.into());
        let (v2x, v2y, v2z): (f64, f64, f64) = (rhs.x.into(), rhs.y.into(), rhs.z.into());

        Self {
            x: NumCast::from(v1y * v2x - v1z * v2y).unwrap(),
            y: NumCast::from(v1z * v2x - v1x * v2z).unwrap(),
            z: NumCast::from(v1x * v2y - v1y * v2x).unwrap(),
        }
    }

    /// Returns the index of the component with the largest value.
    pub fn max_dimension(self) -> usize
    where
        T: PartialOrd,
    {
        if self.x > self.y && self.x > self.z {
            0
        } else if self.y > self.x && self.y > self.z {
            1
        } else {
            2
        }
    }

    /// Returns the dot product of two vectors.
    pub fn dot(self, rhs: Self) -> T {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// The absolute value of the dot product of two vectors.
    pub fn absdot(self, rhs: Self) -> T
    where
        T: Signed,
    {
        self.dot(rhs).abs()
    }

    /// Permute a vector's elements according to the index values
    /// given.
    pub fn permute(self, x: usize, y: usize, z: usize) -> Self
    where
        T: Copy,
    {
        Self {
            x: self[x],
            y: self[y],
            z: self[z],
        }
    }

    /// Construct a local coordinate system given a vector.
    ///
    /// Returns the set of three orthogonal vectors representing
    /// the system.
    pub fn coordinate_system(self) -> (Self, Self, Self)
    where
        T: Signed + PartialOrd + Into<f64> + NumCast + Copy,
        Self: Div<f64, Output = Self>,
    {
        let v1 = self;
        let v2 = if v1.x.abs() > v1.y.abs() {
            Vec3::new(-v1.z, T::zero(), v1.x) / (v1.x * v1.x + v1.z * v1.z).into().sqrt()
        } else {
            Vec3::new(T::zero(), v1.z, -v1.y) / (v1.y * v1.y + v1.z * v1.z).into().sqrt()
        };
        let v3 = v1.cross(v2);

        (v1, v2, v3)
    }
}

impl<T> Vec3<T>
where
    T: Num + Into<pbrt::Float> + Copy,
{
    /// The length of a vector.
    pub fn length(self) -> pbrt::Float {
        self.length_squared().into().sqrt()
    }

    /// Returns the normalization of a vector.
    pub fn normalized(self) -> Vec3<pbrt::Float> {
        self.into_() / self.length()
    }
}

impl<T: Float> Vec3<T> {
    /// Returns true if any component is NaN.
    pub fn has_nans(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
}

impl<T: Ord> Vec3<T> {
    /// Returns the element with the smallest value.
    pub fn min_component(self) -> T {
        self.x.min(self.y).min(self.z)
    }

    /// Returns the element with the largest value.
    pub fn max_component(self) -> T {
        self.x.max(self.y).max(self.z)
    }

    /// Returns a vector containing the min values for each
    /// component of `self` and `other` (the component-wise min).
    pub fn min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    /// Returns a vector containing the max values for each
    /// component of `self` and `other` (the component-wise max).
    pub fn max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }
}

impl<T: Float> Vec3<T> {
    /// Returns the float element with the smallest value.
    pub fn min_component_float(self) -> T {
        self.x.min(self.y).min(self.z)
    }

    /// Returns the float element with the largest value.
    pub fn max_component_float(self) -> T {
        self.x.max(self.y).max(self.z)
    }

    /// Returns a vector containing the min float values for each
    /// component of `self` and `other` (the component-wise min).
    pub fn min_f(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    /// Returns a vector containing the max float values for each
    /// component of `self` and `other` (the component-wise max).
    pub fn max_f(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }
}

impl<T> Index<usize> for Vec3<T> {
    type Output = T;

    /// Index a vector's elements by 0, 1, 2.
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds for Vec3"),
        }
    }
}

impl<T> IndexMut<usize> for Vec3<T> {
    /// Index a vector's elements by 0, 1, 2, mutably.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index out of bounds for Vec3"),
        }
    }
}

impl<T: Add<Output = T>> Add for Vec3<T> {
    type Output = Self;

    /// Add two vectors to get new vector of same type.
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T> AddAssign for Vec3<T>
where
    Self: Add<Output = Self> + Copy,
{
    /// Add assign a vector with another of same type.
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl<T: Sub<Output = T>> Sub for Vec3<T> {
    type Output = Self;

    /// Subtract a vector from another of same type.
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T> SubAssign for Vec3<T>
where
    Self: Sub<Output = Self> + Copy,
{
    /// Subtract assign a vector with another of same type.
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs
    }
}

impl<T, U, V> Mul<U> for Vec3<T>
where
    T: Mul<U, Output = V>,
    U: Copy,
{
    type Output = Vec3<V>;

    /// Multiply a vector by a scalar of same type,
    /// returning a vector.
    fn mul(self, rhs: U) -> Self::Output {
        Self::Output {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T, U> MulAssign<U> for Vec3<T>
where
    Self: Mul<U, Output = Self> + Copy,
{
    /// Multiply assign a vector by a scalar of same type.
    fn mul_assign(&mut self, rhs: U) {
        *self = *self * rhs
    }
}

impl Mul<Vec3i> for i32 {
    type Output = Vec3i;

    /// Multiply a vector by an i32 scalar, returning a new
    /// vector of same type.
    fn mul(self, rhs: Vec3i) -> Vec3i {
        rhs * self
    }
}

impl Mul<Vec3f> for pbrt::Float {
    type Output = Vec3f;

    /// Multiply a vector by a float scalar, returning a new
    /// vector of same type.
    fn mul(self, rhs: Vec3f) -> Vec3f {
        rhs * self
    }
}

impl<T, U> Div<U> for Vec3<T>
where
    T: Mul<U, Output = T>,
    U: Inv<Output = U> + Copy,
{
    type Output = Self;

    /// Divide a vector by a scalar of same type,
    /// returning a new vector.
    fn div(self, rhs: U) -> Self {
        Self {
            x: self.x * rhs.inv(),
            y: self.y * rhs.inv(),
            z: self.z * rhs.inv(),
        }
    }
}

impl<T, U> DivAssign<U> for Vec3<T>
where
    Self: Div<U, Output = Self> + Copy,
{
    /// Divide assign a vector by a scalar of same type.
    fn div_assign(&mut self, rhs: U) {
        *self = *self / rhs
    }
}

impl<T: Neg<Output = T>> Neg for Vec3<T> {
    type Output = Self;

    /// Negate a vector's elements.
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl<T> From<Normal3<T>> for Vec3<T> {
    fn from(n: Normal3<T>) -> Self {
        Self {
            x: n.x,
            y: n.y,
            z: n.z,
        }
    }
}
