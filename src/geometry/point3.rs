use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use num_traits::{real::Real, Float, Num, Signed};

use crate as pbrt;

use super::{bounds3::Bounds3, vec3::Vec3};

/// A 3D point.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub type Point3i = Point3<i32>;
pub type Point3f = Point3<pbrt::Float>;

impl<T> Point3<T> {
    /// Construct a new point with given elements.
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    /// Convert point elements into another type.
    pub fn into_<U>(self) -> Point3<U>
    where
        T: Into<U>,
    {
        Point3 {
            x: self.x.into(),
            y: self.y.into(),
            z: self.z.into(),
        }
    }

    /// Permute a point's elements according to the index values
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
}

impl<T: Num + Copy> Point3<T> {
    /// The squared distance between `self` and `p2`.
    pub fn distance_squared(self, p2: Self) -> T {
        (self - p2).length_squared()
    }

    /// The distance between `self` and `p2`.
    pub fn distance(self, p2: Self) -> pbrt::Float
    where
        T: Into<pbrt::Float>,
    {
        (self - p2).length()
    }

    /// Linearly interpolate between two points. Returns p0 at t==0, p1 at t==1.
    /// Extrapolates for t<0 or t>1.
    pub fn lerp(t: pbrt::Float, p0: Self, p1: Self) -> Self
    where
        Self: Mul<pbrt::Float, Output = Self>,
    {
        p0 * (1.0 - t) + p1 * t
    }

    /// Applies floor to each component.
    pub fn floor(self) -> Self
    where
        T: Real,
    {
        Self {
            x: self.x.floor(),
            y: self.y.floor(),
            z: self.z.floor(),
        }
    }

    /// Applies ceil to each component.
    pub fn ceil(self) -> Self
    where
        T: Real,
    {
        Self {
            x: self.x.ceil(),
            y: self.y.ceil(),
            z: self.z.ceil(),
        }
    }

    /// Applies abs to each component.
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

    pub fn inside(self, b: Bounds3<T>) -> bool
    where
        T: PartialOrd,
    {
        let x_inside = self.x >= b.p_min.x && self.x <= b.p_max.x;
        let y_inside = self.y >= b.p_min.y && self.y <= b.p_max.y;
        let z_inside = self.z >= b.p_min.z && self.z <= b.p_max.z;

        x_inside && y_inside && z_inside
    }

    pub fn inside_exclusive(self, b: Bounds3<T>) -> bool
    where
        T: PartialOrd,
    {
        let x_inside = self.x >= b.p_min.x && self.x < b.p_max.x;
        let y_inside = self.y >= b.p_min.y && self.y < b.p_max.y;
        let z_inside = self.z >= b.p_min.z && self.z < b.p_max.z;

        x_inside && y_inside && z_inside
    }
}

impl<T: Float> Point3<T> {
    /// Returns true if any component is NaN.
    pub fn has_nans(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
}

impl<T: Ord> Point3<T> {
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

impl<T: Float> Point3<T> {
    /// Returns a vector containing the min float values for each
    /// component of `self` and `other` (the component-wise min).
    pub fn min_float(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    /// Returns a vector containing the max float values for each
    /// component of `self` and `other` (the component-wise max).
    pub fn max_float(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }
}

impl<T> Index<usize> for Point3<T> {
    type Output = T;

    /// Index `self`'s elements by 0, 1, 2.
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds for Point3"),
        }
    }
}

impl<T> IndexMut<usize> for Point3<T> {
    /// Index `self`'s elements by 0, 1, 2, mutably.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index out of bounds for Point3"),
        }
    }
}

impl<T: Add<Output = T>> Add for Point3<T> {
    type Output = Self;

    /// Add another point to `self` to get a new point of same type.
    ///
    /// Doesn't make sense mathematically but useful for calculations.
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T> AddAssign for Point3<T>
where
    Self: Add<Output = Self> + Copy,
{
    /// Add assign another point to `self`.
    ///
    /// Doesn't make sense mathematically but useful for calculations.
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl<T: Add<Output = T>> Add<Vec3<T>> for Point3<T> {
    type Output = Self;

    /// Add a vector to `self` to get a new point of same type.
    fn add(self, rhs: Vec3<T>) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T> AddAssign<Vec3<T>> for Point3<T>
where
    Self: Add<Vec3<T>, Output = Self> + Copy,
{
    /// Add assign a vector to `self`.
    fn add_assign(&mut self, rhs: Vec3<T>) {
        *self = *self + rhs
    }
}

impl<T: Sub<Output = T>> Sub for Point3<T> {
    type Output = Vec3<T>;

    /// Subtract two points to get the vector between them.
    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T: Sub<Output = T>> Sub<Vec3<T>> for Point3<T> {
    type Output = Self;

    /// Subtract a vector from `self` to get a new point.
    fn sub(self, rhs: Vec3<T>) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T> SubAssign<Vec3<T>> for Point3<T>
where
    Self: Sub<Vec3<T>, Output = Self> + Copy,
{
    /// Subtract assign a vector from `self`.
    fn sub_assign(&mut self, rhs: Vec3<T>) {
        *self = *self - rhs
    }
}

impl<T, U, V> Mul<U> for Point3<T>
where
    T: Mul<U, Output = V>,
    U: Copy,
{
    type Output = Point3<V>;

    /// Multiply a point by a scalar of same type,
    /// returning a point.
    fn mul(self, rhs: U) -> Self::Output {
        Self::Output {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T, U> MulAssign<U> for Point3<T>
where
    Self: Mul<U, Output = Self> + Copy,
{
    /// Multiply assign `self` by a scalar of same type.
    fn mul_assign(&mut self, rhs: U) {
        *self = *self * rhs
    }
}

impl Mul<Point3i> for i32 {
    type Output = Point3i;

    /// Multiply a point by an i32 scalar, returning a new
    /// point of same type.
    fn mul(self, rhs: Point3i) -> Point3i {
        rhs * self
    }
}

impl Mul<Point3f> for pbrt::Float {
    type Output = Point3f;

    /// Multiply a point by a float scalar, returning a new
    /// point of same type.
    fn mul(self, rhs: Point3f) -> Point3f {
        rhs * self
    }
}

impl<T, U> Div<U> for Point3<T>
where
    T: Div<U>,
    U: Copy,
{
    type Output = Point3<<T as Div<U>>::Output>;

    /// Divide a point by a scalar of same type,
    /// returning a new point.
    fn div(self, rhs: U) -> Self::Output {
        Self::Output {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl<T, U> DivAssign<U> for Point3<T>
where
    Self: Div<U, Output = Self> + Copy,
{
    /// Divide assign a point by a scalar of same type.
    fn div_assign(&mut self, rhs: U) {
        *self = *self / rhs
    }
}

impl<T: Neg<Output = T>> Neg for Point3<T> {
    type Output = Self;

    /// Negate a point's elements.
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl<T> From<Vec3<T>> for Point3<T> {
    fn from(value: Vec3<T>) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}
