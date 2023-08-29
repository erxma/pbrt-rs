use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign};

use crate::{
    self as pbrt,
    geometry::bounds3::Bounds3,
    impl_tuple_math_ops_generic,
    math::{
        interval::Interval,
        tuple::{Tuple, TupleElement},
        vec3::Vec3,
    },
};

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
}

impl<T: TupleElement> Tuple<3, T> for Point3<T> {}

impl_tuple_math_ops_generic!(Point3; 3);

impl<T: TupleElement> Point3<T> {
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

impl<T> From<[T; 3]> for Point3<T> {
    fn from(arr: [T; 3]) -> Self {
        let [x, y, z] = arr;
        Self::new(x, y, z)
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

impl<T> From<Vec3<T>> for Point3<T> {
    fn from(value: Vec3<T>) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point2<T> {
    pub x: T,
    pub y: T,
}

pub type Point2i = Point2<i32>;
pub type Point2f = Point2<pbrt::Float>;

impl<T> Point2<T> {
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

pub type Point3fi = Point3<Interval>;

impl Point3fi {
    pub fn new_fi(values: Point3f, errors: Point3f) -> Self {
        Self::new(
            Interval::new_with_err(values.x, errors.x),
            Interval::new_with_err(values.y, errors.y),
            Interval::new_with_err(values.z, errors.z),
        )
    }

    pub fn new_fi_exact(x: pbrt::Float, y: pbrt::Float, z: pbrt::Float) -> Self {
        Self::new(Interval::new(x), Interval::new(y), Interval::new(z))
    }

    pub fn error(&self) -> Point3f {
        Point3f::new(
            self.x.width() / 2.0,
            self.y.width() / 2.0,
            self.z.width() / 2.0,
        )
    }

    pub fn is_exact(&self) -> bool {
        self.x.width() == 0.0 && self.y.width() == 0.0 && self.z.width() == 0.0
    }

    pub fn midpoints_only(&self) -> Point3f {
        Point3f::new(self.x.midpoint(), self.y.midpoint(), self.z.midpoint())
    }
}

impl From<Point3f> for Point3fi {
    fn from(p: Point3f) -> Self {
        Self::new_fi_exact(p.x, p.y, p.z)
    }
}
