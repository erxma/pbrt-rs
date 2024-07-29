use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign};

use delegate::delegate;
use derive_more::{Add, From, Neg};

use crate::{
    self as pbrt,
    geometry::bounds::{Bounds2f, Bounds2i, Bounds3f, Bounds3i},
    impl_tuple_math_ops,
    math::{interval::Interval, tuple::Tuple},
};

use super::vec::{Vec2f, Vec2i, Vec3f, Vec3fi, Vec3i};

/// A 3D point of i32.
// Wrapper around the vector equivalent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Neg, Add, From)]
#[repr(transparent)]
pub struct Point3i(Vec3i);

impl Tuple<3, i32> for Point3i {}
impl_tuple_math_ops!(Point3i; 3; i32);

impl From<[i32; 3]> for Point3i {
    fn from(arr: [i32; 3]) -> Self {
        let [x, y, z] = arr;
        Self::new(x, y, z)
    }
}

impl Point3i {
    /// Construct a new point with given elements.
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(Vec3i::new(x, y, z))
    }

    delegate! {
        to self.0 {
            #[inline(always)] pub fn x(&self) -> i32;
            #[inline(always)] pub fn y(&self) -> i32;
            #[inline(always)] pub fn z(&self) -> i32;
            #[inline(always)] pub fn x_mut(&mut self) -> &mut i32;
            #[inline(always)] pub fn y_mut(&mut self) -> &mut i32;
            #[inline(always)] pub fn z_mut(&mut self) -> &mut i32;
        }
    }

    /// The squared distance between `self` and `p2`.
    pub fn distance_squared(self, p2: Self) -> i32 {
        (self - p2).length_squared()
    }

    /// The distance between `self` and `p2`.
    pub fn distance(self, p2: Self) -> pbrt::Float {
        (self - p2).length()
    }

    /// Linearly interpolate between two points. Returns p0 at t==0, p1 at t==1.
    /// Extrapolates for t<0 or t>1.
    pub fn lerp(t: pbrt::Float, p0: Self, p1: Self) -> Point3f {
        let p0: Point3f = p0.into();
        let p1: Point3f = p1.into();
        p0 * (1.0 - t) + p1 * t
    }

    pub fn inside(self, b: Bounds3i) -> bool {
        let x_inside = self.x() >= b.p_min.x() && self.x() <= b.p_max.x();
        let y_inside = self.y() >= b.p_min.y() && self.y() <= b.p_max.y();
        let z_inside = self.z() >= b.p_min.z() && self.z() <= b.p_max.z();

        x_inside && y_inside && z_inside
    }

    pub fn inside_exclusive(self, b: Bounds3i) -> bool {
        let x_inside = self.x() >= b.p_min.x() && self.x() < b.p_max.x();
        let y_inside = self.y() >= b.p_min.y() && self.y() < b.p_max.y();
        let z_inside = self.z() >= b.p_min.z() && self.z() < b.p_max.z();

        x_inside && y_inside && z_inside
    }
}

impl Index<usize> for Point3i {
    type Output = i32;

    /// Index `self`'s elements by 0, 1, 2.
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0..=2 => &self.0[index],
            _ => panic!("Index out of bounds for Point3"),
        }
    }
}

impl IndexMut<usize> for Point3i {
    /// Index `self`'s elements by 0, 1, 2, mutably.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0..=2 => &mut self.0[index],
            _ => panic!("Index out of bounds for Point3"),
        }
    }
}

impl Add<Vec3i> for Point3i {
    type Output = Self;

    /// Add a vector to `self` to get a new point of same type.
    fn add(mut self, rhs: Vec3i) -> Self {
        self += rhs;
        self
    }
}

impl AddAssign<Vec3i> for Point3i {
    /// Add assign a vector to `self`.
    fn add_assign(&mut self, rhs: Vec3i) {
        self.0 += rhs;
    }
}

impl Sub for Point3i {
    type Output = Vec3i;

    /// Subtract two points to get the vector between them.
    fn sub(self, rhs: Self) -> Self::Output {
        Vec3i::new(self.x() - rhs.x(), self.y() - rhs.y(), self.z() - rhs.z())
    }
}

impl Sub<Vec3i> for Point3i {
    type Output = Self;

    /// Subtract a vector from `self` to get a new point.
    fn sub(self, rhs: Vec3i) -> Self {
        Self::new(self.x() - rhs.x(), self.y() - rhs.y(), self.z() - rhs.z())
    }
}

impl SubAssign<Vec3i> for Point3i {
    /// Subtract assign a vector from `self`.
    fn sub_assign(&mut self, rhs: Vec3i) {
        *self = *self - rhs
    }
}

/// A 3D point of `f32`, or `f64` if feature `use-f64` is enabled.
// Wrapper around the vector equivalent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Neg, Add, From)]
#[repr(transparent)]
pub struct Point3f(Vec3f);

impl Tuple<3, pbrt::Float> for Point3f {}
impl_tuple_math_ops!(Point3f; 3; pbrt::Float);

impl From<[pbrt::Float; 3]> for Point3f {
    fn from(arr: [pbrt::Float; 3]) -> Self {
        let [x, y, z] = arr;
        Self::new(x, y, z)
    }
}

impl Point3f {
    /// Construct a new point with given elements.
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
        }
    }

    /// The squared distance between `self` and `p2`.
    pub fn distance_squared(self, p2: Self) -> pbrt::Float {
        (self - p2).length_squared()
    }

    /// The distance between `self` and `p2`.
    pub fn distance(self, p2: Self) -> pbrt::Float {
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

    pub fn inside(self, b: Bounds3f) -> bool {
        let x_inside = self.x() >= b.p_min.x() && self.x() <= b.p_max.x();
        let y_inside = self.y() >= b.p_min.y() && self.y() <= b.p_max.y();
        let z_inside = self.z() >= b.p_min.z() && self.z() <= b.p_max.z();

        x_inside && y_inside && z_inside
    }

    pub fn inside_exclusive(self, b: Bounds3f) -> bool {
        let x_inside = self.x() >= b.p_min.x() && self.x() < b.p_max.x();
        let y_inside = self.y() >= b.p_min.y() && self.y() < b.p_max.y();
        let z_inside = self.z() >= b.p_min.z() && self.z() < b.p_max.z();

        x_inside && y_inside && z_inside
    }

    pub fn as_point3i(self) -> Point3i {
        Point3i::new(self.x() as i32, self.y() as i32, self.z() as i32)
    }
}

impl Index<usize> for Point3f {
    type Output = pbrt::Float;

    /// Index `self`'s elements by 0, 1, 2.
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0..=2 => &self.0[index],
            _ => panic!("Index out of bounds for Point3"),
        }
    }
}

impl IndexMut<usize> for Point3f {
    /// Index `self`'s elements by 0, 1, 2, mutably.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0..=2 => &mut self.0[index],
            _ => panic!("Index out of bounds for Point3"),
        }
    }
}

impl From<Point3i> for Point3f {
    fn from(value: Point3i) -> Self {
        Self(value.0.into())
    }
}

impl Add<Vec3f> for Point3f {
    type Output = Self;

    /// Add a vector to `self` to get a new point of same type.
    fn add(mut self, rhs: Vec3f) -> Self {
        self += rhs;
        self
    }
}

impl AddAssign<Vec3f> for Point3f {
    /// Add assign a vector to `self`.
    fn add_assign(&mut self, rhs: Vec3f) {
        self.0 += rhs;
    }
}

impl Sub for Point3f {
    type Output = Vec3f;

    /// Subtract two points to get the vector between them.
    fn sub(self, rhs: Self) -> Self::Output {
        Vec3f::new(self.x() - rhs.x(), self.y() - rhs.y(), self.z() - rhs.z())
    }
}

impl Sub<Vec3f> for Point3f {
    type Output = Self;

    /// Subtract a vector from `self` to get a new point.
    fn sub(self, rhs: Vec3f) -> Self {
        Self::new(self.x() - rhs.x(), self.y() - rhs.y(), self.z() - rhs.z())
    }
}

impl SubAssign<Vec3f> for Point3f {
    /// Subtract assign a vector from `self`.
    fn sub_assign(&mut self, rhs: Vec3f) {
        *self = *self - rhs
    }
}

/// A 2D point of i32.
// Wrapper around the vector equivalent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Neg, Add, From)]
#[repr(transparent)]
pub struct Point2i(Vec2i);

impl Tuple<2, i32> for Point2i {}
impl_tuple_math_ops!(Point2i; 2; i32);

impl From<[i32; 2]> for Point2i {
    fn from(arr: [i32; 2]) -> Self {
        let [x, y] = arr;
        Self::new(x, y)
    }
}

impl Point2i {
    /// Construct a new point with given elements.
    pub const fn new(x: i32, y: i32) -> Self {
        Self(Vec2i::new(x, y))
    }

    delegate! {
        to self.0 {
            #[inline(always)] pub fn x(&self) -> i32;
            #[inline(always)] pub fn y(&self) -> i32;
            #[inline(always)] pub fn x_mut(&mut self) -> &mut i32;
            #[inline(always)] pub fn y_mut(&mut self) -> &mut i32;
        }
    }

    /// The squared distance between `self` and `p2`.
    pub fn distance_squared(self, p2: Self) -> i32 {
        (self - p2).length_squared()
    }

    /// The distance between `self` and `p2`.
    pub fn distance(self, p2: Self) -> pbrt::Float {
        (self - p2).length()
    }

    /// Linearly interpolate between two points. Returns p0 at t==0, p1 at t==1.
    /// Extrapolates for t<0 or t>1.
    pub fn lerp(t: pbrt::Float, p0: Self, p1: Self) -> Point2f {
        let p0: Point2f = p0.into();
        let p1: Point2f = p1.into();
        p0 * (1.0 - t) + p1 * t
    }

    pub fn inside(self, b: Bounds2i) -> bool {
        let x_inside = self.x() >= b.p_min.x() && self.x() <= b.p_max.x();
        let y_inside = self.y() >= b.p_min.y() && self.y() <= b.p_max.y();

        x_inside && y_inside
    }

    pub fn inside_exclusive(self, b: Bounds2i) -> bool {
        let x_inside = self.x() >= b.p_min.x() && self.x() < b.p_max.x();
        let y_inside = self.y() >= b.p_min.y() && self.y() < b.p_max.y();

        x_inside && y_inside
    }
}

impl Index<usize> for Point2i {
    type Output = i32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0..=1 => &self.0[index],
            _ => panic!("Index out of bounds for Point2"),
        }
    }
}

impl IndexMut<usize> for Point2i {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0..=1 => &mut self.0[index],
            _ => panic!("Index out of bounds for Point2"),
        }
    }
}

impl Add<Vec2i> for Point2i {
    type Output = Self;

    /// Add a vector to `self` to get a new point of same type.
    fn add(mut self, rhs: Vec2i) -> Self {
        self += rhs;
        self
    }
}

impl AddAssign<Vec2i> for Point2i {
    /// Add assign a vector to `self`.
    fn add_assign(&mut self, rhs: Vec2i) {
        self.0 += rhs;
    }
}

impl Sub for Point2i {
    type Output = Vec2i;

    /// Subtract two points to get the vector between them.
    fn sub(self, rhs: Self) -> Self::Output {
        Vec2i::new(self.x() - rhs.x(), self.y() - rhs.y())
    }
}

impl Sub<Vec2i> for Point2i {
    type Output = Self;

    /// Subtract a vector from `self` to get a new point.
    fn sub(self, rhs: Vec2i) -> Self {
        Self::new(self.x() - rhs.x(), self.y() - rhs.y())
    }
}

impl SubAssign<Vec2i> for Point2i {
    /// Subtract assign a vector from `self`.
    fn sub_assign(&mut self, rhs: Vec2i) {
        *self = *self - rhs
    }
}

/// A 3D point of `f32`, or `f64` if feature `use-f64` is enabled.
// Wrapper around the vector equivalent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Neg, Add, From)]
#[repr(transparent)]
pub struct Point2f(Vec2f);

impl Tuple<2, pbrt::Float> for Point2f {}
impl_tuple_math_ops!(Point2f; 2; pbrt::Float);

impl From<[pbrt::Float; 2]> for Point2f {
    fn from(arr: [pbrt::Float; 2]) -> Self {
        let [x, y] = arr;
        Self::new(x, y)
    }
}

impl Point2f {
    /// Construct a new point with given elements.
    pub const fn new(x: pbrt::Float, y: pbrt::Float) -> Self {
        Self(Vec2f::new(x, y))
    }

    delegate! {
        to self.0 {
            #[inline(always)] pub fn x(&self) -> pbrt::Float;
            #[inline(always)] pub fn y(&self) -> pbrt::Float;
            #[inline(always)] pub fn x_mut(&mut self) -> &mut pbrt::Float;
            #[inline(always)] pub fn y_mut(&mut self) -> &mut pbrt::Float;
        }
    }

    /// The squared distance between `self` and `p2`.
    pub fn distance_squared(self, p2: Self) -> pbrt::Float {
        (self - p2).length_squared()
    }

    /// The distance between `self` and `p2`.
    pub fn distance(self, p2: Self) -> pbrt::Float {
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

    pub fn inside(self, b: Bounds2f) -> bool {
        let x_inside = self.x() >= b.p_min.x() && self.x() <= b.p_max.x();
        let y_inside = self.y() >= b.p_min.y() && self.y() <= b.p_max.y();

        x_inside && y_inside
    }

    pub fn inside_exclusive(self, b: Bounds2f) -> bool {
        let x_inside = self.x() >= b.p_min.x() && self.x() < b.p_max.x();
        let y_inside = self.y() >= b.p_min.y() && self.y() < b.p_max.y();

        x_inside && y_inside
    }

    pub fn as_point2i(self) -> Point2i {
        Point2i::new(self.x() as i32, self.y() as i32)
    }
}

impl Index<usize> for Point2f {
    type Output = pbrt::Float;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0..=1 => &self.0[index],
            _ => panic!("Index out of bounds for Point2"),
        }
    }
}

impl IndexMut<usize> for Point2f {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0..=1 => &mut self.0[index],
            _ => panic!("Index out of bounds for Point2"),
        }
    }
}

impl From<Point2i> for Point2f {
    fn from(value: Point2i) -> Self {
        Self(value.0.into())
    }
}

impl Add<Vec2f> for Point2f {
    type Output = Self;

    /// Add a vector to `self` to get a new point of same type.
    fn add(mut self, rhs: Vec2f) -> Self {
        self += rhs;
        self
    }
}

impl AddAssign<Vec2f> for Point2f {
    /// Add assign a vector to `self`.
    fn add_assign(&mut self, rhs: Vec2f) {
        self.0 += rhs;
    }
}

impl Sub for Point2f {
    type Output = Vec2f;

    /// Subtract two points to get the vector between them.
    fn sub(self, rhs: Self) -> Self::Output {
        Vec2f::new(self.x() - rhs.x(), self.y() - rhs.y())
    }
}

impl Sub<Vec2f> for Point2f {
    type Output = Self;

    /// Subtract a vector from `self` to get a new point.
    fn sub(self, rhs: Vec2f) -> Self {
        Self::new(self.x() - rhs.x(), self.y() - rhs.y())
    }
}

impl SubAssign<Vec2f> for Point2f {
    /// Subtract assign a vector from `self`.
    fn sub_assign(&mut self, rhs: Vec2f) {
        *self = *self - rhs
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point3fi(Vec3fi);

impl Point3fi {
    pub const fn new(x: Interval, y: Interval, z: Interval) -> Self {
        Self(Vec3fi::new(x, y, z))
    }

    pub fn new_fi(values: Point3f, errors: Point3f) -> Self {
        Self::new(
            Interval::new_with_err(values.x(), errors.x()),
            Interval::new_with_err(values.y(), errors.y()),
            Interval::new_with_err(values.z(), errors.z()),
        )
    }

    pub fn new_fi_exact(x: pbrt::Float, y: pbrt::Float, z: pbrt::Float) -> Self {
        Self::new(Interval::new(x), Interval::new(y), Interval::new(z))
    }

    delegate! {
        to self.0 {
            #[inline(always)] pub fn x(&self) -> Interval;
            #[inline(always)] pub fn y(&self) -> Interval;
            #[inline(always)] pub fn z(&self) -> Interval;
            #[inline(always)] pub fn x_mut(&mut self) -> &mut Interval;
            #[inline(always)] pub fn y_mut(&mut self) -> &mut Interval;
            #[inline(always)] pub fn z_mut(&mut self) -> &mut Interval;
        }
    }

    pub fn error(&self) -> Point3f {
        Point3f::new(
            self.x().width() / 2.0,
            self.y().width() / 2.0,
            self.z().width() / 2.0,
        )
    }

    pub fn is_exact(&self) -> bool {
        self.x().width() == 0.0 && self.y().width() == 0.0 && self.z().width() == 0.0
    }

    pub fn midpoints_only(&self) -> Point3f {
        Point3f::new(
            self.x().midpoint(),
            self.y().midpoint(),
            self.z().midpoint(),
        )
    }
}

impl From<Point3f> for Point3fi {
    fn from(p: Point3f) -> Self {
        Self::new_fi_exact(p.x(), p.y(), p.z())
    }
}
