use std::{
    fmt,
    ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign},
};

use bytemuck::{Pod, Zeroable};
use delegate::delegate;

use crate::{
    self as pbrt,
    geometry::{Bounds2f, Bounds2i, Bounds3f, Bounds3i},
    Float,
};

use super::{
    impl_tuple_math_ops, Interval, Tuple, Vec2Isize, Vec2Usize, Vec2f, Vec2i, Vec3f, Vec3fi, Vec3i,
};

/// A 3D point of i32.
// Wrapper around the vector equivalent.
#[derive(
    Clone,
    Copy,
    Default,
    PartialEq,
    derive_more::Neg,
    derive_more::Add,
    derive_more::From,
    Zeroable,
    Pod,
)]
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
    pub const ZERO: Self = Self::new(0, 0, 0);

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
        Self(self.0 - rhs)
    }
}

impl SubAssign<Vec3i> for Point3i {
    /// Subtract assign a vector from `self`.
    fn sub_assign(&mut self, rhs: Vec3i) {
        *self = *self - rhs
    }
}

impl fmt::Debug for Point3i {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Point3i")
            .field("x", &self.x())
            .field("y", &self.y())
            .field("z", &self.z())
            .finish()
    }
}

impl fmt::Display for Point3i {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Point3i [{}, {}, {}]", self.x(), self.y(), self.z())
    }
}

/// A 3D point of `f32`, or `f64` if feature `use-f64` is enabled.
// Wrapper around the vector equivalent.
#[derive(
    Clone,
    Copy,
    Default,
    PartialEq,
    derive_more::Neg,
    derive_more::Add,
    derive_more::From,
    Zeroable,
    Pod,
)]
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
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

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
        Self(self.0 - rhs)
    }
}

impl SubAssign<Vec3f> for Point3f {
    /// Subtract assign a vector from `self`.
    fn sub_assign(&mut self, rhs: Vec3f) {
        *self = *self - rhs
    }
}

impl fmt::Debug for Point3f {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Point3f")
            .field("x", &self.x())
            .field("y", &self.y())
            .field("z", &self.z())
            .finish()
    }
}

impl fmt::Display for Point3f {
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

/// A 2D point of i32.
// Wrapper around the vector equivalent.
#[derive(
    Clone,
    Copy,
    Default,
    PartialEq,
    derive_more::Neg,
    derive_more::Add,
    derive_more::From,
    Zeroable,
    Pod,
)]
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

    pub fn as_point2f(self) -> Point2f {
        Point2f::new(self.x() as Float, self.y() as Float)
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
        Self(self.0 - rhs)
    }
}

impl SubAssign<Vec2i> for Point2i {
    /// Subtract assign a vector from `self`.
    fn sub_assign(&mut self, rhs: Vec2i) {
        *self = *self - rhs
    }
}

impl fmt::Debug for Point2i {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Point2i")
            .field("x", &self.x())
            .field("y", &self.y())
            .finish()
    }
}

impl fmt::Display for Point2i {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.x(), self.y())
    }
}

/// A 3D point of `f32`, or `f64` if feature `use-f64` is enabled.
// Wrapper around the vector equivalent.
#[derive(
    Clone,
    Copy,
    Default,
    PartialEq,
    derive_more::Neg,
    derive_more::Add,
    derive_more::From,
    Zeroable,
    Pod,
)]
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
    pub const ZERO: Self = Self::new(0.0, 0.0);

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
        Self(self.0 - rhs)
    }
}

impl SubAssign<Vec2f> for Point2f {
    /// Subtract assign a vector from `self`.
    fn sub_assign(&mut self, rhs: Vec2f) {
        *self = *self - rhs
    }
}

impl fmt::Debug for Point2f {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Point2f")
            .field("x", &self.x())
            .field("y", &self.y())
            .finish()
    }
}

impl fmt::Display for Point2f {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(p) = f.precision() {
            write!(f, "[{:.*}, {:.*}]", p, self.x(), p, self.y(),)
        } else {
            write!(f, "[{}, {}]", self.x(), self.y())
        }
    }
}

/// A 2D point of usize.
// Wrapper around the vector equivalent.
#[derive(Clone, Copy, Default, PartialEq, derive_more::Add, derive_more::From)]
#[repr(transparent)]
pub struct Point2Usize(Vec2Usize);

impl Tuple<2, usize> for Point2Usize {}
impl_tuple_math_ops!(Point2Usize; 2; usize);

impl From<[usize; 2]> for Point2Usize {
    fn from(arr: [usize; 2]) -> Self {
        let [x, y] = arr;
        Self::new(x, y)
    }
}

impl Point2Usize {
    /// Construct a new point with given elements.
    pub const fn new(x: usize, y: usize) -> Self {
        Self(Vec2Usize::new(x, y))
    }

    delegate! {
        to self.0 {
            #[inline(always)] pub fn x(&self) -> usize;
            #[inline(always)] pub fn y(&self) -> usize;
            #[inline(always)] pub fn x_mut(&mut self) -> &mut usize;
            #[inline(always)] pub fn y_mut(&mut self) -> &mut usize;
        }
    }

    pub fn as_point2isize(self) -> Point2Isize {
        Point2Isize::new(self.x() as isize, self.y() as isize)
    }
}

impl Index<usize> for Point2Usize {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0..=1 => &self.0[index],
            _ => panic!("Index out of bounds for Point2"),
        }
    }
}

impl IndexMut<usize> for Point2Usize {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0..=1 => &mut self.0[index],
            _ => panic!("Index out of bounds for Point2"),
        }
    }
}

impl Sub for Point2Usize {
    type Output = Vec2Isize;

    /// Subtract two points to get the vector between them.
    fn sub(self, rhs: Self) -> Self::Output {
        let self_i = self.as_point2isize();
        let rhs = rhs.as_point2isize();
        Vec2Isize::new(self_i.x() - rhs.x(), self_i.y() - rhs.y())
    }
}

impl Sub<Vec2Usize> for Point2Usize {
    type Output = Self;

    /// Subtract a vector from `self` to get a new point.
    fn sub(self, rhs: Vec2Usize) -> Self {
        Self(self.0 - rhs)
    }
}

impl SubAssign<Vec2Usize> for Point2Usize {
    /// Subtract assign a vector from `self`.
    fn sub_assign(&mut self, rhs: Vec2Usize) {
        *self = *self - rhs
    }
}

impl fmt::Debug for Point2Usize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Point2Usize")
            .field("x", &self.x())
            .field("y", &self.y())
            .finish()
    }
}

impl fmt::Display for Point2Usize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.x(), self.y())
    }
}

/// A 2D point of isize.
// Wrapper around the vector equivalent.
#[derive(
    Clone, Copy, Default, PartialEq, derive_more::Neg, derive_more::Add, derive_more::From,
)]
#[repr(transparent)]
pub struct Point2Isize(Vec2Isize);

impl Tuple<2, isize> for Point2Isize {}
impl_tuple_math_ops!(Point2Isize; 2; isize);

impl From<[isize; 2]> for Point2Isize {
    fn from(arr: [isize; 2]) -> Self {
        let [x, y] = arr;
        Self::new(x, y)
    }
}

impl Point2Isize {
    /// Construct a new point with given elements.
    pub const fn new(x: isize, y: isize) -> Self {
        Self(Vec2Isize::new(x, y))
    }

    delegate! {
        to self.0 {
            #[inline(always)] pub fn x(&self) -> isize;
            #[inline(always)] pub fn y(&self) -> isize;
            #[inline(always)] pub fn x_mut(&mut self) -> &mut isize;
            #[inline(always)] pub fn y_mut(&mut self) -> &mut isize;
        }
    }

    pub fn as_point2usize(self) -> Point2Usize {
        Point2Usize::new(self.x() as usize, self.y() as usize)
    }
}

impl From<Point2i> for Point2Isize {
    fn from(value: Point2i) -> Self {
        Self::new(value.x() as isize, value.y() as isize)
    }
}

impl Index<usize> for Point2Isize {
    type Output = isize;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0..=1 => &self.0[index],
            _ => panic!("Index out of bounds for Point2"),
        }
    }
}

impl IndexMut<usize> for Point2Isize {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0..=1 => &mut self.0[index],
            _ => panic!("Index out of bounds for Point2"),
        }
    }
}

impl Sub for Point2Isize {
    type Output = Vec2Isize;

    /// Subtract two points to get the vector between them.
    fn sub(self, rhs: Self) -> Self::Output {
        Vec2Isize::new(self.x() - rhs.x(), self.y() - rhs.y())
    }
}

impl Sub<Vec2Isize> for Point2Isize {
    type Output = Self;

    /// Subtract a vector from `self` to get a new point.
    fn sub(self, rhs: Vec2Isize) -> Self {
        Self(self.0 - rhs)
    }
}

impl SubAssign<Vec2Isize> for Point2Isize {
    /// Subtract assign a vector from `self`.
    fn sub_assign(&mut self, rhs: Vec2Isize) {
        *self = *self - rhs
    }
}

impl fmt::Debug for Point2Isize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Point2Isize")
            .field("x", &self.x())
            .field("y", &self.y())
            .finish()
    }
}

impl fmt::Display for Point2Isize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.x(), self.y())
    }
}

#[derive(
    Clone,
    Copy,
    PartialEq,
    derive_more::Sub,
    derive_more::Mul,
    derive_more::Div,
    derive_more::MulAssign,
    derive_more::DivAssign,
)]
pub struct Point3fi(Vec3fi);

impl Point3fi {
    pub const fn new(x: Interval, y: Interval, z: Interval) -> Self {
        Self(Vec3fi::new(x, y, z))
    }

    pub fn new_fi(values: Point3f, errors: Vec3f) -> Self {
        Self::new(
            Interval::new_with_err(values.x(), errors.x()),
            Interval::new_with_err(values.y(), errors.y()),
            Interval::new_with_err(values.z(), errors.z()),
        )
    }

    pub fn new_fi_exact(x: pbrt::Float, y: pbrt::Float, z: pbrt::Float) -> Self {
        Self::new(
            Interval::new_exact(x),
            Interval::new_exact(y),
            Interval::new_exact(z),
        )
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

    pub fn error(&self) -> Vec3f {
        self.0.error()
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

impl Add<Vec3fi> for Point3fi {
    type Output = Self;

    /// Add a vector to `self` to get a new point of same type.
    fn add(mut self, rhs: Vec3fi) -> Self {
        self += rhs;
        self
    }
}

impl AddAssign<Vec3fi> for Point3fi {
    /// Add assign a vector to `self`.
    fn add_assign(&mut self, rhs: Vec3fi) {
        self.0 += rhs;
    }
}

impl Sub<Vec3fi> for Point3fi {
    type Output = Self;

    /// Subtract a vector from `self` to get a new point.
    fn sub(self, rhs: Vec3fi) -> Self {
        Self(self.0 - rhs)
    }
}

impl SubAssign<Vec3fi> for Point3fi {
    /// Subtract assign a vector from `self`.
    fn sub_assign(&mut self, rhs: Vec3fi) {
        *self = *self - rhs
    }
}

impl fmt::Debug for Point3fi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Point3fi")
            .field("x", &self.x())
            .field("y", &self.y())
            .field("z", &self.z())
            .finish()
    }
}

impl fmt::Display for Point3fi {
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
