use std::ops::{Div, Index, IndexMut, Neg};

use num_traits::{Float, NumCast, Signed};

use crate::{self as pbrt, impl_tuple_math_ops_generic, math::routines::safe_asin, PI};

use super::{
    normal3::Normal3,
    tuple::{Tuple, TupleElement},
};

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
    pub const fn new(x: T, y: T, z: T) -> Self {
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
}

impl<T: TupleElement> Tuple<3, T> for Vec3<T> {
    fn from_array(vals: [T; 3]) -> Self {
        Self::new(vals[0], vals[1], vals[2])
    }
}

impl_tuple_math_ops_generic!(Vec3; 3);

impl<T: TupleElement> Vec3<T> {
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

impl<T: TupleElement> Vec3<T>
where
    T: Into<pbrt::Float>,
{
    /// The length of a vector.
    pub fn length(self) -> pbrt::Float {
        self.length_squared().into().sqrt()
    }

    /// Returns the normalization of a vector.
    pub fn normalized(self) -> Vec3<pbrt::Float> {
        self.into_() / self.length()
    }

    #[inline]
    pub fn angle_between(self, other: Self) -> pbrt::Float {
        if self.dot(other).into() < 0.0 {
            PI - 2.0 * safe_asin((self + other).length() / 2.0)
        } else {
            2.0 * safe_asin((other - self).length() / 2.0)
        }
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
