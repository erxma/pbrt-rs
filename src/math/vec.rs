use std::ops::{Add, Sub};

use delegate::delegate;
use derive_more::{From, Index, IndexMut, Neg};
use num_traits::{NumCast, Signed, ToPrimitive};

use crate::{self as pbrt, impl_tuple_math_ops};

use super::{
    interval::Interval,
    normal3::Normal3f,
    tuple::{Tuple, TupleElement},
};

// To facilitate choosing between the implementation from scratch
// ("custom_impl") and glam's, a wrapper is added around the concrete type,
// with inner functionality selectively exposed via the outer functions.
//
// The main disadvantage of this the lack of direct field access (getters
// included instead).
//
// Just for organization, the inner types impl a common internal trait `Vec3`.
// It's not strictly necessary for the structure.
//
// Didn't just expose the common trait as that'd be too unconventional.
//
// Public types aren't generic as glam doesn't offer that.

// ==================== OUTER PUBLIC TYPES ====================

/// A 3-element vector of `i32`.
// Wrapper around the concrete implementation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Index, IndexMut, Neg, From)]
#[repr(transparent)]
pub struct Vec3i(inner::Vec3i);

impl Vec3i {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(inner::Vec3i::new(x, y, z))
    }

    #[inline(always)]
    pub fn x(&self) -> i32 {
        self.0.x
    }

    #[inline(always)]
    pub fn y(&self) -> i32 {
        self.0.y
    }

    #[inline(always)]
    pub fn z(&self) -> i32 {
        self.0.z
    }

    #[inline(always)]
    pub fn x_mut(&mut self) -> &mut i32 {
        &mut self.0.x
    }

    #[inline(always)]
    pub fn y_mut(&mut self) -> &mut i32 {
        &mut self.0.y
    }

    #[inline(always)]
    pub fn z_mut(&mut self) -> &mut i32 {
        &mut self.0.z
    }

    // The following are directly delegated to the inner function
    // of the same name listed in the `Vec3` trait.
    delegate! {
        to self.0 {
            /// The squared length of a vector.
            pub fn length_squared(self) -> i32;
            /// Returns the dot product of two vectors.
            pub fn dot(self, #[newtype] rhs: Self) -> i32;
            /// The absolute value of the dot product of two vectors.
            pub fn absdot(self, #[newtype] rhs: Self) -> i32;
            /// The cross product of two vectors.
            ///
            /// May have precision loss or truncation if required to fit the result in `T`.
            /// No indication of this case will be given.
            #[into]
            pub fn cross(self, #[newtype] rhs: Self) -> Self;
            /// The length of a vector.
            pub fn length(self) -> pbrt::Float;
            /// Returns the normalization of a vector.
            #[into]
            pub fn normalized(self) -> Vec3f;
            pub fn angle_between(self, #[newtype] other: Self) -> pbrt::Float;
    }}

    /// Construct a local coordinate system given a vector.
    ///
    /// Returns the set of three orthogonal float vectors representing
    /// the system.
    pub fn coordinate_system(self) -> (Vec3f, Vec3f, Vec3f) {
        let (v1, v2, v3) = self.0.coordinate_system();
        (Vec3f(v1), Vec3f(v2), Vec3f(v3))
    }
}

impl Tuple<3, i32> for Vec3i {}
impl_tuple_math_ops!(Vec3i; 3; i32);

impl From<[i32; 3]> for Vec3i {
    fn from(arr: [i32; 3]) -> Self {
        let [x, y, z] = arr;
        Self::new(x, y, z)
    }
}

impl Add for Vec3i {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Sub for Vec3i {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

// f version is largely the same idea as i

/// A 3-element vector of `f32`, or `f64` if feature `use-f64` is enabled.
// Wrapper around the concrete implementation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Index, IndexMut, Neg, From)]
#[repr(transparent)]
pub struct Vec3f(inner::Vec3f);

impl Vec3f {
    pub const fn new(x: pbrt::Float, y: pbrt::Float, z: pbrt::Float) -> Self {
        Self(inner::Vec3f::new(x, y, z))
    }

    #[inline(always)]
    pub fn x(&self) -> pbrt::Float {
        self.0.x
    }

    #[inline(always)]
    pub fn y(&self) -> pbrt::Float {
        self.0.y
    }

    #[inline(always)]
    pub fn z(&self) -> pbrt::Float {
        self.0.z
    }

    #[inline(always)]
    pub fn x_mut(&mut self) -> &mut pbrt::Float {
        &mut self.0.x
    }

    #[inline(always)]
    pub fn y_mut(&mut self) -> &mut pbrt::Float {
        &mut self.0.y
    }

    #[inline(always)]
    pub fn z_mut(&mut self) -> &mut pbrt::Float {
        &mut self.0.z
    }

    delegate! {
        to self.0 {
            /// The squared length of a vector.
            pub fn length_squared(self) -> pbrt::Float;
            /// Returns the dot product of two vectors.
            pub fn dot(self, #[newtype] rhs: Self) -> pbrt::Float;
            /// The absolute value of the dot product of two vectors.
            pub fn absdot(self, #[newtype] rhs: Self) -> pbrt::Float;
            /// The cross product of two vectors.
            ///
            /// May have precision loss or truncation if required to fit the result in `T`.
            /// No indication of this case will be given.
            #[into]
            pub fn cross(self, #[newtype] rhs: Self) -> Self;
            /// The length of a vector.
            pub fn length(self) -> pbrt::Float;
            /// Returns the normalization of a vector.
            #[into]
            pub fn normalized(self) -> Self;
            pub fn angle_between(self, #[newtype] other: Self) -> pbrt::Float;
    }}

    /// Construct a local coordinate system given a vector.
    ///
    /// Returns the set of three orthogonal float vectors representing
    /// the system.
    pub fn coordinate_system(self) -> (Self, Self, Self) {
        let (v1, v2, v3) = self.0.coordinate_system();
        (Self(v1), Self(v2), Self(v3))
    }
}

impl Tuple<3, pbrt::Float> for Vec3f {}
impl_tuple_math_ops!(Vec3f; 3; pbrt::Float);

impl From<[pbrt::Float; 3]> for Vec3f {
    fn from(arr: [pbrt::Float; 3]) -> Self {
        let [x, y, z] = arr;
        Self::new(x, y, z)
    }
}

// Convert from normal to vec
impl From<Normal3f> for Vec3f {
    fn from(n: Normal3f) -> Self {
        Self::new(n.x(), n.y(), n.z())
    }
}

impl From<Vec3i> for Vec3f {
    fn from(value: Vec3i) -> Self {
        Self::new(
            value.x() as pbrt::Float,
            value.y() as pbrt::Float,
            value.z() as pbrt::Float,
        )
    }
}

impl Add for Vec3f {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Sub for Vec3f {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

// ==================== HELPER TRAIT FOR INNER TYPES ====================
pub(in crate::math) trait Vec3<T: TupleElement>: Tuple<3, T> {
    /// The type to be used for vecs of float where necessary.
    /// Should always be set to "Self" with T = Float.
    ///
    /// This is here instead of just using "inner::" because having both impls
    /// active (to the compiler) at the same time causes the unselected one
    /// to refer to the wrong type...
    type VecFloat;

    fn length_squared(self) -> T {
        self.dot(self)
    }

    fn dot(self, rhs: Self) -> T;

    fn absdot(self, rhs: Self) -> T
    where
        T: Signed;

    fn cross(self, rhs: Self) -> Self
    where
        T: Into<f64> + NumCast;

    fn length(self) -> pbrt::Float
    where
        T: ToPrimitive;

    fn normalized(self) -> Self::VecFloat
    where
        T: ToPrimitive;

    fn angle_between(self, other: Self) -> pbrt::Float
    where
        T: ToPrimitive;

    fn coordinate_system(self) -> (Self::VecFloat, Self::VecFloat, Self::VecFloat)
    where
        T: ToPrimitive;
}

// ==================== SELECTION OF INNER TYPE ====================
mod inner {
    #[cfg(not(feature = "glam"))]
    pub use super::custom_impl::*;
    #[cfg(feature = "glam")]
    pub use super::glam_impl::*;
}

// ==================== IMPL FROM SCRATCH (CUSTOM) ====================
pub(crate) mod custom_impl {
    use std::ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign};

    use num_traits::{NumCast, Signed, ToPrimitive};

    use crate::{
        self as pbrt, impl_tuple_math_ops_generic,
        math::{
            routines::safe_asin,
            tuple::{Tuple, TupleElement},
        },
    };

    use super::Vec3 as Vec3Trait;

    /// A 3D vector.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Vec3<T> {
        pub x: T,
        pub y: T,
        pub z: T,
    }

    #[allow(dead_code)]
    pub type Vec3i = Vec3<i32>;
    pub type Vec3f = Vec3<pbrt::Float>;

    impl<T> Vec3<T> {
        /// Construct a new vector with given elements.
        pub const fn new(x: T, y: T, z: T) -> Self {
            Self { x, y, z }
        }
    }

    impl<T: TupleElement> Vec3Trait<T> for Vec3<T> {
        type VecFloat = Vec3<pbrt::Float>;

        fn dot(self, rhs: Self) -> T {
            self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
        }

        fn absdot(self, rhs: Self) -> T
        where
            T: Signed,
        {
            self.dot(rhs).abs()
        }

        fn cross(self, rhs: Self) -> Self
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

        fn length(self) -> pbrt::Float
        where
            T: ToPrimitive,
        {
            let sqlen: pbrt::Float = NumCast::from(self.length_squared()).unwrap();
            sqlen.sqrt()
        }

        fn normalized(self) -> Vec3f
        where
            T: ToPrimitive,
        {
            let vec_f: Vec3f = self.num_cast().unwrap();
            vec_f / vec_f.length()
        }

        #[inline]
        fn angle_between(self, other: Self) -> pbrt::Float
        where
            T: ToPrimitive,
        {
            let dot: pbrt::Float = NumCast::from(self.dot(other)).unwrap();
            if dot < 0.0 {
                pbrt::PI - 2.0 * safe_asin((self + other).length() / 2.0)
            } else {
                2.0 * safe_asin((other - self).length() / 2.0)
            }
        }

        fn coordinate_system(self) -> (Vec3f, Vec3f, Vec3f)
        where
            T: ToPrimitive,
        {
            let v1: Vec3f = self.num_cast().unwrap();
            let v2 = if v1.x.abs() > v1.y.abs() {
                Vec3::new(-v1.z, 0.0, v1.x) / (v1.x * v1.x + v1.z * v1.z).sqrt()
            } else {
                Vec3::new(0.0, v1.z, -v1.y) / (v1.y * v1.y + v1.z * v1.z).sqrt()
            };
            let v3 = v1.cross(v2);

            (v1, v2, v3)
        }
    }

    impl<T: TupleElement> Tuple<3, T> for Vec3<T> {}

    impl_tuple_math_ops_generic!(Vec3; 3);

    impl<T> From<[T; 3]> for Vec3<T> {
        fn from(arr: [T; 3]) -> Self {
            let [x, y, z] = arr;
            Self::new(x, y, z)
        }
    }

    impl<T> Index<usize> for Vec3<T> {
        type Output = T;

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
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            match index {
                0 => &mut self.x,
                1 => &mut self.y,
                2 => &mut self.z,
                _ => panic!("Index out of bounds for Vec3"),
            }
        }
    }

    impl<T: AddAssign + Copy> Add for Vec3<T> {
        type Output = Self;

        fn add(mut self, rhs: Self) -> Self::Output {
            self += rhs;
            self
        }
    }

    impl<T: SubAssign + Copy> Sub for Vec3<T> {
        type Output = Self;

        fn sub(mut self, rhs: Self) -> Self::Output {
            self -= rhs;
            self
        }
    }
}

// ==================== GLAM IMPL ====================
#[cfg(feature = "glam")]
mod glam_impl {
    use super::Vec3 as Vec3Trait;
    use crate::{self as pbrt, math::tuple::Tuple};

    use delegate::delegate;

    pub type Vec3i = glam::IVec3;

    #[cfg(feature = "use-f64")]
    pub type Vec3f = glam::DVec3;
    #[cfg(not(feature = "use-f64"))]
    pub type Vec3f = glam::Vec3;

    impl Vec3Trait<i32> for Vec3i {
        type VecFloat = Vec3f;

        delegate! {
            to self {
                fn dot(self, rhs: Self) -> i32;
                fn cross(self, rhs: Self) -> Self;
            }
        }

        fn absdot(self, rhs: Self) -> i32 {
            self.dot(rhs).abs()
        }

        fn length(self) -> pbrt::Float {
            #[cfg(feature = "use-f64")]
            return self.as_dvec3().length();
            #[cfg(not(feature = "use-f64"))]
            return self.as_vec3().length();
        }

        fn normalized(self) -> Vec3f {
            #[cfg(feature = "use-f64")]
            return self.as_dvec3().normalize();
            #[cfg(not(feature = "use-f64"))]
            return self.as_vec3().normalize();
        }

        fn angle_between(self, other: Self) -> pbrt::Float {
            #[cfg(feature = "use-f64")]
            return self.as_dvec3().angle_between(other.as_dvec3());
            #[cfg(not(feature = "use-f64"))]
            return self.as_vec3().angle_between(other.as_vec3());
        }

        fn coordinate_system(self) -> (Vec3f, Vec3f, Vec3f) {
            #[cfg(feature = "use-f64")]
            let v1: Vec3f = self.as_dvec3();
            #[cfg(not(feature = "use-f64"))]
            let v1: Vec3f = self.as_vec3();

            let v2 = if v1.x.abs() > v1.y.abs() {
                Vec3f::new(-v1.z, 0.0, v1.x) / (v1.x * v1.x + v1.z * v1.z).sqrt()
            } else {
                Vec3f::new(0.0, v1.z, -v1.y) / (v1.y * v1.y + v1.z * v1.z).sqrt()
            };
            let v3 = v1.cross(v2);

            (v1, v2, v3)
        }
    }

    impl Tuple<3, i32> for Vec3i {}

    impl Vec3Trait<pbrt::Float> for Vec3f {
        type VecFloat = Vec3f;

        delegate! {
            to self {
                fn dot(self, rhs: Self) -> pbrt::Float;
                fn cross(self, rhs: Self) -> Self;
                fn length(self) -> pbrt::Float;
                #[call(normalize)]
                fn normalized(self) -> Vec3f;
                fn angle_between(self, other: Self) -> pbrt::Float;
            }
        }

        fn absdot(self, rhs: Self) -> pbrt::Float {
            self.dot(rhs).abs()
        }

        fn coordinate_system(self) -> (Vec3f, Vec3f, Vec3f) {
            let v1: Vec3f = self;
            let v2 = if v1.x.abs() > v1.y.abs() {
                Vec3f::new(-v1.z, 0.0, v1.x) / (v1.x * v1.x + v1.z * v1.z).sqrt()
            } else {
                Vec3f::new(0.0, v1.z, -v1.y) / (v1.y * v1.y + v1.z * v1.z).sqrt()
            };
            let v3 = v1.cross(v2);

            (v1, v2, v3)
        }
    }

    impl Tuple<3, pbrt::Float> for Vec3f {}
}

// ==================== VEC3FI - NO GLAM EQUIVALENT ====================

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3fi(custom_impl::Vec3<Interval>);

impl Vec3fi {
    pub const fn new(x: Interval, y: Interval, z: Interval) -> Self {
        Self(custom_impl::Vec3::new(x, y, z))
    }

    pub fn new_fi(values: Vec3f, errors: Vec3f) -> Self {
        Self::new(
            Interval::new_with_err(values.x(), errors.x()),
            Interval::new_with_err(values.y(), errors.y()),
            Interval::new_with_err(values.z(), errors.z()),
        )
    }

    pub fn new_fi_exact(x: pbrt::Float, y: pbrt::Float, z: pbrt::Float) -> Self {
        Self::new(Interval::new(x), Interval::new(y), Interval::new(z))
    }

    pub fn with_intervals(x: Interval, y: Interval, z: Interval) -> Self {
        Self::new(x, y, z)
    }

    #[inline(always)]
    pub fn x(&self) -> Interval {
        self.0.x
    }

    #[inline(always)]
    pub fn y(&self) -> Interval {
        self.0.y
    }

    #[inline(always)]
    pub fn z(&self) -> Interval {
        self.0.z
    }

    #[inline(always)]
    pub fn x_mut(&mut self) -> &mut Interval {
        &mut self.0.x
    }

    #[inline(always)]
    pub fn y_mut(&mut self) -> &mut Interval {
        &mut self.0.y
    }

    #[inline(always)]
    pub fn z_mut(&mut self) -> &mut Interval {
        &mut self.0.z
    }

    pub fn error(&self) -> Vec3f {
        Vec3f::new(
            self.x().width() / 2.0,
            self.y().width() / 2.0,
            self.z().width() / 2.0,
        )
    }

    pub fn is_exact(&self) -> bool {
        self.x().width() == 0.0 && self.y().width() == 0.0 && self.z().width() == 0.0
    }

    pub fn midpoints_only(&self) -> Vec3f {
        Vec3f::new(
            self.x().midpoint(),
            self.y().midpoint(),
            self.z().midpoint(),
        )
    }
}

impl From<Vec3f> for Vec3fi {
    fn from(v: Vec3f) -> Self {
        Self::new_fi_exact(v.x(), v.y(), v.z())
    }
}

// ==================== OTHERS ====================
// May have glam equivalent, but just using custom_impl for now
// as used ops are all simple
pub type Vec3B = custom_impl::Vec3<bool>;
pub type Vec3Usize = custom_impl::Vec3<usize>;
