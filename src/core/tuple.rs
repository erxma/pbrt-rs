use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use num_traits::{Float, Num, NumAssign, Signed};

pub trait Tuple<const N: usize, T: TupleElement>:
    Copy
    + PartialEq
    + IndexMut<usize, Output = T>
    + Add
    + Sub
    + Mul<T, Output = Self>
    + Div<T, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign<T>
    + DivAssign<T>
    + From<[T; N]>
    + Into<[T; N]>
{
    /// Returns a `Self` with the absolute values of the components.
    fn abs(mut self) -> Self
    where
        T: Signed,
    {
        for i in 0..N {
            self[i] = self[i].abs();
        }
        self
    }

    /// Permute `self`'s elements according to the index values given.
    fn permute(self, indices: [usize; N]) -> Self {
        let mut ret = self;
        for i in 0..N {
            ret[i] = self[indices[i]];
        }
        ret
    }

    /// Returns true if any component is NaN.
    fn has_nan(self) -> bool
    where
        T: Float,
    {
        (0..N).any(|i| self[i].is_nan())
    }

    /// Returns the element with the smallest value.
    fn min_component(self) -> T {
        (0..N)
            .map(|i| self[i])
            .min_by(|x, y| {
                x.partial_cmp(y)
                    .expect("All tuple values need to be comparable - is there a NaN?")
            })
            .unwrap()
    }

    /// Returns the element with the largest value.
    fn max_component(self) -> T
    where
        T: Float,
    {
        (0..N)
            .map(|i| self[i])
            .max_by(|x, y| {
                x.partial_cmp(y)
                    .expect("All tuple values need to be comparable - is there a NaN?")
            })
            .unwrap()
    }

    /// Returns the index of the component with the max value.
    fn max_dimension(self) -> usize {
        (0..N)
            .min_by(|&i1, &i2| {
                self[i1]
                    .partial_cmp(&self[i2])
                    .expect("All tuple values need to be comparable - is there a NaN?")
            })
            .unwrap()
    }

    /// Returns the index of the component with the min value.
    fn min_dimension(self) -> usize {
        (0..N)
            .max_by(|&i1, &i2| {
                self[i1]
                    .partial_cmp(&self[i2])
                    .expect("All tuple values need to be comparable - is there a NaN?")
            })
            .unwrap()
    }

    /// Returns a `Self` containing the min values for each
    /// component of `self` and `other` (the component-wise min).
    fn min(mut self, other: Self) -> Self {
        for i in 0..N {
            self[i] = match self[i].partial_cmp(&other[i]) {
                Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal) => self[i],
                Some(std::cmp::Ordering::Greater) => other[i],
                None => panic!("All tuple values need to be comparable - is there a NaN?"),
            }
        }
        self
    }

    /// Returns a `Self` containing the max values for each
    /// component of `self` and `other` (the component-wise max).
    fn max(mut self, other: Self) -> Self
    where
        T: PartialOrd,
    {
        for i in 0..N {
            self[i] = match self[i].partial_cmp(&other[i]) {
                Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal) => self[i],
                Some(std::cmp::Ordering::Less) => other[i],
                None => panic!("All tuple values need to be comparable - is there a NaN?"),
            }
        }
        self
    }

    /// Returns `self` with `ceil` applied component-wise.
    fn ceil(mut self) -> Self
    where
        T: Float,
    {
        for i in 0..N {
            self[i] = self[i].ceil();
        }
        self
    }

    /// Returns `self` with `floor` applied component-wise.
    fn floor(mut self) -> Self
    where
        T: Float,
    {
        for i in 0..N {
            self[i] = self[i].floor();
        }
        self
    }
}

pub trait TupleElement: Num + NumAssign + PartialOrd + Copy + Debug {}

impl<T> TupleElement for T where T: Num + NumAssign + PartialOrd + Copy + Debug {}

macro_rules! impl_tuple_math_ops {
    ($name:ty; $n:expr; $t:ty) => {
        impl std::ops::Mul<$t> for $name {
            type Output = Self;

            #[inline]
            fn mul(mut self, rhs: $t) -> Self {
                self *= rhs;
                self
            }
        }

        impl std::ops::Mul<$name> for $t {
            type Output = $name;

            #[inline]
            fn mul(self, rhs: $name) -> $name {
                rhs * self
            }
        }

        impl std::ops::Div<$t> for $name {
            type Output = Self;

            #[inline]
            fn div(mut self, rhs: $t) -> Self {
                self /= rhs;
                self
            }
        }

        impl std::ops::AddAssign for $name {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                for i in 0..$n {
                    self[i] += rhs[i];
                }
            }
        }

        impl std::ops::SubAssign for $name {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                for i in 0..$n {
                    self[i] -= rhs[i];
                }
            }
        }

        impl std::ops::MulAssign<$t> for $name {
            #[inline]
            fn mul_assign(&mut self, rhs: $t) {
                for i in 0..$n {
                    self[i] *= rhs;
                }
            }
        }

        impl std::ops::DivAssign<$t> for $name {
            #[inline]
            fn div_assign(&mut self, rhs: $t) {
                for i in 0..$n {
                    self[i] /= rhs;
                }
            }
        }

        impl From<$name> for [$t; $n] {
            #[inline]
            fn from(tuple: $name) -> Self {
                std::array::from_fn(|i| tuple[i])
            }
        }
    };
}
pub(crate) use impl_tuple_math_ops;

macro_rules! impl_tuple_math_ops_generic {
    ($name:ident; $n:expr) => {
        impl<T> $name<T> {
            /// Convert vector elements into another type.
            pub fn into_<U>(self) -> $name<U>
            where
                T: Into<U> + Copy,
            {
                let vals = std::array::from_fn(|i| self[i].into());
                $name::from(vals)
            }

            pub fn num_cast<U>(self) -> Option<$name<U>>
            where
                T: num_traits::ToPrimitive + Copy,
                U: NumCast + Default,
            {
                let mut vals: [U; $n] = Default::default();
                for i in 0..$n {
                    let casted = NumCast::from(self[i])?;
                    vals[i] = casted;
                }
                Some($name::from(vals))
            }
        }

        impl<Rhs, T> std::ops::Mul<Rhs> for $name<T>
        where
            T: std::ops::Mul<Rhs, Output = T> + Copy,
            Rhs: Copy,
        {
            type Output = Self;

            #[inline]
            fn mul(mut self, rhs: Rhs) -> Self {
                for i in 0..$n {
                    self[i] = self[i] * rhs;
                }
                self
            }
        }

        impl std::ops::Mul<$name<i32>> for i32 {
            type Output = $name<i32>;

            #[inline]
            fn mul(self, rhs: $name<i32>) -> $name<i32> {
                rhs * self
            }
        }

        impl std::ops::Mul<$name<$crate::core::Float>> for $crate::core::Float {
            type Output = $name<$crate::core::Float>;

            #[inline]
            fn mul(self, rhs: $name<$crate::core::Float>) -> $name<$crate::core::Float> {
                rhs * self
            }
        }

        impl std::ops::Mul<$name<$crate::core::Interval>> for $crate::core::Interval {
            type Output = $name<$crate::core::Interval>;

            #[inline]
            fn mul(self, rhs: $name<$crate::core::Interval>) -> $name<$crate::core::Interval> {
                rhs * self
            }
        }

        impl<Rhs, T> std::ops::Div<Rhs> for $name<T>
        where
            T: std::ops::Div<Rhs, Output = T> + Copy,
            Rhs: Copy,
        {
            type Output = Self;

            #[inline]
            fn div(mut self, rhs: Rhs) -> Self {
                for i in 0..$n {
                    self[i] = self[i] / rhs;
                }
                self
            }
        }

        impl<T: std::ops::AddAssign + Copy> std::ops::AddAssign for $name<T> {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                for i in 0..$n {
                    self[i] += rhs[i];
                }
            }
        }

        impl<T: std::ops::SubAssign + Copy> std::ops::SubAssign for $name<T> {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                for i in 0..$n {
                    self[i] -= rhs[i];
                }
            }
        }

        impl<T: std::ops::MulAssign + Copy> std::ops::MulAssign<T> for $name<T> {
            #[inline]
            fn mul_assign(&mut self, rhs: T) {
                for i in 0..$n {
                    self[i] *= rhs;
                }
            }
        }

        impl<T: std::ops::DivAssign + Copy> std::ops::DivAssign<T> for $name<T> {
            #[inline]
            fn div_assign(&mut self, rhs: T) {
                for i in 0..$n {
                    self[i] /= rhs;
                }
            }
        }

        impl<T: std::ops::Neg<Output = T> + Copy> std::ops::Neg for $name<T> {
            type Output = Self;

            #[inline]
            fn neg(mut self) -> Self {
                for i in 0..$n {
                    self[i] = -self[i];
                }
                self
            }
        }

        impl<T: Copy> From<$name<T>> for [T; $n] {
            #[inline]
            fn from(tuple: $name<T>) -> Self {
                std::array::from_fn(|i| tuple[i])
            }
        }
    };
}
pub(crate) use impl_tuple_math_ops_generic;
