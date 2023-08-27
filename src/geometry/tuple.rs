use std::ops::{Add, AddAssign, Div, DivAssign, IndexMut, Mul, MulAssign, Sub, SubAssign};

pub trait Tuple<const N: usize, T: TupleElement>:
    Copy
    + IndexMut<usize, Output = T>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<T, Output = Self>
    + Div<T, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign<T>
    + DivAssign<T>
    + PartialEq
{
    fn from_array(vals: [T; N]) -> Self;
}

pub trait TupleElement:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + PartialEq
{
}

impl<T> TupleElement for T where
    T: Copy
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + PartialEq
{
}

#[macro_export]
macro_rules! impl_tuple_math_ops {
    ($name:ty; $n:expr; $t:ty) => {
        impl std::ops::Add for $name {
            type Output = Self;

            #[inline]
            fn add(mut self, rhs: Self) -> Self {
                self += rhs;
                self
            }
        }

        impl std::ops::Sub for $name {
            type Output = Self;

            #[inline]
            fn sub(mut self, rhs: Self) -> Self {
                self -= rhs;
                self
            }
        }

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
    };
}

#[macro_export]
macro_rules! impl_tuple_math_ops_generic {
    ($name:ident; $n:expr) => {
        impl<T: std::ops::Add<Output = T> + Copy> std::ops::Add for $name<T> {
            type Output = Self;

            #[inline]
            fn add(mut self, rhs: Self) -> Self {
                for i in 0..$n {
                    self[i] = self[i] + rhs[i];
                }
                self
            }
        }

        impl<T: std::ops::Sub<Output = T> + Copy> std::ops::Sub for $name<T> {
            type Output = Self;

            #[inline]
            fn sub(mut self, rhs: Self) -> Self {
                for i in 0..$n {
                    self[i] = self[i] - rhs[i];
                }
                self
            }
        }

        impl<T: std::ops::Mul<Output = T> + Copy> std::ops::Mul<T> for $name<T> {
            type Output = Self;

            #[inline]
            fn mul(mut self, rhs: T) -> Self {
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

        impl std::ops::Mul<$name<crate::Float>> for crate::Float {
            type Output = $name<crate::Float>;

            #[inline]
            fn mul(self, rhs: $name<crate::Float>) -> $name<crate::Float> {
                rhs * self
            }
        }

        impl<T: std::ops::Div<Output = T> + Copy> std::ops::Div<T> for $name<T> {
            type Output = Self;

            #[inline]
            fn div(mut self, rhs: T) -> Self {
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
    };
}
