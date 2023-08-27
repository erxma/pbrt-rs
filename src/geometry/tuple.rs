use std::ops::{Add, AddAssign, Div, DivAssign, IndexMut, Mul, MulAssign, Sub, SubAssign};

pub trait Tuple<const N: usize, T>:
    Copy
    + IndexMut<usize, Output = T>
    + Add
    + Sub
    + Mul<T, Output = Self>
    + Div<T, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign<T>
    + DivAssign<T>
{
    fn from_array(vals: [T; N]) -> Self;
}

#[macro_export]
macro_rules! impl_tuple_math_ops {
    ($name:ty; $n:expr; $t:ty) => {
        impl Add for $name {
            type Output = Self;

            #[inline]
            fn add(mut self, rhs: Self) -> Self {
                self += rhs;
                self
            }
        }

        impl Sub for $name {
            type Output = Self;

            #[inline]
            fn sub(mut self, rhs: Self) -> Self {
                self -= rhs;
                self
            }
        }

        impl Mul<$t> for $name {
            type Output = Self;

            #[inline]
            fn mul(mut self, rhs: $t) -> Self {
                self *= rhs;
                self
            }
        }

        impl Div<$t> for $name {
            type Output = Self;

            #[inline]
            fn div(mut self, rhs: $t) -> Self {
                self /= rhs;
                self
            }
        }

        impl AddAssign for $name {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                for i in 0..$n {
                    self[i] += rhs[i];
                }
            }
        }

        impl SubAssign for $name {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                for i in 0..$n {
                    self[i] -= rhs[i];
                }
            }
        }

        impl MulAssign<$t> for $name {
            #[inline]
            fn mul_assign(&mut self, rhs: $t) {
                for i in 0..$n {
                    self[i] *= rhs;
                }
            }
        }

        impl DivAssign<$t> for $name {
            #[inline]
            fn div_assign(&mut self, rhs: $t) {
                for i in 0..$n {
                    self[i] /= rhs;
                }
            }
        }
    };
}
