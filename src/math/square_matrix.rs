use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign};

use crate::{geometry::tuple::Tuple, Float};

#[derive(Clone, Debug, PartialEq)]
pub struct SquareMatrix<const N: usize> {
    m: [[Float; N]; N],
}

impl<const N: usize> SquareMatrix<N> {
    pub const ZERO: Self = Self { m: [[0.0; N]; N] };
    pub const IDENTITY: Self = Self::identity();

    pub const fn new(values: [[Float; N]; N]) -> Self {
        Self { m: values }
    }

    pub const fn diagonal(values: [Float; N]) -> Self {
        let mut ret = Self::ZERO;

        let mut i = 0;
        while i < N {
            ret.m[i][i] = values[i];
            i += 1;
        }

        ret
    }

    const fn identity() -> Self {
        let mut m = Self::ZERO;
        let mut i = 0;
        while i < N {
            m.m[i][i] = 1.0;
            i += 1;
        }

        m
    }

    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    #[inline]
    pub fn transpose(&self) -> Self {
        todo!()
    }

    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        todo!()
    }

    #[inline]
    pub fn determinant(&self) -> Float {
        todo!()
    }

    #[inline]
    pub fn mul<R>(&self, rhs: &impl Tuple<N, Float>) -> R
    where
        R: Tuple<N, Float>,
    {
        let mut res = R::from_array([0.0; N]);

        for i in 0..N {
            for j in 0..N {
                res[i] += self[i][j] * rhs[j];
            }
        }

        res
    }
}

impl<const N: usize> Default for SquareMatrix<N> {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<const N: usize> Add for SquareMatrix<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = self;
        ret += rhs;
        ret
    }
}

impl<const N: usize> AddAssign for SquareMatrix<N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            for j in 0..N {
                self.m[i][j] += rhs.m[i][j];
            }
        }
    }
}

impl<const N: usize> Sub for SquareMatrix<N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = self;
        ret -= rhs;
        ret
    }
}

impl<const N: usize> SubAssign for SquareMatrix<N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            for j in 0..N {
                self.m[i][j] -= rhs.m[i][j];
            }
        }
    }
}

impl<const N: usize> Mul for SquareMatrix<N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl<const N: usize> Mul for &SquareMatrix<N> {
    type Output = SquareMatrix<N>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut m = [[0.0; N]; N];

        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    m[i][j] = self[i][k].mul_add(rhs[k][j], m[i][j]);
                }
            }
        }

        SquareMatrix { m }
    }
}

impl<const N: usize> Index<usize> for SquareMatrix<N> {
    type Output = [Float; N];

    fn index(&self, index: usize) -> &Self::Output {
        &self.m[index]
    }
}

impl<const N: usize> IndexMut<usize> for SquareMatrix<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.m[index]
    }
}
