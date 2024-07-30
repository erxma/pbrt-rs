use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign};

use inherent::inherent;
use itertools::iproduct;

use crate::Float;

use super::{routines::difference_of_products, Tuple};

#[derive(Clone, Debug, PartialEq)]
pub struct SquareMatrix<const N: usize> {
    m: [[Float; N]; N],
}

impl<const N: usize> SquareMatrix<N> {
    pub const ZERO: Self = Self { m: [[0.0; N]; N] };
    pub const IDENTITY: Self = Self::identity();

    /// Construct a new matrix with the given values.
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

    /// Compute the transposed matrix of `self`.
    #[inline]
    pub fn transpose(&self) -> Self {
        let mut ret = self.clone();

        for (i, j) in iproduct!(0..N, 0..N) {
            ret[i][j] = self.m[j][i];
        }

        ret
    }

    #[inline]
    pub fn mul<R>(&self, rhs: &impl Tuple<N, Float>) -> R
    where
        R: Tuple<N, Float>,
    {
        let mut res = R::from([0.0; N]);

        for i in 0..N {
            for j in 0..N {
                res[i] += self[i][j] * rhs[j];
            }
        }

        res
    }

    #[inline]
    pub fn linear_least_squares<const ROWS: usize>(
        a: &[[Float; N]; ROWS],
        b: &[[Float; N]; ROWS],
    ) -> Option<Self>
    where
        Self: Invert,
    {
        let mut ata = Self::ZERO;
        let mut atb = Self::ZERO;

        for (i, j, r) in iproduct!(0..N, 0..N, 0..ROWS) {
            ata[i][j] += a[r][i] * a[r][j];
            atb[i][j] += a[r][i] * b[r][j];
        }

        let atai = ata.inverse()?;

        Some((atai * atb).transpose())
    }
}

pub trait Invert: Sized {
    /// Compute the inverse matrix of `self`.
    fn inverse(&self) -> Option<Self>;
}

impl SquareMatrix<1> {
    #[inline]
    pub fn determinant(&self) -> Float {
        self.m[0][0]
    }
}

impl SquareMatrix<2> {
    #[inline]
    pub fn determinant(&self) -> Float {
        difference_of_products(self.m[0][0], self.m[1][1], self.m[0][1], self.m[1][0])
    }
}

impl SquareMatrix<3> {
    #[inline]
    pub fn determinant(&self) -> Float {
        let minor12 =
            difference_of_products(self.m[1][1], self.m[2][2], self.m[1][2], self.m[2][1]);
        let minor02 =
            difference_of_products(self.m[1][0], self.m[2][2], self.m[1][2], self.m[2][0]);
        let minor01 =
            difference_of_products(self.m[1][0], self.m[2][1], self.m[1][1], self.m[2][0]);

        self.m[0][2].mul_add(
            minor01,
            difference_of_products(self.m[0][0], minor12, self.m[0][1], minor02),
        )
    }
}

#[inherent]
impl Invert for SquareMatrix<3> {
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det == 0.0 {
            return None;
        }
        let inv_det = 1.0 / det;

        let m = &self.m;
        let mut ret = *m;

        ret[0][0] = inv_det * difference_of_products(m[1][1], m[2][2], m[1][2], m[2][1]);
        ret[1][0] = inv_det * difference_of_products(m[1][2], m[2][0], m[1][0], m[2][2]);
        ret[2][0] = inv_det * difference_of_products(m[1][0], m[2][1], m[1][1], m[2][0]);
        ret[0][1] = inv_det * difference_of_products(m[0][2], m[2][1], m[0][1], m[2][2]);
        ret[1][1] = inv_det * difference_of_products(m[0][0], m[2][2], m[0][2], m[2][0]);
        ret[2][1] = inv_det * difference_of_products(m[0][1], m[2][0], m[0][0], m[2][1]);
        ret[0][2] = inv_det * difference_of_products(m[0][1], m[1][2], m[0][2], m[1][1]);
        ret[1][2] = inv_det * difference_of_products(m[0][2], m[1][0], m[0][0], m[1][2]);
        ret[2][2] = inv_det * difference_of_products(m[0][0], m[1][1], m[0][1], m[1][0]);

        Some(Self::new(ret))
    }
}

impl SquareMatrix<4> {
    #[inline]
    // TODO: Currently not providing version for N > 4 because of complexity of constraining.
    // But it isn't expected to be used anyway
    pub fn determinant(&self) -> Float {
        let m = &self.m;
        let s0 = difference_of_products(m[0][0], m[1][1], m[1][0], m[0][1]);
        let s1 = difference_of_products(m[0][0], m[1][2], m[1][0], m[0][2]);
        let s2 = difference_of_products(m[0][0], m[1][3], m[1][0], m[0][3]);

        let s3 = difference_of_products(m[0][1], m[1][2], m[1][1], m[0][2]);
        let s4 = difference_of_products(m[0][1], m[1][3], m[1][1], m[0][3]);
        let s5 = difference_of_products(m[0][2], m[1][3], m[1][2], m[0][3]);

        let c0 = difference_of_products(m[2][0], m[3][1], m[3][0], m[2][1]);
        let c1 = difference_of_products(m[2][0], m[3][2], m[3][0], m[2][2]);
        let c2 = difference_of_products(m[2][0], m[3][3], m[3][0], m[2][3]);

        let c3 = difference_of_products(m[2][1], m[3][2], m[3][1], m[2][2]);
        let c4 = difference_of_products(m[2][1], m[3][3], m[3][1], m[2][3]);
        let c5 = difference_of_products(m[2][2], m[3][3], m[3][2], m[2][3]);

        difference_of_products(s0, c5, s1, c4)
            + difference_of_products(s2, c3, -s3, c2)
            + difference_of_products(s5, c0, s4, c1)
    }
}

#[inherent]
impl Invert for SquareMatrix<4> {
    // FIXME: Better impl available in book, need to impl inner_product first
    #[allow(clippy::needless_range_loop)]
    pub fn inverse(&self) -> Option<Self> {
        let mut indxc = [0; 4];
        let mut indxr = [0; 4];
        let mut ipiv = [0; 4];
        let mut minv = self.m;
        for i in 0..4 {
            let mut irow = 0;
            let mut icol = 0;
            let mut big: Float = 0.0;
            // Choose pivot
            for j in 0..4 {
                if ipiv[j] != 1 {
                    for k in 0..4 {
                        if ipiv[k] == 0 {
                            if minv[j][k].abs() >= big {
                                big = minv[j][k].abs();
                                irow = j;
                                icol = k;
                            }
                        } else if ipiv[k] > 1 {
                            return None;
                        }
                    }
                }
            }
            ipiv[icol] += 1;
            // Swap rows _irow_ and _icol_ for pivot
            if irow != icol {
                for k in 0..4 {
                    // Swap
                    (minv[irow][k], minv[icol][k]) = (minv[icol][k], minv[irow][k])
                }
            }
            indxr[i] = irow;
            indxc[i] = icol;
            if minv[icol][icol] == 0.0 {
                return None;
            }

            // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
            let pivinv = 1. / minv[icol][icol];
            minv[icol][icol] = 1.;
            for j in 0..4 {
                minv[icol][j] *= pivinv;
            }

            // Subtract this row from others to zero out their columns
            for j in 0..4 {
                if j != icol {
                    let save = minv[j][icol];
                    minv[j][icol] = 0.0;
                    for k in 0..4 {
                        minv[j][k] -= minv[icol][k] * save;
                    }
                }
            }
        }
        // Swap columns to reflect permutation
        for j in (0..=3).rev() {
            if indxr[j] != indxc[j] {
                for k in 0..4 {
                    // Swap
                    (minv[k][indxr[j]], minv[k][indxc[j]]) = (minv[k][indxc[j]], minv[k][indxr[j]]);
                }
            }
        }

        Some(Self::new(minv))
    }
}

impl<const N: usize> Default for SquareMatrix<N> {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<const N: usize> Add for SquareMatrix<N> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
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

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::Float;
    use approx::{assert_relative_eq, AbsDiffEq, RelativeEq};

    impl<const N: usize> AbsDiffEq for SquareMatrix<N> {
        type Epsilon = <Float as AbsDiffEq>::Epsilon;

        fn default_epsilon() -> Self::Epsilon {
            Float::default_epsilon()
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            iproduct!(0..N, 0..N)
                .all(|(i, j)| Float::abs_diff_eq(&self.m[i][j], &other.m[i][j], epsilon))
        }
    }

    impl<const N: usize> RelativeEq for SquareMatrix<N> {
        fn default_max_relative() -> Self::Epsilon {
            Float::default_max_relative()
        }

        fn relative_eq(
            &self,
            other: &Self,
            epsilon: Self::Epsilon,
            max_relative: Self::Epsilon,
        ) -> bool {
            iproduct!(0..N, 0..N).all(|(i, j)| {
                Float::relative_eq(&self.m[i][j], &other.m[i][j], epsilon, max_relative)
            })
        }
    }

    #[test]
    fn inverse() {
        let mat = SquareMatrix::new([
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 3.0, 1.0, 2.0],
            [2.0, 3.0, 1.0, 0.0],
            [1.0, 0.0, 2.0, 1.0],
        ]);

        let inv = mat.inverse().unwrap();
        let inv_expected = SquareMatrix::new([
            [-3.0, -0.5, 1.5, 1.0],
            [1.0, 0.25, -0.25, -0.5],
            [3.0, 0.25, -1.25, -0.5],
            [-3.0, 0.0, 1.0, 1.0],
        ]);

        assert_relative_eq!(inv, inv_expected);
    }

    #[test]
    fn inverse_singular_none() {
        let mat = SquareMatrix::<4>::ZERO;
        let result = mat.inverse();

        dbg!(result.clone());
        assert!(result.is_none());
    }
}
