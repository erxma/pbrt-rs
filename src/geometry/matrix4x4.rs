use std::ops::Mul;

use crate::Float;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Matrix4x4 {
    pub m: [[Float; 4]; 4],
}

impl Matrix4x4 {
    pub const IDENTITY: Self = Self {
        m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    /// Construct a new matrix with the given values.
    pub fn new(mat: [[Float; 4]; 4]) -> Self {
        Self { m: mat }
    }

    /// Compute the transposed matrix of `self`.
    pub fn transpose(&self) -> Self {
        Self {
            m: [
                [self.m[0][0], self.m[1][0], self.m[2][0], self.m[3][0]],
                [self.m[0][1], self.m[1][1], self.m[2][1], self.m[3][1]],
                [self.m[0][2], self.m[1][2], self.m[2][2], self.m[3][2]],
                [self.m[0][3], self.m[1][3], self.m[2][3], self.m[3][3]],
            ],
        }
    }

    /// Compute the inverse matrix of `self`.
    #[allow(clippy::needless_range_loop)]
    pub fn inverse(&self) -> Result<Self, &str> {
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
                            return Err("Singular matrix in MatrixInvert");
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
                return Err("Singular matrix in MatrixInvert");
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

        Ok(Self::new(minv))
    }
}

impl Mul for Matrix4x4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let mut result = Self::default();
        for i in 0..4 {
            for j in 0..4 {
                result.m[i][j] = self.m[i][0] * rhs.m[0][j]
                    + self.m[i][1] * rhs.m[1][j]
                    + self.m[i][2] * rhs.m[2][j]
                    + self.m[i][3] * rhs.m[3][j];
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix4x4;

    use crate::Float;
    use approx::{assert_relative_eq, AbsDiffEq, RelativeEq};

    impl AbsDiffEq for Matrix4x4 {
        type Epsilon = <Float as AbsDiffEq>::Epsilon;

        fn default_epsilon() -> Self::Epsilon {
            Float::default_epsilon()
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            for i in 0..4 {
                for j in 0..4 {
                    if Float::abs_diff_ne(&self.m[i][j], &other.m[i][j], epsilon) {
                        return false;
                    }
                }
            }

            true
        }
    }

    impl RelativeEq for Matrix4x4 {
        fn default_max_relative() -> Self::Epsilon {
            Float::default_max_relative()
        }

        fn relative_eq(
            &self,
            other: &Self,
            epsilon: Self::Epsilon,
            max_relative: Self::Epsilon,
        ) -> bool {
            for i in 0..4 {
                for j in 0..4 {
                    if Float::relative_ne(&self.m[i][j], &other.m[i][j], epsilon, max_relative) {
                        return false;
                    }
                }
            }

            true
        }
    }

    #[test]
    fn inverse() {
        let mat = Matrix4x4::new([
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 3.0, 1.0, 2.0],
            [2.0, 3.0, 1.0, 0.0],
            [1.0, 0.0, 2.0, 1.0],
        ]);

        let inv = mat.inverse().unwrap();
        let inv_expected = Matrix4x4::new([
            [-3.0, -0.5, 1.5, 1.0],
            [1.0, 0.25, -0.25, -0.5],
            [3.0, 0.25, -1.25, -0.5],
            [-3.0, 0.0, 1.0, 1.0],
        ]);

        assert_relative_eq!(inv, inv_expected);
    }

    #[test]
    fn inverse_singular_err() {
        let mat = Matrix4x4::default();
        let result = mat.inverse();

        assert!(result.is_err());
    }
}
