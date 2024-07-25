mod color_space;
mod colors;

pub use color_space::{RGBColorSpace, RGBSigmoidPolynomial, RGBToSpectrumTable};
pub use colors::{RGB, XYZ};

use crate::math::{point::Point2f, square_matrix::SquareMatrix};

#[inline]
pub fn white_balance(_src_white: Point2f, _target_white: Point2f) -> SquareMatrix<3> {
    todo!()
}
