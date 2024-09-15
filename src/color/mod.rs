mod color_space;
mod colors;

pub use color_space::{
    CoefficientTable, RGBColorSpace, RGBSigmoidPolynomial, RGBToSpectrumTable, SRGB,
};
pub use colors::{RGB, XYZ};

use crate::core::{Point2f, SquareMatrix};

#[inline]
pub fn white_balance(_src_white: Point2f, _target_white: Point2f) -> SquareMatrix<3> {
    todo!()
}
