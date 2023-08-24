use crate::{geometry::point2::Point2f, math::square_matrix::SquareMatrix};

use super::{
    color::{RGB, XYZ},
    spectrum::{spectrum_to_xyz, DenselySampledSpectrum, Spectrum},
};

#[derive(Clone, Debug)]
pub struct RGBColorSpace<'a> {
    pub r: Point2f,
    pub g: Point2f,
    pub b: Point2f,
    pub w: Point2f,
    pub illuminant: DenselySampledSpectrum,
    pub xyz_from_rgb: SquareMatrix<3>,
    pub rgb_from_xyz: SquareMatrix<3>,

    rgb_to_spectrum_table: &'a RGBToSpectrumTable,
}

impl<'a> RGBColorSpace<'a> {
    #[allow(non_snake_case)]
    pub fn new(
        r: Point2f,
        g: Point2f,
        b: Point2f,
        illuminant: impl Spectrum,
        rgb_to_spectrum_table: &'a RGBToSpectrumTable,
    ) -> Self {
        // Compute whitepoint primaries and XYZ coordinates
        let W = spectrum_to_xyz(&illuminant);
        let w = W.xy();
        let R = XYZ::from_xyy(r, None);
        let G = XYZ::from_xyy(g, None);
        let B = XYZ::from_xyy(b, None);

        // Construct XYZ color space conversion matrices
        let rgb = SquareMatrix::new([[R.x, G.x, B.x], [R.y, G.y, B.y], [R.z, G.z, B.z]]);
        let rgb_inverse = rgb
            .inverse()
            .expect("RGB matrix for RGBColorSpace is expected to be invertible. Are the given RGB values invalid?");
        let C: XYZ = rgb_inverse.mul(&W);

        let xyz_from_rgb = rgb * SquareMatrix::diagonal([C.x, C.y, C.z]);
        let rgb_from_xyz = xyz_from_rgb.inverse().unwrap();

        Self {
            r,
            g,
            b,
            w,
            illuminant: DenselySampledSpectrum::new(&illuminant, None, None),
            xyz_from_rgb,
            rgb_from_xyz,
            rgb_to_spectrum_table,
        }
    }

    /// Convert an XYZ triplet into `self`'s color space.
    pub fn to_rgb(&self, xyz: XYZ) -> RGB {
        self.rgb_from_xyz.mul(&xyz)
    }

    /// Convert an RGB triplet in `self`'s color space to XYZ.
    pub fn to_xyz(&self, rgb: RGB) -> XYZ {
        self.xyz_from_rgb.mul(&rgb)
    }

    /// Compute the matrix to convert from `self`'s color space to the `target` space.
    pub fn conversion_matrix(&self, target: &Self) -> SquareMatrix<3> {
        if std::ptr::eq(self, target) {
            return SquareMatrix::IDENTITY;
        }

        &target.rgb_from_xyz * &self.xyz_from_rgb
    }
}

#[derive(Debug)]
pub struct RGBToSpectrumTable {}
