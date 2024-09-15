use std::{
    cmp::min,
    ops::{Index, IndexMut},
    sync::LazyLock,
};

use crate::{
    core::{evaluate_polynomial, find_interval, lerp, Float, Point2f, SquareMatrix},
    sampling::spectrum::{DenselySampledSpectrum, Spectrum, ILLUMD65},
    util::data::SRGB_TABLE,
};

use super::{RGB, XYZ};

/// Represents an RGB color space, defining how RGB values
/// map to physical color representations.
#[derive(Clone, Debug)]
pub struct RGBColorSpace {
    /// xyY chromaticity coordinates of the red primary.
    pub r: Point2f,
    /// xyY chromaticity coordinates of the green primary.
    pub g: Point2f,
    /// xyY chromaticity coordinates of the blue primary.
    pub b: Point2f,
    /// xyY chromaticity coordinates of the illuminant whitepoint.
    pub w: Point2f,
    /// Spectral power distribution of the whitepoint, i.e. the light source
    /// color defining the whitepoint.
    pub illuminant: DenselySampledSpectrum,
    /// Matrix transforming from RGB color space to CIE XYZ color space.
    pub xyz_from_rgb: SquareMatrix<3>,
    /// Matrix transforming from CIE XYZ color space to RGB color space.
    pub rgb_from_xyz: SquareMatrix<3>,

    /// Table mapping RGB values to full spectral distributions.
    rgb_to_spectrum_table: &'static RGBToSpectrumTable,
}

pub static SRGB: LazyLock<RGBColorSpace> = LazyLock::new(|| {
    RGBColorSpace::new(
        Point2f::new(0.64, 0.33),
        Point2f::new(0.3, 0.6),
        Point2f::new(0.15, 0.06),
        &*ILLUMD65,
        &SRGB_TABLE,
    )
});

impl RGBColorSpace {
    /// Construct a new color space with the given `x`, `y`, `z` whitepoint chromaticities,
    /// `illuminant` spectrum (treated as the spectral distribution for RGB white),
    /// and table for converting RGB values to spectral distributions.
    #[allow(non_snake_case)]
    pub fn new(
        r: Point2f,
        g: Point2f,
        b: Point2f,
        illuminant: &impl Spectrum,
        rgb_to_spectrum_table: &'static RGBToSpectrumTable,
    ) -> Self {
        // Compute whitepoint primaries and XYZ coordinates
        let W = illuminant.to_xyz();
        let w = W.xy();
        let R = XYZ::from_xyy(r, None);
        let G = XYZ::from_xyy(g, None);
        let B = XYZ::from_xyy(b, None);

        // Construct XYZ color space conversion matrices
        let rgb = SquareMatrix::new([[R.x, G.x, B.x], [R.y, G.y, B.y], [R.z, G.z, B.z]]);
        let rgb_inverse = rgb
            .inverse()
            .expect("RGB matrix for RGBColorSpace is expected to be invertible. Are the given RGB values invalid?");
        let C: XYZ = rgb_inverse * W;

        let xyz_from_rgb = rgb * SquareMatrix::diagonal([C.x, C.y, C.z]);
        let rgb_from_xyz = xyz_from_rgb.inverse().unwrap();

        Self {
            r,
            g,
            b,
            w,
            illuminant: DenselySampledSpectrum::new(illuminant, None, None),
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

    pub fn to_rgb_coeffs(&self, rgb: RGB) -> RGBSigmoidPolynomial {
        self.rgb_to_spectrum_table.convert(rgb.clamp_zero())
    }
}

#[derive(Clone, Debug)]
pub struct RGBToSpectrumTable {
    z_nodes: [Float; Self::RESOLUTION],
    coeffs: CoefficientTable,
}

impl RGBToSpectrumTable {
    pub const RESOLUTION: usize = 64;

    pub fn new(z_nodes: [Float; Self::RESOLUTION], coeffs: CoefficientTable) -> Self {
        Self { z_nodes, coeffs }
    }

    pub fn convert(&self, rgb: RGB) -> RGBSigmoidPolynomial {
        // Handle uniform RGB values
        if rgb[0] == rgb[1] && rgb[1] == rgb[2] {
            return RGBSigmoidPolynomial::new(
                0.0,
                0.0,
                RGBSigmoidPolynomial::inverse_sigmoid(rgb[0]),
            );
        }

        // Find max component and compute remapped component values
        let max_i = if rgb[0] > rgb[1] {
            if rgb[0] > rgb[2] {
                0
            } else {
                2
            }
        } else if rgb[1] > rgb[2] {
            1
        } else {
            2
        };

        let z = rgb[max_i];
        let x = rgb[(max_i + 1) % 3] * (Self::RESOLUTION - 1) as Float / z;
        let y = rgb[(max_i + 2) % 3] * (Self::RESOLUTION - 1) as Float / z;

        // Compute integer indices and offsets for coefficient interpolation
        let xi = min(x as usize, Self::RESOLUTION - 2);
        let yi = min(y as usize, Self::RESOLUTION - 2);
        let zi = find_interval(Self::RESOLUTION, |i| self.z_nodes[i] < z).unwrap();

        let dx = x - xi as Float;
        let dy = y - yi as Float;
        let dz = (z - self.z_nodes[zi]) / (self.z_nodes[zi + 1] - self.z_nodes[zi]);

        // Trilinearly interpolate sigmoid polynomial efficients c
        let c = |c_i| {
            let co = |dx, dy, dz| self.coeffs[(max_i, zi + dz, yi + dy, xi + dx, c_i)];
            lerp(
                lerp(
                    lerp(co(0, 0, 0), co(1, 0, 0), dx),
                    lerp(co(0, 1, 0), co(1, 1, 0), dx),
                    dy,
                ),
                lerp(
                    lerp(co(0, 0, 1), co(1, 0, 1), dx),
                    lerp(co(0, 1, 1), co(1, 1, 1), dx),
                    dy,
                ),
                dz,
            )
        };

        RGBSigmoidPolynomial::new(c(0), c(1), c(2))
    }
}

#[derive(Clone, Debug)]
pub struct CoefficientTable {
    vals: Vec<Float>,
}

impl CoefficientTable {
    const RESOLUTION: usize = RGBToSpectrumTable::RESOLUTION;

    pub fn new(vals: Vec<Float>) -> Self {
        assert_eq!(vals.len(), 3 * Self::RESOLUTION.pow(3) * 3);
        Self { vals }
    }
}

impl Index<(usize, usize, usize, usize, usize)> for CoefficientTable {
    type Output = Float;

    fn index(&self, index: (usize, usize, usize, usize, usize)) -> &Self::Output {
        let (rgb_max_i, z, y, x, c_i) = index;
        let mut vec_idx = rgb_max_i;
        vec_idx = vec_idx * Self::RESOLUTION + z;
        vec_idx = vec_idx * Self::RESOLUTION + y;
        vec_idx = vec_idx * Self::RESOLUTION + x;
        vec_idx = vec_idx * 3 + c_i;
        &self.vals[vec_idx]
    }
}

impl IndexMut<(usize, usize, usize, usize, usize)> for CoefficientTable {
    fn index_mut(&mut self, index: (usize, usize, usize, usize, usize)) -> &mut Self::Output {
        let (rgb_max_i, z, y, x, c_i) = index;
        let mut vec_idx = rgb_max_i;
        vec_idx = vec_idx * Self::RESOLUTION + z;
        vec_idx = vec_idx * Self::RESOLUTION + y;
        vec_idx = vec_idx * Self::RESOLUTION + x;
        vec_idx = vec_idx * 3 + c_i;
        &mut self.vals[vec_idx]
    }
}

/// A reflectance spectrum, represented as a sigmoid applied
/// to a quadratic polynomial.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RGBSigmoidPolynomial {
    /// Coefficient of lambda^2 in the polynomial.
    c0: Float,
    /// Coefficient of lambda^1 in the polynomial.
    c1: Float,
    /// Constant in the polynomial.
    c2: Float,
}

impl RGBSigmoidPolynomial {
    pub fn new(c0: Float, c1: Float, c2: Float) -> Self {
        Self { c0, c1, c2 }
    }

    pub fn at(&self, lambda: Float) -> Float {
        Self::sigmoid(evaluate_polynomial(lambda, &[self.c2, self.c1, self.c0]))
    }

    /// Max value of the spectral distribution over the visible wavelength range (360-830nm),
    /// which can be obtained as the max of the polynomial.
    pub fn max_value(&self) -> Float {
        let max_endpoint = self.at(360.0).max(self.at(830.0));

        // Value found by setting polynomial's derivative to 0.
        let lambda = -self.c1 / (2.0 * self.c0);

        // Return the max out of the endpoints values and value at lambda
        if (360.0..=830.0).contains(&lambda) {
            max_endpoint.max(self.at(lambda))
        } else {
            max_endpoint
        }
    }

    /// Sigmoid function with a special case to handle `+Infinity` and `-Infinity`,
    /// In these cases, `1.0` or `0.0` are returned, respectively.
    /// Useful for coefficients in ideally absorptive/reflective spectra.
    fn sigmoid(x: Float) -> Float {
        if x.is_infinite() {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            // Standard sigmoid function
            0.5 + x / (2.0 * (1.0 + x * x).sqrt())
        }
    }

    fn inverse_sigmoid(x: Float) -> Float {
        (x - 0.5) / (x * (1.0 - x)).sqrt()
    }
}
