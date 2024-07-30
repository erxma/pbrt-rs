use std::{
    cmp::min,
    ops::{Index, IndexMut},
};

use crate::{
    math::{evaluate_polynomial, lerp, Point2f, SquareMatrix},
    sampling::spectrum::{spectrum_to_xyz, DenselySampledSpectrum, SpectrumEnum},
    Float,
};

use super::{RGB, XYZ};

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
        illuminant: SpectrumEnum,
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
        let zi = self
            .z_nodes
            .binary_search_by(|&node| node.partial_cmp(&z).unwrap())
            .unwrap();

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

impl Index<(usize, usize, usize, usize, usize)> for CoefficientTable {
    type Output = Float;

    fn index(&self, index: (usize, usize, usize, usize, usize)) -> &Self::Output {
        let (rgb_max_i, z, y, x, c_i) = index;
        &self.vals[rgb_max_i][z][y][x][c_i]
    }
}

impl IndexMut<(usize, usize, usize, usize, usize)> for CoefficientTable {
    fn index_mut(&mut self, index: (usize, usize, usize, usize, usize)) -> &mut Self::Output {
        let (rgb_max_i, z, y, x, c_i) = index;
        &mut self.vals[rgb_max_i][z][y][x][c_i]
    }
}

const RESOLUTION: usize = RGBToSpectrumTable::RESOLUTION;

#[derive(Clone, Debug)]
struct CoefficientTable {
    vals: [[[[[Float; 3]; RESOLUTION]; RESOLUTION]; RESOLUTION]; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RGBSigmoidPolynomial {
    c0: Float,
    c1: Float,
    c2: Float,
}

impl RGBSigmoidPolynomial {
    pub fn new(c0: Float, c1: Float, c2: Float) -> Self {
        Self { c0, c1, c2 }
    }

    pub fn at(&self, lambda: Float) -> Float {
        Self::sigmoid(evaluate_polynomial(lambda, &[self.c2, self.c1, self.c0]))
    }

    pub fn max_value(&self) -> Float {
        let max_endpoint = self.at(360.0).max(self.at(830.0));

        let lambda = -self.c1 / (2.0 * self.c0);

        if (360.0..830.0).contains(&lambda) {
            max_endpoint.max(self.at(lambda))
        } else {
            max_endpoint
        }
    }

    fn sigmoid(x: Float) -> Float {
        if x.is_infinite() {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            0.5 + x / (2.0 * (1.0 + x * x).sqrt())
        }
    }

    fn inverse_sigmoid(x: Float) -> Float {
        (x - 0.5) / (x * (1.0 - x)).sqrt()
    }
}
