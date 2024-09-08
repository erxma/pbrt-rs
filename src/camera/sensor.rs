use std::{array, sync::LazyLock};

use crate::{
    color::{self, RGBColorSpace, RGB, XYZ},
    math::{SquareMatrix, Tuple},
    sampling::spectrum::{
        self, DenselySampledSpectrum, PiecewiseLinearSpectrum, SampledSpectrum, SampledWavelengths,
        Spectrum, SpectrumEnum, LAMBDA_MAX, LAMBDA_MIN,
    },
    util::data::SWATCH_REFLECTANCES_INTERLEAVED,
    Float,
};

#[derive(Clone, Debug)]
pub struct PixelSensor {
    r_bar: DenselySampledSpectrum,
    g_bar: DenselySampledSpectrum,
    b_bar: DenselySampledSpectrum,
    imaging_ratio: Float,
    pub xyz_from_sensor_rgb: SquareMatrix<3>,
}

static SWATCH_REFLECTANCES: LazyLock<
    [PiecewiseLinearSpectrum; PixelSensor::N_SWATCH_REFLECTANCES],
> = LazyLock::new(|| {
    core::array::from_fn(|i| {
        PiecewiseLinearSpectrum::from_interleaved(&SWATCH_REFLECTANCES_INTERLEAVED[i], false)
    })
});

impl PixelSensor {
    pub const N_SWATCH_REFLECTANCES: usize = 24;

    pub fn with_rgb_matching(
        output_color_space: &RGBColorSpace,
        r: &impl Spectrum,
        g: &impl Spectrum,
        b: &impl Spectrum,
        sensor_illum: &impl Spectrum,
        imaging_ratio: Float,
    ) -> Self {
        let r_bar = DenselySampledSpectrum::new(r, None, None);
        let g_bar = DenselySampledSpectrum::new(g, None, None);
        let b_bar = DenselySampledSpectrum::new(b, None, None);

        // Compute XYZ from camera RGB matrix:
        // Compute rgb_camera values for training swatches
        let rgb_camera: [[Float; 3]; PixelSensor::N_SWATCH_REFLECTANCES] = array::from_fn(|row| {
            let rgb: RGB = project_reflectance(
                &SWATCH_REFLECTANCES[row],
                sensor_illum,
                &r_bar,
                &g_bar,
                &b_bar,
            );
            rgb.into()
        });

        let sensor_white_g = sensor_illum.inner_product(&g_bar);
        let sensor_white_y = sensor_illum.inner_product(&*spectrum::Y);

        let xyz_output: [[Float; 3]; PixelSensor::N_SWATCH_REFLECTANCES] = array::from_fn(|row| {
            let xyz = project_reflectance::<XYZ>(
                &SWATCH_REFLECTANCES[row],
                &output_color_space.illuminant,
                &*spectrum::X,
                &*spectrum::Y,
                &*spectrum::Z,
            ) * (sensor_white_y / sensor_white_g);
            xyz.into()
        });

        // Initialize XYZFromSensorRGB using linear least squares
        let xyz_from_sensor_rgb = SquareMatrix::linear_least_squares(&rgb_camera, &xyz_output)
            .expect("sensor XYZ from RGB matrix could not be solved");

        PixelSensor {
            r_bar,
            g_bar,
            b_bar,
            imaging_ratio,
            xyz_from_sensor_rgb,
        }
    }

    pub fn with_xyz_matching(
        output_color_space: &RGBColorSpace,
        sensor_illum: Option<&SpectrumEnum>,
        imaging_ratio: Float,
    ) -> Self {
        let r_bar = DenselySampledSpectrum::new(&*spectrum::X, None, None);
        let g_bar = DenselySampledSpectrum::new(&*spectrum::Y, None, None);
        let b_bar = DenselySampledSpectrum::new(&*spectrum::Z, None, None);

        // Compute white balancing matrix for XYZ PixelSensor
        let xyz_from_sensor_rgb = match sensor_illum {
            Some(sensor) => {
                let src_white = sensor.to_xyz().xy();
                let target_white = output_color_space.w;
                color::white_balance(src_white, target_white)
            }
            None => SquareMatrix::IDENTITY,
        };

        Self {
            r_bar,
            g_bar,
            b_bar,
            imaging_ratio,
            xyz_from_sensor_rgb,
        }
    }

    #[allow(non_snake_case)]
    pub fn to_sensor_rgb(&self, L: &SampledSpectrum, lambda: &SampledWavelengths) -> RGB {
        let L = L.safe_div(&lambda.pdf());
        RGB::new(
            (self.r_bar.sample(lambda) * &L).average().unwrap(),
            (self.g_bar.sample(lambda) * &L).average().unwrap(),
            (self.b_bar.sample(lambda) * L).average().unwrap(),
        ) * self.imaging_ratio
    }
}

#[inline]
fn project_reflectance<Triplet: Tuple<3, Float>>(
    reflectance: &impl Spectrum,
    illum: &impl Spectrum,
    b1: &impl Spectrum,
    b2: &impl Spectrum,
    b3: &impl Spectrum,
) -> Triplet {
    let mut result = Triplet::from([0.0; 3]);
    let mut g_integral = 0.0;

    let mut lambda = LAMBDA_MIN;
    while lambda <= LAMBDA_MAX {
        g_integral += b2.at(lambda) * illum.at(lambda);
        result[0] += b1.at(lambda) * reflectance.at(lambda) * illum.at(lambda);
        result[1] += b2.at(lambda) * reflectance.at(lambda) * illum.at(lambda);
        result[2] += b3.at(lambda) * reflectance.at(lambda) * illum.at(lambda);
        lambda += 1.0;
    }

    result / g_integral
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use color::SRGB;

    use super::*;

    #[test]
    fn to_sensor_rgb_srgb() {
        let sensor = PixelSensor::with_xyz_matching(&SRGB, None, 1.0);
        let lambda = SampledWavelengths::from_parts(
            [591.47546, 700.3552, 452.77136, 525.49225],
            [0.003408977, 0.001265176, 0.0027623207, 0.003908024],
        );
        let l = SampledSpectrum::new([0.00357853, 0.0025127477, 0.005482404, 0.004613267]);
        let result = sensor.to_sensor_rgb(&l, &lambda);
        assert_relative_eq!(result.r, 0.47140643);
        assert_relative_eq!(result.g, 0.4532867);
        assert_relative_eq!(result.b, 0.889924);

        let lambda = SampledWavelengths::from_parts(
            [507.79996, 572.4091, 658.3696, 425.15646],
            [0.0037592475, 0.0037075484, 0.0020110987, 0.002166194],
        );
        let l = SampledSpectrum::new([0.0047808043, 0.0039641955, 0.002973517, 0.0043246187]);
        let result = sensor.to_sensor_rgb(&l, &lambda);
        assert_relative_eq!(result.r, 0.38974532);
        assert_relative_eq!(result.g, 0.42723745);
        assert_relative_eq!(result.b, 0.5760964);
    }
}
