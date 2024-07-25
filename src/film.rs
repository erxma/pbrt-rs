use std::{array, sync::OnceLock};

use derive_builder::Builder;

use crate::{
    color::{self, RGBColorSpace, RGB, XYZ},
    math::{square_matrix::SquareMatrix, tuple::Tuple},
    sampling::spectrum::{
        self, spectra, DenselySampledSpectrum, PiecewiseLinearSpectrum, SampledSpectrum,
        SampledWavelengths, Spectrum, LAMBDA_MAX, LAMBDA_MIN,
    },
    util::data::SWATCH_REFLECTANCES_INTERLEAVED,
    Float,
};

pub trait Film {}

#[derive(Clone, Debug)]
pub struct PixelSensor {
    r_bar: DenselySampledSpectrum,
    g_bar: DenselySampledSpectrum,
    b_bar: DenselySampledSpectrum,
    imaging_ratio: Float,
    xyz_from_sensor_rgb: SquareMatrix<3>,
}

pub const N_SWATCH_REFLECTANCES: usize = 24;
static SWATCH_REFLECTANCES: OnceLock<[PiecewiseLinearSpectrum; N_SWATCH_REFLECTANCES]> =
    OnceLock::new();

fn swatch_reflectances() -> &'static [PiecewiseLinearSpectrum; N_SWATCH_REFLECTANCES] {
    SWATCH_REFLECTANCES.get_or_init(|| {
        core::array::from_fn(|i| {
            PiecewiseLinearSpectrum::from_interleaved(&SWATCH_REFLECTANCES_INTERLEAVED[i], false)
        })
    })
}

impl PixelSensor {
    pub fn builder<'a>() -> PixelSensorBuilder<'a> {
        PixelSensorBuilder::default()
    }
}

impl PixelSensor {
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

#[derive(Builder)]
#[builder(
    name = "PixelSensorBuilder",
    public,
    setter(strip_option),
    build_fn(private, name = "build_params")
)]
struct PixelSensorParams<'a> {
    #[builder(default)]
    rgb_matching: Option<[&'a dyn Spectrum; 3]>,
    output_color_space: &'a RGBColorSpace<'a>,
    #[builder(default)]
    sensor_illum: Option<&'a dyn Spectrum>,
    imaging_ratio: Float,
}

impl<'a> PixelSensorBuilder<'a> {
    pub fn build(&self) -> Result<PixelSensor, PixelSensorBuilderError> {
        let params = self.build_params()?;

        match params.rgb_matching {
            Some(_) => Self::build_with_rgb_matching(params),
            None => Self::build_with_xyz_matching(params),
        }
    }

    fn build_with_rgb_matching(
        params: PixelSensorParams,
    ) -> Result<PixelSensor, PixelSensorBuilderError> {
        let [r, g, b] = params.rgb_matching.unwrap();
        let r_bar = DenselySampledSpectrum::new(r, None, None);
        let g_bar = DenselySampledSpectrum::new(g, None, None);
        let b_bar = DenselySampledSpectrum::new(b, None, None);
        let sensor_illum = params
            .sensor_illum
            .ok_or(PixelSensorBuilderError::ValidationError(
                "When RGB matching is used, sensor_illum must be provided".to_string(),
            ))?;

        // Compute XYZ from camera RGB matrix:

        // Compute rgb_camera values for training swatches
        let rgb_camera: [[Float; 3]; N_SWATCH_REFLECTANCES] = array::from_fn(|row| {
            let rgb: RGB = project_reflectance(
                &swatch_reflectances()[row],
                sensor_illum,
                &r_bar,
                &g_bar,
                &b_bar,
            );
            rgb.into()
        });

        let sensor_white_g = spectrum::inner_product(sensor_illum, &g_bar);
        let sensor_white_y = spectrum::inner_product(sensor_illum, spectra::y());

        let xyz_output: [[Float; 3]; N_SWATCH_REFLECTANCES] = array::from_fn(|row| {
            let xyz = project_reflectance::<XYZ>(
                &swatch_reflectances()[row],
                &params.output_color_space.illuminant,
                spectra::x(),
                spectra::y(),
                spectra::z(),
            ) * (sensor_white_y / sensor_white_g);
            xyz.into()
        });

        // Initialize XYZFromSensorRGB using linear least squares
        let xyz_from_sensor_rgb = SquareMatrix::linear_least_squares(&rgb_camera, &xyz_output)
            .ok_or(PixelSensorBuilderError::ValidationError(
                "Sensor XYZ from RGB matrix could not be solved.".to_string(),
            ))?;

        Ok(PixelSensor {
            r_bar,
            g_bar,
            b_bar,
            imaging_ratio: params.imaging_ratio,
            xyz_from_sensor_rgb,
        })
    }

    fn build_with_xyz_matching(
        params: PixelSensorParams,
    ) -> Result<PixelSensor, PixelSensorBuilderError> {
        let r_bar = DenselySampledSpectrum::new(spectra::x(), None, None);
        let g_bar = DenselySampledSpectrum::new(spectra::y(), None, None);
        let b_bar = DenselySampledSpectrum::new(spectra::z(), None, None);

        // Compute white balancing matrix for XYZ PixelSensor
        let xyz_from_sensor_rgb = match params.sensor_illum {
            Some(sensor_illum) => {
                let src_white = spectrum::spectrum_to_xyz(sensor_illum).xy();
                let target_white = params.output_color_space.w;
                color::white_balance(src_white, target_white)
            }
            None => SquareMatrix::IDENTITY,
        };

        Ok(PixelSensor {
            r_bar,
            g_bar,
            b_bar,
            imaging_ratio: params.imaging_ratio,
            xyz_from_sensor_rgb,
        })
    }
}

#[inline]
fn project_reflectance<Triplet: Tuple<3, Float>>(
    reflectance: &impl Spectrum,
    illum: &dyn Spectrum,
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
