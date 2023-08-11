use crate::{math::routines::lerp, Float};

const LAMBDA_MIN: Float = 360.0;
const LAMBDA_MAX: Float = 830.0;

pub trait Spectrum {
    fn at(&self, lambda: Float) -> Float;
    fn max_value(&self) -> Float;
}

#[derive(Clone, Debug)]
pub struct ConstantSpectrum {
    c: Float,
}

impl ConstantSpectrum {
    pub fn new(c: Float) -> Self {
        Self { c }
    }
}

impl Spectrum for ConstantSpectrum {
    fn at(&self, _lambda: Float) -> Float {
        self.c
    }

    fn max_value(&self) -> Float {
        self.c
    }
}

#[derive(Clone, Debug)]
pub struct DenselySampledSpectrum {
    lambda_min: u32,
    lambda_max: u32,
    values: Vec<Float>,
}

impl DenselySampledSpectrum {
    pub fn new(spec: &impl Spectrum, lambda_min: Option<u32>, lambda_max: Option<u32>) -> Self {
        let lambda_min = lambda_min.unwrap_or(LAMBDA_MIN as u32);
        let lambda_max = lambda_max.unwrap_or(LAMBDA_MAX as u32);

        assert!(lambda_max >= lambda_min);

        let mut values = Vec::with_capacity((lambda_max - lambda_min + 1) as usize);
        for lambda in lambda_min..=lambda_max {
            values.push(spec.at(lambda as Float));
        }

        Self {
            lambda_min,
            lambda_max,
            values,
        }
    }
}

impl Spectrum for DenselySampledSpectrum {
    fn at(&self, lambda: Float) -> Float {
        assert!(
            lambda >= 0.0,
            "Spectrum wavelength lambda should be nonnegative"
        );

        let lambda = lambda.round() as u32;

        if lambda > self.lambda_max {
            0.0
        } else {
            let offset = (lambda - self.lambda_min) as usize;
            self.values[offset]
        }
    }

    fn max_value(&self) -> Float {
        *self
            .values
            .iter()
            .max_by(|a, b| {
                a.partial_cmp(b)
                    .expect("There should not be NaNs in spectrum")
            })
            .unwrap()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SpectrumSample {
    pub lambda: Float,
    pub value: Float,
}

pub struct PiecewiseLinearSpectrum {
    samples: Vec<SpectrumSample>,
}

impl PiecewiseLinearSpectrum {
    pub fn new(samples: &[SpectrumSample]) -> Self {
        assert!(
            samples.windows(2).all(|w| w[0].lambda < w[1].lambda),
            "Lambdas should be sorted ascending (strictly, since duplicates should not exist)"
        );

        Self {
            samples: samples.to_vec(),
        }
    }

    fn find_interval(&self, lambda: Float) -> usize {
        // Binary search for
        match self
            .samples
            .binary_search_by(|&sample| sample.lambda.partial_cmp(&lambda).unwrap())
        {
            // Exact match
            Ok(i) => i,
            Err(i) => {
                if i > 0 && i < self.samples.len() - 1 {
                    // Subtract 1 to get index of largest less than lambda
                    // (In case of no match, binary search's return would be 1 over this)
                    i - 1
                } else {
                    panic!("lambda should be within the covered range of samples")
                }
            }
        }
    }
}

#[macro_export]
macro_rules! spectrum_samples {
    ( $( [$lambda:expr, $value:expr] ),* ) => {
        &[
            $( SpectrumSample { lambda: $lambda, value: $value } ),*
        ]
    };
}

pub use spectrum_samples;

impl Spectrum for PiecewiseLinearSpectrum {
    fn at(&self, lambda: Float) -> Float {
        // Handle corner cases
        if self.samples.is_empty()
            || lambda < self.samples[0].lambda
            || lambda > self.samples[self.samples.len() - 1].lambda
        {
            return 0.0;
        }

        // Find lambda offsets to samples and interpolate
        // Sample with largest lambda <=lambda
        let lower_i = self.find_interval(lambda);
        let lower_sample = &self.samples[lower_i];
        let higher_sample = &self.samples[lower_i + 1];
        // Lerp position between the two samples (by lambda)
        let t = (lambda - lower_sample.lambda) / (higher_sample.lambda - lower_sample.lambda);

        lerp(t, lower_sample.value, higher_sample.value)
    }

    fn max_value(&self) -> Float {
        if self.samples.is_empty() {
            0.0
        } else {
            self.samples
                .iter()
                .max_by(|sample_a, sample_b| {
                    sample_a
                        .lambda
                        .partial_cmp(&sample_b.lambda)
                        .expect("There should not be NaNs in spectrum")
                })
                .unwrap()
                .value
        }
    }
}

/// Compute the blackbody emitted radiance at a given `temp` in Kelvin
/// for wavelength `lambda`.
fn blackbody(lambda: Float, temp: Float) -> Float {
    // TODO: Return 0, or panic?
    if temp <= 0.0 {
        return 0.0;
    }

    // Speed of light
    const C: Float = 299792458.0;
    // Planck's constant
    const H: Float = 6.62606957e-34;
    // Boltzmann constant
    const K_B: Float = 1.3806488e-23;

    // Scale nanometers to meters
    let l = lambda * 1e-9;

    // Emitted radiance
    // TODO: Possible speedup for exp
    (2.0 * H * C * C) / (l.powi(5) * (((H * C) / (l * K_B * temp)).exp() - 1.0))
}

/// A normalized blackbody spectral distribution.
/// (The max value at any wavelength is 1.0)
#[derive(Clone, Debug)]
pub struct BlackbodySpectrum {
    /// Temperature of the blackbody in Kelvin.
    temp: Float,
    /// Blackbody normalization constant derived from the temperature
    normalization_factor: Float,
}

impl BlackbodySpectrum {
    /// Construct a new spectrum for a blackbody at the given `temp`.
    pub fn new(temp: Float) -> Self {
        // Wavelength (in meters) where emitted radiance is at maximum, for this temp
        let lambda_max = 2.877721e-3 / temp;
        Self {
            temp,
            normalization_factor: 1.0 / blackbody(lambda_max * 1e9, temp),
        }
    }
}

impl Spectrum for BlackbodySpectrum {
    fn at(&self, lambda: Float) -> Float {
        blackbody(lambda, self.temp) * self.normalization_factor
    }

    /// Max value of the normalized blackbody spectrum,
    /// which by definition is always `1.0`.
    fn max_value(&self) -> Float {
        1.0
    }
}
