use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use crate::{math::routines::lerp, Float};

const LAMBDA_MIN: Float = 360.0;
const LAMBDA_MAX: Float = 830.0;

const N_SPECTRUM_SAMPLES: usize = 4;

pub trait Spectrum {
    fn at(&self, lambda: Float) -> Float;
    fn max_value(&self) -> Float;
    fn sample(&self, wavelengths: &SampledWavelengths) -> SampledSpectrum {
        let values = wavelengths.lambdas().map(|l| self.at(l));
        SampledSpectrum { values }
    }
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

    fn sample(&self, _wavelengths: &SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::new([self.c; N_SPECTRUM_SAMPLES])
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
    #[cfg(pbrt_double_as_float)]
    const H: Float = 6.62606957e-34;
    #[cfg(not(pbrt_double_as_float))]
    const H: Float = 6.6260697e-34;
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

#[derive(Clone, Debug, PartialEq)]
pub struct SampledSpectrum {
    values: [Float; N_SPECTRUM_SAMPLES],
}

impl SampledSpectrum {
    pub fn new(values: [Float; N_SPECTRUM_SAMPLES]) -> Self {
        Self { values }
    }

    pub fn with_single_value(c: Float) -> Self {
        Self {
            values: [c; N_SPECTRUM_SAMPLES],
        }
    }

    /// Returns true if all values in `self` are zero,
    /// false otherwise.
    pub fn is_all_zero(&self) -> bool {
        self.values.iter().all(|&v| v == 0.0)
    }

    pub fn safe_div(&self, rhs: &Self) -> Self {
        // Copy values from self and div by rhs values
        let mut ret = self.clone();
        ret.safe_div_assign(rhs);
        ret
    }

    pub fn safe_div_assign(&mut self, rhs: &Self) {
        for i in 0..N_SPECTRUM_SAMPLES {
            if rhs[i] != 0.0 {
                self[i] -= rhs[i];
            } else {
                self[i] = 0.0;
            }
        }
    }

    pub fn lerp(t: Float, s1: &Self, s2: &Self) -> Self {
        (1.0 - t) * s1 + t * s2
    }

    pub fn sqrt(&self) -> Self {
        // TODO: These fors could be replaced with map if
        // there's no performance issue
        let mut ret = self.clone();

        for i in 0..N_SPECTRUM_SAMPLES {
            ret[i] = ret[i].sqrt();
        }

        ret
    }

    pub fn clamp(&self, min: Float, max: Float) -> Self {
        let mut ret = self.clone();

        for i in 0..N_SPECTRUM_SAMPLES {
            ret[i] = ret[i].clamp(min, max);
        }

        ret
    }

    pub fn clamp_zero(&self) -> Self {
        let mut ret = self.clone();

        for i in 0..N_SPECTRUM_SAMPLES {
            ret[i] = ret[i].max(0.0);
        }

        ret
    }

    pub fn powi(&self, n: i32) -> Self {
        let mut ret = self.clone();

        for i in 0..N_SPECTRUM_SAMPLES {
            ret[i] = ret[i].powi(n);
        }

        ret
    }

    pub fn powf(&self, n: Float) -> Self {
        let mut ret = self.clone();

        for i in 0..N_SPECTRUM_SAMPLES {
            ret[i] = ret[i].powf(n);
        }

        ret
    }

    pub fn exp(&self) -> Self {
        let mut ret = self.clone();

        for i in 0..N_SPECTRUM_SAMPLES {
            ret[i] = ret[i].exp();
        }

        ret
    }

    pub fn max_component_value(&self) -> Option<Float> {
        self.values
            .iter()
            .max_by(|a, b| {
                a.partial_cmp(b)
                    .expect("There should not be NaNs in spectrum")
            })
            .copied()
    }

    pub fn min_component_value(&self) -> Option<Float> {
        self.values
            .iter()
            .min_by(|a, b| {
                a.partial_cmp(b)
                    .expect("There should not be NaNs in spectrum")
            })
            .copied()
    }

    pub fn average(&self) -> Option<Float> {
        if self.values.is_empty() {
            None
        } else {
            let sum: Float = self.values.iter().sum();
            Some(sum / N_SPECTRUM_SAMPLES as Float)
        }
    }
}

impl Index<usize> for SampledSpectrum {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<usize> for SampledSpectrum {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl Add for SampledSpectrum {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        // Take self and add rhs values to it
        let mut ret = self;
        ret += &rhs;

        ret
    }
}

impl Add for &SampledSpectrum {
    type Output = SampledSpectrum;

    fn add(self, rhs: Self) -> Self::Output {
        // Clone self and add rhs values to it
        let mut ret = self.clone();
        ret += rhs;

        ret
    }
}

impl AddAssign<&Self> for SampledSpectrum {
    fn add_assign(&mut self, rhs: &Self) {
        for i in 0..N_SPECTRUM_SAMPLES {
            self[i] += rhs[i];
        }
    }
}

impl Sub for SampledSpectrum {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        // Take self and sub rhs values from it
        let mut ret = self;
        ret -= &rhs;

        ret
    }
}

impl Sub for &SampledSpectrum {
    type Output = SampledSpectrum;

    fn sub(self, rhs: Self) -> Self::Output {
        // Clone self and sub rhs values from it
        let mut ret = self.clone();
        ret -= rhs;

        ret
    }
}

impl SubAssign<&Self> for SampledSpectrum {
    fn sub_assign(&mut self, rhs: &Self) {
        for i in 0..N_SPECTRUM_SAMPLES {
            self[i] -= rhs[i];
        }
    }
}

impl Mul for SampledSpectrum {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        // Take self and mul by rhs values
        let mut ret = self;
        ret *= &rhs;

        ret
    }
}

impl Mul for &SampledSpectrum {
    type Output = SampledSpectrum;

    fn mul(self, rhs: Self) -> Self::Output {
        // Clone self and mul by rhs values
        let mut ret = self.clone();
        ret *= rhs;

        ret
    }
}

impl MulAssign<&Self> for SampledSpectrum {
    fn mul_assign(&mut self, rhs: &Self) {
        for i in 0..N_SPECTRUM_SAMPLES {
            self[i] *= rhs[i];
        }
    }
}

impl Mul<Float> for SampledSpectrum {
    type Output = Self;

    fn mul(self, rhs: Float) -> Self {
        // Take self and mul by rhs
        let mut ret = self;
        ret *= rhs;

        ret
    }
}

impl Mul<Float> for &SampledSpectrum {
    type Output = SampledSpectrum;

    fn mul(self, rhs: Float) -> Self::Output {
        // Clone self and mul by rhs
        let mut ret = self.clone();
        ret *= rhs;

        ret
    }
}

impl Mul<SampledSpectrum> for Float {
    type Output = SampledSpectrum;

    fn mul(self, rhs: SampledSpectrum) -> Self::Output {
        rhs * self
    }
}

impl Mul<&SampledSpectrum> for Float {
    type Output = SampledSpectrum;

    fn mul(self, rhs: &SampledSpectrum) -> Self::Output {
        rhs * self
    }
}

impl MulAssign<Float> for SampledSpectrum {
    fn mul_assign(&mut self, rhs: Float) {
        for i in 0..N_SPECTRUM_SAMPLES {
            self[i] *= rhs;
        }
    }
}

impl Div for SampledSpectrum {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        // Take self and div by rhs values
        let mut ret = self;
        ret /= &rhs;

        ret
    }
}

impl Div for &SampledSpectrum {
    type Output = SampledSpectrum;

    fn div(self, rhs: Self) -> Self::Output {
        // Clone self and div by rhs values
        let mut ret = self.clone();
        ret /= rhs;

        ret
    }
}

impl DivAssign<&Self> for SampledSpectrum {
    fn div_assign(&mut self, rhs: &Self) {
        for i in 0..N_SPECTRUM_SAMPLES {
            self[i] /= rhs[i];
        }
    }
}

impl Div<Float> for SampledSpectrum {
    type Output = Self;

    fn div(self, rhs: Float) -> Self {
        // Take self and div by rhs
        let mut ret = self;
        ret /= rhs;

        ret
    }
}

impl Div<Float> for &SampledSpectrum {
    type Output = SampledSpectrum;

    fn div(self, rhs: Float) -> Self::Output {
        assert!(rhs != 0.0, "Cannot divide spectrum values by zero");

        // Clone self and div by rhs
        let mut ret = self.clone();
        ret /= rhs;

        ret
    }
}

impl DivAssign<Float> for SampledSpectrum {
    fn div_assign(&mut self, rhs: Float) {
        assert!(rhs != 0.0, "Cannot divide spectrum values by zero");

        for i in 0..N_SPECTRUM_SAMPLES {
            self[i] /= rhs;
        }
    }
}

#[derive(Clone, Debug)]
pub struct SampledWavelengths {
    lambdas: [Float; N_SPECTRUM_SAMPLES],
    pdf: [Float; N_SPECTRUM_SAMPLES],
}

impl SampledWavelengths {
    pub fn sample_uniform(u: Float, lambda_min: Option<Float>, lambda_max: Option<Float>) -> Self {
        let lambda_min = lambda_min.unwrap_or(LAMBDA_MIN);
        let lambda_max = lambda_max.unwrap_or(LAMBDA_MAX);

        let mut lambdas = [0.0; N_SPECTRUM_SAMPLES];

        // Sample first wavelength using u
        lambdas[0] = lerp(lambda_min, lambda_max, u);

        // Initialize lambda for remaining wavelengths
        let delta = (lambda_max - lambda_min) / N_SPECTRUM_SAMPLES as Float;

        for i in 1..N_SPECTRUM_SAMPLES {
            lambdas[i] =
                (lambdas[i - 1] + delta - lambda_min) % (lambda_max - lambda_min) + lambda_min;
        }

        // Compute PDF for sample wavelengths
        let pdf = [1.0 / (lambda_max - lambda_min); N_SPECTRUM_SAMPLES];

        Self { lambdas, pdf }
    }

    pub fn lambdas(&self) -> &[Float; N_SPECTRUM_SAMPLES] {
        &self.lambdas
    }

    pub fn pdf(&self) -> SampledSpectrum {
        SampledSpectrum::new(self.pdf)
    }

    pub fn terminate_secondary(&mut self) {
        if !self.secondary_terminated() {
            // Update wavelength probabilities for termination
            self.pdf[0] /= N_SPECTRUM_SAMPLES as Float;
            for p in self.pdf.iter_mut().skip(1) {
                *p = 0.0;
            }
        }
    }

    pub fn secondary_terminated(&self) -> bool {
        self.pdf.iter().skip(1).all(|&x| x == 0.0)
    }
}

impl Index<usize> for SampledWavelengths {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        &self.lambdas[index]
    }
}

impl IndexMut<usize> for SampledWavelengths {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.lambdas[index]
    }
}
