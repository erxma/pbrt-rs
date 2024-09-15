use crate::{
    color::{RGBColorSpace, RGBSigmoidPolynomial, RGB, XYZ},
    core::{find_interval, lerp, Float},
    util::data::{CIE_ILLUM_D6500, CIE_LAMBDA, CIE_X, CIE_Y, CIE_Z, N_CIE_SPECTRUM_SAMPLES},
};
use approx::{AbsDiffEq, RelativeEq};
use delegate::delegate;
use enum_as_inner::EnumAsInner;
use enum_dispatch::enum_dispatch;
use ordered_float::NotNan;
use overload::overload;
use std::{
    array, fmt,
    iter::Sum,
    ops::{self, Add, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    sync::LazyLock,
};

pub const LAMBDA_MIN: Float = 360.0;
pub const LAMBDA_MAX: Float = 830.0;

const N_SPECTRUM_SAMPLES: usize = 4;

/// X component of CIE XYZ matching curves, sampled at 1-nm increments from 360nm to 830nm.
pub static X: LazyLock<SpectrumEnum> = LazyLock::new(|| {
    let samples: [SpectrumSample; N_CIE_SPECTRUM_SAMPLES] = array::from_fn(|i| SpectrumSample {
        lambda: CIE_LAMBDA[i],
        value: CIE_X[i],
    });
    let pls = PiecewiseLinearSpectrum::new(&samples);
    DenselySampledSpectrum::new(&pls, None, None).into()
});

/// Y component of CIE XYZ matching curves, sampled at 1-nm increments from 360nm to 830nm.
pub static Y: LazyLock<SpectrumEnum> = LazyLock::new(|| {
    let samples: [SpectrumSample; N_CIE_SPECTRUM_SAMPLES] = array::from_fn(|i| SpectrumSample {
        lambda: CIE_LAMBDA[i],
        value: CIE_Y[i],
    });
    let pls = PiecewiseLinearSpectrum::new(&samples);
    DenselySampledSpectrum::new(&pls, None, None).into()
});

/// Z component of CIE XYZ matching curves, sampled at 1-nm increments from 360nm to 830nm.
pub static Z: LazyLock<SpectrumEnum> = LazyLock::new(|| {
    let samples: [SpectrumSample; N_CIE_SPECTRUM_SAMPLES] = array::from_fn(|i| SpectrumSample {
        lambda: CIE_LAMBDA[i],
        value: CIE_Z[i],
    });
    let pls = PiecewiseLinearSpectrum::new(&samples);
    DenselySampledSpectrum::new(&pls, None, None).into()
});

pub static ILLUMD65: LazyLock<SpectrumEnum> =
    LazyLock::new(|| PiecewiseLinearSpectrum::from_interleaved(&CIE_ILLUM_D6500, true).into());

pub const CIE_Y_INTEGRAL: Float = 106.856895;

#[inline]
pub fn visible_wavelengths_pdf(lambda: Float) -> Float {
    if (LAMBDA_MIN..=LAMBDA_MAX).contains(&lambda) {
        #[cfg(not(feature = "use-f64"))]
        return 0.003939804 / (0.0072 * (lambda - 538.0)).cosh().powi(2);
        #[cfg(feature = "use-f64")]
        return 0.0039398042 / (0.0072 * (lambda - 538.0)).cosh().powi(2);
    } else {
        0.0
    }
}

#[inline]
pub fn sample_visible_wavelengths(u: Float) -> Float {
    #[cfg(not(feature = "use-f64"))]
    return 538.0 - 138.88889 * (0.85691062 - 1.827502 * u).atanh();
    #[cfg(feature = "use-f64")]
    return 538.0 - 138.888889 * (0.85691062 - 1.82750197 * u).atanh();
}

#[enum_dispatch]
#[derive(Clone, Debug, EnumAsInner)]
pub enum SpectrumEnum {
    Constant(ConstantSpectrum),
    DenselySampled(DenselySampledSpectrum),
    PiecewiseLinear(PiecewiseLinearSpectrum),
    Blackbody(BlackbodySpectrum),
    RgbAlbedo(RgbAlbedoSpectrum),
    RgbUnbounded(RgbUnboundedSpectrum),
    RgbIlluminant(RgbIlluminantSpectrum),
}

impl SpectrumEnum {
    delegate! {
        #[through(Spectrum)]
        to self {
            pub fn at(&self, lambda: Float) -> Float;
            pub fn max_value(&self) -> Float;
            pub fn sample(&self, wavelengths: &SampledWavelengths) -> SampledSpectrum;
            pub fn inner_product(&self, other: &impl Spectrum) -> Float;
            pub fn to_xyz(&self) -> XYZ;
            pub fn to_photometric(&self) -> Float;
        }
    }
}

#[enum_dispatch(SpectrumEnum)]
pub trait Spectrum {
    fn at(&self, lambda: Float) -> Float;
    fn max_value(&self) -> Float;
    fn sample(&self, wavelengths: &SampledWavelengths) -> SampledSpectrum {
        let values = wavelengths.lambdas().map(|l| self.at(l));
        SampledSpectrum::new(values)
    }

    /// Compute the inner product (sum of element-wise products) with a Riemann sum
    /// over integer wavelengths.
    fn inner_product(&self, other: &impl Spectrum) -> Float {
        let mut integral = 0.0;

        let mut lambda = LAMBDA_MIN;
        while lambda <= LAMBDA_MAX {
            integral += self.at(lambda) * other.at(lambda);
            lambda += 1.0;
        }

        integral
    }

    /// Compute the XYZ coefficients of this distribution,
    /// i.e. this distribution's inner product with the X, Y, Z matching curves.
    fn to_xyz(&self) -> XYZ {
        XYZ::new(
            self.inner_product(&*X),
            self.inner_product(&*Y),
            self.inner_product(&*Z),
        ) / CIE_Y_INTEGRAL
    }

    fn to_photometric(&self) -> Float {
        let mut y = 0.0;
        let mut lambda = LAMBDA_MIN;
        while lambda <= LAMBDA_MAX {
            y += Y.at(lambda) * self.at(lambda);
            lambda += 1.0;
        }
        y
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DenselySampledSpectrum {
    lambda_min: usize,
    lambda_max: usize,
    values: Vec<NotNan<Float>>,
}

impl DenselySampledSpectrum {
    pub fn new(spec: &impl Spectrum, lambda_min: Option<usize>, lambda_max: Option<usize>) -> Self {
        let lambda_min = lambda_min.unwrap_or(LAMBDA_MIN as usize);
        let lambda_max = lambda_max.unwrap_or(LAMBDA_MAX as usize);

        assert!(lambda_max >= lambda_min);

        let mut values = Vec::with_capacity(lambda_max - lambda_min + 1);
        for lambda in lambda_min..=lambda_max {
            values.push(NotNan::new(spec.at(lambda as Float)).expect(
                "There should not be NaNs in spectrum, but one was found when sampling `spec`",
            ));
        }

        Self {
            lambda_min,
            lambda_max,
            values,
        }
    }

    pub fn scaled(mut self, factor: Float) -> Self {
        for v in self.values.iter_mut() {
            *v *= factor;
        }
        self
    }
}

impl Spectrum for DenselySampledSpectrum {
    fn at(&self, lambda: Float) -> Float {
        assert!(
            lambda >= 0.0,
            "Spectrum wavelength lambda should be nonnegative"
        );

        let lambda = lambda.round() as usize;

        if (self.lambda_min..=self.lambda_max).contains(&lambda) {
            let offset = lambda - self.lambda_min;
            *self.values[offset]
        } else {
            0.0
        }
    }

    fn max_value(&self) -> Float {
        **self.values.iter().max().unwrap()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SpectrumSample {
    pub lambda: Float,
    pub value: Float,
}

#[derive(Clone, Debug)]
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

    pub fn from_interleaved(interleaved_samples: &[Float], normalize: bool) -> Self {
        assert_eq!(
            0,
            interleaved_samples.len() % 2,
            "Interleaved format should pairs, and thus an even number of values"
        );

        let n = interleaved_samples.len() / 2;
        let mut samples = Vec::with_capacity(n + 2);

        // Extend samples to cover range of visible wavelengths if needed.
        if interleaved_samples[0] > LAMBDA_MIN {
            samples.push(SpectrumSample {
                lambda: LAMBDA_MIN - 1.0,
                value: interleaved_samples[1],
            });
        }

        for i in 0..n {
            samples.push(SpectrumSample {
                lambda: interleaved_samples[2 * i],
                value: interleaved_samples[2 * i + 1],
            });
        }

        if samples.last().unwrap().lambda < LAMBDA_MAX {
            samples.push(SpectrumSample {
                lambda: LAMBDA_MAX + 1.0,
                ..*samples.last().unwrap()
            });
        }

        let mut spec = PiecewiseLinearSpectrum::new(&samples);

        if normalize {
            // Normalize to have luminance of 1.
            let inner = CIE_Y_INTEGRAL / spec.inner_product(&*Y);
            spec = spec.scale(inner);
        }

        spec
    }

    pub fn scale(mut self, s: Float) -> Self {
        for sample in self.samples.iter_mut() {
            sample.value *= s;
        }
        self
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
        let lower_i =
            find_interval(self.samples.len(), |i| self.samples[i].lambda <= lambda).unwrap();
        let lower_sample = &self.samples[lower_i];

        if lambda == lower_sample.lambda {
            lower_sample.value
        } else {
            let higher_sample = &self.samples[lower_i + 1];
            // Lerp position between the two samples (by lambda)
            let t = (lambda - lower_sample.lambda) / (higher_sample.lambda - lower_sample.lambda);

            lerp(lower_sample.value, higher_sample.value, t)
        }
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
    #[cfg(feature = "use-f64")]
    const H: Float = 6.62606957e-34;
    #[cfg(not(feature = "use-f64"))]
    const H: Float = 6.6260697e-34;
    // Boltzmann constant
    const K_B: Float = 1.3806488e-23;

    // Scale nanometers to meters
    let l = lambda * 1e-9;

    // Emitted radiance
    // OPTIMIZATION: Possible speedup for exp
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
        #[cfg(feature = "use-f64")]
        let lambda_max = 2.8977721e-3 / temp;
        #[cfg(not(feature = "use-f64"))]
        let lambda_max = 2.897772e-3 / temp;
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
                self[i] /= rhs[i];
            } else {
                self[i] = 0.0;
            }
        }
    }

    pub fn lerp(t: Float, s1: &Self, s2: &Self) -> Self {
        (1.0 - t) * s1 + t * s2
    }

    pub fn sqrt(&self) -> Self {
        Self {
            values: self.values.map(|sample| sample.sqrt()),
        }
    }

    pub fn clamp(&self, min: Float, max: Float) -> Self {
        Self {
            values: self.values.map(|sample| sample.clamp(min, max)),
        }
    }

    pub fn clamp_zero(&self) -> Self {
        Self {
            values: self.values.map(|sample| sample.max(0.0)),
        }
    }

    pub fn powi(&self, n: i32) -> Self {
        Self {
            values: self.values.map(|sample| sample.powi(n)),
        }
    }

    pub fn powf(&self, n: Float) -> Self {
        Self {
            values: self.values.map(|sample| sample.powf(n)),
        }
    }

    pub fn exp(&self) -> Self {
        Self {
            values: self.values.map(|sample| sample.exp()),
        }
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

    /// Compute XYZ coefficients for `self`, using a Monte Carlo estimate
    /// using the sampled XYZ spectral values at the given `wavelengths`.
    pub fn to_xyz(&self, wavelengths: &SampledWavelengths) -> XYZ {
        // Sample the X, Y, Z matching curves at lambda
        let x = X.sample(wavelengths);
        let y = Y.sample(wavelengths);
        let z = Z.sample(wavelengths);

        // Evaluate estimator to compute (x, y, z) coefficients
        let pdf = wavelengths.pdf();
        XYZ::new(
            (x * self).safe_div(&pdf).average().unwrap(),
            (y * self).safe_div(&pdf).average().unwrap(),
            (z * self).safe_div(&pdf).average().unwrap(),
        ) / CIE_Y_INTEGRAL
    }

    pub fn y(&self, wavelengths: &SampledWavelengths) -> Float {
        // Sample the Y matching curve at lambda
        let y = Y.sample(wavelengths);

        // Evaluate estimator to compute y coefficient
        let pdf = wavelengths.pdf();

        (&y * self).safe_div(&pdf).average().unwrap()
    }

    /// Convert spectrum sampled at `wavelengths` to RGB in a given color space `cs`,
    /// via XYZ.
    pub fn to_rgb(&self, wavelengths: &SampledWavelengths, cs: &RGBColorSpace) -> RGB {
        let xyz = self.to_xyz(wavelengths);
        cs.to_rgb(xyz)
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

overload!((lhs: &mut SampledSpectrum) += (rhs: ?SampledSpectrum) {
    for i in 0..N_SPECTRUM_SAMPLES {
        lhs[i] += rhs[i];
    }
});

impl Sum for SampledSpectrum {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::with_single_value(0.0), |sum, item| sum + item)
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

overload!((lhs: ?SampledSpectrum) * (rhs: ?SampledSpectrum) -> SampledSpectrum {
    let mut ret = lhs.clone();
    for i in 0..N_SPECTRUM_SAMPLES {
        ret[i] *= rhs[i];
    }
    ret
});

overload!((lhs: &mut SampledSpectrum) *= (rhs: ?SampledSpectrum) {
    for i in 0..N_SPECTRUM_SAMPLES {
        lhs[i] *= rhs[i];
    }
});

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

impl fmt::Display for SampledSpectrum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.values).finish()
    }
}

impl AbsDiffEq for SampledSpectrum {
    type Epsilon = <Float as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        Float::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.values
            .iter()
            .zip(other.values.iter())
            .all(|(lhs, rhs)| lhs.abs_diff_eq(rhs, epsilon))
    }
}

impl RelativeEq for SampledSpectrum {
    fn default_max_relative() -> Self::Epsilon {
        Float::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.values
            .iter()
            .zip(other.values.iter())
            .all(|(lhs, rhs)| lhs.relative_eq(rhs, epsilon, max_relative))
    }
}

#[derive(Clone, Debug)]
pub struct SampledWavelengths {
    lambdas: [Float; N_SPECTRUM_SAMPLES],
    pdf: [Float; N_SPECTRUM_SAMPLES],
}

impl SampledWavelengths {
    pub fn from_parts(
        lambdas: [Float; N_SPECTRUM_SAMPLES],
        pdf: [Float; N_SPECTRUM_SAMPLES],
    ) -> Self {
        Self { lambdas, pdf }
    }

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

    pub fn sample_visible(u: Float) -> Self {
        let lambdas: [Float; N_SPECTRUM_SAMPLES] = array::from_fn(|i| {
            let mut up = u + (i as Float) / (N_SPECTRUM_SAMPLES as Float);
            if up > 1.0 {
                up -= 1.0;
            }

            sample_visible_wavelengths(up)
        });

        let pdf = lambdas.map(visible_wavelengths_pdf);

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

#[derive(Clone, Debug)]
pub struct RgbAlbedoSpectrum {
    rsp: RGBSigmoidPolynomial,
}

impl RgbAlbedoSpectrum {
    pub fn new(cs: &RGBColorSpace, rgb: RGB) -> Self {
        assert!(
            (0.0..=1.0).contains(&rgb.r)
                && (0.0..=1.0).contains(&rgb.g)
                && (0.0..=1.0).contains(&rgb.b)
        );
        Self {
            rsp: cs.to_rgb_coeffs(rgb),
        }
    }
}

impl Spectrum for RgbAlbedoSpectrum {
    fn at(&self, lambda: Float) -> Float {
        self.rsp.at(lambda)
    }

    fn max_value(&self) -> Float {
        self.rsp.max_value()
    }
}

#[derive(Clone, Debug)]
pub struct RgbUnboundedSpectrum {
    scale: Float,
    rsp: RGBSigmoidPolynomial,
}

impl RgbUnboundedSpectrum {
    pub fn new(cs: &RGBColorSpace, rgb: RGB) -> Self {
        let m = rgb.r.max(rgb.g).max(rgb.b);

        let scale = 2.0 * m;
        let rsp = cs.to_rgb_coeffs(if scale != 0.0 {
            rgb / scale
        } else {
            RGB::new(0.0, 0.0, 0.0)
        });

        Self { scale, rsp }
    }
}

impl Spectrum for RgbUnboundedSpectrum {
    fn at(&self, lambda: Float) -> Float {
        self.scale * self.rsp.at(lambda)
    }

    fn max_value(&self) -> Float {
        self.scale * self.rsp.max_value()
    }
}

#[derive(Clone, Debug, Copy)]
pub struct RgbIlluminantSpectrum {
    scale: Float,
    rsp: RGBSigmoidPolynomial,
    illuminant: &'static DenselySampledSpectrum,
}

impl RgbIlluminantSpectrum {
    pub fn new(cs: &'static RGBColorSpace, rgb: RGB) -> Self {
        let m = rgb.r.max(rgb.g).max(rgb.b);

        let scale = 2.0 * m;
        let rsp = cs.to_rgb_coeffs(if scale != 0.0 {
            rgb / scale
        } else {
            RGB::new(0.0, 0.0, 0.0)
        });

        Self {
            scale,
            rsp,
            illuminant: &cs.illuminant,
        }
    }
}

impl Spectrum for RgbIlluminantSpectrum {
    fn at(&self, lambda: Float) -> Float {
        self.scale * self.rsp.at(lambda) * self.illuminant.at(lambda)
    }

    fn max_value(&self) -> Float {
        self.scale * self.rsp.max_value() * self.illuminant.max_value()
    }

    fn to_photometric(&self) -> Float {
        self.illuminant.to_photometric()
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;

    use crate::color::SRGB;

    use super::*;

    #[test]
    fn rgb_illuminant_spectrum_photometric() {
        let spec = RgbIlluminantSpectrum::new(&SRGB, RGB::new(0.4, 0.45, 0.5));
        assert_relative_eq!(spec.at(360.0), 0.25344774);
        assert_relative_eq!(spec.at(394.0), 0.34775504);
        assert_relative_eq!(spec.at(513.0), 0.5024529);
        assert_relative_eq!(spec.at(644.0), 0.33175462);
        assert_relative_eq!(spec.at(788.0), 0.2123907);
        assert_relative_eq!(spec.at(830.0), 0.18764316);
    }

    #[test]
    fn test_visible_wavelengths_pdf() {
        assert_relative_eq!(visible_wavelengths_pdf(529.2491), 0.0039242054);
        assert_relative_eq!(visible_wavelengths_pdf(620.24963), 0.0028269484);
        assert_relative_eq!(visible_wavelengths_pdf(760.7133), 0.0005891933);
        assert_relative_eq!(visible_wavelengths_pdf(394.56876), 0.0015735145);
    }
}
