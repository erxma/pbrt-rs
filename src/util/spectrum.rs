use crate::Float;

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
