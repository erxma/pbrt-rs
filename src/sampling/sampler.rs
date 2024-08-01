use crate::{
    math::{Point2f, Point2i},
    util::rng::{self, Rng},
    Float,
};
use enum_dispatch::enum_dispatch;
use rand::Rng as _;
use thiserror::Error;

#[enum_dispatch]
#[derive(Clone)]
pub enum SamplerEnum {
    Independent(IndependentSampler),
}

#[enum_dispatch(SamplerEnum)]
pub trait Sampler {
    /// The number of samples to be taken per pixel, for this sampler.
    fn samples_per_pixel(&self) -> usize;

    /// Request the next single dimension of the sample point.
    fn get_1d(&mut self) -> Float;
    /// Request the next 2 dimensions of the sample point.
    fn get_2d(&mut self) -> Point2f;
    /// The 2D sample used to determine the point on the film plane that is sampled.
    fn get_pixel_2d(&mut self) -> Point2f;

    /// Begin work on a given sample at a given pixel.
    /// Errors if the index would exceed the expected `samples_per_pixel`.
    fn start_pixel_sample(
        &mut self,
        p: Point2i,
        sample_index: usize,
        dimension: usize,
    ) -> Result<(), SamplerError>;
}

#[derive(Error, Debug)]
pub enum SamplerError {
    #[error(
        "Sampler expects {expected_count} samples, but attempted to start one at index {index}"
    )]
    SampleCountExceeded { expected_count: usize, index: usize },
}

#[derive(Clone)]
pub struct IndependentSampler {
    samples_per_pixel: usize,
    seed: u64,
    rng: Rng,
}

impl IndependentSampler {
    pub fn new(samples_per_pixel: usize, seed: Option<u64>) -> Self {
        Self {
            samples_per_pixel,
            seed: seed.unwrap_or_default(),
            rng: Default::default(),
        }
    }
}

impl Sampler for IndependentSampler {
    fn samples_per_pixel(&self) -> usize {
        self.samples_per_pixel
    }

    fn get_1d(&mut self) -> Float {
        self.rng.gen()
    }

    fn get_2d(&mut self) -> Point2f {
        Point2f::new(self.rng.gen(), self.rng.gen())
    }

    fn get_pixel_2d(&mut self) -> Point2f {
        self.get_2d()
    }

    fn start_pixel_sample(
        &mut self,
        p: Point2i,
        sample_index: usize,
        dimension: usize,
    ) -> Result<(), SamplerError> {
        if sample_index < self.samples_per_pixel {
            // Reset RNG to a deterministic state based on the coords p.
            // Also advance by an offet into the sequence based on the sample index,
            // so samples start far apart.
            let inc = sample_index * 65536 + dimension;
            self.rng = Rng::new(rng::hash(&p, self.seed as u32), inc as u64);
            Ok(())
        } else {
            Err(SamplerError::SampleCountExceeded {
                expected_count: self.samples_per_pixel,
                index: sample_index,
            })
        }
    }
}
