use crate::{
    camera::CameraSample,
    core::{Float, Point2f, Point2i, Vec2f},
    imaging::Filter,
    util::rng::{self, Rng},
};
use enum_dispatch::enum_dispatch;
use thiserror::Error;

#[enum_dispatch]
#[derive(Clone)]
pub enum SamplerEnum {
    Independent(IndependentSampler),
}

#[enum_dispatch(SamplerEnum)]
pub trait Sampler: Send + Sync {
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

    fn get_camera_sample(&mut self, p_pixel: Point2i, filter: &impl Filter) -> CameraSample {
        let fs = filter.sample(self.get_pixel_2d());

        // Map pixel coord (discrete) to image sample pos (continuous) by shifting 0.5,
        // which is the coordinate mapping being used.
        let p_film = p_pixel.as_point2f() + fs.p + Vec2f::new(0.5, 0.5);
        let time = self.get_1d();
        let p_lens = self.get_2d();
        let filter_weight = fs.weight;
        CameraSample {
            p_film,
            p_lens,
            time,
            filter_weight,
        }
    }
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
        let rng = match seed {
            Some(seed) => Rng::from_seed(seed),
            None => Rng::default(),
        };

        let sampler_seed = seed.unwrap_or_default();

        Self {
            samples_per_pixel,
            seed: sampler_seed,
            rng,
        }
    }
}

impl Sampler for IndependentSampler {
    fn samples_per_pixel(&self) -> usize {
        self.samples_per_pixel
    }

    fn get_1d(&mut self) -> Float {
        self.rng.uniform_float()
    }

    fn get_2d(&mut self) -> Point2f {
        Point2f::new(self.rng.uniform_float(), self.rng.uniform_float())
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
