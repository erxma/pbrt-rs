use std::io::Cursor;

use bytemuck::NoUninit;
use rand::{distributions::Uniform, Rng as _, SeedableRng};
use rand_pcg::Pcg32;

use crate::{math::ONE_MINUS_EPSILON, Float};

#[derive(Clone)]
pub struct Rng {
    pcg: Pcg32,
    uniform: Uniform<Float>,
}

impl Rng {
    pub fn new(state: u64, inc: u64) -> Self {
        Self {
            pcg: Pcg32::new(state, inc),
            uniform: Uniform::new_inclusive(0.0, ONE_MINUS_EPSILON),
        }
    }

    pub fn from_seed(seed: u64) -> Self {
        Self {
            pcg: Pcg32::seed_from_u64(seed),
            uniform: Uniform::from(0.0..1.0),
        }
    }

    pub fn advance(&mut self, delta: u64) {
        self.pcg.advance(delta)
    }

    pub fn uniform_float(&mut self) -> Float {
        self.pcg.sample(self.uniform)
    }
}

impl Default for Rng {
    fn default() -> Self {
        const PCG32_DEFAULT_STATE: u64 = 0xcafef00dd15ea5e5;
        const PCG32_DEFAULT_STREAM: u64 = 0xa02bdbf7bb3c0a7;
        Self::new(PCG32_DEFAULT_STATE, PCG32_DEFAULT_STREAM)
    }
}

pub fn hash<T: NoUninit>(source: &T, seed: u32) -> u64 {
    let bytes = bytemuck::bytes_of(source);
    murmur3::murmur3_x86_128(&mut Cursor::new(bytes), seed).unwrap() as u64
}
