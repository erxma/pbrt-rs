use std::io::Cursor;

use bytemuck::NoUninit;
use delegate::delegate;
use rand::{RngCore, SeedableRng};
use rand_pcg::Pcg32;

#[derive(Clone)]
pub struct Rng(Pcg32);

impl Rng {
    pub fn new(state: u64, inc: u64) -> Self {
        Self(Pcg32::new(state, inc))
    }

    pub fn advance(&mut self, delta: u64) {
        self.0.advance(delta)
    }
}

impl RngCore for Rng {
    delegate! {
        to self.0 {
            fn next_u32(&mut self) -> u32;
            fn next_u64(&mut self) -> u64;
            fn fill_bytes(&mut self, dest: &mut [u8]);
            fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error>;
        }
    }
}

impl SeedableRng for Rng {
    type Seed = <Pcg32 as SeedableRng>::Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        Self(Pcg32::from_seed(seed))
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
