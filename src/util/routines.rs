use std::hash::Hasher;

use crate::core::Float;

pub fn find_interval(size: usize, pred: impl Fn(usize) -> bool) -> Option<usize> {
    // If < 2, no valid result
    if size < 2 {
        return None;
    }

    // Perform a binary search
    let mut left = 0;
    let mut right = size - 1;

    while left < right {
        let mid = left + (right - left) / 2;

        // If pred is true, move left bound up
        if pred(mid) {
            left = mid + 1;
        } else {
            // Otherwise, move right bound down
            right = mid;
        }
    }

    if left == 0 {
        // No index satisfies pred, return 0
        Some(0)
    } else if left >= size - 1 {
        // All indices satisfy pred, return sz - 2 to stay in bounds
        Some(size - 2)
    } else {
        Some(left - 1)
    }
}

#[inline]
pub fn encode_morton_3(x: f32, y: f32, z: f32) -> u32 {
    (left_shift_3(z.to_bits()) << 2) | (left_shift_3(y.to_bits()) << 1) | left_shift_3(x.to_bits())
}

#[inline]
fn left_shift_3(mut x: u32) -> u32 {
    if x == (1 << 10) {
        x -= 1;
    }
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    x = (x | (x << 8)) & 0b00000011000000001111000000001111;
    x = (x | (x << 4)) & 0b00000011000011000011000011000011;
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;

    x
}

pub trait HasherFloat: Hasher {
    /// Like [Hasher::finish], but produces a `Float` in [0.0, 1.0) instead.
    #[inline]
    fn finish_float(&self) -> Float {
        // Quick, simple scale by `u64::MAX + 1`.
        // This approach of scaling integers to floats has its issues,
        // but they aren't really irrelevant here since the `u64`s are
        // roughly uniform over the range.
        const INV_2_64: Float = 1.0 / (u64::MAX as Float + 1.0);
        (self.finish() as Float) * INV_2_64
    }
}

impl<T: Hasher> HasherFloat for T {}

#[inline]
pub fn permutation_element(mut i: usize, n: usize, seed: usize) -> usize {
    let mut w = n - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;

    loop {
        i ^= seed;
        i *= 0xe170893d;
        i ^= seed >> 16;
        i ^= (i & w) >> 4;
        i ^= seed >> 8;
        i *= 0x0929eb3f;
        i ^= seed >> 23;
        i ^= (i & w) >> 1;
        i *= 1 | seed >> 27;
        i *= 0x6935fa69;
        i ^= (i & w) >> 11;
        i *= 0x74dcb303;
        i ^= (i & w) >> 2;
        i *= 0x9e501cc3;
        i ^= (i & w) >> 2;
        i *= 0xc860a3df;
        i &= w;
        i ^= i >> 5;

        if i >= n {
            break;
        }
    }

    (i + seed) % n
}
