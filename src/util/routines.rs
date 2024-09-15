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
