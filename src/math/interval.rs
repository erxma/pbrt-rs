use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::Float;

use super::float_utility::{next_float_down, next_float_up};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    low: Float,
    high: Float,
}

impl Interval {
    pub fn new(val: Float) -> Self {
        Self {
            low: val,
            high: val,
        }
    }

    pub fn new_with_err(val: Float, err: Float) -> Self {
        if err == 0.0 {
            Self::new(val)
        } else {
            Self {
                low: next_float_down(val - err),
                high: next_float_up(val + err),
            }
        }
    }

    pub fn lower_bound(&self) -> f32 {
        self.low
    }

    pub fn upper_bound(&self) -> f32 {
        self.high
    }

    pub fn midpoint(&self) -> f32 {
        (self.low + self.high) * 0.5
    }

    pub fn width(&self) -> f32 {
        self.high - self.low
    }

    pub fn contains(&self, val: Float) -> bool {
        val >= self.low && val <= self.high
    }

    pub fn contains_interval(&self, other: &Self) -> bool {
        self.low <= self.high && self.high >= other.low
    }

    pub fn squared(self) -> Self {
        if self.low >= 0.0 {
            // Both positive. Just square low and high
            Self {
                low: next_float_down(self.low * self.low),
                high: next_float_up(self.high * self.high),
            }
        } else if self.high <= 0.0 {
            // Both negative, so square reverses order
            Self {
                low: next_float_down(self.high * self.high),
                high: next_float_up(self.low * self.low),
            }
        } else {
            // Straddles 0, so min is 0, max is larger abs val squared
            let larger_abs = self.low.abs().max(self.high.abs());
            Self {
                low: 0.0,
                high: next_float_up(larger_abs * larger_abs),
            }
        }
    }

    pub fn sqrt(self) -> Self {
        Self {
            low: next_float_down(self.low.sqrt()).max(0.0),
            high: next_float_up(self.high.sqrt()),
        }
    }

    pub fn abs(self) -> Self {
        if self.low > 0.0 {
            // Definitely positive, no change
            self
        } else if self.high <= 0.0 {
            // Definitely negative
            Self {
                low: -self.high,
                high: -self.low,
            }
        } else {
            // Interval straddles zero
            Self {
                low: 0.0,
                high: -self.low.max(self.high),
            }
        }
    }

    pub fn min(self, other: Self) -> f32 {
        self.low.min(other.low)
    }

    pub fn max(self, other: Self) -> f32 {
        self.high.max(other.high)
    }

    pub fn floor(self) -> f32 {
        self.low.floor()
    }

    pub fn ceil(self) -> f32 {
        self.high.ceil()
    }
}

impl Add for Interval {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            low: next_float_down(self.low + rhs.low),
            high: next_float_up(self.high + rhs.high),
        }
    }
}

impl Sub for Interval {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            low: next_float_down(self.low - rhs.high),
            high: next_float_up(self.high - rhs.low),
        }
    }
}

impl Mul for Interval {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let prods = [
            self.low * rhs.low,
            self.low * rhs.high,
            self.high * rhs.low,
            self.high * rhs.high,
        ];

        let low = next_float_down(prods.into_iter().reduce(f32::min).unwrap());
        let high = next_float_up(prods.into_iter().reduce(f32::max).unwrap());

        Self { low, high }
    }
}

impl Div for Interval {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let low;
        let high;
        if rhs.low < 0.0 && rhs.high > 0.0 {
            // Divisor straddles zero...so return interval of everything
            low = f32::NEG_INFINITY;
            high = f32::INFINITY;
        } else {
            let quots = [
                self.low / rhs.low,
                self.low / rhs.high,
                self.high / rhs.low,
                self.high / rhs.high,
            ];
            low = next_float_down(quots.into_iter().reduce(f32::min).unwrap());
            high = next_float_up(quots.into_iter().reduce(f32::max).unwrap());
        }

        Self { low, high }
    }
}

impl AddAssign for Interval {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl SubAssign for Interval {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs
    }
}

impl MulAssign for Interval {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs
    }
}

impl DivAssign for Interval {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs
    }
}

impl From<Interval> for f32 {
    fn from(interval: Interval) -> Self {
        interval.midpoint()
    }
}
