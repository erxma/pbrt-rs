use std::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use crate::Float;

use super::float_utility::{next_float_down, next_float_up};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    low: Float,
    high: Float,
}

impl Interval {
    pub const fn new(low: Float, high: Float) -> Self {
        Self { low, high }
    }

    pub const fn new_exact(val: Float) -> Self {
        Self::new(val, val)
    }

    pub fn new_with_err(val: Float, err: Float) -> Self {
        if err == 0.0 {
            Self::new_exact(val)
        } else {
            Self::new(next_float_down(val - err), next_float_up(val + err))
        }
    }

    pub fn lower_bound(&self) -> Float {
        self.low
    }

    pub fn upper_bound(&self) -> Float {
        self.high
    }

    pub fn midpoint(&self) -> Float {
        (self.low + self.high) / 2.0
    }

    pub fn width(&self) -> Float {
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

    pub fn min(self, other: Self) -> Float {
        self.low.min(other.low)
    }

    pub fn max(self, other: Self) -> Float {
        self.high.max(other.high)
    }

    pub fn floor(self) -> Float {
        self.low.floor()
    }

    pub fn ceil(self) -> Float {
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

        let low = next_float_down(prods.into_iter().reduce(Float::min).unwrap());
        let high = next_float_up(prods.into_iter().reduce(Float::max).unwrap());

        Self { low, high }
    }
}

impl Mul<Float> for Interval {
    type Output = Self;

    fn mul(self, f: Float) -> Self {
        if f > 0.0 {
            Self::new(next_float_down(self.low * f), next_float_up(self.high * f))
        } else {
            Self::new(next_float_down(self.high * f), next_float_up(self.low * f))
        }
    }
}

impl Mul<Interval> for Float {
    type Output = Interval;

    fn mul(self, i: Interval) -> Self::Output {
        if self > 0.0 {
            Interval::new(next_float_down(self * i.low), next_float_up(self * i.high))
        } else {
            Interval::new(next_float_down(self * i.high), next_float_up(self * i.low))
        }
    }
}

impl Add<Float> for Interval {
    type Output = Interval;

    fn add(self, f: Float) -> Self::Output {
        self + Interval::new_exact(f)
    }
}

impl Add<Interval> for Float {
    type Output = Interval;

    fn add(self, i: Interval) -> Self::Output {
        Interval::new_exact(self) + i
    }
}

impl Div for Interval {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let low;
        let high;
        if rhs.low < 0.0 && rhs.high > 0.0 {
            // Divisor straddles zero...so return interval of everything
            low = Float::NEG_INFINITY;
            high = Float::INFINITY;
        } else {
            let quots = [
                self.low / rhs.low,
                self.low / rhs.high,
                self.high / rhs.low,
                self.high / rhs.high,
            ];
            low = next_float_down(quots.into_iter().reduce(Float::min).unwrap());
            high = next_float_up(quots.into_iter().reduce(Float::max).unwrap());
        }

        Self { low, high }
    }
}

impl Div<Float> for Interval {
    type Output = Self;

    fn div(self, f: Float) -> Self::Output {
        match f {
            // Divisor is negative
            ..0.0 => Self::new(next_float_down(self.high / f), next_float_up(self.low / f)),
            // Divisor is zero...
            0.0 => {
                // If both ends are same sign, they will divide to the same infinity.
                // Otherwise, it's -inf to (or) inf.
                if self.low.signum() == self.high.signum() {
                    Self::new_exact(self.low / f)
                } else {
                    Self::new(Float::NEG_INFINITY, Float::INFINITY)
                }
            }
            // Divisor is NaN, propagate
            _ if f.is_nan() => Self::new(Float::NAN, Float::NAN),
            // Divisor is positive
            _ => Self::new(next_float_down(self.low / f), next_float_up(self.high / f)),
        }
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

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(p) = f.precision() {
            write!(f, "{:.*}..{:.*}", p, self.low, p, self.high)
        } else {
            write!(f, "{}..{}", self.low, self.high)
        }
    }
}
