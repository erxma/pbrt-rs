use std::ops::{Add, Div, Mul, Neg, Sub};

use num_traits::Num;

pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

impl<T: Num> Complex<T> {
    pub fn new(real: T, imaginary: T) -> Self {
        Self {
            re: real,
            im: imaginary,
        }
    }

    pub fn real(val: T) -> Self {
        Self {
            re: val,
            im: T::zero(),
        }
    }
}

impl<T: Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl<T: Num> Add for Complex<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<T: Num> Sub for Complex<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl<T: Num + Copy> Mul for Complex<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<T: num_traits::Float> Div for Complex<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let scale = T::one() / (rhs.re * rhs.re + rhs.im * rhs.im);
        Self {
            re: scale * (self.re * rhs.re + self.im * rhs.im),
            im: scale * (self.im * rhs.re - self.re * rhs.im),
        }
    }
}

impl<T: Num> Add<T> for Complex<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        self + Self::real(rhs)
    }
}

impl<T: Num> Sub<T> for Complex<T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        self - Self::real(rhs)
    }
}

impl<T: Num + Copy> Mul<T> for Complex<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        self * Self::real(rhs)
    }
}

impl<T: num_traits::Float> Div<T> for Complex<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        self / Self::real(rhs)
    }
}
