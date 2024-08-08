use std::ops::{Div, DivAssign, Index, IndexMut, Mul, MulAssign};

use derive_more::{Add, Neg, Sub};

use crate::{
    math::{impl_tuple_math_ops, Point2f, Tuple},
    Float,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Add, Sub, Neg)]
pub struct XYZ {
    pub x: Float,
    pub y: Float,
    pub z: Float,
}

impl Tuple<3, Float> for XYZ {}

impl XYZ {
    pub fn new(x: Float, y: Float, z: Float) -> Self {
        Self { x, y, z }
    }

    pub fn from_xyy(xy: Point2f, Y: Option<Float>) -> XYZ {
        #![allow(non_snake_case)]

        let Y = Y.unwrap_or(1.0);
        if xy.y() == 0.0 {
            XYZ::new(0.0, 0.0, 0.0)
        } else {
            XYZ::new(xy.x() * Y / xy.y(), Y, (1.0 - xy.x() - xy.y()) * Y / xy.y())
        }
    }

    pub fn average(self) -> Float {
        (self.x + self.y + self.z) / 3.0
    }

    pub fn xy(&self) -> Point2f {
        Point2f::new(
            self.x / (self.x + self.y + self.z),
            self.y / (self.x + self.y + self.z),
        )
    }
}

impl_tuple_math_ops!(XYZ; 3; Float);

impl Mul for XYZ {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl MulAssign for XYZ {
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
    }
}

impl Div for XYZ {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = self;
        ret /= rhs;
        ret
    }
}

impl DivAssign for XYZ {
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}

impl From<[Float; 3]> for XYZ {
    fn from(arr: [Float; 3]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }
}

impl Index<usize> for XYZ {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index for XYZ must be within 0..=2"),
        }
    }
}

impl IndexMut<usize> for XYZ {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index for XYZ must be within 0..=2"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Add, Sub)]
pub struct RGB {
    pub r: Float,
    pub g: Float,
    pub b: Float,
}

impl RGB {
    pub const fn new(r: Float, g: Float, b: Float) -> Self {
        Self { r, g, b }
    }

    pub fn clamp_zero(self) -> Self {
        Self {
            r: self.r.max(0.0),
            g: self.g.max(0.0),
            b: self.b.max(0.0),
        }
    }
}

impl Tuple<3, Float> for RGB {}

impl_tuple_math_ops!(RGB; 3; Float);

impl From<[Float; 3]> for RGB {
    fn from(arr: [Float; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }
}

impl Index<usize> for RGB {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.r,
            1 => &self.g,
            2 => &self.b,
            _ => panic!("Index for RGB must be within 0..3"),
        }
    }
}

impl IndexMut<usize> for RGB {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.r,
            1 => &mut self.g,
            2 => &mut self.b,
            _ => panic!("Index for RGB must be within 0..3"),
        }
    }
}
