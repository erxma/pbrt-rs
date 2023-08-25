use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::{geometry::point2::Point2f, math::tuple::Tuple, Float};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct XYZ {
    pub x: Float,
    pub y: Float,
    pub z: Float,
}

impl Tuple<3, Float> for XYZ {
    fn from_array(vals: [Float; 3]) -> Self {
        Self::new(vals[0], vals[1], vals[2])
    }
}

impl XYZ {
    pub fn new(x: Float, y: Float, z: Float) -> Self {
        Self { x, y, z }
    }

    #[allow(non_snake_case)]
    pub fn from_xyy(xy: Point2f, Y: Option<Float>) -> XYZ {
        let Y = Y.unwrap_or(1.0);
        if xy.y == 0.0 {
            XYZ::new(0.0, 0.0, 0.0)
        } else {
            XYZ::new(xy.x * Y / xy.y, Y, (1.0 - xy.x - xy.y) * Y / xy.y)
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

impl Add for XYZ {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = self;
        ret += rhs;
        ret
    }
}

impl AddAssign for XYZ {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for XYZ {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = self;
        ret -= rhs;
        ret
    }
}

impl SubAssign for XYZ {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Mul for XYZ {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = self;
        ret *= rhs;
        ret
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

impl Mul<Float> for XYZ {
    type Output = Self;

    fn mul(self, rhs: Float) -> Self::Output {
        let mut ret = self;
        ret *= rhs;
        ret
    }
}

impl Mul<XYZ> for Float {
    type Output = XYZ;

    fn mul(self, rhs: XYZ) -> Self::Output {
        rhs * self
    }
}

impl MulAssign<Float> for XYZ {
    fn mul_assign(&mut self, rhs: Float) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl Div<Float> for XYZ {
    type Output = Self;

    fn div(self, rhs: Float) -> Self::Output {
        let mut ret = self;
        ret /= rhs;
        ret
    }
}

impl DivAssign<Float> for XYZ {
    fn div_assign(&mut self, rhs: Float) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl Neg for XYZ {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
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

impl Tuple<3, Float> for RGB {
    fn from_array(vals: [Float; 3]) -> Self {
        Self::new(vals[0], vals[1], vals[2])
    }
}

impl Index<usize> for RGB {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.r,
            1 => &self.g,
            2 => &self.b,
            _ => panic!("Index for RGB must be within 0..=2"),
        }
    }
}

impl IndexMut<usize> for RGB {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.r,
            1 => &mut self.g,
            2 => &mut self.b,
            _ => panic!("Index for RGB must be within 0..=2"),
        }
    }
}
