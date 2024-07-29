use crate::{math::vec::Vec3f, Float};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OctahedralVec {
    x: u16,
    y: u16,
}

impl OctahedralVec {}

/// Encodes a value in `[-1.0, 1.0]` to the integer encoding.
fn encode(f: Float) -> u16 {
    (((f + 1.0) / 2.0).clamp(0.0, 1.0) * 65535.0).round() as u16
}

impl From<Vec3f> for OctahedralVec {
    fn from(v: Vec3f) -> Self {
        let v = v / (v.x().abs() + v.y().abs() + v.z().abs());
        if v.z() >= 0.0 {
            Self {
                x: encode(v.x()),
                y: encode(v.y()),
            }
        } else {
            // Encode with z < 0.0
            Self {
                x: encode((1.0 * v.y().abs()) * v.x().signum()),
                y: encode((1.0 * v.x().abs()) * v.y().signum()),
            }
        }
    }
}

impl From<OctahedralVec> for Vec3f {
    fn from(o: OctahedralVec) -> Self {
        let mut x = -1.0 + 2.0 * (o.x as Float / 65535.0);
        let mut y = -1.0 + 2.0 * (o.y as Float / 65535.0);
        let z = 1.0 - x.abs() - y.abs();

        // Reparameterize directions in the z < 0.0 portion of the octahedron
        if z < 0.0 {
            x = (1.0 - y.abs()) * x.signum();
            y = (1.0 - x.abs()) * y.signum();
        }

        Self::new(x, y, z)
    }
}
