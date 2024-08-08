pub mod camera;
pub mod color;
pub mod geometry;
pub mod image;
pub mod integrate;
pub mod lights;
pub mod materials;
pub mod math;
pub mod medium;
pub mod memory;
pub mod parallel;
pub mod primitives;
pub mod sampling;
pub mod shapes;
pub mod util;

// Choice of representation of floats
#[cfg(feature = "use-f64")]
pub mod float {
    use crate::Float;
    pub const MACHINE_EPSILON: Float = f64::EPSILON * 0.5;
    pub const PI: Float = std::f64::consts::PI;
    pub const SQRT_2: Float = std::f64::consts::SQRT_2;
}

#[cfg(feature = "use-f64")]
pub type Float = f64;

#[cfg(not(feature = "use-f64"))]
pub mod float {
    use crate::Float;
    pub const MACHINE_EPSILON: Float = f32::EPSILON * 0.5;
    pub const PI: Float = std::f32::consts::PI;
    pub const SQRT_2: Float = std::f32::consts::SQRT_2;
}

#[cfg(not(feature = "use-f64"))]
pub type Float = f32;
