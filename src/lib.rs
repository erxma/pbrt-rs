pub mod base;
pub mod geometry;
pub mod math;
pub mod media;
pub mod sampling;
pub mod util;

// Choice of representation of floats
#[cfg(feature = "float-as-double")]
mod f64_float {
    pub type Float = f64;
    pub const MACHINE_EPSILON: Float = f64::EPSILON * 0.5;
    pub const PI: Float = std::f64::consts::PI;
}

#[cfg(feature = "float-as-double")]
pub use f64_float::*;

#[cfg(not(feature = "float-as-double"))]
mod f32_float {
    pub type Float = f32;
    pub const MACHINE_EPSILON: Float = f32::EPSILON * 0.5;
    pub const PI: Float = std::f32::consts::PI;
}

#[cfg(not(feature = "float-as-double"))]
pub use f32_float::*;
