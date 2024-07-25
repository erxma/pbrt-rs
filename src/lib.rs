pub mod camera;
pub mod color;
pub mod film;
pub mod geometry;
pub mod lights;
pub mod math;
pub mod media;
pub mod memory;
pub mod sampling;
pub mod shapes;
pub mod util;

// Choice of representation of floats
#[cfg(feature = "use-f64")]
mod f64_float {
    pub type Float = f64;
    pub const MACHINE_EPSILON: Float = f64::EPSILON * 0.5;
    pub const PI: Float = std::f64::consts::PI;
}

#[cfg(feature = "use-f64")]
pub use f64_float::*;

#[cfg(not(feature = "use-f64"))]
mod f32_float {
    pub type Float = f32;
    pub const MACHINE_EPSILON: Float = f32::EPSILON * 0.5;
    pub const PI: Float = std::f32::consts::PI;
}

#[cfg(not(feature = "use-f64"))]
pub use f32_float::*;
