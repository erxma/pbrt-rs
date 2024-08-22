pub mod bxdf;
pub mod camera;
pub mod color;
pub mod geometry;
pub mod image;
pub mod integrators;
pub mod lights;
pub mod materials;
pub mod math;
pub mod media;
pub mod memory;
pub mod parallel;
pub mod primitives;
pub mod sampling;
pub mod scenes;
pub mod shapes;
pub mod util;

// Choice of representation of floats
#[cfg(feature = "use-f64")]
pub mod float {
    use crate::Float;
    pub use std::f64::consts::*;
    pub const MACHINE_EPSILON: Float = f64::EPSILON * 0.5;
}

#[cfg(feature = "use-f64")]
pub type Float = f64;

#[cfg(not(feature = "use-f64"))]
pub mod float {
    use crate::Float;
    pub use std::f32::consts::*;
    pub const MACHINE_EPSILON: Float = f32::EPSILON * 0.5;
}

#[cfg(not(feature = "use-f64"))]
pub type Float = f32;
