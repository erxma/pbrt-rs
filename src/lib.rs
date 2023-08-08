pub mod camera;
pub mod geometry;
pub mod math;
pub mod media;
pub mod sampler;

// Choice of representation of floats
#[cfg(pbrt_float_as_double)]
pub type Float = f64;

#[cfg(not(pbrt_float_as_double))]
pub type Float = f32;

#[cfg(pbrt_float_as_double)]
pub const MACHINE_EPSILON: Float = f64::EPSILON * 0.5;

#[cfg(not(pbrt_float_as_double))]
pub const MACHINE_EPSILON: Float = f32::EPSILON * 0.5;
