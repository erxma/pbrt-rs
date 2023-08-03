pub mod camera;
pub mod geometry;
pub mod media;
pub mod sampler;

// Choice of representation of floats
#[cfg(pbrt_float_as_double)]
pub type Float = f64;

#[cfg(not(pbrt_float_as_double))]
pub type Float = f32;
