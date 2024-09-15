use super::Float;

#[cfg(feature = "use-f64")]
mod constants_f64 {
    use super::Float;
    pub const MACHINE_EPSILON: Float = f64::EPSILON * 0.5;
    pub const PI: Float = std::f64::consts::PI;
    pub const SQRT_2: Float = std::f64::consts::SQRT_2;
    pub const FRAC_1_PI: Float = std::f64::consts::FRAC_1_PI;
    pub const FRAC_PI_2: Float = std::f64::consts::FRAC_PI_2;
    pub const FRAC_PI_4: Float = std::f64::consts::FRAC_PI_4;
    pub const INV_4_PI: Float = 0.07957747154594767;
    pub const INV_2_PI: Float = 0.15915494309189535;
}

#[cfg(not(feature = "use-f64"))]
mod constants_f32 {
    use super::Float;
    pub const MACHINE_EPSILON: Float = f32::EPSILON * 0.5;
    pub const PI: Float = std::f32::consts::PI;
    pub const SQRT_2: Float = std::f32::consts::SQRT_2;
    pub const FRAC_1_PI: Float = std::f32::consts::FRAC_1_PI;
    pub const FRAC_PI_2: Float = std::f32::consts::FRAC_PI_2;
    pub const FRAC_PI_4: Float = std::f32::consts::FRAC_PI_4;
    pub const INV_4_PI: Float = 0.07957747;
    pub const INV_2_PI: Float = 0.15915494;
}

#[cfg(not(feature = "use-f64"))]
pub use constants_f32::*;
#[cfg(feature = "use-f64")]
pub use constants_f64::*;

pub const ONE_MINUS_EPSILON: Float = 1.0 - Float::EPSILON;
pub const SHADOW_EPSILON: Float = 0.0001;
