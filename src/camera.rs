use crate::{
    film::Film,
    geometry::ray::{Ray, RayDifferential},
    math::point::Point2f,
    sampling::spectrum::{SampledSpectrum, SampledWavelengths},
    Float,
};

use self::camera_common::CameraCommon;

pub trait Camera: CameraCommon {
    /// Compute the ray corresponding to a given image `sample`, if one exists.
    /// `self` might model dispersion in its lens, in which case `wavelengths` may
    /// be modified via `TerminateSecondary`.
    ///
    /// The returned ray (if any) will be normalized.
    fn generate_ray(
        &self,
        sample: CameraSample,
        wavelengths: &mut SampledWavelengths,
    ) -> Option<CameraRay>;

    /// Compute the ray corresponding to a given image `sample`,
    /// if one exists, and the corresponding rays (differentials)
    /// for adjacent pixels.
    ///
    /// `self` might model dispersion in its lens, in which case `wavelengths` may
    /// be modified via `TerminateSecondary`.
    ///
    /// The returned ray (if any) will be normalized.
    fn generate_ray_differential(
        &self,
        sample: CameraSample,
        wavelengths: &mut SampledWavelengths,
    ) -> Option<CameraRayDifferential>;

    /// Borrow `self`'s film.
    fn film(&self) -> &dyn Film;

    /// Map a uniform random sample `u` (in range `[0,1)`)
    /// to a time when `self`'s shutter is open.
    fn sample_time(&self, u: Float) -> Float;

    /// Borrow `self`'s camera transform.
    fn camera_transform(&self) -> &CameraTransform;
}

mod camera_common {
    #[derive(Clone)]
    pub struct CameraBase {}

    pub trait CameraCommon {
        fn base(&self) -> CameraBase;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraSample {
    pub p_film: Point2f,
    pub p_lens: Point2f,
    pub time: Float,
    pub filter_weight: Float,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CameraRay<'a> {
    pub ray: Ray<'a>,
    pub weight: SampledSpectrum,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CameraRayDifferential<'a> {
    pub ray: RayDifferential<'a>,
    pub weight: SampledSpectrum,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CameraTransform {}
