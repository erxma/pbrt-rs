pub mod film;
pub mod sensor;

use delegate::delegate;
use derive_builder::Builder;
use enum_dispatch::enum_dispatch;
use film::Film;

use crate::{
    geometry::{
        bounds::Bounds2f,
        ray::{Ray, RayDifferential},
        transform::Transform,
    },
    math::{Point2f, Vec3f},
    media::medium::Medium,
    sampling::spectrum::{SampledSpectrum, SampledWavelengths},
    Float,
};

#[enum_dispatch]
pub enum Camera<'a> {
    ProjectiveCamera(ProjectiveCamera<'a>),
}

impl<'a> Camera<'a> {
    delegate! {
       #[through(CameraTrait)]
       to self {
           /// Compute the ray corresponding to a given image `sample`, if one exists.
           /// `self` might model dispersion in its lens, in which case `wavelengths` may
           /// be modified via `TerminateSecondary`.
           ///
           /// The returned ray (if any) will be normalized.
           pub fn generate_ray(
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
           pub fn generate_ray_differential(
               &self,
               sample: CameraSample,
               wavelengths: &mut SampledWavelengths,
           ) -> Option<CameraRayDifferential>;

           /// Borrow `self`'s film.
           pub fn film(&self) -> &Film;

           /// Map a uniform random sample `u` (in range `[0,1)`)
           /// to a time when `self`'s shutter is open.
           pub fn sample_time(&self, u: Float) -> Float;

           /// Borrow `self`'s camera transform.
           pub fn camera_transform(&self) -> &CameraTransform;
        }
    }
}

#[enum_dispatch(Camera)]
pub trait CameraTrait {
    fn generate_ray(
        &self,
        sample: CameraSample,
        wavelengths: &mut SampledWavelengths,
    ) -> Option<CameraRay>;

    fn generate_ray_differential(
        &self,
        sample: CameraSample,
        wavelengths: &mut SampledWavelengths,
    ) -> Option<CameraRayDifferential>;

    fn film(&self) -> &Film;

    fn sample_time(&self, u: Float) -> Float;

    fn camera_transform(&self) -> &CameraTransform;
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

#[derive(Clone, Debug)]
pub struct ProjectiveCamera<'a> {
    transform: CameraTransform,
    shutter_open: Float,
    shutter_close: Float,
    film: &'a Film<'a>,
    medium: &'a Medium,

    screen_from_camera: Transform,
    camera_from_raster: Transform,
    raster_from_screen: Transform,
    screen_from_raster: Transform,
    lens_radius: Float,
    focal_distance: Float,
}

#[derive(Builder)]
#[builder(
    name = "ProjectiveCameraBuilder",
    build_fn(private, name = "build_params")
)]
pub struct ProjectiveCameraParams<'a> {
    transform: CameraTransform,
    shutter_open: Float,
    shutter_close: Float,
    film: &'a Film<'a>,
    medium: &'a Medium,

    screen_from_camera: Transform,
    screen_window: Bounds2f,
    lens_radius: Float,
    focal_distance: Float,
}

impl<'a> ProjectiveCameraBuilder<'a> {
    pub fn build(&self) -> Result<ProjectiveCamera<'_>, ProjectiveCameraBuilderError> {
        let params = self.build_params()?;

        // Compute projective camera transforms

        let ndc_from_screen = Transform::scale(
            1.0 / (params.screen_window.p_max.x() - params.screen_window.p_min.x()),
            1.0 / (params.screen_window.p_max.y() - params.screen_window.p_min.y()),
            1.0,
        ) * Transform::translate(Vec3f::new(
            -params.screen_window.p_min.x(),
            -params.screen_window.p_max.y(),
            0.0,
        ));

        let full_resolution: Point2f = params.film.full_resolution().into();
        let raster_from_ndc = Transform::scale(full_resolution.x(), -full_resolution.y(), 1.0);

        let raster_from_screen = raster_from_ndc * ndc_from_screen;
        let screen_from_raster = raster_from_screen.inverse();

        let camera_from_raster = params.screen_from_camera.inverse() * screen_from_raster.clone();

        Ok(ProjectiveCamera {
            transform: params.transform,
            shutter_open: params.shutter_open,
            shutter_close: params.shutter_close,
            film: params.film,
            medium: params.medium,

            screen_from_camera: params.screen_from_camera,
            camera_from_raster,
            raster_from_screen,
            screen_from_raster,
            lens_radius: params.lens_radius,
            focal_distance: params.focal_distance,
        })
    }
}

impl<'a> CameraTrait for ProjectiveCamera<'a> {
    fn generate_ray(
        &self,
        sample: CameraSample,
        wavelengths: &mut SampledWavelengths,
    ) -> Option<CameraRay> {
        todo!()
    }

    fn generate_ray_differential(
        &self,
        sample: CameraSample,
        wavelengths: &mut SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        todo!()
    }

    fn film(&self) -> &Film {
        &self.film
    }

    fn sample_time(&self, u: Float) -> Float {
        todo!()
    }

    fn camera_transform(&self) -> &CameraTransform {
        &self.transform
    }
}
