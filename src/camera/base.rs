use std::{ops::Mul, sync::Arc};

use super::{film::Film, perspective::PerspectiveCamera, OrthographicCamera};
use crate::{
    geometry::{Bounds2f, Ray, RayDifferential, Transform},
    math::{lerp, Point2f, Vec3f},
    media::MediumEnum,
    sampling::spectrum::{SampledSpectrum, SampledWavelengths},
    Float,
};
use delegate::delegate;
use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub enum CameraEnum {
    Orthographic(OrthographicCamera),
    Perspective(PerspectiveCamera),
}

impl CameraEnum {
    delegate! {
       #[through(Camera)]
       to self {
           /// Compute the ray corresponding to a given image `sample`, if one exists.
           /// `self` might model dispersion in its lens, in which case `wavelengths` may
           /// be modified via `TerminateSecondary`.
           ///
           /// The returned ray (if any) will be normalized.
           pub fn generate_ray(
               &self,
               sample: CameraSample,
               wavelengths: &SampledWavelengths,
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
               wavelengths: &SampledWavelengths,
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

#[enum_dispatch(CameraEnum)]
pub trait Camera {
    fn generate_ray(
        &self,
        sample: CameraSample,
        wavelengths: &SampledWavelengths,
    ) -> Option<CameraRay>;

    fn generate_ray_differential(
        &self,
        sample: CameraSample,
        wavelengths: &SampledWavelengths,
    ) -> Option<CameraRayDifferential>;

    fn film(&self) -> &Film;

    fn sample_time(&self, u: Float) -> Float {
        lerp(u, self.shutter_open(), self.shutter_close())
    }

    fn camera_transform(&self) -> &CameraTransform;

    fn shutter_open(&self) -> Float;
    fn shutter_close(&self) -> Float;
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

impl<'a> CameraRay<'a> {
    pub fn new(ray: Ray<'a>) -> Self {
        Self::with_weight(ray, SampledSpectrum::with_single_value(1.0))
    }

    pub fn with_weight(ray: Ray<'a>, weight: SampledSpectrum) -> Self {
        Self { ray, weight }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CameraRayDifferential<'a> {
    pub ray: RayDifferential<'a>,
    pub weight: SampledSpectrum,
}

impl<'a> CameraRayDifferential<'a> {
    pub fn new(ray: RayDifferential<'a>) -> Self {
        Self::with_weight(ray, SampledSpectrum::with_single_value(1.0))
    }

    pub fn with_weight(ray: RayDifferential<'a>, weight: SampledSpectrum) -> Self {
        Self { ray, weight }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CameraTransform {
    pub world_from_render: Transform,
}

impl CameraTransform {
    pub fn new(world_from_camera: Transform) -> Self {
        todo!()
    }

    pub fn render_from_camera<T>(&self, item: T) -> T
    where
        for<'a> &'a Transform: Mul<T, Output = T>,
    {
        &self.world_from_render * item
    }
}

#[derive(Clone, Debug)]
pub(super) struct ProjectiveCamera {
    pub transform: CameraTransform,
    pub shutter_open: Float,
    pub shutter_close: Float,
    pub film: Arc<Film>,
    pub medium: Arc<MediumEnum>,

    pub _screen_from_camera: Transform,
    pub camera_from_raster: Transform,
    pub _raster_from_screen: Transform,
    pub _screen_from_raster: Transform,
    pub lens_radius: Float,
    pub focal_distance: Float,
}

pub(super) struct ProjectiveCameraParams {
    pub transform: CameraTransform,
    pub shutter_open: Float,
    pub shutter_close: Float,
    pub film: Arc<Film>,
    pub medium: Arc<MediumEnum>,

    pub screen_from_camera: Transform,
    pub screen_window: Bounds2f,
    pub lens_radius: Float,
    pub focal_distance: Float,
}

impl ProjectiveCamera {
    pub(super) fn new(params: ProjectiveCameraParams) -> Self {
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

        Self {
            transform: params.transform,
            shutter_open: params.shutter_open,
            shutter_close: params.shutter_close,
            film: params.film,
            medium: params.medium,

            _screen_from_camera: params.screen_from_camera,
            camera_from_raster,
            _raster_from_screen: raster_from_screen,
            _screen_from_raster: screen_from_raster,
            lens_radius: params.lens_radius,
            focal_distance: params.focal_distance,
        }
    }
}
