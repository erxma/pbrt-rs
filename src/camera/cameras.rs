use std::{ops::Mul, sync::Arc};

use super::film::Film;
use crate::{
    geometry::{Bounds2f, Differentials, Ray, RayDifferential, Transform},
    math::{lerp, Point2f, Point3f, Vec3f},
    media::MediumEnum,
    sampling::spectrum::{SampledSpectrum, SampledWavelengths},
    util::sampling::sample_uniform_disk_concentric,
    Float,
};
use delegate::delegate;
use derive_builder::Builder;
use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub enum CameraEnum {
    Orthographic(OrthographicCamera),
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
    world_from_render: Transform,
}

impl CameraTransform {
    pub fn render_from_camera<T>(&self, item: T) -> T
    where
        for<'a> &'a Transform: Mul<T, Output = T>,
    {
        &self.world_from_render * item
    }
}

#[derive(Clone, Debug)]
struct ProjectiveCamera {
    transform: CameraTransform,
    shutter_open: Float,
    shutter_close: Float,
    film: Arc<Film>,
    medium: Arc<MediumEnum>,

    _screen_from_camera: Transform,
    camera_from_raster: Transform,
    _raster_from_screen: Transform,
    _screen_from_raster: Transform,
    lens_radius: Float,
    focal_distance: Float,
}

struct ProjectiveCameraParams {
    transform: CameraTransform,
    shutter_open: Float,
    shutter_close: Float,
    film: Arc<Film>,
    medium: Arc<MediumEnum>,

    screen_from_camera: Transform,
    screen_window: Bounds2f,
    lens_radius: Float,
    focal_distance: Float,
}

impl ProjectiveCamera {
    fn new(params: ProjectiveCameraParams) -> Self {
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

#[derive(Clone, Debug)]
pub struct OrthographicCamera {
    projective: ProjectiveCamera,
    _dx_camera: Vec3f,
    _dy_camera: Vec3f,
}

#[derive(Builder)]
#[builder(
    name = "OrthographicCameraBuilder",
    public,
    build_fn(private, name = "build_params")
)]
struct OrthographicCameraParams {
    transform: CameraTransform,
    shutter_open: Float,
    shutter_close: Float,
    film: Arc<Film>,
    medium: Arc<MediumEnum>,

    screen_from_camera: Transform,
    screen_window: Bounds2f,
    lens_radius: Float,
    focal_distance: Float,
}

impl OrthographicCameraBuilder {
    pub fn build(&self) -> Result<OrthographicCamera, OrthographicCameraBuilderError> {
        let params = self.build_params()?;

        let projective_params = ProjectiveCameraParams {
            transform: params.transform,
            shutter_open: params.shutter_open,
            shutter_close: params.shutter_close,
            film: params.film,
            medium: params.medium,
            screen_from_camera: params.screen_from_camera,
            screen_window: params.screen_window,
            lens_radius: params.lens_radius,
            focal_distance: params.focal_distance,
        };
        let projective = ProjectiveCamera::new(projective_params);

        // Compute differential changes in origin for orthographic cam rays
        let dx_camera = &projective.camera_from_raster * Vec3f::new(1.0, 0.0, 0.0);
        let dy_camera = &projective.camera_from_raster * Vec3f::new(0.0, 1.0, 0.0);

        Ok(OrthographicCamera {
            projective,
            _dx_camera: dx_camera,
            _dy_camera: dy_camera,
        })
    }
}

impl OrthographicCamera {
    pub fn builder() -> OrthographicCameraBuilder {
        OrthographicCameraBuilder::create_empty()
    }
}

impl Camera for OrthographicCamera {
    fn generate_ray(
        &self,
        _sample: CameraSample,
        _wavelengths: &SampledWavelengths,
    ) -> Option<CameraRay> {
        todo!()
    }

    fn generate_ray_differential(
        &self,
        _sample: CameraSample,
        _wavelengths: &SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        todo!()
    }

    fn film(&self) -> &Film {
        &self.projective.film
    }

    fn camera_transform(&self) -> &CameraTransform {
        &self.projective.transform
    }

    fn shutter_open(&self) -> Float {
        self.projective.shutter_open
    }

    fn shutter_close(&self) -> Float {
        self.projective.shutter_close
    }
}

#[derive(Clone, Debug)]
pub struct PerspectiveCamera {
    projective: ProjectiveCamera,
    dx_camera: Vec3f,
    dy_camera: Vec3f,
    cos_total_width: Float,
}

#[derive(Builder)]
#[builder(
    name = "PerspectiveCameraBuilder",
    public,
    build_fn(private, name = "build_params")
)]
struct PerspectiveCameraParams {
    // Camera base fields
    transform: CameraTransform,
    shutter_open: Float,
    shutter_close: Float,
    film: Arc<Film>,
    medium: Arc<MediumEnum>,

    // For setting ProjectiveCamera fields
    fov: Float,
    screen_window: Bounds2f,
    lens_radius: Float,
    focal_distance: Float,
}

impl PerspectiveCameraBuilder {
    pub fn build(&self) -> Result<PerspectiveCamera, PerspectiveCameraBuilderError> {
        let params = self.build_params()?;

        // Perspective transform
        let screen_from_camera = Transform::perspective(params.fov, 0.01, 1000.0);

        // Build projective base
        let projective_params = ProjectiveCameraParams {
            transform: params.transform,
            shutter_open: params.shutter_open,
            shutter_close: params.shutter_close,
            film: params.film,
            medium: params.medium,
            screen_from_camera,
            screen_window: params.screen_window,
            lens_radius: params.lens_radius,
            focal_distance: params.focal_distance,
        };
        let projective = ProjectiveCamera::new(projective_params);

        // Compute differential changes in origin for cam rays
        let dx_camera = &projective.camera_from_raster * Vec3f::new(1.0, 0.0, 0.0)
            - &projective.camera_from_raster * Vec3f::new(0.0, 0.0, 0.0);
        let dy_camera = &projective.camera_from_raster * Vec3f::new(0.0, 1.0, 0.0)
            - &projective.camera_from_raster * Vec3f::new(0.0, 0.0, 0.0);

        // Compute cosine of maximum view angle
        let radius: Point2f = projective.film.filter().radius().into();
        let p_corner = Point3f::new(-radius.x(), -radius.y(), 0.0);
        let w_corner_camera = Vec3f::from(&projective.camera_from_raster * p_corner).normalized();
        let cos_total_width = w_corner_camera.z();

        Ok(PerspectiveCamera {
            projective,
            dx_camera,
            dy_camera,
            cos_total_width,
        })
    }
}

impl PerspectiveCamera {
    pub fn builder() -> PerspectiveCameraBuilder {
        Default::default()
    }
}

impl Camera for PerspectiveCamera {
    fn generate_ray(
        &self,
        sample: CameraSample,
        _wavelengths: &SampledWavelengths,
    ) -> Option<CameraRay> {
        // Compute raster and camera sample positions
        let p_film = Point3f::new(sample.p_film.x(), sample.p_film.y(), 0.0);
        let p_camera = &self.projective.camera_from_raster * p_film;

        // All rays originate from camera origin
        let o = Point3f::ZERO;
        // Direction is a vec from origin to point on near plane
        let dir = Vec3f::from(p_camera).normalized();

        let mut ray = Ray::new(
            o,
            dir,
            self.sample_time(sample.time),
            Some(&self.projective.medium),
        );

        // Modify ray for depth of field
        if self.projective.lens_radius > 0.0 {
            // Sample point on lens
            let p_lens =
                self.projective.lens_radius * sample_uniform_disk_concentric(sample.p_lens);
            // Compute point on plane of focus
            let t_focus = self.projective.focal_distance / ray.dir.z();
            let p_focus = ray.at(t_focus);
            // Update ray for effect of lens
            ray.o = Point3f::new(p_lens.x(), p_lens.y(), 0.0);
            ray.dir = (p_focus - ray.o).normalized();
        }

        let camera_ray = CameraRay::new(self.camera_transform().render_from_camera(ray));

        Some(camera_ray)
    }

    fn generate_ray_differential(
        &self,
        sample: CameraSample,
        _wavelengths: &SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        // Compute raster and camera sample positions
        let p_film = Point3f::new(sample.p_film.x(), sample.p_film.y(), 0.0);
        let p_camera = &self.projective.camera_from_raster * p_film;

        // All rays originate from camera origin
        let o = Point3f::ZERO;
        // Direction is a vec from origin to point on near plane
        let dir = Vec3f::from(p_camera).normalized();

        let mut ray = Ray::new(
            o,
            dir,
            self.sample_time(sample.time),
            Some(&self.projective.medium),
        );

        // Modify ray for depth of field,
        // and compute offset rays for differentials accounting for lens
        let rx_dir;
        let ry_dir;
        if self.projective.lens_radius > 0.0 {
            // Sample point on lens
            let p_lens =
                self.projective.lens_radius * sample_uniform_disk_concentric(sample.p_lens);
            // Compute point on plane of focus
            let t_focus = self.projective.focal_distance / ray.dir.z();
            let p_focus = ray.at(t_focus);
            // Update ray for effect of lens
            ray.o = Point3f::new(p_lens.x(), p_lens.y(), 0.0);
            ray.dir = (p_focus - ray.o).normalized();

            let dx = Vec3f::from(p_camera + self.dx_camera).normalized();
            let dy = Vec3f::from(p_camera + self.dy_camera).normalized();
            let dx_p_focus = Point3f::ZERO + t_focus * dx;
            let dy_p_focus = Point3f::ZERO + t_focus * dy;
            rx_dir = (dx_p_focus - ray.o).normalized();
            ry_dir = (dy_p_focus - ray.o).normalized();
        } else {
            rx_dir = Vec3f::from(p_camera + self.dx_camera).normalized();
            ry_dir = Vec3f::from(p_camera + self.dy_camera).normalized();
        }

        let differentials = Differentials {
            rx_origin: ray.o,
            ry_origin: ray.o,
            rx_dir,
            ry_dir,
        };
        let ray_diff = RayDifferential::new(ray, differentials);
        let camera_ray_diff =
            CameraRayDifferential::new(self.camera_transform().render_from_camera(ray_diff));

        Some(camera_ray_diff)
    }

    fn film(&self) -> &Film {
        &self.projective.film
    }

    fn camera_transform(&self) -> &CameraTransform {
        &self.projective.transform
    }

    fn shutter_open(&self) -> Float {
        self.projective.shutter_open
    }

    fn shutter_close(&self) -> Float {
        self.projective.shutter_close
    }
}
