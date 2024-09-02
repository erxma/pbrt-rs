use std::{ops::Mul, sync::Arc};

use super::{film::Film, perspective::PerspectiveCamera, OrthographicCamera};
use crate::{
    geometry::{Bounds2f, Frame, Ray, RayDifferential, Transform},
    math::{lerp, Normal3f, Point2f, Point3f, Vec3f},
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
    fn min_pos_differential_x(&self) -> Vec3f;
    fn min_pos_differential_y(&self) -> Vec3f;
    fn min_dir_differential_x(&self) -> Vec3f;
    fn min_dir_differential_y(&self) -> Vec3f;

    fn approximate_dp_dxy(
        &self,
        p: Point3f,
        n: Normal3f,
        time: Float,
        samples_per_pixel: usize,
    ) -> (Vec3f, Vec3f) {
        // Compute tangent plane equation for ray diff intersections
        let p_cam = self.camera_transform().camera_from_render(p);
        let down_z_from_cam =
            Transform::rotate_from_to(Vec3f::from(p_cam).normalized(), Vec3f::FORWARD);
        let p_down_z = &down_z_from_cam * p_cam;
        let n_down_z = &down_z_from_cam * self.camera_transform().camera_from_render(n);
        let d = n_down_z.z() * p_down_z.z();

        // Find intersection points for approximated cam differential rays
        let x_ray = Ray::new(
            Point3f::ZERO + self.min_pos_differential_x(),
            Vec3f::FORWARD + self.min_dir_differential_x(),
            0.0,
            None,
        );
        let tx = -(n_down_z.dot(x_ray.o.into()) - d) / n_down_z.dot(x_ray.dir.into());
        let y_ray = Ray::new(
            Point3f::ZERO + self.min_pos_differential_y(),
            Vec3f::FORWARD + self.min_dir_differential_y(),
            0.0,
            None,
        );
        let ty = -(n_down_z.dot(y_ray.o.into()) - d) / n_down_z.dot(y_ray.dir.into());
        let px = x_ray.at(tx);
        let py = y_ray.at(ty);

        // Estimate dp/dx and dp/dy in tangent plane at intersection point
        let spp_scale = (1.0 / (samples_per_pixel as Float).sqrt()).max(0.125);
        let dpdx = spp_scale
            * self
                .camera_transform()
                .render_from_camera(down_z_from_cam.inverse() * (px - p_down_z));
        let dpdy = spp_scale
            * self
                .camera_transform()
                .render_from_camera(down_z_from_cam.inverse() * (py - p_down_z));

        (dpdx, dpdy)
    }
}

pub(super) fn find_minimum_differentials(cam: &impl Camera) -> (Vec3f, Vec3f, Vec3f, Vec3f) {
    let mut min_pos_differential_x = Vec3f::INFINITY;
    let mut min_pos_differential_y = Vec3f::INFINITY;
    let mut min_dir_differential_x = Vec3f::INFINITY;
    let mut min_dir_differential_y = Vec3f::INFINITY;

    let wavelengths = SampledWavelengths::sample_visible(0.5);

    const N: usize = 512;
    for i in 0..N {
        let p_film = Point2f::new(
            i as Float / (N - 1) as Float * cam.film().full_resolution().x() as Float,
            i as Float / (N - 1) as Float * cam.film().full_resolution().y() as Float,
        );
        let sample = CameraSample {
            p_film,
            p_lens: Point2f::new(0.5, 0.5),
            time: 0.5,
            filter_weight: 1.0,
        };

        let cam_ray_diff = cam.generate_ray_differential(sample, &wavelengths);
        if cam_ray_diff.is_none() {
            break;
        }
        let cam_ray_diff = cam_ray_diff.unwrap();

        let ray = cam_ray_diff.ray.ray;
        let diffs = cam_ray_diff.ray.differentials.unwrap();

        let diff_o_x = cam
            .camera_transform()
            .camera_from_render(diffs.rx_origin - ray.o);
        if diff_o_x.length_squared() < min_pos_differential_x.length_squared() {
            min_pos_differential_x = diff_o_x;
        }
        let diff_o_y = cam
            .camera_transform()
            .camera_from_render(diffs.ry_origin - ray.o);
        if diff_o_y.length_squared() < min_pos_differential_y.length_squared() {
            min_pos_differential_y = diff_o_y;
        }

        let dir = ray.dir.normalized();
        let rx_dir = diffs.rx_dir.normalized();
        let ry_dir = diffs.ry_dir.normalized();

        let frame = Frame::from_z(dir);
        let dir_f = frame.to_local(dir);
        let dx_f = frame.to_local(rx_dir).normalized();
        let dy_f = frame.to_local(ry_dir).normalized();

        if (dx_f - dir_f).length_squared() < min_dir_differential_x.length_squared() {
            min_dir_differential_x = dx_f - dir_f;
        }
        if (dy_f - dir_f).length_squared() < min_dir_differential_y.length_squared() {
            min_dir_differential_y = dy_f - dir_f;
        }
    }

    (
        min_pos_differential_x,
        min_pos_differential_y,
        min_dir_differential_x,
        min_dir_differential_y,
    )
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
    pub render_from_camera: Transform,
    pub world_from_render: Transform,
}

impl CameraTransform {
    pub fn new(world_from_camera: Transform) -> Self {
        // Compute world from render
        let p_camera = &world_from_camera * Point3f::ZERO;
        let world_from_render = Transform::translate(p_camera.into());

        // Compute render from camera
        let render_from_world = world_from_render.inverse();
        let render_from_camera = render_from_world * world_from_camera;

        Self {
            render_from_camera,
            world_from_render,
        }
    }

    pub fn render_from_camera<T>(&self, item: T) -> T
    where
        for<'a> &'a Transform: Mul<T, Output = T>,
    {
        &self.render_from_camera * item
    }

    pub fn camera_from_render<T>(&self, item: T) -> T
    where
        for<'a> &'a Transform: Mul<T, Output = T>,
    {
        &self.render_from_camera.inverse() * item
    }

    pub fn world_from_render<T>(&self, item: T) -> T
    where
        for<'a> &'a Transform: Mul<T, Output = T>,
    {
        &self.world_from_render * item
    }

    pub fn render_from_world<T>(&self, item: T) -> T
    where
        for<'a> &'a Transform: Mul<T, Output = T>,
    {
        &self.world_from_render.inverse() * item
    }
}

#[derive(Clone, Debug)]
pub(super) struct ProjectiveCamera {
    pub transform: CameraTransform,
    pub shutter_open: Float,
    pub shutter_close: Float,
    pub film: Arc<Film>,
    pub medium: Option<Arc<MediumEnum>>,
    pub min_pos_differential_x: Vec3f,
    pub min_pos_differential_y: Vec3f,
    pub min_dir_differential_x: Vec3f,
    pub min_dir_differential_y: Vec3f,

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
    pub medium: Option<Arc<MediumEnum>>,

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
            min_pos_differential_x: Default::default(),
            min_pos_differential_y: Default::default(),
            min_dir_differential_x: Default::default(),
            min_dir_differential_y: Default::default(),

            _screen_from_camera: params.screen_from_camera,
            camera_from_raster,
            _raster_from_screen: raster_from_screen,
            _screen_from_raster: screen_from_raster,
            lens_radius: params.lens_radius,
            focal_distance: params.focal_distance,
        }
    }
}
