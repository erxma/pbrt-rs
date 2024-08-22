use std::sync::Arc;

use derive_builder::Builder;

use super::{
    base::{ProjectiveCamera, ProjectiveCameraParams},
    film::Film,
    Camera, CameraRay, CameraRayDifferential, CameraSample, CameraTransform,
};
use crate::{
    geometry::{Bounds2f, Differentials, Ray, RayDifferential, Transform},
    math::{Point2f, Point3f, Vec3f},
    media::MediumEnum,
    sampling::spectrum::SampledWavelengths,
    util::sampling::sample_uniform_disk_concentric,
    Float,
};

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
