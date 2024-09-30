use std::{ops::Range, sync::Arc};

use super::{
    base::{
        Camera, CameraRay, CameraRayDifferential, CameraSample, CameraTransform, ProjectiveCamera,
        ProjectiveCameraParams,
    },
    film::Film,
};
use crate::{
    core::{Bounds2f, Float, Transform, Vec3f},
    media::MediumEnum,
    sampling::spectrum::SampledWavelengths,
};
use bon::bon;

#[derive(Debug)]
pub struct OrthographicCamera {
    projective: ProjectiveCamera,
    _dx_camera: Vec3f,
    _dy_camera: Vec3f,
}

#[bon]
impl OrthographicCamera {
    #[builder]
    pub fn new(
        world_from_camera: Transform,
        shutter_period: Range<Float>,
        film: Film,
        medium: Option<Arc<MediumEnum>>,

        screen_window: Bounds2f,
        lens_radius: Float,
        focal_distance: Float,
    ) -> Self {
        let screen_from_camera = Transform::orthographic(0.0, 1.0);

        let projective_params = ProjectiveCameraParams {
            transform: CameraTransform::new(world_from_camera),
            shutter_period,
            film,
            medium,
            screen_from_camera,
            screen_window,
            lens_radius,
            focal_distance,
        };
        let projective = ProjectiveCamera::new(projective_params);

        // Compute differential changes in origin for orthographic cam rays
        let dx_camera = &projective.camera_from_raster * Vec3f::new(1.0, 0.0, 0.0);
        let dy_camera = &projective.camera_from_raster * Vec3f::new(0.0, 1.0, 0.0);

        let mut result = OrthographicCamera {
            projective,
            _dx_camera: dx_camera,
            _dy_camera: dy_camera,
        };

        // Compute min differentials
        result.projective.min_pos_differential_x = dx_camera;
        result.projective.min_pos_differential_y = dy_camera;
        result.projective.min_dir_differential_x = Vec3f::ZERO;
        result.projective.min_dir_differential_y = Vec3f::ZERO;

        result
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

    fn shutter_period(&self) -> Range<Float> {
        self.projective.shutter_period.clone()
    }

    fn min_pos_differential_x(&self) -> Vec3f {
        self.projective.min_pos_differential_x
    }

    fn min_pos_differential_y(&self) -> Vec3f {
        self.projective.min_pos_differential_y
    }

    fn min_dir_differential_x(&self) -> Vec3f {
        self.projective.min_dir_differential_x
    }

    fn min_dir_differential_y(&self) -> Vec3f {
        self.projective.min_dir_differential_y
    }
}
