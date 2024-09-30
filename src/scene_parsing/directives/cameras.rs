use crate::{
    core::{Float, Transform},
    scene_parsing::{
        common::{impl_from_entity, param_map, EntityDirective, FromEntity, ParseContext},
        PbrtParseError,
    },
};

use super::Film;

#[derive(Clone, Debug, PartialEq)]
pub enum Camera {
    Orthographic(OrthographicCamera),
    Perspective(PerspectiveCamera),
}

impl Camera {
    pub fn update_with_film(&mut self, film_info: &Film) {
        match self {
            Camera::Orthographic(cam) => {
                let aspect = *cam
                    .frame_aspect_ratio
                    .get_or_insert(film_info.aspect_ratio());
                if aspect > 1.0 {
                    cam.screen_window
                        .get_or_insert([-aspect, -1.0, aspect, 1.0]);
                } else {
                    cam.screen_window
                        .get_or_insert([-1.0, -aspect, 1.0, aspect]);
                }
            }
            Camera::Perspective(cam) => {
                let aspect = *cam
                    .frame_aspect_ratio
                    .get_or_insert(film_info.aspect_ratio());
                if aspect > 1.0 {
                    cam.screen_window
                        .get_or_insert([-aspect, -1.0, aspect, 1.0]);
                } else {
                    cam.screen_window
                        .get_or_insert([-1.0, -aspect, 1.0, aspect]);
                }
            }
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::Perspective(PerspectiveCamera {
            fov_degs: 90.0,
            ..Default::default()
        })
    }
}

impl FromEntity for Camera {
    fn from_entity(entity: EntityDirective, ctx: &ParseContext) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "Camera");

        match entity.subtype {
            "orthographic" => {
                OrthographicCamera::from_entity(entity, ctx).map(Camera::Orthographic)
            }
            "perspective" => PerspectiveCamera::from_entity(entity, ctx).map(Camera::Perspective),
            "realistic" => todo!(),
            "spherical" => todo!(),
            invalid_type => Err(PbrtParseError::UnrecognizedSubtype {
                entity: "Camera".to_string(),
                type_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrthographicCamera {
    transform: Transform,
    shutter_open: Float,
    shutter_close: Float,
    frame_aspect_ratio: Option<Float>,
    screen_window: Option<[Float; 4]>,
    lens_radius: Float,
    focal_distance: Float,
}

impl Default for OrthographicCamera {
    fn default() -> Self {
        Self {
            transform: Transform::IDENTITY,
            shutter_open: 0.0,
            shutter_close: 1.0,
            frame_aspect_ratio: None,
            screen_window: None,
            lens_radius: 0.0,
            focal_distance: 10e30,
        }
    }
}

impl_from_entity! {
    OrthographicCamera,
    CTM => transform,
    has_defaults {
        "shutteropen" => shutter_open,
        "shutterclose" => shutter_close,
        "frameaspectratio" => frame_aspect_ratio,
        "screenwindow" => screen_window,
        "lensradius" => lens_radius,
        "focaldistance" => focal_distance,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PerspectiveCamera {
    transform: Transform,
    shutter_open: Float,
    shutter_close: Float,
    frame_aspect_ratio: Option<Float>,
    screen_window: Option<[Float; 4]>,
    lens_radius: Float,
    focal_distance: Float,
    fov_degs: Float,
}

impl Default for PerspectiveCamera {
    fn default() -> Self {
        Self {
            transform: Transform::IDENTITY,
            shutter_open: 0.0,
            shutter_close: 1.0,
            frame_aspect_ratio: None,
            screen_window: None,
            lens_radius: 0.0,
            focal_distance: 10e30,
            fov_degs: 90.0,
        }
    }
}

impl_from_entity! {
    PerspectiveCamera,
    CTM => transform,
    has_defaults {
        "shutteropen" => shutter_open,
        "shutterclose" => shutter_close,
        "frameaspectratio" => frame_aspect_ratio,
        "screenwindow" => screen_window,
        "lensradius" => lens_radius,
        "focaldistance" => focal_distance,
        "fov" => fov_degs
    }
}

#[cfg(test)]
mod test {
    use crate::scene_parsing::common::entity_directive;

    use super::*;

    #[test]
    fn test_orthographic() {
        assert_eq!(
            Camera::from_entity(
                entity_directive(
                    &mut r#"Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4"#
                )
                .unwrap(),
                &Default::default()
            ),
            Ok(Camera::Orthographic(OrthographicCamera {
                shutter_open: 1.2,
                shutter_close: 2.4,
                ..Default::default()
            }))
        );
    }
}
