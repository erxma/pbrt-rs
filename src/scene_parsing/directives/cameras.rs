use crate::{
    core::Float,
    scene_parsing::{
        common::{impl_try_from_parameter_map, param_map, EntityDirective},
        PbrtParseError,
    },
};
use winnow::{
    ascii::{alpha1, space1},
    combinator::{cut_err, delimited, fail, terminated, trace},
    dispatch,
    error::StrContext,
    prelude::*,
};

#[derive(Clone, Debug, PartialEq)]
pub enum Camera {
    Orthographic(OrthographicCamera),
}

impl<'a> TryFrom<EntityDirective<'a>> for Camera {
    type Error = PbrtParseError;

    fn try_from(entity: EntityDirective) -> Result<Self, Self::Error> {
        assert_eq!(entity.identifier, "Camera");

        match entity.subtype {
            "orthographic" => {
                OrthographicCamera::try_from(entity.param_map).map(Camera::Orthographic)
            }
            invalid_type => Err(PbrtParseError::UnrecognizedSubtype {
                entity: "Camera".to_string(),
                type_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrthographicCamera {
    shutter_open: Float,
    shutter_close: Float,
    frame_aspect_ratio: Option<Float>,
    screen_window: Option<Float>,
    lens_radius: Float,
    focal_distance: Float,
}

impl Default for OrthographicCamera {
    fn default() -> Self {
        Self {
            shutter_open: 0.0,
            shutter_close: 1.0,
            frame_aspect_ratio: None,
            screen_window: None,
            lens_radius: 0.0,
            focal_distance: 10e30,
        }
    }
}

impl_try_from_parameter_map! {
    OrthographicCamera,
    has_defaults {
        "shutteropen" => shutter_open,
        "shutterclose" => shutter_close,
        "frameaspectratio" => frame_aspect_ratio,
        "screenwindow" => screen_window,
        "lensradius" => lens_radius,
        "focaldistance" => focal_distance,
    }
}

#[cfg(test)]
mod test {
    use crate::scene_parsing::common::entity_directive;

    use super::*;

    #[test]
    fn test_orthographic() {
        assert_eq!(
            Camera::try_from(
                entity_directive(
                    &mut r#"Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4"#
                )
                .unwrap()
            ),
            Ok(Camera::Orthographic(OrthographicCamera {
                shutter_open: 1.2,
                shutter_close: 2.4,
                ..Default::default()
            }))
        );
    }
}
