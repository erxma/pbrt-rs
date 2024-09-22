use crate::{core::Float, scene_parsing::common::param_map};
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

pub fn camera_directive(input: &mut &str) -> PResult<Camera> {
    trace(
        "camera_directive",
        dispatch! { cut_err(terminated(delimited('"', alpha1, '"'), space1));
            "orthographic" => orthographic_camera_params,
            _=> fail.context(StrContext::Label("camera type"))
        },
    )
    .parse_next(input)
}

fn orthographic_camera_params(input: &mut &str) -> PResult<Camera> {
    /*
    let params = |input: &mut &str| {
        let items = expected_params_map(vec![
            "float shutteropen",
            "float shutterclose",
            "float frameaspectratio",
            "float screenwindow",
            "float lensradius",
            "float focaldistance",
        ]);
        let found_params = param_map(items).parse_next(input)?;
        let mut ortho = OrthographicCamera::default();
        for (k, v) in found_params {
            match k.as_str() {
                "shutteropen" => {
                    ortho.shutter_open = *v.as_single().unwrap().as_float().unwrap();
                }
                "shutterclose" => {
                    ortho.shutter_close = *v.as_single().unwrap().as_float().unwrap();
                }
                "frameaspectratio" => {
                    ortho.frame_aspect_ratio = Some(*v.as_single().unwrap().as_float().unwrap());
                }
                "screenwindow" => {
                    ortho.screen_window = Some(*v.as_single().unwrap().as_float().unwrap());
                }
                "lensradius" => {
                    ortho.lens_radius = *v.as_single().unwrap().as_float().unwrap();
                }
                "focaldistance" => {
                    ortho.focal_distance = *v.as_single().unwrap().as_float().unwrap();
                }
                _ => unreachable!(),
            }
        }

        Ok(Camera::Orthographic(ortho))
    };

    trace("orthographic_camera_params", params).parse_next(input)
    */
    todo!()
}

#[cfg(test)]
mod test {
    use super::*;

    /*
    #[test]
    fn test_orthographic() {
        assert_eq!(
            Ok(GlobalDirective::Camera(Camera::Orthographic(
                OrthographicCamera {
                    shutter_open: 1.2,
                    shutter_close: 2.4,
                    ..Default::default()
                }
            ))),
            global_directive(false).parse(
                &mut r#"Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4"#
            )
        );
    }
    */
}
