use winnow::{
    ascii::{alpha1, space1},
    combinator::{cut_err, delimited, fail, terminated, trace},
    dispatch,
    error::StrContext,
    prelude::*,
};

use crate::{
    core::Float,
    scene_parsing::common::{impl_try_from_parameter_map, Alpha},
};

#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Sphere(Sphere),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sphere {
    alpha: Alpha,
    radius: Float,
    z_min: Option<Float>,
    z_max: Option<Float>,
    phi_max: Float,
}

impl Default for Sphere {
    fn default() -> Self {
        Self {
            alpha: Alpha::Constant(1.0),
            radius: 1.0,
            z_min: None,
            z_max: None,
            phi_max: 360.0,
        }
    }
}

pub fn shape_directive(input: &mut &str) -> PResult<Shape> {
    /*
    trace(
        "shape_directive",
        dispatch! { cut_err(terminated(delimited('"', alpha1, '"'), space1));
            "sphere" => sphere_params,
            _=> fail.context(StrContext::Label("shape type"))
        },
    )
    .parse_next(input)
    */
    todo!()
}
