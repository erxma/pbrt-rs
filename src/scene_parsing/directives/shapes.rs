use winnow::{
    ascii::{alpha1, space1},
    combinator::{cut_err, delimited, fail, terminated, trace},
    dispatch,
    error::StrContext,
    prelude::*,
};

use crate::{core::Float, scene_parsing::common::Alpha};

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
    trace(
        "shape_directive",
        dispatch! { cut_err(terminated(delimited('"', alpha1, '"'), space1));
            "sphere" => sphere_params,
            _=> fail.context(StrContext::Label("shape type"))
        },
    )
    .parse_next(input)
}

fn sphere_params(input: &mut &str) -> PResult<Shape> {
    /*
    let params = |input: &mut &str| {
        let items = expected_params_map(vec![
            "float alpha",
            //"texture alpha",
            "float radius",
            "float zmin",
            "float zmax",
            "float phimax",
        ]);
        let found_params = param_map(items).parse_next(input)?;
        let mut sphere = Sphere::default();
        for (k, v) in found_params {
            match k.as_str() {
                "alpha" => {
                    sphere.alpha = Alpha::Constant(*v.as_single().unwrap().as_float().unwrap());
                }
                "radius" => {
                    sphere.radius = *v.as_single().unwrap().as_float().unwrap();
                    sphere.z_min.get_or_insert(-sphere.radius);
                    sphere.z_max.get_or_insert(sphere.radius);
                }
                "zmin" => {
                    sphere.z_min = Some(*v.as_single().unwrap().as_float().unwrap());
                }
                "zmax" => {
                    sphere.z_max = Some(*v.as_single().unwrap().as_float().unwrap());
                }
                "phimax" => {
                    sphere.phi_max = *v.as_single().unwrap().as_float().unwrap();
                }
                _ => unreachable!(),
            }
        }

        Ok(Shape::Sphere(sphere))
    };

    trace("sphere_params", params).parse_next(input)
    */
    todo!()
}
