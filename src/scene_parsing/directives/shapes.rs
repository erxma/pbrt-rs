use winnow::{
    ascii::{alpha1, space1},
    combinator::{cut_err, delimited, fail, terminated, trace},
    dispatch,
    error::StrContext,
    prelude::*,
};

use crate::{
    core::Float,
    scene_parsing::{
        common::{impl_try_from_parameter_map, Alpha, EntityDirective},
        PbrtParseError,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Sphere(Sphere),
}

impl<'a> TryFrom<EntityDirective<'a>> for Shape {
    type Error = PbrtParseError;

    fn try_from(entity: EntityDirective) -> Result<Self, Self::Error> {
        assert_eq!(entity.identifier, "Shape");

        match entity.subtype {
            "sphere" => Sphere::try_from(entity.param_map).map(Shape::Sphere),
            invalid_type => Err(PbrtParseError::UnrecognizedSubtype {
                entity: "Shape".to_string(),
                type_name: invalid_type.to_owned(),
            }),
        }
    }
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

impl_try_from_parameter_map! {
    Sphere,
    has_defaults {
        "alpha" => alpha,
        "radius" => radius,
        "zmin" => z_min,
        "zmax" => z_max,
        "phimax" => phi_max,
    }
}
