use crate::{
    core::Float,
    scene_parsing::common::{
        impl_from_entity, Alpha, EntityDirective, FromEntity, ParseContext, PbrtParseError,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Sphere(Sphere),
}

impl FromEntity for Shape {
    fn from_entity(entity: EntityDirective, ctx: &ParseContext) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "Shape");

        match entity.subtype {
            "sphere" => Sphere::from_entity(entity, ctx).map(Shape::Sphere),
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Shape".to_string(),
                variant_name: invalid_type.to_owned(),
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

impl_from_entity! {
    Sphere,
    has_defaults {
        "alpha" => alpha,
        "radius" => radius,
        "zmin" => z_min,
        "zmax" => z_max,
        "phimax" => phi_max,
    }
}
