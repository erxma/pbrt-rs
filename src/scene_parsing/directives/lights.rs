use crate::{
    core::{Float, Point3f},
    scene_parsing::{
        common::{impl_try_from_parameter_map, EntityDirective, Spectrum},
        PbrtParseError,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum Light {
    Distant(DirectionalLight),
}

impl<'a> TryFrom<EntityDirective<'a>> for Light {
    type Error = PbrtParseError;

    fn try_from(entity: EntityDirective) -> Result<Self, Self::Error> {
        assert_eq!(entity.identifier, "LightSource");

        match entity.subtype {
            "distant" => DirectionalLight::try_from(entity.param_map).map(Light::Distant),
            invalid_type => Err(PbrtParseError::UnrecognizedSubtype {
                entity: "LightSource".to_string(),
                type_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectionalLight {
    illuminance: Option<Float>,
    scale: Float,
    radiance: Option<Spectrum>,
    from: Point3f,
    to: Point3f,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            illuminance: None,
            scale: 1.0,
            radiance: None,
            from: Point3f::ZERO,
            to: Point3f::new(0.0, 0.0, 1.0),
        }
    }
}

impl_try_from_parameter_map! {
    DirectionalLight,
    has_defaults {
        "illuminance" => illuminance,
        "scale" => scale,
        "L" => radiance,
        "from" => from,
        "to" => to,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{color::RGB, scene_parsing::common::entity_directive};

    #[test]
    fn directional() {
        assert_eq!(
            Light::try_from(
                entity_directive(
                    &mut r#"LightSource "distant" "rgb L" [0.2 .6   0] "point from" [10 12 5.9]"#
                )
                .unwrap()
            ),
            Ok(Light::Distant(DirectionalLight {
                radiance: Some(Spectrum::Rgb(RGB::new(0.2, 0.6, 0.0))),
                from: Point3f::new(10.0, 12.0, 5.9),
                ..Default::default()
            }))
        );
    }
}
