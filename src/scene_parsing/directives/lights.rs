use crate::{
    core::{Float, Point3f, Transform},
    scene_parsing::common::{
        impl_from_entity, EntityDirective, FromEntity, ParseContext, PbrtParseError, Spectrum,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum Light {
    Distant(DirectionalLight),
    Infinite(InfiniteLight),
}

impl FromEntity for Light {
    fn from_entity(entity: EntityDirective, ctx: &ParseContext) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "LightSource");

        match entity.subtype {
            "distant" => DirectionalLight::from_entity(entity, ctx).map(Light::Distant),
            "infinite" => InfiniteLight::from_entity(entity, ctx).map(Light::Infinite),
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "LightSource".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectionalLight {
    pub render_from_light: Transform,
    pub illuminance: Option<Float>,
    pub scale: Float,
    pub radiance: Option<Spectrum>,
    pub from: Point3f,
    pub to: Point3f,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            render_from_light: Transform::IDENTITY,
            illuminance: None,
            scale: 1.0,
            radiance: None,
            from: Point3f::ZERO,
            to: Point3f::new(0.0, 0.0, 1.0),
        }
    }
}

impl_from_entity! {
    DirectionalLight,
    has_defaults {
        "illuminance" => illuminance,
        "scale" => scale,
        "L" => radiance,
        "from" => from,
        "to" => to,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InfiniteLight {
    pub illuminance: Option<Float>,
    pub scale: Float,
    // filename: PathBuf,
    // portal: [Point3f; 4];
    pub radiance: Option<Spectrum>,
}

impl Default for InfiniteLight {
    fn default() -> Self {
        Self {
            illuminance: None,
            scale: 1.0,
            radiance: None,
        }
    }
}

impl_from_entity! {
    InfiniteLight,
    has_defaults {
        "illuminance" => illuminance,
        "scale" => scale,
        "L" => radiance,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{color::RGB, scene_parsing::common::entity_directive};

    #[test]
    fn directional() {
        assert_eq!(
            Light::from_entity(
                entity_directive(
                    &mut r#"LightSource "distant" "rgb L" [0.2 .6   0] "point from" [10 12 5.9]"#
                )
                .unwrap(),
                &Default::default(),
            )
            .unwrap(),
            Light::Distant(DirectionalLight {
                radiance: Some(Spectrum::Rgb(RGB::new(0.2, 0.6, 0.0))),
                from: Point3f::new(10.0, 12.0, 5.9),
                ..Default::default()
            })
        );
    }
}
