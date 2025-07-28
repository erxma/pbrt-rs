use std::path::PathBuf;

use crate::{
    core::{Float, Point3f, Transform},
    scene_parsing::common::{
        impl_from_entity, params_map_to_fields, EntityDirective, FromEntity, GraphicsState,
        PbrtParseError, SpectrumDesc,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum LightDesc {
    Distant(DirectionalLight),
    Infinite(InfiniteLight),
}

impl FromEntity for LightDesc {
    fn from_entity(entity: EntityDirective, state: &GraphicsState) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "LightSource");

        match entity.subtype {
            "distant" => DirectionalLight::from_entity(entity, state).map(LightDesc::Distant),
            "infinite" => InfiniteLight::from_entity(entity, state).map(LightDesc::Infinite),
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "LightSource".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectionalLight {
    pub world_from_light: Transform,
    pub illuminance: Option<Float>,
    pub scale: Float,
    pub radiance: Option<SpectrumDesc>,
    pub from: Point3f,
    pub to: Point3f,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            world_from_light: Transform::IDENTITY,
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
    CTM => world_from_light,
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
    pub radiance: Option<SpectrumDesc>,
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

#[derive(Clone, Debug, PartialEq)]
pub enum AreaLightDesc {
    Diffuse(DiffuseAreaLight),
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiffuseAreaLight {
    pub power: Option<Float>,
    pub scale: Float,
    pub emission: Option<EmissionDesc>,
    pub two_sided: bool,
}

impl FromEntity for AreaLightDesc {
    fn from_entity(entity: EntityDirective, state: &GraphicsState) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "AreaLightSource");

        match entity.subtype {
            "diffuse" => DiffuseAreaLight::from_entity(entity, state).map(Self::Diffuse),
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "AreaLightSource".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

impl Default for DiffuseAreaLight {
    fn default() -> Self {
        Self {
            power: None,
            scale: 1.0,
            emission: None,
            two_sided: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum EmissionDesc {
    ImageFile(PathBuf),
    Spectrum(SpectrumDesc),
}

impl FromEntity for DiffuseAreaLight {
    fn from_entity(
        mut entity: EntityDirective,
        _state: &GraphicsState,
    ) -> Result<Self, PbrtParseError> {
        let mut result = Self::default();

        // Emission can be specified as image file, spectrum, or none.
        entity
            .param_map
            .check_mutually_exclusive(&["filename"], &["spectrum"])?;

        result.emission = if let Some(filename) = entity.param_map.remove("filename") {
            Some(EmissionDesc::ImageFile(filename.try_into()?))
        } else if let Some(spectrum) = entity.param_map.remove("spectrum") {
            Some(EmissionDesc::Spectrum(spectrum.try_into()?))
        } else {
            None
        };

        params_map_to_fields! {
            entity.param_map => result,
            has_defaults {
                power = "power",
                scale = "scale",
                two_sided = "twosided"
            }
        }

        entity.param_map.check_no_remaining_params()?;

        Ok(result)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{color::RGB, scene_parsing::common::entity_directive};

    #[test]
    fn directional() {
        assert_eq!(
            LightDesc::from_entity(
                entity_directive(
                    &mut r#"LightSource "distant" "rgb L" [0.2 .6   0] "point3 from" [10 12 5.9]"#
                )
                .unwrap(),
                &Default::default(),
            )
            .unwrap(),
            LightDesc::Distant(DirectionalLight {
                radiance: Some(SpectrumDesc::Rgb(RGB::new(0.2, 0.6, 0.0))),
                from: Point3f::new(10.0, 12.0, 5.9),
                ..Default::default()
            })
        );
    }
}
