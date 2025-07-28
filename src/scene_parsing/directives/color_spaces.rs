use crate::scene_parsing::common::{EntityDirective, FromEntity, GraphicsState, PbrtParseError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSpaceDesc {
    Srgb,
}

impl Default for ColorSpaceDesc {
    fn default() -> Self {
        Self::Srgb
    }
}

impl FromEntity for ColorSpaceDesc {
    fn from_entity(
        entity: EntityDirective,
        _state: &GraphicsState,
    ) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "ColorSpace");

        if !entity.param_map.is_empty() {
            return Err(PbrtParseError::UnexpectedParameter(
                "ColorSpace directive doesn't expect any parameters".to_string(),
            ));
        }

        match entity.subtype {
            "srgb" => Ok(Self::Srgb),
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "ColorSpace".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}
