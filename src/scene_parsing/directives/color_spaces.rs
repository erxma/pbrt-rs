use crate::scene_parsing::{common::EntityDirective, PbrtParseError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSpace {
    Srgb,
}

impl<'a> TryFrom<EntityDirective<'a>> for ColorSpace {
    type Error = PbrtParseError;

    fn try_from(entity: EntityDirective) -> Result<Self, Self::Error> {
        assert_eq!(entity.identifier, "ColorSpace");

        if !entity.param_map.is_empty() {
            return Err(PbrtParseError::UnexpectedParameter(
                "ColorSpace directive doesn't expect any parameters".to_string(),
            ));
        }

        match entity.subtype {
            "srgb" => Ok(Self::Srgb),
            invalid_type => Err(PbrtParseError::UnrecognizedSubtype {
                entity: "ColorSpace".to_string(),
                type_name: invalid_type.to_owned(),
            }),
        }
    }
}
