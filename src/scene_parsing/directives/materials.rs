use crate::{
    color::RGB,
    scene_parsing::{
        common::{impl_from_entity, EntityDirective, FromEntity, ParseContext, PbrtParseError},
        directives::{textures::ConstantSpectrumTexture, SpectrumTextureDesc},
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum MaterialDesc {
    Diffuse(DiffuseMaterial),
}

impl FromEntity for MaterialDesc {
    fn from_entity(entity: EntityDirective, ctx: &ParseContext) -> Result<Self, PbrtParseError> {
        // Use specific function for the subtype
        match entity.subtype {
            "diffuse" => DiffuseMaterial::from_entity(entity, ctx).map(Self::Diffuse),
            // Unrecognized
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Material".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiffuseMaterial {
    reflectance: SpectrumTextureDesc,
}

impl Default for DiffuseMaterial {
    fn default() -> Self {
        Self {
            reflectance: ConstantSpectrumTexture::with_rgb(RGB::new(0.5, 0.5, 0.5)).into(),
        }
    }
}

impl_from_entity! {
    DiffuseMaterial,
    has_defaults {
        "reflectance" => reflectance
    }
}
