use crate::{
    color::RGB,
    scene_parsing::{
        common::{
            impl_from_entity, params_map_to_fields, EntityDirective, FromEntity, ParseContext,
            PbrtParseError, Spectrum,
        },
        directives::{
            textures::{ConstantFloatTexture, ConstantSpectrumTexture},
            FloatTextureDesc, SpectrumTextureDesc,
        },
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum MaterialDesc {
    Diffuse(DiffuseMaterial),
    Dielectric(DielectricMaterial),
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
    pub reflectance: SpectrumTextureDesc,
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

#[derive(Clone, Debug, PartialEq)]
pub struct DielectricMaterial {
    pub roughness: Option<FloatTextureDesc>,
    pub u_roughness: Option<FloatTextureDesc>,
    pub v_roughness: Option<FloatTextureDesc>,
    pub remap_roughness: bool,
    pub eta: Spectrum,
}

impl Default for DielectricMaterial {
    fn default() -> Self {
        Self {
            roughness: Some(ConstantFloatTexture { value: 0.0 }.into()),
            u_roughness: Some(ConstantFloatTexture { value: 0.0 }.into()),
            v_roughness: Some(ConstantFloatTexture { value: 0.0 }.into()),
            remap_roughness: true,
            eta: Spectrum::Constant(1.5),
        }
    }
}

impl FromEntity for DielectricMaterial {
    fn from_entity(
        mut entity: EntityDirective,
        _ctx: &ParseContext,
    ) -> Result<Self, PbrtParseError> {
        let mut result = Self::default();

        entity
            .param_map
            .check_mutually_exclusive(&["roughness"], &["uroughness", "vroughness"])?;

        params_map_to_fields! {
            entity.param_map => result,
            has_defaults {
                roughness = "roughness",
                u_roughness = "uroughness",
                v_roughness = "vroughness",
                eta = "eta",
            }
        }

        if entity.param_map.contains_key("roughness") {
            result.u_roughness = None;
            result.v_roughness = None;
        } else {
            result.roughness = None;
        }

        entity.param_map.check_no_remaining_params()?;

        Ok(result)
    }
}
