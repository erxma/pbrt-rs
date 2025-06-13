use enum_as_inner::EnumAsInner;
use winnow::{
    ascii::{alpha1, alphanumeric1, multispace1},
    combinator::{delimited, seq, trace},
    PResult, Parser as _,
};

use crate::{
    color::RGB,
    core::Float,
    scene_parsing::common::{param_map, ParameterMap, ParseContext, PbrtParseError, Spectrum},
};

#[derive(Clone, Debug, derive_more::From, EnumAsInner)]
pub enum TextureDesc {
    Float(FloatTextureDesc),
    Spectrum(SpectrumTextureDesc),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextureDirective<'a> {
    pub name: &'a str,
    pub subtype: &'a str,
    pub class: &'a str,
    pub param_map: ParameterMap,
}

pub fn texture_directive<'a>(input: &mut &'a str) -> PResult<TextureDirective<'a>> {
    trace(
        "texture_directive",
        seq! { TextureDirective {
            _: ("Texture", multispace1),
            name:delimited('"', alphanumeric1, '"'),
            _: multispace1,
            subtype: delimited('"', alpha1, '"'),
            _: multispace1,
            class: delimited('"', alpha1, '"'),
            _: multispace1,
            param_map: param_map
        }},
    )
    .parse_next(input)
}

impl TextureDesc {
    pub fn from_directive(
        directive: TextureDirective,
        ctx: &ParseContext,
    ) -> Result<(String, Self), PbrtParseError> {
        let name = directive.name.to_string();
        let texture = match directive.class {
            "float" => FloatTextureDesc::from_directive(directive, ctx).map(Self::from)?,
            "spectrum" => SpectrumTextureDesc::from_directive(directive, ctx).map(Self::from)?,
            invalid_type => {
                return Err(PbrtParseError::UnrecognizedVariant {
                    entity: "Texture".to_string(),
                    variant_name: invalid_type.to_owned(),
                });
            }
        };

        Ok((name, texture))
    }
}

#[derive(Clone, Debug, derive_more::From)]
pub enum FloatTextureDesc {
    Constant(ConstantFloatTexture),
}

impl FloatTextureDesc {
    pub fn from_directive(
        directive: TextureDirective,
        _ctx: &ParseContext,
    ) -> Result<Self, PbrtParseError> {
        let texture = match directive.subtype {
            "constant" => {
                ConstantFloatTexture::from_directive(directive).map(FloatTextureDesc::Constant)?
            }
            invalid_type => {
                return Err(PbrtParseError::UnrecognizedVariant {
                    entity: "Float Texture".to_string(),
                    variant_name: invalid_type.to_owned(),
                });
            }
        };

        Ok(texture)
    }
}

#[derive(Clone, Debug, derive_more::From)]
pub enum SpectrumTextureDesc {
    Constant(ConstantSpectrumTexture),
}

impl SpectrumTextureDesc {
    pub fn from_directive(
        directive: TextureDirective,
        _ctx: &ParseContext,
    ) -> Result<Self, PbrtParseError> {
        let texture = match directive.subtype {
            "constant" => ConstantSpectrumTexture::from_directive(directive)
                .map(SpectrumTextureDesc::Constant)?,
            invalid_type => {
                return Err(PbrtParseError::UnrecognizedVariant {
                    entity: "Spectrum Texture".to_string(),
                    variant_name: invalid_type.to_owned(),
                });
            }
        };

        Ok(texture)
    }
}

macro_rules! struct_from_param_map {
    (
        $struct_name:ty,
        $(
            required {
                $(
                    $required_name:literal => $required_field:ident
                ),* $(,)?
            }
        )?
        $(
            has_defaults {
                $(
                    $defaulted_name:literal => $defaulted_field:ident
                ),* $(,)?
            }
        )?
        $(
            has_defaults_textures {
                $(
                    $defaulted_texture_name:literal => $defaulted_texture_field:ident
                ),* $(,)?
            }
        )?
    ) => {
        impl $struct_name {
            #[allow(unused_variables)]
            fn from_directive(
                mut directive: TextureDirective,
            ) -> Result<Self, PbrtParseError> {
                let mut result = <$struct_name>::default();

                $(
                    $(
                        if let Some(value) = directive.param_map.remove($required_name) {
                            result.$required_field = value.try_into()?;
                        } else {
                            return Err(PbrtParseError::MissingRequiredParameter($required_name.to_string()));
                        }
                    )*
                )?

                $(
                    $(
                        if let Some(value) = directive.param_map.remove($defaulted_name) {
                            result.$defaulted_field = value.try_into()?;
                        }
                    )*
                )?

                $(
                    $(
                        if let Some(value) = directive.param_map.remove($defaulted_texture_name) {
                            result.$defaulted_texture_field = ConstantTextureData::try_from_with_class(value, directive.class)?;
                        }
                    )*
                )?

                if let Some(unexpected_name) = directive.param_map.into_keys().next() {
                    return Err(PbrtParseError::UnexpectedParameter(unexpected_name));
                }

                Ok(result)
            }
        }
    };
}

#[derive(Clone, Debug)]
pub struct ConstantFloatTexture {
    pub value: Float,
}

impl Default for ConstantFloatTexture {
    fn default() -> Self {
        Self { value: 1.0 }
    }
}

struct_from_param_map! {
    ConstantFloatTexture,
    has_defaults {
        "value" => value
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantSpectrumTexture {
    pub value: Spectrum,
}

impl Default for ConstantSpectrumTexture {
    fn default() -> Self {
        Self {
            value: Spectrum::Rgb(RGB::new(1.0, 1.0, 1.0)),
        }
    }
}

struct_from_param_map! {
    ConstantSpectrumTexture,
    has_defaults {
        "value" => value
    }
}
