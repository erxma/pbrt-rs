use enum_as_inner::EnumAsInner;
use winnow::{
    ascii::{alpha1, alphanumeric1, multispace1},
    combinator::{delimited, seq, trace},
    PResult, Parser as _,
};

use crate::{
    core::Float,
    scene_parsing::{
        common::{param_map, ParameterMap, ParseContext, Spectrum, Value},
        PbrtParseError,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum Texture {
    Constant(ConstantTexture),
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

impl Texture {
    pub fn from_directive(
        directive: TextureDirective,
        _ctx: &ParseContext,
    ) -> Result<(String, Self), PbrtParseError> {
        let name = directive.name.to_string();
        let texture = match directive.subtype {
            "constant" => ConstantTexture::from_directive(directive).map(Texture::Constant)?,
            invalid_type => {
                return Err(PbrtParseError::UnrecognizedSubtype {
                    entity: "Texture".to_string(),
                    type_name: invalid_type.to_owned(),
                });
            }
        };

        Ok((name, texture))
    }
}

#[derive(Clone, Debug, PartialEq, EnumAsInner)]
pub enum ConstantTextureData {
    Spectrum(Spectrum),
    Float(Float),
}

impl ConstantTextureData {
    fn try_from_with_class(value: Value, class: &str) -> Result<Self, PbrtParseError> {
        match class {
            "spectrum" => match value {
                Value::Rgb(_) | Value::BlackbodyTemp(_) => {
                    Ok(Self::Spectrum(Spectrum::try_from(value)?))
                }
                _ => Err(PbrtParseError::IncorrectType {
                    expected: "spectrum texture".to_string(),
                    found: value.to_string(),
                }),
            },

            "float" => value
                .into_float()
                .map_err(|found_value| PbrtParseError::IncorrectType {
                    expected: "float texture".to_string(),
                    found: found_value.to_string(),
                })
                .map(|f| Self::Float(f as Float)),

            _ => Err(PbrtParseError::IncorrectType {
                expected: "spectrum or float texture".to_string(),
                found: value.to_string(),
            }),
        }
    }
}

macro_rules! impl_texture_from_directive {
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

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantTexture {
    pub value: ConstantTextureData,
}

impl Default for ConstantTexture {
    fn default() -> Self {
        Self {
            value: ConstantTextureData::Float(1.0),
        }
    }
}

impl_texture_from_directive! {
    ConstantTexture,
    has_defaults_textures {
        "value" => value
    }
}
