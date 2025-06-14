use enum_as_inner::EnumAsInner;
use winnow::{
    ascii::{alpha1, alphanumeric1, multispace1},
    combinator::{delimited, seq, trace},
    PResult, Parser as _,
};

use crate::{
    color::RGB,
    core::Float,
    scene_parsing::common::{
        param_map, params_map_to_fields, ParameterMap, ParseContext, PbrtParseError, Spectrum,
    },
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

pub trait FromTextureDirective {
    fn from_directive(
        directive: TextureDirective,
        ctx: &ParseContext,
    ) -> Result<Self, PbrtParseError>
    where
        Self: Sized;
}

impl FromTextureDirective for TextureDesc {
    fn from_directive(
        directive: TextureDirective,
        ctx: &ParseContext,
    ) -> Result<Self, PbrtParseError> {
        match directive.subtype {
            "float" => FloatTextureDesc::from_directive(directive, ctx).map(Self::Float),
            "spectrum" => SpectrumTextureDesc::from_directive(directive, ctx).map(Self::Spectrum),
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Texture".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug)]
pub enum FloatTextureDesc {
    Constant(ConstantFloatTexture),
}

impl FromTextureDirective for FloatTextureDesc {
    fn from_directive(
        directive: TextureDirective,
        ctx: &ParseContext,
    ) -> Result<Self, PbrtParseError> {
        assert_eq!(directive.subtype, "float");

        match directive.class {
            "constant" => ConstantFloatTexture::from_directive(directive, ctx).map(Self::Constant),
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Float Texture".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, derive_more::From)]
pub enum SpectrumTextureDesc {
    Constant(ConstantSpectrumTexture),
}

impl FromTextureDirective for SpectrumTextureDesc {
    fn from_directive(
        directive: TextureDirective,
        ctx: &ParseContext,
    ) -> Result<Self, PbrtParseError> {
        assert_eq!(directive.subtype, "spectrum");

        match directive.class {
            "constant" => {
                ConstantSpectrumTexture::from_directive(directive, ctx).map(Self::Constant)
            }
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Spectrum Texture".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
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

impl FromTextureDirective for ConstantFloatTexture {
    fn from_directive(
        mut directive: TextureDirective,
        _ctx: &ParseContext,
    ) -> Result<Self, PbrtParseError> {
        let mut result = Self::default();
        params_map_to_fields! {
            directive.param_map => result,
            has_defaults {
                value = "value"
            }
        }
        directive.param_map.check_no_remaining_params()?;

        Ok(result)
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

impl FromTextureDirective for ConstantSpectrumTexture {
    fn from_directive(
        mut directive: TextureDirective,
        _ctx: &ParseContext,
    ) -> Result<Self, PbrtParseError> {
        let mut result = Self::default();
        params_map_to_fields! {
            directive.param_map => result,
            has_defaults {
                value = "value"
            }
        }
        directive.param_map.check_no_remaining_params()?;

        Ok(result)
    }
}
