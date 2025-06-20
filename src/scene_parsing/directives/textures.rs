use enum_as_inner::EnumAsInner;
use winnow::{
    ascii::{alpha1, alphanumeric1, multispace1},
    combinator::{delimited, seq, trace},
    ModalResult, Parser as _,
};

use crate::{
    color::RGB,
    core::Float,
    materials::{PointTransformMapping, TextureMapping2DEnum, TextureMapping3DEnum, UvMapping},
    scene_parsing::common::{
        param_map, params_map_to_fields, GraphicsState, ParameterMap, PbrtParseError, Spectrum,
        Value,
    },
};

#[derive(Clone, Debug, PartialEq, derive_more::From, EnumAsInner)]
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

pub fn texture_directive<'a>(input: &mut &'a str) -> ModalResult<TextureDirective<'a>> {
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

/// Common trait for descriptions that can be converted from a `TextureDirective`.
pub trait FromTextureDirective {
    fn from_directive(
        directive: TextureDirective,
        state: &GraphicsState,
    ) -> Result<Self, PbrtParseError>
    where
        Self: Sized;
}

impl FromTextureDirective for TextureDesc {
    fn from_directive(
        directive: TextureDirective,
        state: &GraphicsState,
    ) -> Result<Self, PbrtParseError> {
        // Use specific function for the subtype (float/spectrum)
        match directive.subtype {
            "float" => FloatTextureDesc::from_directive(directive, state).map(Self::Float),
            "spectrum" => SpectrumTextureDesc::from_directive(directive, state).map(Self::Spectrum),
            // Unrecognized
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Texture".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

/// Enum of the descriptions for all classes of float-type texture implementations.
#[derive(Clone, Debug, PartialEq, derive_more::From)]
pub enum FloatTextureDesc {
    Constant(ConstantFloatTexture),
    Checkerboard2D(CheckerboardFloatTexture2D),
    Checkerboard3D(CheckerboardFloatTexture3D),
    Named(String),
}

impl FromTextureDirective for FloatTextureDesc {
    fn from_directive(
        directive: TextureDirective,
        state: &GraphicsState,
    ) -> Result<Self, PbrtParseError> {
        assert_eq!(directive.subtype, "float");

        match directive.class {
            "constant" => {
                ConstantFloatTexture::from_directive(directive, state).map(Self::Constant)
            }
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Float Texture".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

impl TryFrom<Value> for FloatTextureDesc {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let texture = match value {
            // A float value is interpreted as a constant texture with that value
            Value::Float(value) => ConstantFloatTexture { value }.into(),
            Value::Texture(name) => Self::Named(name),
            _ => {
                return Err(PbrtParseError::IncorrectType {
                    expected: "float".to_string(),
                    found: value,
                })
            }
        };

        Ok(texture)
    }
}

impl TryFrom<Value> for Option<FloatTextureDesc> {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        FloatTextureDesc::try_from(value).map(Some)
    }
}

/// Enum of the descriptions for all classes of spectrum-type texture implementations.
#[derive(Clone, Debug, PartialEq, derive_more::From)]
pub enum SpectrumTextureDesc {
    Constant(ConstantSpectrumTexture),
    Checkerboard2D(CheckerboardSpectrumTexture2D),
    Checkerboard3D(CheckerboardSpectrumTexture3D),
    Named(String),
}

impl FromTextureDirective for SpectrumTextureDesc {
    fn from_directive(
        mut directive: TextureDirective,
        state: &GraphicsState,
    ) -> Result<Self, PbrtParseError> {
        assert_eq!(directive.subtype, "spectrum");

        // Use specific function conversion for the class of texture
        match directive.class {
            "constant" => {
                ConstantSpectrumTexture::from_directive(directive, state).map(Self::Constant)
            }
            "checkerboard" => {
                // Checkerboard has 2D and 3D cases, depending on parameter `dimension` (defaults to 2).
                let dimension = get_checkerboard_dimension(&mut directive)?;
                match dimension {
                    2 => CheckerboardSpectrumTexture2D::from_directive(directive, state)
                        .map(Self::Checkerboard2D),
                    3 => CheckerboardSpectrumTexture3D::from_directive(directive, state)
                        .map(Self::Checkerboard3D),
                    _ => Err(PbrtParseError::InvalidValue {
                        expected: "2 or 3".to_string(),
                        found: Value::Integer(dimension as i64),
                    }),
                }
            }
            // Unrecognized
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Spectrum Texture".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

impl TryFrom<Value> for SpectrumTextureDesc {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let texture = match value {
            // An RGB value is interpreted as a constant texture with that color
            Value::Rgb(rgb) => ConstantSpectrumTexture::with_rgb(rgb).into(),
            Value::Texture(name) => Self::Named(name),
            _ => {
                return Err(PbrtParseError::IncorrectType {
                    expected: "RGB".to_string(),
                    found: value,
                })
            }
        };

        Ok(texture)
    }
}

impl TryFrom<Value> for Option<SpectrumTextureDesc> {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        SpectrumTextureDesc::try_from(value).map(Some)
    }
}

#[derive(Clone, Debug, PartialEq)]
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
        _state: &GraphicsState,
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

const CONSTANT_FLOAT_TEXTURE_0: FloatTextureDesc =
    FloatTextureDesc::Constant(ConstantFloatTexture { value: 0.0 });
const CONSTANT_FLOAT_TEXTURE_1: FloatTextureDesc =
    FloatTextureDesc::Constant(ConstantFloatTexture { value: 1.0 });

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantSpectrumTexture {
    pub value: Spectrum,
}

impl ConstantSpectrumTexture {
    /// Create a texture of a constant RGB color.
    pub const fn with_rgb(rgb: RGB) -> Self {
        Self {
            value: Spectrum::Rgb(rgb),
        }
    }
}

impl Default for ConstantSpectrumTexture {
    fn default() -> Self {
        Self::with_rgb(RGB::new(1.0, 1.0, 1.0))
    }
}

const CONSTANT_SPECTRUM_TEXTURE_0: SpectrumTextureDesc =
    SpectrumTextureDesc::Constant(ConstantSpectrumTexture::with_rgb(RGB::new(0.0, 0.0, 0.0)));
const CONSTANT_SPECTRUM_TEXTURE_1: SpectrumTextureDesc =
    SpectrumTextureDesc::Constant(ConstantSpectrumTexture::with_rgb(RGB::new(1.0, 1.0, 1.0)));

impl FromTextureDirective for ConstantSpectrumTexture {
    fn from_directive(
        mut directive: TextureDirective,
        _state: &GraphicsState,
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

/// Helper to determine the dimension of a checkerboard texture.
fn get_checkerboard_dimension(directive: &mut TextureDirective) -> Result<usize, PbrtParseError> {
    assert_eq!(directive.class, "checkerboard");

    // Get param value for "dimension", default to 2
    directive
        .param_map
        .remove("dimension")
        .unwrap_or(Value::Integer(2))
        .try_into()
}

#[derive(Clone, Debug, PartialEq)]
pub struct CheckerboardFloatTexture2D {
    pub tex1: Box<FloatTextureDesc>,
    pub tex2: Box<FloatTextureDesc>,
    pub mapping: TextureMapping2DEnum,
}

impl Default for CheckerboardFloatTexture2D {
    fn default() -> Self {
        // Default to black tex1, white tex2, identity UV mapping
        Self {
            tex1: Box::new(CONSTANT_FLOAT_TEXTURE_0.clone()),
            tex2: Box::new(CONSTANT_FLOAT_TEXTURE_1.clone()),
            mapping: UvMapping::new(1.0, 1.0, 0.0, 0.0).into(),
        }
    }
}

impl FromTextureDirective for CheckerboardFloatTexture2D {
    fn from_directive(
        mut directive: TextureDirective,
        state: &GraphicsState,
    ) -> Result<Self, PbrtParseError> {
        let mut result = Self::default();

        params_map_to_fields! {
            directive.param_map => result,
            has_defaults {
                tex1 = "tex1" => Box::new,
                tex2 = "tex2" => Box::new,
            }
        }
        // Determine mapping based on params.
        // (Note this always overwrites Self::default)
        result.mapping = TextureMapping2DEnum::from_directive(&mut directive, state)?;

        directive.param_map.check_no_remaining_params()?;

        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CheckerboardSpectrumTexture2D {
    pub tex1: Box<SpectrumTextureDesc>,
    pub tex2: Box<SpectrumTextureDesc>,
    pub mapping: TextureMapping2DEnum,
}

impl Default for CheckerboardSpectrumTexture2D {
    fn default() -> Self {
        // Default to black tex1, white tex2, identity UV mapping
        Self {
            tex1: Box::new(CONSTANT_SPECTRUM_TEXTURE_0.clone()),
            tex2: Box::new(CONSTANT_SPECTRUM_TEXTURE_1.clone()),
            mapping: UvMapping::new(1.0, 1.0, 0.0, 0.0).into(),
        }
    }
}

impl FromTextureDirective for CheckerboardSpectrumTexture2D {
    fn from_directive(
        mut directive: TextureDirective,
        state: &GraphicsState,
    ) -> Result<Self, PbrtParseError> {
        let mut result = Self::default();

        params_map_to_fields! {
            directive.param_map => result,
            has_defaults {
                tex1 = "tex1" => Box::new,
                tex2 = "tex2" => Box::new,
            }
        }
        // Determine mapping based on params.
        // (Note this always overwrites Self::default)
        result.mapping = TextureMapping2DEnum::from_directive(&mut directive, state)?;

        directive.param_map.check_no_remaining_params()?;

        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CheckerboardFloatTexture3D {
    pub tex1: Box<FloatTextureDesc>,
    pub tex2: Box<FloatTextureDesc>,
    pub mapping: TextureMapping3DEnum,
}

impl Default for CheckerboardFloatTexture3D {
    fn default() -> Self {
        // Default to tex1 = 0.0, tex2 = 1.0, point transform mapping
        Self {
            tex1: Box::new(CONSTANT_FLOAT_TEXTURE_0.clone()),
            tex2: Box::new(CONSTANT_FLOAT_TEXTURE_1.clone()),
            mapping: PointTransformMapping.into(),
        }
    }
}

impl FromTextureDirective for CheckerboardFloatTexture3D {
    fn from_directive(
        mut directive: TextureDirective,
        _state: &GraphicsState,
    ) -> Result<Self, PbrtParseError> {
        let mut result = Self::default();

        params_map_to_fields! {
            directive.param_map => result,
            has_defaults {
                tex1 = "tex1" => Box::new,
                tex2 = "tex2" => Box::new,
            }
        }

        // There is currently only one possible 3D mapping,
        // so it's not accepted as a parameter

        directive.param_map.check_no_remaining_params()?;

        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CheckerboardSpectrumTexture3D {
    pub tex1: Box<SpectrumTextureDesc>,
    pub tex2: Box<SpectrumTextureDesc>,
    pub mapping: TextureMapping3DEnum,
}

impl Default for CheckerboardSpectrumTexture3D {
    fn default() -> Self {
        // Default to tex1 = 0.0, tex2 = 1.0, point transform mapping
        Self {
            tex1: Box::new(CONSTANT_SPECTRUM_TEXTURE_0.clone()),
            tex2: Box::new(CONSTANT_SPECTRUM_TEXTURE_1.clone()),
            mapping: PointTransformMapping.into(),
        }
    }
}

impl FromTextureDirective for CheckerboardSpectrumTexture3D {
    fn from_directive(
        mut directive: TextureDirective,
        _state: &GraphicsState,
    ) -> Result<Self, PbrtParseError> {
        let mut result = Self::default();

        params_map_to_fields! {
            directive.param_map => result,
            has_defaults {
                tex1 = "tex1" => Box::new,
                tex2 = "tex2" => Box::new,
            }
        }

        // There is currently only one possible 3D mapping,
        // so it's not accepted as a parameter

        directive.param_map.check_no_remaining_params()?;

        Ok(result)
    }
}

// Differs from standard trait method in that it needs to borrow only
impl TextureMapping2DEnum {
    fn from_directive(
        directive: &mut TextureDirective,
        _state: &GraphicsState,
    ) -> Result<Self, PbrtParseError> {
        // Get param value for string "mapping", default to "uv"
        let mapping_param = directive
            .param_map
            .remove("mapping")
            .unwrap_or(Value::String("uv".to_string()));
        let mapping_name: String = mapping_param.clone().try_into()?;

        match mapping_name.as_str() {
            "uv" => {
                let mut result = UvMapping::new(1.0, 1.0, 0.0, 0.0);
                params_map_to_fields! {
                    directive.param_map => result,
                    has_defaults {
                        su = "uscale",
                        sv = "vscale",
                        du = "udelta",
                        dv = "vdelta"
                    }
                }

                Ok(result.into())
            }
            "spherical" | "cylindrical" | "planar" => unimplemented!(),
            _ => Err(PbrtParseError::InvalidValue {
                expected: "uv, spherical, cylindrical, or planar".to_string(),
                found: mapping_param,
            }),
        }
    }
}

#[cfg(test)]
mod test {
    use maplit::{convert_args, hashmap};

    use super::*;

    #[test]
    fn spectrum_checkerboard_2d_with_uv_scale() {
        // Parse directive and check that resulting desc struct is as expected
        let directive = TextureDirective {
            name: "Texture",
            subtype: "spectrum",
            class: "checkerboard",
            param_map: convert_args!(hashmap!(
                "uscale" => Value::Float(16.0),
                "vscale" => Value::Float(16.0),
                "tex1" => Value::Rgb(RGB::new(0.1, 0.1, 0.1)),
                "tex2" => Value::Rgb(RGB::new(0.8, 0.8, 0.8)),
            ))
            .into(),
        };

        let result = TextureDesc::from_directive(directive, &GraphicsState::default()).unwrap();

        let expected_tex1 = ConstantSpectrumTexture::with_rgb(RGB::new(0.1, 0.1, 0.1)).into();
        let expected_tex2 = ConstantSpectrumTexture::with_rgb(RGB::new(0.8, 0.8, 0.8)).into();
        let expected = TextureDesc::Spectrum(SpectrumTextureDesc::Checkerboard2D(
            CheckerboardSpectrumTexture2D {
                tex1: Box::new(expected_tex1),
                tex2: Box::new(expected_tex2),
                mapping: UvMapping::new(16.0, 16.0, 0.0, 0.0).into(),
            },
        ));

        assert_eq!(result, expected);
    }
}
