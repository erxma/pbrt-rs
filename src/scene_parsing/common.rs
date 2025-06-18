use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    str::FromStr,
};

use delegate::delegate;
use enum_as_inner::EnumAsInner;
use itertools::Itertools;
use num_traits::NumCast;
use strum::{EnumDiscriminants, EnumString};
use thiserror::Error;
use winnow::{
    ascii::{alpha1, alphanumeric1, float, multispace1, space1},
    combinator::{
        alt, cut_err, delimited, eof, fail, opt, preceded, separated, separated_pair, terminated,
        trace,
    },
    error::{AddContext, ErrMode, ErrorKind, ParserError, StrContext, StrContextValue},
    prelude::*,
    stream::Stream,
    token::take_until,
};

use crate::{
    color::RGB,
    core::{Float, Normal3f, Point2f, Point3f, Transform, Vec2f, Vec3f},
};

use super::directives::{
    texture_directive, transform_directive, TextureDirective, TransformDirective,
};

#[derive(Clone, Debug, PartialEq, EnumAsInner, strum::Display)]
pub enum Value {
    Int(Int),
    IntArray(Vec<Int>),
    Float(Float),
    FloatArray(Vec<Float>),
    Bool(bool),
    Str(String),
    Point2(Point2f),
    Vec2(Vec2f),
    Point3(Point3f),
    Vec3(Vec3f),
    Normal(Normal3f),
    Rgb(RGB),
    BlackbodyTemp(Float),
    TextureName(String),
}

pub(super) type Int = i64;

#[derive(Clone, Debug, PartialEq, EnumString, strum::Display)]
pub(super) enum ValueType {
    #[strum(serialize = "integer")]
    Int,
    #[strum(serialize = "float")]
    Float,
    #[strum(serialize = "bool")]
    Bool,
    #[strum(serialize = "string")]
    Str,
    #[strum(serialize = "point2")]
    Point2,
    #[strum(serialize = "vector2")]
    Vec2,
    #[strum(serialize = "point3")]
    Point3,
    #[strum(serialize = "vector3")]
    Vec3,
    #[strum(serialize = "normal")]
    Normal,
    #[strum(serialize = "rgb")]
    Rgb,
    #[strum(serialize = "blackbody")]
    Blackbody,
    #[strum(serialize = "texture")]
    TextureName,
}

macro_rules! impl_num_try_from_value {
    ($ty:ty, $from:ident, $from_arr:ident) => {
        impl TryFrom<Value> for $ty {
            type Error = PbrtParseError;

            fn try_from(value: Value) -> Result<Self, Self::Error> {
                match value {
                    Value::$from(val) => val.try_into().map_err(|_| value),
                    // Also allow implicit conversion from single elem array to single
                    Value::$from_arr(ref arr) if arr.len() == 1 => {
                        NumCast::from(arr[0]).ok_or(value)
                    }
                    _ => Err(value),
                }
                .map_err(|found_val| PbrtParseError::IncorrectType {
                    expected: stringify!($ty).to_string(),
                    found: found_val,
                })
            }
        }

        impl TryFrom<Value> for Option<$ty> {
            type Error = PbrtParseError;

            fn try_from(value: Value) -> Result<Self, Self::Error> {
                <$ty>::try_from(value).map(Some)
            }
        }

        impl<const N: usize> TryFrom<Value> for [$ty; N] {
            type Error = PbrtParseError;

            fn try_from(value: Value) -> Result<Self, Self::Error> {
                let incorrect_type_err = PbrtParseError::IncorrectType {
                    expected: format!("{}[{N}]", stringify!($ty)),
                    found: value.clone(),
                };

                if let Value::$from_arr(vec) = value {
                    let len = vec.len();
                    let converted: Result<Vec<_>, _> =
                        vec.into_iter().map(|int| int.try_into()).collect();

                    // Irrefutable when target type is same as source representation
                    #[allow(irrefutable_let_patterns)]
                    if let Ok(vec) = converted {
                        vec.try_into().map_err(|_| PbrtParseError::IncorrectLength {
                            expected: N,
                            found: len,
                        })
                    } else {
                        Err(incorrect_type_err)
                    }
                } else {
                    Err(incorrect_type_err)
                }
            }
        }

        impl<const N: usize> TryFrom<Value> for Option<[$ty; N]> {
            type Error = PbrtParseError;

            fn try_from(value: Value) -> Result<Self, Self::Error> {
                <[$ty; N]>::try_from(value).map(Some)
            }
        }

        impl TryFrom<Value> for Vec<$ty> {
            type Error = PbrtParseError;

            fn try_from(value: Value) -> Result<Self, Self::Error> {
                let incorrect_type_err = PbrtParseError::IncorrectType {
                    expected: format!("{}[]", stringify!($ty)),
                    found: value.clone(),
                };

                if let Value::$from_arr(vec) = value {
                    let converted: Result<Vec<_>, _> =
                        vec.into_iter().map(|num| num.try_into()).collect();

                    converted.map_err(|_| incorrect_type_err)
                } else {
                    Err(incorrect_type_err)
                }
            }
        }

        impl TryFrom<Value> for Option<Vec<$ty>> {
            type Error = PbrtParseError;

            fn try_from(value: Value) -> Result<Self, Self::Error> {
                Vec::try_from(value).map(Some)
            }
        }
    };
}

impl_num_try_from_value!(usize, Int, IntArray);
impl_num_try_from_value!(u8, Int, IntArray);
impl_num_try_from_value!(u64, Int, IntArray);
impl_num_try_from_value!(Float, Float, FloatArray);

impl TryFrom<Value> for bool {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        value
            .into_bool()
            .map_err(|found_val| PbrtParseError::IncorrectType {
                expected: "bool".to_string(),
                found: found_val,
            })
    }
}

impl TryFrom<Value> for Option<bool> {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        bool::try_from(value).map(Some)
    }
}

impl TryFrom<Value> for String {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        value
            .into_str()
            .map_err(|found_val| PbrtParseError::IncorrectType {
                expected: ValueType::Str.to_string(),
                found: found_val,
            })
    }
}

impl TryFrom<Value> for Option<String> {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        String::try_from(value).map(Some)
    }
}

macro_rules! impl_tuple_try_from_value {
    ($ty:ty, $from:ident, $from_arr:ident, $n:expr) => {
        impl TryFrom<Value> for $ty {
            type Error = PbrtParseError;

            fn try_from(value: Value) -> Result<Self, Self::Error> {
                if let Value::$from(value) = value {
                    Ok(value)
                } else {
                    Err(PbrtParseError::IncorrectType {
                        expected: ValueType::$from.to_string(),
                        found: value,
                    })
                }
            }
        }

        impl TryFrom<Value> for Option<$ty> {
            type Error = PbrtParseError;

            fn try_from(value: Value) -> Result<Self, Self::Error> {
                <$ty>::try_from(value).map(Some)
            }
        }

        impl TryFrom<Value> for Vec<$ty> {
            type Error = PbrtParseError;

            fn try_from(value: Value) -> Result<Self, Self::Error> {
                let incorrect_type_err = PbrtParseError::IncorrectType {
                    expected: format!("{}[]", stringify!($ty)),
                    found: value.clone(),
                };

                if let Value::$from_arr(vec) = value {
                    if vec.len() % $n != 0 {
                        return Err(PbrtParseError::InvalidValue {
                            expected: format!(
                                "array length that is a multiple of {}",
                                stringify!($n)
                            ),
                            found: Value::Int(vec.len() as i64),
                        });
                    }

                    let result: Result<Vec<_>, _> = vec.chunks($n).map(<$ty>::try_from).collect();
                    result.map_err(|_| incorrect_type_err)
                } else {
                    Err(incorrect_type_err)
                }
            }
        }

        impl TryFrom<Value> for Option<Vec<$ty>> {
            type Error = PbrtParseError;

            fn try_from(value: Value) -> Result<Self, Self::Error> {
                Vec::try_from(value).map(Some)
            }
        }
    };
}

impl_tuple_try_from_value!(Point2f, Point2, FloatArray, 2);
impl_tuple_try_from_value!(Vec2f, Vec2, FloatArray, 2);
impl_tuple_try_from_value!(Point3f, Point3, FloatArray, 3);
impl_tuple_try_from_value!(Vec3f, Vec3, FloatArray, 3);
impl_tuple_try_from_value!(Normal3f, Normal, FloatArray, 3);
impl_tuple_try_from_value!(RGB, Rgb, FloatArray, 3);

impl TryFrom<Value> for PathBuf {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        value
            .into_str()
            .map(PathBuf::from)
            .map_err(|found_val| PbrtParseError::IncorrectType {
                expected: "PathBuf".to_string(),
                found: found_val,
            })
    }
}

pub(super) fn value_type(input: &mut &str) -> PResult<ValueType> {
    cut_err(alphanumeric1.verify_map(|ty| ValueType::from_str(ty).ok()))
        .context(StrContext::Expected(StrContextValue::Description(
            "variable type",
        )))
        .parse_next(input)
}

#[derive(Debug, PartialEq, EnumAsInner)]
pub(super) enum Literal {
    Atomic(AtomicLiteral),
    Array(Vec<AtomicLiteral>),
}

pub(super) fn literal(input: &mut &str) -> PResult<Literal> {
    // FUTURE: cut_err verify is not available in winnow yet
    let array = delimited(
        ('[', opt(multispace1)),
        separated(1.., atomic_literal, multispace1),
        (opt(multispace1), ']'),
    )
    .verify(|arr: &Vec<_>| arr.iter().map(AtomicLiteralType::from).all_equal());
    let val = alt((
        atomic_literal.map(Literal::Atomic),
        array.map(Literal::Array),
        fail.context(StrContext::Label("parameter value"))
            .context(StrContext::Expected(StrContextValue::Description("number")))
            .context(StrContext::Expected(StrContextValue::Description("bool")))
            .context(StrContext::Expected(StrContextValue::Description("string")))
            .context(StrContext::Expected(StrContextValue::Description(
                "homegeneous array",
            ))),
    ));

    trace("literal", val).parse_next(input)
}

#[derive(Debug, PartialEq, EnumDiscriminants, EnumAsInner)]
#[strum_discriminants(name(AtomicLiteralType))]
pub(super) enum AtomicLiteral {
    Num(f64),
    Bool(bool),
    Str(String),
}

pub(super) fn atomic_literal(input: &mut &str) -> PResult<AtomicLiteral> {
    let boolean = alt(("true".value(true), "false".value(false)));
    let string = delimited('"', take_until(0.., '"'), '"');
    let val = alt((
        float.map(AtomicLiteral::Num),
        boolean.map(AtomicLiteral::Bool),
        string.map(|s: &str| AtomicLiteral::Str(s.to_string())),
    ));

    trace("atomic_literal", val).parse_next(input)
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Alpha {
    Constant(Float),
}

impl TryFrom<Value> for Alpha {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        // TODO: Update once textures are available
        Float::try_from(value).map(Alpha::Constant)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Spectrum {
    Constant(Float),
    Rgb(RGB),
    BlackbodyTemp(Float),
}

impl TryFrom<Value> for Spectrum {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Float(val) => Ok(Self::Constant(val)),
            Value::Rgb(rgb) => Ok(Self::Rgb(rgb)),
            Value::BlackbodyTemp(temp) => Ok(Self::BlackbodyTemp(temp)),
            _ => Err(PbrtParseError::IncorrectType {
                expected: "spectrum".to_string(),
                found: value,
            }),
        }
    }
}

impl TryFrom<Value> for Option<Spectrum> {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Spectrum::try_from(value).map(Some)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct ParameterMap(HashMap<String, Value>);

impl ParameterMap {
    // Expose necessary methods of inner HashMap
    delegate! {
        to self.0 {
            pub fn contains_key(&self, key: &str) -> bool;
            pub fn remove(&mut self, key: &str) -> Option<Value>;
            pub fn is_empty(&self) -> bool;
        }
    }

    /// If `self` has no items left, returns `Ok(())``.
    ///
    /// Otherwise, returns an error with one of the remaining keys.
    ///
    /// Used for raising an error for having irrelevant/unrecognized parameters for a directive.
    pub(super) fn check_no_remaining_params(self) -> Result<(), PbrtParseError> {
        if let Some(unexpected_name) = self.0.into_keys().next() {
            Err(PbrtParseError::UnexpectedParameter(unexpected_name))
        } else {
            Ok(())
        }
    }

    pub(super) fn check_mutually_exclusive(
        &self,
        set1: &[&str],
        set2: &[&str],
    ) -> Result<(), PbrtParseError> {
        let uses_set1 = set1.iter().any(|s| self.contains_key(s));
        let uses_set2 = set2.iter().any(|s| self.contains_key(s));

        if uses_set1 && uses_set2 {
            Err(PbrtParseError::MutuallyExclusiveParameters)
        } else {
            Ok(())
        }
    }
}

impl FromIterator<(String, Value)> for ParameterMap {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(iter: T) -> Self {
        Self(HashMap::from_iter(iter))
    }
}

impl From<HashMap<String, Value>> for ParameterMap {
    fn from(value: HashMap<String, Value>) -> Self {
        Self(value)
    }
}

pub(super) fn param_map(input: &mut &str) -> PResult<ParameterMap> {
    // Already seen param names, to check for uniqueness.
    let mut seen = HashSet::new();
    // Parses one parameter:
    let possible_param = |input: &mut &str| {
        let start = input.checkpoint();
        // Parse anything matching the format for a single param.
        let (name, value) = param.parse_next(input)?;
        // Add to set of seen param names.
        // If it already was, fail
        if !seen.insert(name.clone()) {
            input.reset(&start);
            return Err(
                ErrMode::from_error_kind(input, ErrorKind::Verify).add_context(
                    input,
                    &start,
                    StrContext::Expected(StrContextValue::Description(
                        "parameter to appear at most once",
                    )),
                ),
            );
        }

        Ok((name, value))
    };

    // Parse an entire parameter list (i.e. up until something that doesn't match the param format)
    // And convert it to a map.
    // Must get at least one param to parse Ok
    let param_map = separated(1.., possible_param, multispace1)
        .map(|list: Vec<_>| ParameterMap::from_iter(list));

    let mut traced = trace("param_map", param_map);
    traced.parse_next(input)
}

fn param_name<'a>(input: &mut &'a str) -> PResult<&'a str> {
    alphanumeric1
        .context(StrContext::Expected(StrContextValue::Description(
            "parameter name",
        )))
        .parse_next(input)
}

fn param(input: &mut &str) -> PResult<(String, Value)> {
    let full_param = separated_pair(
        delimited('"', separated_pair(value_type, space1, param_name), '"'),
        space1,
        literal,
    )
    .verify_map(|((ty, name), val)| {
        let val = match ty {
            ValueType::Int => match val {
                // FIXME: Can't assume `as`` is valid; array could have floats
                Literal::Atomic(val) => Value::Int(val.into_num().ok()? as Int),
                Literal::Array(arr) => {
                    let int_arr = arr
                        .into_iter()
                        .map(|v| v.into_num().map(|v| v as Int))
                        .collect::<Result<_, _>>()
                        .ok()?;
                    Value::IntArray(int_arr)
                }
            },
            // FIXME: Currently allows f64 saturating to f32 infinity
            ValueType::Float => match val {
                Literal::Atomic(val) => Value::Float(val.into_num().ok()? as Float),
                Literal::Array(arr) => {
                    let f_arr = arr
                        .into_iter()
                        .map(|v| v.into_num().map(|v| v as Float))
                        .collect::<Result<_, _>>()
                        .ok()?;
                    Value::FloatArray(f_arr)
                }
            },
            // FIXME: After refactor, no longer automatically supports same type name
            // for either single or array. Should be easy as only the atomic ones will
            // have this case
            ValueType::Bool => Value::Bool(*val.as_atomic()?.as_bool()?),
            ValueType::Str => Value::Str(val.into_atomic().ok()?.into_str().ok()?),
            ValueType::Point2 => {
                let arr = val.as_array()?;
                if arr.len() != 2 {
                    return None;
                }
                Value::Point2(Point2f::new(
                    *arr[0].as_num()? as Float,
                    *arr[1].as_num()? as Float,
                ))
            }
            ValueType::Vec2 => {
                let arr = val.as_array()?;
                if arr.len() != 2 {
                    return None;
                }
                Value::Vec2(Vec2f::new(
                    *arr[0].as_num()? as Float,
                    *arr[1].as_num()? as Float,
                ))
            }
            ValueType::Point3 => {
                let arr = val.as_array()?;
                if arr.len() != 3 {
                    return None;
                }
                Value::Point3(Point3f::new(
                    *arr[0].as_num()? as Float,
                    *arr[1].as_num()? as Float,
                    *arr[2].as_num()? as Float,
                ))
            }
            ValueType::Vec3 => {
                let arr = val.as_array()?;
                if arr.len() != 3 {
                    return None;
                }
                Value::Vec3(Vec3f::new(
                    *arr[0].as_num()? as Float,
                    *arr[1].as_num()? as Float,
                    *arr[2].as_num()? as Float,
                ))
            }
            ValueType::Normal => {
                let arr = val.as_array()?;
                if arr.len() != 3 {
                    return None;
                }
                Value::Normal(Normal3f::new(
                    *arr[0].as_num()? as Float,
                    *arr[1].as_num()? as Float,
                    *arr[2].as_num()? as Float,
                ))
            }
            ValueType::Rgb => {
                let arr = val.as_array()?;
                if arr.len() != 3 {
                    return None;
                }
                Value::Rgb(RGB::new(
                    *arr[0].as_num()? as Float,
                    *arr[1].as_num()? as Float,
                    *arr[2].as_num()? as Float,
                ))
            }
            ValueType::Blackbody => Value::BlackbodyTemp(*val.as_atomic()?.as_num()? as Float),
            ValueType::TextureName => Value::TextureName(val.into_atomic().ok()?.into_str().ok()?),
        };

        Some((name.to_owned(), val))
    })
    .context(StrContext::Label("parameter expression"));

    trace("param", full_param).parse_next(input)
}

#[derive(Clone, Debug)]
pub struct GraphicsState {
    pub current_transform: Transform,
    pub current_material_name: Option<String>,
    pub reverse_orientation: bool,
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            current_transform: Transform::IDENTITY,
            current_material_name: None,
            reverse_orientation: false,
        }
    }
}

pub(super) trait FromEntity {
    fn from_entity(entity: EntityDirective, state: &GraphicsState) -> Result<Self, PbrtParseError>
    where
        Self: Sized;
}

macro_rules! impl_from_entity {
    (
        $struct_name:ty,
        $(CTM => $transform_field:ident$(,)?)?
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
    ) => {
        impl crate::scene_parsing::common::FromEntity for $struct_name {
            #[allow(unused_variables)]
            fn from_entity(
                mut entity: crate::scene_parsing::common::EntityDirective,
                state: &crate::scene_parsing::common::GraphicsState,
            ) -> Result<Self, PbrtParseError> {
                let mut result = <$struct_name>::default();

                $(result.$transform_field = state.current_transform.clone().into();)?

                $(
                    $(
                        if let Some(value) = entity.param_map.remove($required_name) {
                            result.$required_field = value.try_into()?;
                        } else {
                            return Err(PbrtParseError::MissingRequiredParameter($required_name.to_string()));
                        }
                    )*
                )?

                $(
                    $(
                        if let Some(value) = entity.param_map.remove($defaulted_name) {
                            result.$defaulted_field = value.try_into()?;
                        }
                    )*
                )?

                entity.param_map.check_no_remaining_params()?;

                Ok(result)
            }
        }
    };
}
pub(super) use impl_from_entity;

// Similar to impl_from_entity, allowing use with more manual steps
macro_rules! params_map_to_fields {
    (
        $param_map:expr => $result:ident,
        $(
            required {
                $(
                    $required_field:ident = $required_name:literal $(=> $required_convert:expr)?
                ),* $(,)?
            }
        )?
        $(
            has_defaults {
                $(
                    $defaulted_field:ident = $defaulted_name:literal $(=> $defaulted_convert:expr)?
                ),* $(,)?
            }
        )?
    ) => {
        $(
            $(
                params_map_to_fields!(@required $param_map => $result, $required_field = $required_name $(=> $required_convert)?);
            )*
        )?

        $(
            $(
                params_map_to_fields!(@defaulted $param_map => $result, $defaulted_field = $defaulted_name $(=> $defaulted_convert)?);
            )*
        )?
    };
    (@required $param_map:expr => $result:ident, $field:ident = $name:literal) => {
        params_map_to_fields!(@required $param_map => $result, $field = $name => std::convert::identity)
    };
    (@required $param_map:expr => $result:ident, $field:ident = $name:literal => $convert_fn:expr) => {
        if let Some(value) = $param_map.remove($name) {
            $result.$field = $convert_fn(value.try_into()?);
        } else {
            return Err(PbrtParseError::MissingRequiredParameter($name.to_string()));
        }
    };
    (@defaulted $param_map:expr => $result:ident, $field:ident = $name:literal) => {
        params_map_to_fields!(@defaulted $param_map => $result, $field = $name => std::convert::identity)
    };
    (@defaulted $param_map:expr => $result:ident, $field:ident = $name:literal => $convert_fn:expr) => {
        if let Some(value) = $param_map.remove($name) {
            $result.$field = $convert_fn(value.try_into()?);
        }
    }
}
pub(super) use params_map_to_fields;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Directive<'a> {
    Entity(EntityDirective<'a>),
    Transform(TransformDirective),
    Texture(TextureDirective<'a>),
    WorldBegin,
    AttributeBegin,
    AttributeEnd,
    ReverseOrientation,
}

pub(super) fn directive<'a>(input: &mut &'a str) -> PResult<Directive<'a>> {
    let parser = terminated(
        alt((
            "WorldBegin".map(|_| Directive::WorldBegin),
            "AttributeBegin".map(|_| Directive::AttributeBegin),
            "AttributeEnd".map(|_| Directive::AttributeEnd),
            "ReverseOrientation".map(|_| Directive::ReverseOrientation),
            // Order here matters to prevent entity_directive from
            // eating part of a texture_directive, thinking the third "string"
            // is part of the next directive.
            // One alterative is to use dispatch
            texture_directive.map(Directive::Texture),
            transform_directive.map(Directive::Transform),
            entity_directive.map(Directive::Entity),
        )),
        alt((multispace1, eof)),
    );

    trace("directive", parser).parse_next(input)
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct EntityDirective<'a> {
    pub identifier: &'a str,
    pub subtype: &'a str,
    pub param_map: ParameterMap,
}

pub(super) fn entity_directive<'a>(input: &mut &'a str) -> PResult<EntityDirective<'a>> {
    let parser = |input: &mut &'a str| {
        let (identifier, subtype) =
            separated_pair(alpha1, multispace1, delimited('"', alphanumeric1, '"'))
                .parse_next(input)?;
        let param_map = opt(preceded(multispace1, param_map)).parse_next(input)?;

        Ok(EntityDirective {
            identifier,
            subtype,
            param_map: param_map.unwrap_or_default(),
        })
    };

    trace("entity_directive", parser).parse_next(input)
}

#[derive(Debug, Error)]
pub enum PbrtParseError {
    #[error("failed to read scene file")]
    IoError(#[from] std::io::Error),
    #[error("format error in scene file: {message}")]
    FormatError { message: String },

    #[error("directive is illegal in the current section: `{0}`")]
    IllegalForSection(String),
    #[error("missing required global option: `{0}`")]
    MissingRequiredOption(String),
    #[error("unrecognized directive: `{0}`")]
    UnrecognizedDirective(String),
    #[error("the `{0}` directive should only appear once, but was repeated")]
    RepeatedDirective(String),
    #[error("name `{0}` is already defined and cannot be redefined")]
    RedefinedName(String),
    #[error("attribute scope was not closed with AttributeEnd")]
    UnclosedAttributeScope,

    #[error("unexpected parameter for this entity: `{0}`")]
    UnexpectedParameter(String),
    #[error("entity is missing required parameter `{0}`")]
    MissingRequiredParameter(String),
    #[error("entity specifies mutually exclusive parameters")]
    MutuallyExclusiveParameters,
    #[error(
        "incorrect type for this parameter (expected type convertable to {expected}, found {found})",
    )]
    IncorrectType { expected: String, found: Value },
    #[error("invalid value for this parameter (expected {expected}, found {found})")]
    InvalidValue { expected: String, found: Value },
    #[error("incorrect length for this array (expected {expected}, found {found})")]
    IncorrectLength { expected: usize, found: usize },
    #[error("unrecognized variant \"{variant_name}\" for {entity}")]
    UnrecognizedVariant {
        entity: String,
        variant_name: String,
    },
}

impl From<winnow::error::ContextError> for PbrtParseError {
    fn from(value: winnow::error::ContextError) -> Self {
        Self::FormatError {
            message: value.to_string(),
        }
    }
}

#[cfg(test)]
mod test {
    use core::fmt;

    use maplit::{convert_args, hashmap};
    use winnow::stream::{AsBStr, StreamIsPartial};

    use super::*;

    fn assert_parses_to<I, O, E>(parser: impl Parser<I, O, E>, input: I, expected_output: O)
    where
        I: Stream + StreamIsPartial + AsBStr,
        O: PartialEq + fmt::Debug,
        E: ParserError<I> + fmt::Debug + fmt::Display,
    {
        let output = must_parse_ok(parser, input, false);
        assert_eq!(output, expected_output, "Parsed result does not match");
    }

    fn must_parse_ok<I, O, E>(
        mut parser: impl Parser<I, O, E>,
        input: I,
        print_ok_result: bool,
    ) -> O
    where
        I: Stream + StreamIsPartial + AsBStr,
        O: fmt::Debug,
        E: ParserError<I> + fmt::Debug + fmt::Display,
    {
        let result = parser.parse(input);
        assert!(
            result.is_ok(),
            "Parsing returned an error:\n{}",
            result.unwrap_err(),
        );

        let output = result.unwrap();
        if print_ok_result {
            println!("Successful parse:\n{:#?}", output);
        }
        output
    }

    #[test]
    fn test_atomic_literal() {
        assert_eq!(
            atomic_literal.parse(&mut "90"),
            Ok(AtomicLiteral::Num(90.0))
        );
        assert_eq!(
            atomic_literal.parse(&mut "90.0"),
            Ok(AtomicLiteral::Num(90.0))
        );
        assert_eq!(atomic_literal.parse(&mut ".2"), Ok(AtomicLiteral::Num(0.2)));
        assert_eq!(atomic_literal.parse(&mut "0"), Ok(AtomicLiteral::Num(0.0)));

        assert_eq!(
            atomic_literal.parse(&mut "true"),
            Ok(AtomicLiteral::Bool(true))
        );
        assert_eq!(
            atomic_literal.parse(&mut "false"),
            Ok(AtomicLiteral::Bool(false))
        );
        assert_eq!(
            atomic_literal.parse(&mut "\"parse\""),
            Ok(AtomicLiteral::Str("parse".to_owned()))
        );
        assert_eq!(
            atomic_literal.parse(&mut "\"\""),
            Ok(AtomicLiteral::Str("".to_owned()))
        );
        assert_parses_to(
            atomic_literal,
            &mut "\"file.png\"",
            AtomicLiteral::Str("file.png".to_owned()),
        );
    }

    #[test]
    fn atomic_literal_err() {
        assert!(atomic_literal.parse(&mut "parse").is_err());
    }

    #[test]
    fn test_literal() {
        assert_eq!(
            literal.parse(&mut "90"),
            Ok(Literal::Atomic(AtomicLiteral::Num(90.0)))
        );
        assert_eq!(
            literal.parse(&mut "[90 0 .2 9.9]"),
            Ok(Literal::Array(vec![
                AtomicLiteral::Num(90.0),
                AtomicLiteral::Num(0.0),
                AtomicLiteral::Num(0.2),
                AtomicLiteral::Num(9.9)
            ]))
        );
        assert_eq!(
            literal.parse(&mut "[ .4 .45 .5 ]"),
            Ok(Literal::Array(vec![
                AtomicLiteral::Num(0.4),
                AtomicLiteral::Num(0.45),
                AtomicLiteral::Num(0.5),
            ]))
        );
    }

    #[test]
    fn literal_err_mixed_array() {
        assert!(literal.parse(&mut "[90 0 .2 9.9 true]").is_err());
        assert!(literal.parse(&mut r#"[90 0 .2 "9.9" 9.9]"#).is_err());
    }

    #[test]
    fn test_param() {
        assert_parses_to(
            param,
            &mut r#""float foo" 1.0"#,
            ("foo".to_string(), Value::Float(1.0)),
        );
    }

    #[test]
    fn tuple_params_single() {
        assert_parses_to(
            param,
            &mut r#""rgb foo" [0.5 .6   0]"#,
            ("foo".to_string(), Value::Rgb(RGB::new(0.5, 0.6, 0.0))),
        );

        assert_parses_to(
            param,
            &mut r#""point2 foo" [.6   0]"#,
            ("foo".to_string(), Value::Point2(Point2f::new(0.6, 0.0))),
        );

        assert_parses_to(
            param,
            &mut r#""vector2 foo" [.6   0]"#,
            ("foo".to_string(), Value::Vec2(Vec2f::new(0.6, 0.0))),
        );

        assert_parses_to(
            param,
            &mut r#""point3 foo" [0.5 .6   0]"#,
            (
                "foo".to_string(),
                Value::Point3(Point3f::new(0.5, 0.6, 0.0)),
            ),
        );

        assert_parses_to(
            param,
            &mut r#""vector3 foo" [0.5 .6   0]"#,
            ("foo".to_string(), Value::Vec3(Vec3f::new(0.5, 0.6, 0.0))),
        );

        assert_parses_to(
            param,
            &mut r#""normal foo" [0.5 .6   0]"#,
            (
                "foo".to_string(),
                Value::Normal(Normal3f::new(0.5, 0.6, 0.0)),
            ),
        );
    }

    #[test]
    fn tuple_params_wrong_length() {
        assert!(param.parse(&mut r#""rgb foo" [0.5 .6 0 0.1]"#).is_err());
        assert!(param.parse(&mut r#""point2 foo" [0.5 .6 0]"#).is_err());
        assert!(param.parse(&mut r#""vector2 foo" [0.5 .6 0]"#).is_err());
        assert!(param.parse(&mut r#""point3 foo" [0.5 .6]"#).is_err());
        assert!(param.parse(&mut r#""vector3 foo" [0.5 .6]"#).is_err());
    }

    #[test]
    fn parameter_map_simple_singles() {
        assert_parses_to(
            param_map,
            &mut r#""float foo" 1.0 "integer bar" 2"#,
            convert_args!(hashmap! (
                "foo" => Value::Float(1.0),
                "bar" => Value::Int(2)
            ))
            .into(),
        );
    }

    #[test]
    fn tuple_vec_from_array_param() {
        let result =
            <Vec<RGB>>::try_from(Value::FloatArray(vec![0.5, 0.6, 0.0, 1.0, 0.3, 0.2])).unwrap();
        assert_eq!(
            result,
            vec![RGB::new(0.5, 0.6, 0.0), RGB::new(1.0, 0.3, 0.2)]
        );

        let result =
            <Vec<Point3f>>::try_from(Value::FloatArray(vec![0.5, 0.6, 0.0, 1.0, 0.3, 0.2]))
                .unwrap();
        assert_eq!(
            result,
            vec![Point3f::new(0.5, 0.6, 0.0), Point3f::new(1.0, 0.3, 0.2)]
        );

        let result =
            <Vec<Vec2f>>::try_from(Value::FloatArray(vec![0.5, 0.6, 0.0, 1.0, 0.3, 0.2])).unwrap();
        assert_eq!(
            result,
            vec![
                Vec2f::new(0.5, 0.6),
                Vec2f::new(0.0, 1.0),
                Vec2f::new(0.3, 0.2)
            ]
        );
    }

    #[test]
    fn tuple_vec_from_array_param_wrong_length() {
        assert!(
            <Vec<Point3f>>::try_from(Value::FloatArray(vec![0.5, 0.6, 0.0, 1.0, 0.3])).is_err()
        );

        assert!(<Vec<Vec2f>>::try_from(Value::FloatArray(vec![0.5, 0.6, 0.0])).is_err());
    }

    #[test]
    fn test_entity_directive() {
        assert_parses_to(
            entity_directive,
            &mut r#"Sampler "independent" "integer pixelsamples" 128"#,
            EntityDirective {
                identifier: "Sampler",
                subtype: "independent",
                param_map: ParameterMap(convert_args!(hashmap!(
                    "pixelsamples" => Value::Int(128)
                ))),
            },
        );

        assert_parses_to(
            entity_directive,
            &mut r#"Film "rgb" "string filename" "simple.png"
                    "integer xresolution" [400] "integer yresolution" [400]"#,
            EntityDirective {
                identifier: "Film",
                subtype: "rgb",
                param_map: ParameterMap(convert_args!(hashmap!(
                    "filename" => Value::Str("simple.png".to_string()),
                    "xresolution" => Value::IntArray(vec![400]),
                    "yresolution" => Value::IntArray(vec![400]),
                ))),
            },
        );

        assert_parses_to(
            entity_directive,
            &mut r#"LightSource "infinite" "rgb L" [ .4 .45 .5 ]"#,
            EntityDirective {
                identifier: "LightSource",
                subtype: "infinite",
                param_map: ParameterMap(convert_args!(hashmap!(
                    "L" => Value::Rgb(RGB::new(0.4, 0.45, 0.5))
                ))),
            },
        );
    }

    #[test]
    fn test_entity_directive_empty_param_map() {
        assert!(param_map.parse("").is_err());

        assert_parses_to(
            entity_directive,
            r#"Integrator "volpath""#,
            EntityDirective {
                identifier: "Integrator",
                subtype: "volpath",
                param_map: ParameterMap(HashMap::new()),
            },
        );
    }
}
