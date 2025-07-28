use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    str::FromStr as _,
};

use delegate::delegate;
use enum_as_inner::EnumAsInner;
use itertools::Itertools as _;
use num_traits::NumCast;
use strum::EnumDiscriminants;
use thiserror::Error;
use winnow::{
    ascii::{alpha1, alphanumeric1, float, multispace1, space1},
    combinator::{
        alt, cut_err, delimited, eof, fail, opt, preceded, separated, separated_pair, terminated,
        trace,
    },
    error::{AddContext, ErrMode, ParserError, StrContext, StrContextValue},
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

#[derive(Clone, Debug, PartialEq, strum::Display, strum::EnumDiscriminants)]
#[strum_discriminants(
    name(ValueType),
    derive(strum::Display, strum::EnumString),
    strum(serialize_all = "lowercase")
)]
pub enum Value {
    Integer(Int),
    Float(Float),
    Bool(bool),
    String(String),
    Point2(Point2f),
    Vector2(Vec2f),
    Point3(Point3f),
    Vector3(Vec3f),
    Normal(Normal3f),
    Rgb(RGB),
    Blackbody(Float),
    Texture(String),
    Array(Vec<Value>),
}

impl ValueType {
    fn is_tuple_type(&self) -> bool {
        matches!(
            self,
            ValueType::Point2
                | ValueType::Vector2
                | ValueType::Point3
                | ValueType::Vector3
                | ValueType::Normal
                | ValueType::Rgb
        )
    }
}

pub(super) type Int = i64;

macro_rules! impl_num_try_from_value {
    ($ty:ty, $from:ident) => {
        impl TryFrom<Value> for $ty {
            type Error = PbrtParseError;

            fn try_from(value: Value) -> Result<Self, Self::Error> {
                match value {
                    Value::$from(val) => val.try_into().map_err(|_| value),
                    // Also allow implicit conversion from single elem array to single
                    Value::Array(ref arr) if arr.len() == 1 => {
                        arr[0].clone().try_into().or(Err(value))
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

                if let Value::Array(vec) = value {
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

                if let Value::Array(vec) = value {
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

impl_num_try_from_value!(usize, Integer);
impl_num_try_from_value!(u8, Integer);
impl_num_try_from_value!(u64, Integer);
impl_num_try_from_value!(Float, Float);

macro_rules! impl_try_from_value {
    ($ty:ty, $from:ident) => {
        impl TryFrom<Value> for $ty {
            type Error = PbrtParseError;

            fn try_from(value: Value) -> Result<Self, Self::Error> {
                if let Value::$from(value) = value {
                    Ok(value)
                } else {
                    Err(PbrtParseError::IncorrectType {
                        expected: stringify!($ty).to_string(),
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

                if let Value::Array(arr) = value {
                    let converted: Result<Vec<_>, _> =
                        arr.into_iter().map(<$ty>::try_from).collect();
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

impl_try_from_value!(bool, Bool);
impl_try_from_value!(Point2f, Point2);
impl_try_from_value!(Vec2f, Vector2);
impl_try_from_value!(Point3f, Point3);
impl_try_from_value!(Vec3f, Vector3);
impl_try_from_value!(Normal3f, Normal);
impl_try_from_value!(RGB, Rgb);
impl_try_from_value!(String, String);

impl TryFrom<Value> for PathBuf {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        String::try_from(value).map(PathBuf::from)
    }
}

pub(super) fn value_type(input: &mut &str) -> ModalResult<ValueType> {
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

pub(super) fn literal(input: &mut &str) -> ModalResult<Literal> {
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

pub(super) fn atomic_literal(input: &mut &str) -> ModalResult<AtomicLiteral> {
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
pub enum Spectrum {
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
            Value::Blackbody(temp) => Ok(Self::BlackbodyTemp(temp)),
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

pub(super) fn param_map(input: &mut &str) -> ModalResult<ParameterMap> {
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
            return Err(ErrMode::from_input(input).add_context(
                input,
                &start,
                StrContext::Expected(StrContextValue::Description(
                    "parameter to appear at most once",
                )),
            ));
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

fn param_name<'a>(input: &mut &'a str) -> ModalResult<&'a str> {
    alphanumeric1
        .context(StrContext::Expected(StrContextValue::Description(
            "parameter name",
        )))
        .parse_next(input)
}

fn param(input: &mut &str) -> ModalResult<(String, Value)> {
    let full_param = separated_pair(
        delimited('"', separated_pair(value_type, space1, param_name), '"'),
        space1,
        literal,
    )
    .verify_map(|((ty, name), literal)| {
        let val = literal_to_value(literal, ty)?;
        Some((name.to_owned(), val))
    })
    .context(StrContext::Label("parameter expression"));

    trace("param", full_param).parse_next(input)
}

fn literal_to_value(value: Literal, value_type: ValueType) -> Option<Value> {
    match value {
        Literal::Atomic(atomic) => atomic_literal_to_value(atomic, value_type),
        Literal::Array(atomics) => {
            if value_type.is_tuple_type() {
                array_literal_to_tuple_values(atomics, value_type)
            } else if value_type == ValueType::Array {
                None
            } else {
                let vec = atomics
                    .into_iter()
                    .map(|a| atomic_literal_to_value(a, value_type))
                    .collect::<Option<_>>()?;
                Some(Value::Array(vec))
            }
        }
    }
}

fn array_literal_to_tuple_values(
    atomics: Vec<AtomicLiteral>,
    value_type: ValueType,
) -> Option<Value> {
    let n = match value_type {
        ValueType::Point2 | ValueType::Vector2 => 2,
        ValueType::Point3 | ValueType::Vector3 | ValueType::Normal | ValueType::Rgb => 3,
        _ => panic!("called array_literal_to_tuple_values for non-tuple value type"),
    };

    if atomics.len() % n != 0 {
        return None;
    }

    let mut vec = Vec::new();
    for chunk in atomics.chunks(n) {
        let val = match value_type {
            ValueType::Point2 => Value::Point2(Point2f::new(
                *chunk[0].as_num()? as Float,
                *chunk[1].as_num()? as Float,
            )),
            ValueType::Vector2 => Value::Vector2(Vec2f::new(
                *chunk[0].as_num()? as Float,
                *chunk[1].as_num()? as Float,
            )),
            ValueType::Point3 => Value::Point3(Point3f::new(
                *chunk[0].as_num()? as Float,
                *chunk[1].as_num()? as Float,
                *chunk[2].as_num()? as Float,
            )),
            ValueType::Vector3 => Value::Vector3(Vec3f::new(
                *chunk[0].as_num()? as Float,
                *chunk[1].as_num()? as Float,
                *chunk[2].as_num()? as Float,
            )),
            ValueType::Normal => Value::Normal(Normal3f::new(
                *chunk[0].as_num()? as Float,
                *chunk[1].as_num()? as Float,
                *chunk[2].as_num()? as Float,
            )),
            ValueType::Rgb => Value::Rgb(RGB::new(
                *chunk[0].as_num()? as Float,
                *chunk[1].as_num()? as Float,
                *chunk[2].as_num()? as Float,
            )),
            _ => unreachable!(),
        };

        vec.push(val);
    }

    if vec.len() == 1 {
        Some(vec.remove(0))
    } else {
        Some(Value::Array(vec))
    }
}

fn atomic_literal_to_value(atomic: AtomicLiteral, value_type: ValueType) -> Option<Value> {
    let val = match value_type {
        ValueType::Integer => Value::Integer(atomic.into_num().ok().and_then(NumCast::from)?),
        // FIXME: Currently allows f64 saturating to f32 infinity
        ValueType::Float => Value::Float(atomic.into_num().ok().and_then(NumCast::from)?),
        ValueType::Bool => Value::Bool(atomic.into_bool().ok()?),
        ValueType::String => Value::String(atomic.into_str().ok()?),
        ValueType::Blackbody => Value::Blackbody(atomic.into_num().ok().and_then(NumCast::from)?),
        ValueType::Texture => Value::Texture(atomic.into_str().ok()?),
        _ => panic!("tried to convert atomic literal to value that requires array"),
    };

    Some(val)
}

#[derive(Clone, Debug)]
pub struct GraphicsState {
    pub current_transform: Transform,
    pub current_material_name: Option<String>,
    pub reverse_orientation: bool,
    pub current_area_light_index: Option<usize>,
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            current_transform: Transform::IDENTITY,
            current_material_name: None,
            reverse_orientation: false,
            current_area_light_index: None,
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

pub(super) fn directive<'a>(input: &mut &'a str) -> ModalResult<Directive<'a>> {
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

pub(super) fn entity_directive<'a>(input: &mut &'a str) -> ModalResult<EntityDirective<'a>> {
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
        E: ParserError<I>,
        <E as ParserError<I>>::Inner: ParserError<I> + fmt::Debug + fmt::Display,
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
        E: ParserError<I>,
        <E as ParserError<I>>::Inner: ParserError<I> + fmt::Debug + fmt::Display,
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

    macro_rules! array_value {
        () => (
            Value::Array(Vec::new())
        );
        ($type:ident; $elem:expr; $n:expr) => (
            Value::Array(vec![Value::$type($elem); n])
        );
        ($type:ident; $($x:expr),+ $(,)?) => (
            Value::Array(vec![$(Value::$type($x)),+])
        );
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
            ("foo".to_string(), Value::Vector2(Vec2f::new(0.6, 0.0))),
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
            ("foo".to_string(), Value::Vector3(Vec3f::new(0.5, 0.6, 0.0))),
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
                "bar" => Value::Integer(2)
            ))
            .into(),
        );
    }

    #[test]
    fn tuple_vec_from_array_param() {
        assert_parses_to(
            param,
            &mut r#""rgb foo" [0.5 .6   0 1.0 0.3 0.2 ]"#,
            (
                "foo".to_string(),
                array_value![Rgb; RGB::new(0.5, 0.6, 0.0), RGB::new(1.0, 0.3, 0.2)],
            ),
        );

        assert_parses_to(
            param,
            &mut r#""point3 foo" [ 0.5 .6   0 1.0 0.3 0.2]"#,
            (
                "foo".to_string(),
                array_value![Point3; Point3f::new(0.5, 0.6, 0.0), Point3f::new(1.0, 0.3, 0.2)],
            ),
        );

        assert_parses_to(
            param,
            &mut r#""vector2 foo" [ 0.5 .6   0 1.0 0.3 0.2 ]"#,
            (
                "foo".to_string(),
                array_value![Vector2; Vec2f::new(0.5, 0.6), Vec2f::new(0.0, 1.0), Vec2f::new(0.3, 0.2)],
            ),
        );
    }

    #[test]
    fn tuple_vec_from_array_param_wrong_length() {
        assert!(param
            .parse(&mut r#""point3 foo" [0.5 .6 0 1.0 0.3]"#)
            .is_err());
        assert!(param.parse(&mut r#""vector2 foo" [0.5 .6 0]"#).is_err());
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
                    "pixelsamples" => Value::Integer(128)
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
                    "filename" => Value::String("simple.png".to_string()),
                    "xresolution" => array_value![Integer; 400],
                    "yresolution" => array_value![Integer; 400],
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
