use std::{
    collections::{HashMap, HashSet},
    str::FromStr,
};

use enum_as_inner::EnumAsInner;
use itertools::Itertools;
use strum::{EnumDiscriminants, EnumString};
use thiserror::Error;
use winnow::{
    ascii::{alpha1, alphanumeric0, alphanumeric1, float, multispace1, space1},
    combinator::{alt, cut_err, delimited, fail, separated, separated_pair, seq, trace},
    error::{AddContext, ErrMode, ErrorKind, ParserError, StrContext, StrContextValue},
    prelude::*,
    stream::Stream,
};

use crate::core::Float;

use super::directives::{transform_directive, TransformDirective};

#[derive(Clone, Debug, PartialEq, EnumAsInner, EnumDiscriminants)]
#[strum_discriminants(derive(strum::Display))]
#[strum_discriminants(name(ValueType))]
pub(super) enum Value {
    Single(SingleValue),
    Array(Vec<SingleValue>),
}

impl TryFrom<Value> for SingleValue {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        value
            .into_single()
            .map_err(|_| PbrtParseError::IncorrectType {
                expected: ValueType::Single.to_string(),
                found: ValueType::Array.to_string(),
            })
    }
}

impl TryFrom<Value> for Float {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        SingleValue::try_from(value)?
            .into_float()
            .map_err(|found_val| PbrtParseError::IncorrectType {
                expected: SingleValueType::Float.to_string(),
                found: SingleValueType::from(found_val).to_string(),
            })
    }
}

impl TryFrom<Value> for Option<Float> {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Float::try_from(value).map(Some)
    }
}

#[derive(Clone, Debug, PartialEq, EnumDiscriminants, EnumAsInner)]
#[strum_discriminants(derive(EnumString, strum::Display))]
#[strum_discriminants(name(SingleValueType))]
pub(super) enum SingleValue {
    #[strum_discriminants(strum(serialize = "integer"))]
    Int(i32),
    #[strum_discriminants(strum(serialize = "float"))]
    Float(Float),
    #[strum_discriminants(strum(serialize = "bool"))]
    Bool(bool),
    #[strum_discriminants(strum(serialize = "string"))]
    Str(String),
}

pub(super) fn single_ty(input: &mut &str) -> PResult<SingleValueType> {
    alphanumeric1
        .verify_map(|ty| SingleValueType::from_str(ty).ok())
        .context(StrContext::Expected(StrContextValue::Description(
            "parameter type",
        )))
        .parse_next(input)
}

#[derive(Debug, PartialEq)]
pub(super) enum Literal {
    Atomic(AtomicLiteral),
    Array(Vec<AtomicLiteral>),
}

pub(super) fn literal(input: &mut &str) -> PResult<Literal> {
    // FUTURE: cut_err verify is not available in winnow yet
    let array = delimited('[', separated(1.., atomic_literal, space1), ']')
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
    let string = delimited('"', alphanumeric0, '"');
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
    Texture(()),
}

pub(super) type ParameterMap = HashMap<String, Value>;

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
        if seen.insert(name.clone()) {
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
    let param_map = cut_err(separated(0.., possible_param, multispace1))
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
        delimited('"', separated_pair(single_ty, space1, param_name), '"'),
        space1,
        literal,
    )
    .verify_map(|((ty, name), val)| {
        let conversion = |single: AtomicLiteral| match ty {
            SingleValueType::Int => single.into_num().map(|v| SingleValue::Int(v as i32)),
            // FIXME: Currently allows f64 saturating to f32 infinity
            SingleValueType::Float => single.into_num().map(|v| SingleValue::Float(v as Float)),
            SingleValueType::Bool => single.into_bool().map(SingleValue::Bool),
            SingleValueType::Str => single.into_str().map(SingleValue::Str),
        };
        let val = match val {
            Literal::Atomic(val) => conversion(val).map(Value::Single).ok()?,
            Literal::Array(arr) => arr
                .into_iter()
                .map(conversion)
                .collect::<Result<Vec<_>, _>>()
                .map(Value::Array)
                .ok()?,
        };

        Some((name.to_owned(), val))
    })
    .context(StrContext::Label("value for the declared type"));

    trace("param", full_param).parse_next(input)
}

macro_rules! impl_try_from_parameter_map {
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
    ) => {
        impl TryFrom<crate::scene_parsing::common::ParameterMap> for $struct_name {
            type Error = crate::scene_parsing::common::PbrtParseError;

            fn try_from(
                mut params: crate::scene_parsing::common::ParameterMap,
            ) -> Result<Self, Self::Error> {
                let mut result = <$struct_name>::default();

                $(
                    $(
                        if let Some(value) = params.remove($required_name) {
                            result.$required_field = value.try_into()?;
                        } else {
                            return Err(Self::Error::MissingRequiredParameter($required_name.to_string()));
                        }
                    )*
                )?

                $(
                    $(
                        if let Some(value) = params.remove($defaulted_name) {
                            result.$defaulted_field = value.try_into()?;
                        }
                    )*
                )?

                if let Some(unexpected_name) = params.into_keys().next() {
                    return Err(Self::Error::UnexpectedParameter(unexpected_name));
                }

                Ok(result)
            }
        }
    };
}
pub(super) use impl_try_from_parameter_map;

pub(super) enum Directive<'a> {
    Entity(EntityDirective<'a>),
    Transform(TransformDirective),
    WorldBegin,
    AttributeBegin,
    AttributeEnd,
}

pub(super) fn directive<'a>(input: &mut &'a str) -> PResult<Directive<'a>> {
    alt((
        entity_directive.map(Directive::Entity),
        transform_directive.map(Directive::Transform),
        "WorldBegin".map(|_| Directive::WorldBegin),
        "AttributeBegin".map(|_| Directive::AttributeBegin),
        "AttributeEnd".map(|_| Directive::AttributeEnd),
    ))
    .parse_next(input)
}

pub(super) struct EntityDirective<'a> {
    identifier: &'a str,
    subtype: &'a str,
    param_map: ParameterMap,
}

pub(super) fn entity_directive<'a>(input: &mut &'a str) -> PResult<EntityDirective<'a>> {
    trace(
        "entity_directive",
        seq! { EntityDirective {
            identifier: alpha1,
            _: multispace1,
            subtype: delimited('"', alpha1, '"'),
            _: multispace1,
            param_map: param_map
        }},
    )
    .parse_next(input)
}

#[derive(Debug, Error)]
pub enum PbrtParseError {
    #[error("directive is illegal in the current section: {0}")]
    IllegalForSection(String),
    #[error("missing required global option: {0}")]
    MissingRequiredOption(String),

    #[error("missing required parameter {0}")]
    MissingRequiredParameter(String),
    #[error("unexpected parameter for this entity: {0}")]
    UnexpectedParameter(String),
    #[error("incorrect type for parameter (expected {expected}, found {found})")]
    IncorrectType { expected: String, found: String },
}

#[cfg(test)]
mod test {
    use core::fmt;

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
            Ok(AtomicLiteral::Str("".to_owned())),
            atomic_literal.parse(&mut "\"\"")
        );
    }

    #[test]
    fn test_atomic_literal_err() {
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
    }

    #[test]
    fn test_literal_err_mixed_array() {
        assert!(literal.parse(&mut "[90 0 .2 9.9 true]").is_err());
        assert!(literal.parse(&mut r#"[90 0 .2 "9.9" 9.9]"#).is_err());
    }

    #[test]
    fn test_param() {
        assert_parses_to(
            param,
            &mut r#""float foo" 1.0"#,
            ("foo".to_string(), Value::Single(SingleValue::Float(1.0))),
        );
    }
}
