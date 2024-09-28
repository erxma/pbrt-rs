use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    str::FromStr,
};

use enum_as_inner::EnumAsInner;
use itertools::Itertools;
use strum::{EnumDiscriminants, EnumString};
use thiserror::Error;
use winnow::{
    ascii::{alpha1, alphanumeric0, alphanumeric1, float, multispace1, space1},
    combinator::{
        alt, cut_err, delimited, eof, fail, separated, separated_pair, seq, terminated, trace,
    },
    error::{AddContext, ErrMode, ErrorKind, ParserError, StrContext, StrContextValue},
    prelude::*,
    stream::Stream,
};

use crate::{color::RGB, core::Float};

use super::directives::{transform_directive, TransformDirective};

#[derive(Clone, Debug, PartialEq, EnumAsInner, EnumDiscriminants)]
#[strum_discriminants(derive(EnumString, strum::Display))]
#[strum_discriminants(name(ValueType))]
pub(super) enum Value {
    #[strum_discriminants(strum(serialize = "integer"))]
    Int(i32),
    #[strum_discriminants(strum(serialize = "float"))]
    Float(Float),
    #[strum_discriminants(strum(serialize = "bool"))]
    Bool(bool),
    #[strum_discriminants(strum(serialize = "string"))]
    Str(String),
    #[strum_discriminants(strum(serialize = "rgb"))]
    Rgb(RGB),
    #[strum_discriminants(strum(serialize = "blackbody"))]
    BlackbodyTemp(Float),
}

impl TryFrom<Value> for Float {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        value
            .into_float()
            .map_err(|found_val| PbrtParseError::IncorrectType {
                expected: ValueType::Float.to_string(),
                found: ValueType::from(found_val).to_string(),
            })
    }
}

impl TryFrom<Value> for Option<Float> {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Float::try_from(value).map(Some)
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

impl TryFrom<Value> for Alpha {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        // TODO: Update once textures are available
        Float::try_from(value).map(Alpha::Constant)
    }
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
        delimited('"', separated_pair(value_type, space1, param_name), '"'),
        space1,
        literal,
    )
    .verify_map(|((ty, name), val)| {
        let val = match ty {
            ValueType::Int => Value::Int(*val.as_atomic()?.as_num()? as i32),
            // FIXME: Currently allows f64 saturating to f32 infinity
            // FIXME: After refactor, no longer automatically supports same type name
            // for either single or array. Should be easy as only the atomic ones will
            // have this case
            ValueType::Float => Value::Float(*val.as_atomic()?.as_num()? as Float),
            ValueType::Bool => Value::Bool(*val.as_atomic()?.as_bool()?),
            ValueType::Str => Value::Str(val.into_atomic().ok()?.into_str().ok()?),
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
            ValueType::BlackbodyTemp => Value::BlackbodyTemp(*val.as_atomic()?.as_num()? as Float),
        };

        Some((name.to_owned(), val))
    })
    .context(StrContext::Label("parameter expression"));

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

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Directive<'a> {
    Entity(EntityDirective<'a>),
    Transform(TransformDirective),
    WorldBegin,
    AttributeBegin,
    AttributeEnd,
}

pub(super) fn directive<'a>(input: &mut &'a str) -> PResult<Directive<'a>> {
    terminated(
        alt((
            entity_directive.map(Directive::Entity),
            transform_directive.map(Directive::Transform),
            "WorldBegin".map(|_| Directive::WorldBegin),
            "AttributeBegin".map(|_| Directive::AttributeBegin),
            "AttributeEnd".map(|_| Directive::AttributeEnd),
        )),
        alt((multispace1, eof)),
    )
    .parse_next(input)
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct EntityDirective<'a> {
    pub identifier: &'a str,
    pub subtype: &'a str,
    pub param_map: ParameterMap,
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

#[derive(Debug, Error, PartialEq)]
pub enum PbrtParseError {
    #[error("directive is illegal in the current section: {0}")]
    IllegalForSection(String),
    #[error("missing required global option: {0}")]
    MissingRequiredOption(String),
    #[error("directive is unrecognized or illegal in the current section: {0}")]
    UnrecognizedOrIllegalDirective(String),
    #[error("the `{0}` directive should only appear once, but was repeated")]
    RepeatedDirective(String),

    #[error("missing required parameter {0}")]
    MissingRequiredParameter(String),
    #[error("unexpected parameter for this entity: {0}")]
    UnexpectedParameter(String),
    #[error("incorrect type for parameter (expected {expected}, found {found})")]
    IncorrectType { expected: String, found: String },
    #[error("unrecognized subtype \"{type_name}\" for {entity}")]
    UnrecognizedSubtype { entity: String, type_name: String },
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
            Ok(AtomicLiteral::Str("".to_owned())),
            atomic_literal.parse(&mut "\"\"")
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
    fn rgb_param() {
        assert_parses_to(
            param,
            &mut r#""rgb foo" [0.5 .6   0]"#,
            ("foo".to_string(), Value::Rgb(RGB::new(0.5, 0.6, 0.0))),
        );
    }

    #[test]
    fn rgb_param_wrong_length() {
        assert!(param.parse(&mut r#""rgb foo" [0.5 .6 0 0.1]"#).is_err());
    }

    #[test]
    fn parameter_map_simple_singles() {
        assert_parses_to(
            param_map,
            &mut r#""float foo" 1.0 "integer bar" 2"#,
            convert_args!(hashmap! (
                "foo" => Value::Float(1.0),
                "bar" => Value::Int(2)
            )),
        );
    }
}
