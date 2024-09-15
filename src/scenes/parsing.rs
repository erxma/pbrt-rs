use std::{
    collections::{HashMap, HashSet},
    str::FromStr,
};

use derive_builder::Builder;
use enum_as_inner::EnumAsInner;
use itertools::Itertools;
use strum::{EnumDiscriminants, EnumString};
use winnow::{
    ascii::{
        alpha1, alphanumeric0, alphanumeric1, float, multispace0, multispace1, space0, space1,
    },
    combinator::{
        alt, cut_err, delimited, eof, fail, opt, peek, preceded, separated, separated_pair,
        terminated, trace,
    },
    dispatch,
    error::{AddContext, ErrMode, ErrorKind, ParserError, StrContext, StrContextValue},
    prelude::*,
    stream::Stream,
};

use crate::core::Float;

#[derive(Debug)]
pub struct Scene {
    options: Options,
    world: World,
}

#[derive(Debug, Builder)]
pub struct Options {
    #[builder(private)]
    camera: Camera,
}

#[derive(Debug, Builder)]
pub struct World {
    #[builder(private, default)]
    shapes: Vec<Shape>,
}

pub enum Directive {
    Global(GlobalDirective),
}

#[derive(Clone, Debug, PartialEq, EnumDiscriminants)]
#[strum_discriminants(derive(Hash))]
pub enum GlobalDirective {
    Camera(Camera),
}

#[derive(Clone, Debug, PartialEq, EnumDiscriminants)]
#[strum_discriminants(derive(Hash))]
pub enum WorldDirective {
    Shape(Shape),
}

#[derive(Clone, Debug, PartialEq, EnumAsInner, EnumDiscriminants)]
#[strum_discriminants(name(ValueType))]
pub enum Value {
    Single(SingleValue),
    Array(Vec<SingleValue>),
}

#[derive(Clone, Debug, PartialEq, EnumDiscriminants, EnumAsInner)]
#[strum_discriminants(derive(EnumString))]
#[strum_discriminants(name(SingleValueType))]
pub enum SingleValue {
    #[strum_discriminants(strum(serialize = "integer"))]
    Int(i32),
    #[strum_discriminants(strum(serialize = "float"))]
    Float(Float),
    #[strum_discriminants(strum(serialize = "bool"))]
    Bool(bool),
    #[strum_discriminants(strum(serialize = "string"))]
    Str(String),
}

#[derive(Debug, PartialEq)]
enum Literal {
    Atomic(AtomicLiteral),
    Array(Vec<AtomicLiteral>),
}

#[derive(Debug, PartialEq, EnumDiscriminants, EnumAsInner)]
#[strum_discriminants(name(AtomicLiteralType))]
enum AtomicLiteral {
    Num(f64),
    Bool(bool),
    Str(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Camera {
    Orthographic(OrthographicCamera),
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrthographicCamera {
    shutter_open: Float,
    shutter_close: Float,
    frame_aspect_ratio: Option<Float>,
    screen_window: Option<Float>,
    lens_radius: Float,
    focal_distance: Float,
}

impl Default for OrthographicCamera {
    fn default() -> Self {
        Self {
            shutter_open: 0.0,
            shutter_close: 1.0,
            frame_aspect_ratio: None,
            screen_window: None,
            lens_radius: 0.0,
            focal_distance: 10e30,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Sphere(Sphere),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sphere {
    alpha: Alpha,
    radius: Float,
    z_min: Option<Float>,
    z_max: Option<Float>,
    phi_max: Float,
}

impl Default for Sphere {
    fn default() -> Self {
        Self {
            alpha: Alpha::Constant(1.0),
            radius: 1.0,
            z_min: None,
            z_max: None,
            phi_max: 360.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Alpha {
    Constant(Float),
    Texture(()),
}

pub fn scene(ignore_unrecognized_directives: bool) -> impl FnMut(&mut &str) -> PResult<Scene> {
    move |input| {
        separated_pair(
            trace(
                "global_options",
                global_options(ignore_unrecognized_directives),
            ),
            ("WorldBegin", multispace0),
            world(ignore_unrecognized_directives),
        )
        .map(|(global_options, world)| Scene {
            options: global_options,
            world,
        })
        .parse_next(input)
    }
}

fn global_options(
    ignore_unrecognized_directives: bool,
) -> impl FnMut(&mut &str) -> PResult<Options> {
    move |input| {
        let mut seen_directives: HashSet<GlobalDirectiveDiscriminants> = HashSet::new();
        let unseen_directive = move |input: &mut &str| {
            let start = input.checkpoint();
            let dir = global_directive(ignore_unrecognized_directives).parse_next(input)?;
            if seen_directives.insert(dir.clone().into()) {
                Ok(dir)
            } else {
                input.reset(&start);
                Err(ErrMode::from_error_kind(input, ErrorKind::Verify)
                    .add_context(
                        input,
                        &start,
                        StrContext::Expected(StrContextValue::Description(
                            "directive to appear at most once",
                        )),
                    )
                    .cut())
            }
        };

        let dirs: Vec<_> = terminated(separated(0.., unseen_directive, multispace1), multispace0)
            .parse_next(input)?;

        let mut options = OptionsBuilder::create_empty();
        for dir in dirs {
            options.add_option(dir);
        }
        options.build().map_err(|_| {
            ErrMode::from_error_kind(input, ErrorKind::Verify)
                .add_context(
                    input,
                    &input.checkpoint(),
                    StrContext::Expected(StrContextValue::Description(
                        "all required global options",
                    )),
                )
                .cut()
        })
    }
}

fn world(ignore_unrecognized_directives: bool) -> impl FnMut(&mut &str) -> PResult<World> {
    move |input| {
        terminated(
            separated(
                0..,
                world_directive(ignore_unrecognized_directives),
                multispace1,
            ),
            (multispace0, peek(eof)),
        )
        .map(|dirs: Vec<_>| {
            let mut options = WorldBuilder::create_empty();
            for d in dirs {
                options.add_option(d);
            }
            options.build().unwrap()
        })
        .parse_next(input)
    }
}

impl WorldBuilder {
    pub fn add_option(&mut self, directive: WorldDirective) {
        match directive {
            WorldDirective::Shape(s) => self.shapes.get_or_insert(Vec::new()).push(s),
        };
    }
}

impl OptionsBuilder {
    pub fn add_option(&mut self, directive: GlobalDirective) {
        match directive {
            GlobalDirective::Camera(cam) => self.camera(cam),
        };
    }
}

fn global_directive(
    ignore_unrecognized: bool,
) -> impl FnMut(&mut &str) -> PResult<GlobalDirective> {
    move |input| {
        trace(
            "global_directive",
            dispatch! { cut_err(terminated(alpha1, space0));
                "Camera" => camera_directive.map(GlobalDirective::Camera),
                "WorldBegin" => fail,
                _ if ignore_unrecognized => fail,
                _ => cut_err(fail)
            }
            .context(StrContext::Label("global directive")),
        )
        .parse_next(input)
    }
}

fn world_directive(ignore_unrecognized: bool) -> impl FnMut(&mut &str) -> PResult<WorldDirective> {
    move |input| {
        trace(
            "world_directive",
            dispatch! { cut_err(terminated(alpha1, space0));
                "Shape" => shape_directive.map(WorldDirective::Shape),
                _ if ignore_unrecognized => fail,
                _ => cut_err(fail)
            }
            .context(StrContext::Label("world directive")),
        )
        .parse_next(input)
    }
}

fn camera_directive(input: &mut &str) -> PResult<Camera> {
    trace(
        "camera_directive",
        dispatch! { cut_err(terminated(delimited('"', alpha1, '"'), space1));
            "orthographic" => orthographic_camera_params,
            _=> fail.context(StrContext::Label("camera type"))
        },
    )
    .parse_next(input)
}

fn orthographic_camera_params(input: &mut &str) -> PResult<Camera> {
    let params = |input: &mut &str| {
        let items = expected_params_map(vec![
            "float shutteropen",
            "float shutterclose",
            "float frameaspectratio",
            "float screenwindow",
            "float lensradius",
            "float focaldistance",
        ]);
        let found_params = param_list(items).parse_next(input)?;
        let mut ortho = OrthographicCamera::default();
        for (k, v) in found_params {
            match k.as_str() {
                "shutteropen" => {
                    ortho.shutter_open = *v.as_single().unwrap().as_float().unwrap();
                }
                "shutterclose" => {
                    ortho.shutter_close = *v.as_single().unwrap().as_float().unwrap();
                }
                "frameaspectratio" => {
                    ortho.frame_aspect_ratio = Some(*v.as_single().unwrap().as_float().unwrap());
                }
                "screenwindow" => {
                    ortho.screen_window = Some(*v.as_single().unwrap().as_float().unwrap());
                }
                "lensradius" => {
                    ortho.lens_radius = *v.as_single().unwrap().as_float().unwrap();
                }
                "focaldistance" => {
                    ortho.focal_distance = *v.as_single().unwrap().as_float().unwrap();
                }
                _ => unreachable!(),
            }
        }

        Ok(Camera::Orthographic(ortho))
    };

    trace("orthographic_camera_params", params).parse_next(input)
}

fn shape_directive(input: &mut &str) -> PResult<Shape> {
    trace(
        "shape_directive",
        dispatch! { cut_err(terminated(delimited('"', alpha1, '"'), space1));
            "sphere" => sphere_params,
            _=> fail.context(StrContext::Label("shape type"))
        },
    )
    .parse_next(input)
}

fn sphere_params(input: &mut &str) -> PResult<Shape> {
    let params = |input: &mut &str| {
        let items = expected_params_map(vec![
            "float alpha",
            //"texture alpha",
            "float radius",
            "float zmin",
            "float zmax",
            "float phimax",
        ]);
        let found_params = param_list(items).parse_next(input)?;
        let mut sphere = Sphere::default();
        for (k, v) in found_params {
            match k.as_str() {
                "alpha" => {
                    sphere.alpha = Alpha::Constant(*v.as_single().unwrap().as_float().unwrap());
                }
                "radius" => {
                    sphere.radius = *v.as_single().unwrap().as_float().unwrap();
                    sphere.z_min.get_or_insert(-sphere.radius);
                    sphere.z_max.get_or_insert(sphere.radius);
                }
                "zmin" => {
                    sphere.z_min = Some(*v.as_single().unwrap().as_float().unwrap());
                }
                "zmax" => {
                    sphere.z_max = Some(*v.as_single().unwrap().as_float().unwrap());
                }
                "phimax" => {
                    sphere.phi_max = *v.as_single().unwrap().as_float().unwrap();
                }
                _ => unreachable!(),
            }
        }

        Ok(Shape::Sphere(sphere))
    };

    trace("sphere_params", params).parse_next(input)
}

/// Helper for listing the possible parameters for a directive with strings
fn expected_params_map(descriptions: Vec<&str>) -> HashMap<String, (ValueType, SingleValueType)> {
    let mut desc = (
        separated_pair(single_ty, space1, param_name),
        opt(preceded(space1, "[]")).map(|is_arr| {
            if is_arr.is_some() {
                ValueType::Array
            } else {
                ValueType::Single
            }
        }),
    )
        .map(|((single_ty, name), ty)| (name.to_owned(), (ty, single_ty)));

    HashMap::from_iter(
        descriptions
            .into_iter()
            .map(|s| desc.parse(s).expect("description must be valid")),
    )
}

fn param_list(
    possible: HashMap<String, (ValueType, SingleValueType)>,
) -> impl FnMut(&mut &str) -> PResult<HashMap<String, Value>> {
    move |input: &mut &str| {
        // Will be used in a moment to check for uniqueness in the listed params.
        let mut unseen = possible.clone();
        // Parses one parameter that matches an entry in the possible params:
        let possible_param = |input: &mut &str| {
            let start = input.checkpoint();
            // Parse anything matching the format for a single param.
            let (name, value) = param.parse_next(input)?;
            // First, verify that parameter name should be possible here.
            if !possible.contains_key(&name) {
                input.reset(&start);
                return Err(ErrMode::from_error_kind(input, ErrorKind::Verify)
                    .add_context(
                        input,
                        &start,
                        StrContext::Label("parameter for this directive"),
                    )
                    .cut());
            }
            let (ty, single_ty) = possible[&name];
            // Then, verify that the type is correct for that param.
            if !match value {
                // (check for single)
                Value::Single(ref single) => ty == ValueType::Single && single_ty == single.into(),
                // (check for array)
                Value::Array(ref arr) => {
                    ty == ValueType::Array && arr.iter().all(|v| single_ty == v.into())
                }
            } {
                input.reset(&start);
                return Err(ErrMode::from_error_kind(input, ErrorKind::Verify)
                    .add_context(
                        input,
                        &start,
                        StrContext::Label("type for the named parameter"),
                    )
                    .cut());
            }
            // Finally, verify that this parameter hasn't already been seen
            // (by removing from the captured "unseen" map)
            if unseen.remove(&name).is_none() {
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
        let expected_param_list = cut_err(separated(0.., possible_param, multispace1))
            .map(|list: Vec<_>| HashMap::from_iter(list));

        let mut traced = trace("expected_param_list", expected_param_list);
        traced.parse_next(input)
    }
}

fn single_ty(input: &mut &str) -> PResult<SingleValueType> {
    alphanumeric1
        .verify_map(|ty| SingleValueType::from_str(ty).ok())
        .context(StrContext::Expected(StrContextValue::Description(
            "parameter type",
        )))
        .parse_next(input)
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

fn literal(input: &mut &str) -> PResult<Literal> {
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

fn atomic_literal(input: &mut &str) -> PResult<AtomicLiteral> {
    let boolean = alt(("true".value(true), "false".value(false)));
    let string = delimited('"', alphanumeric0, '"');
    let val = alt((
        float.map(AtomicLiteral::Num),
        boolean.map(AtomicLiteral::Bool),
        string.map(|s: &str| AtomicLiteral::Str(s.to_string())),
    ));

    trace("atomic_literal", val).parse_next(input)
}

#[cfg(test)]
mod test {
    use super::*;
    use std::fmt::{Debug, Display};
    use winnow::stream::{AsBStr, StreamIsPartial};

    fn assert_parses_to<I, O, E>(parser: impl Parser<I, O, E>, input: I, expected_output: O)
    where
        I: Stream + StreamIsPartial + AsBStr,
        O: PartialEq + Debug,
        E: ParserError<I> + Debug + Display,
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
        O: Debug,
        E: ParserError<I> + Debug + Display,
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

    #[test]
    fn test_orthographic() {
        assert_eq!(
            Ok(GlobalDirective::Camera(Camera::Orthographic(
                OrthographicCamera {
                    shutter_open: 1.2,
                    shutter_close: 2.4,
                    ..Default::default()
                }
            ))),
            global_directive(false).parse(
                &mut r#"Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4"#
            )
        );
    }

    #[test]
    fn test_directive_invalid_name() {
        assert!(
            global_directive(false).parse(
                &mut r#"ThisIsNotARealDirective "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4"#
            ).is_err()
        );
    }

    #[test]
    fn test_global_options() {
        must_parse_ok(
            global_options(false),
            r#"Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4"#,
            true,
        );
    }

    #[test]
    fn test_global_options_fail_on_repeat_directive() {
        assert!(global_options(false)
            .parse(
                r#"Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4
                   Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4
                   WorldBegin"#,
            )
            .is_err());
    }

    #[test]
    fn test_scene() {
        must_parse_ok(
            scene(false),
            r#"Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4
                    WorldBegin
                    Shape "sphere" "float radius" 0.25"#,
            true,
        );
    }
}
