use std::io::Read;

use crate::core::Transform;

use super::{
    common::{directive, Directive},
    directives::{Camera, Shape, TransformDirective},
    PbrtParseError,
};

#[derive(Debug)]
pub struct Scene {
    options: Options,
    world: World,
}

#[derive(Debug, derive_builder::Builder)]
pub struct Options {
    camera: Camera,
}

#[derive(Debug, Default)]
pub struct World {
    shapes: Vec<Shape>,
}

pub fn parse_pbrt_file(
    mut file: impl Read,
    ignore_unrecognized_directives: bool,
) -> Result<Scene, PbrtParseError> {
    let mut buf = String::new();
    file.read_to_string(&mut buf);

    let mut input: &str = buf.as_str();
    let options = parse_options_section(&mut input)?;
    let world = parse_world_section(&mut input)?;

    Ok(Scene { options, world })
}

fn parse_options_section(input: &mut &str) -> Result<Options, PbrtParseError> {
    let mut current_transform = Transform::IDENTITY;
    let mut options_builder = OptionsBuilder::create_empty();

    while let Ok(directive) = directive(input) {
        match directive {
            Directive::Entity(entity) => match entity.identifier {
                "Camera" => {
                    options_builder.camera(entity.try_into()?);
                }
                invalid_name => {
                    return Err(PbrtParseError::UnrecognizedOrIllegalDirective(
                        invalid_name.to_owned(),
                    ))
                }
            },
            Directive::Transform(transform_directive) => {
                current_transform = Transform::from(transform_directive) * current_transform;
            }
            Directive::WorldBegin => {
                break;
            }
            Directive::AttributeBegin => {
                return Err(PbrtParseError::IllegalForSection(
                    "AttributeBegin".to_string(),
                ));
            }
            Directive::AttributeEnd => {
                return Err(PbrtParseError::IllegalForSection(
                    "AttributeEnd".to_string(),
                ));
            }
        }
    }

    options_builder
        .build()
        .map_err(|err| PbrtParseError::MissingRequiredOption(err.to_string()))
}

fn parse_world_section(input: &mut &str) -> Result<World, PbrtParseError> {
    let mut current_transform = Transform::IDENTITY;
    let mut world = World::default();

    while let Ok(directive) = directive(input) {
        match directive {
            Directive::Entity(entity) => match entity.identifier {
                "Shape" => {
                    world.shapes.push(entity.try_into()?);
                }
                invalid_name => {
                    return Err(PbrtParseError::UnrecognizedOrIllegalDirective(
                        invalid_name.to_owned(),
                    ))
                }
            },
            Directive::Transform(transform_directive) => {
                current_transform = Transform::from(transform_directive) * current_transform;
            }
            Directive::WorldBegin => {
                return Err(PbrtParseError::IllegalForSection("WorldBegin".to_string()));
            }
            Directive::AttributeBegin => {
                todo!()
            }
            Directive::AttributeEnd => {
                todo!()
            }
        }
    }

    Ok(world)
}

/*
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

        let options = Options::new(dirs);
        options.map_err(|_| {
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
                let mut world = World::default();
                for d in dirs {
                    world.add_option(d);
                    }
                    world
                    })
        .parse_next(input)
        }
        }
*/

#[cfg(test)]
mod test {
    use std::io::Cursor;

    use super::*;
    use winnow::stream::{AsBStr, StreamIsPartial};

    fn file_must_parse_ok(input: &mut impl Read, print_ok_result: bool) -> Scene {
        let result = parse_pbrt_file(input, false);
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

    /*
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
    */

    #[test]
    fn test_scene_parse_ok() {
        file_must_parse_ok(
            &mut Cursor::new(
                r#"Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4
                WorldBegin
                Shape "sphere" "float radius" 0.25"#,
            ),
            true,
        );
    }
}
