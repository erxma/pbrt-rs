use std::{cell::OnceCell, io::Read};

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

#[derive(Debug)]
pub struct Options {
    camera: Camera,
}

#[derive(Debug, Default)]
struct OptionsBuilder {
    camera: OnceCell<Camera>,
}

impl OptionsBuilder {
    fn empty() -> Self {
        Self::default()
    }

    fn build(mut self) -> Result<Options, PbrtParseError> {
        let camera = self
            .camera
            .take()
            .ok_or(PbrtParseError::MissingRequiredOption("Camera".to_string()))?;
        Ok(Options { camera })
    }
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
    let options = parse_options_section(&mut input, ignore_unrecognized_directives)?;
    let world = parse_world_section(&mut input, ignore_unrecognized_directives)?;

    Ok(Scene { options, world })
}

fn parse_options_section(
    input: &mut &str,
    ignore_unrecognized_directives: bool,
) -> Result<Options, PbrtParseError> {
    let mut current_transform = Transform::IDENTITY;
    let mut options_builder = OptionsBuilder::empty();

    while let Ok(directive) = directive(input) {
        match directive {
            Directive::Entity(entity) => match entity.identifier {
                "Camera" => {
                    options_builder
                        .camera
                        .set(entity.try_into()?)
                        .map_err(|_| PbrtParseError::RepeatedDirective("Camera".to_string()))?;
                }
                invalid_name => {
                    if !ignore_unrecognized_directives {
                        return Err(PbrtParseError::UnrecognizedOrIllegalDirective(
                            invalid_name.to_owned(),
                        ));
                    }
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

fn parse_world_section(
    input: &mut &str,
    ignore_unrecognized_directives: bool,
) -> Result<World, PbrtParseError> {
    let mut current_transform = Transform::IDENTITY;
    let mut world = World::default();

    while let Ok(directive) = directive(input) {
        match directive {
            Directive::Entity(entity) => match entity.identifier {
                "Shape" => {
                    world.shapes.push(entity.try_into()?);
                }
                invalid_name => {
                    if !ignore_unrecognized_directives {
                        return Err(PbrtParseError::UnrecognizedOrIllegalDirective(
                            invalid_name.to_owned(),
                        ));
                    }
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

    #[test]
    fn test_invalid_directive_name() {
        assert!(
            parse_options_section(
                &mut r#"Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4
                        ThisIsNotARealDirective "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4"#,
                false
            ).is_err()
        );
    }

    #[test]
    fn test_ignore_invalid_directive() {
        assert!(
            parse_options_section(
                &mut r#"Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4
                        ThisIsNotARealDirective "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4"#,
                true
            ).is_ok()
        );
    }

    #[test]
    fn test_global_options_ok() {
        assert!(parse_options_section(
            &mut r#"Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4"#,
            false
        )
        .is_ok());
    }

    #[test]
    fn test_global_options_fail_on_repeat_directive() {
        assert!(parse_options_section(
            &mut r#"Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4
                   Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4
                   WorldBegin"#,
            false
        )
        .is_err());
    }

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
