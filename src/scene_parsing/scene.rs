use std::{cell::OnceCell, io::Read};

use crate::core::Transform;

use super::{
    common::{directive, Directive, FromEntity, ParseContext},
    directives::{Camera, ColorSpace, Film, Filter, Integrator, Light, Sampler, Shape},
    PbrtParseError,
};

#[derive(Debug)]
pub struct SceneDescription {
    pub options: Options,
    pub world: World,
}

#[derive(Debug)]
pub struct Options {
    pub camera: Camera,
    pub sampler: Sampler,
    pub color_space: ColorSpace,
    pub film: Film,
    pub filter: Filter,
    pub integrator: Integrator,
}

#[derive(Debug, Default)]
struct OptionsBuilder {
    camera: OnceCell<Camera>,
    sampler: OnceCell<Sampler>,
    color_space: OnceCell<ColorSpace>,
    film: OnceCell<Film>,
    filter: OnceCell<Filter>,
    integrator: OnceCell<Integrator>,
}

impl OptionsBuilder {
    fn empty() -> Self {
        Self::default()
    }

    fn build(mut self) -> Result<Options, PbrtParseError> {
        let film = self.film.take().unwrap_or_default();
        let mut camera = self.camera.take().unwrap_or_default();
        let sampler = self.sampler.take().unwrap_or_default();
        let color_space = self.color_space.take().unwrap_or_default();
        let filter = self.filter.take().unwrap_or_default();
        let integrator = self.integrator.take().unwrap_or_default();

        // Camera may use film to determine some defaults, but it may come
        // before film, so it's provided here
        camera.update_with_film(&film);

        Ok(Options {
            camera,
            sampler,
            color_space,
            film,
            filter,
            integrator,
        })
    }
}

#[derive(Debug, Default)]
pub struct World {
    pub shapes: Vec<Shape>,
    pub lights: Vec<Light>,
}

pub(super) fn parse_pbrt_file(
    mut file: impl Read,
    ignore_unrecognized_directives: bool,
) -> Result<SceneDescription, PbrtParseError> {
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;
    buf = strip_comments(buf);

    let mut input: &str = buf.as_str();
    let options = parse_options_section(&mut input, ignore_unrecognized_directives)?;
    let world = parse_world_section(&mut input, &options, ignore_unrecognized_directives)?;

    Ok(SceneDescription { options, world })
}

/// Remove all comments, which start with a # character and continue to the end of the line.
///
/// Also trims any remaining trailing whitespace.
fn strip_comments(input: String) -> String {
    input
        .lines()
        .map(|line| {
            // For each line, look for first '#' char.
            // Keep slice up to that '#', and also trim remaining trailing whitespace.
            // If no '#', no change.
            if let Some(comment_start) = line.find('#') {
                line[..comment_start].trim_end()
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_options_section(
    input: &mut &str,
    ignore_unrecognized_directives: bool,
) -> Result<Options, PbrtParseError> {
    let options_builder = OptionsBuilder::empty();
    let mut context = ParseContext::default();

    while let Ok(directive) = directive(input) {
        match directive {
            Directive::Entity(entity) => match entity.identifier {
                "Camera" => {
                    options_builder
                        .camera
                        .set(Camera::from_entity(entity, &context)?)
                        .map_err(|_| PbrtParseError::RepeatedDirective("Camera".to_string()))?;
                }
                "Sampler" => {
                    options_builder
                        .sampler
                        .set(Sampler::from_entity(entity, &context)?)
                        .map_err(|_| PbrtParseError::RepeatedDirective("Sampler".to_string()))?;
                }
                "ColorSpace" => {
                    options_builder
                        .color_space
                        .set(ColorSpace::from_entity(entity, &context)?)
                        .map_err(|_| PbrtParseError::RepeatedDirective("ColorSpace".to_string()))?;
                }
                "Film" => {
                    options_builder
                        .film
                        .set(Film::from_entity(entity, &context)?)
                        .map_err(|_| PbrtParseError::RepeatedDirective("Film".to_string()))?;
                }
                "Filter" => {
                    options_builder
                        .filter
                        .set(Filter::from_entity(entity, &context)?)
                        .map_err(|_| PbrtParseError::RepeatedDirective("Filter".to_string()))?;
                }
                "Integrator" => {
                    options_builder
                        .integrator
                        .set(Integrator::from_entity(entity, &context)?)
                        .map_err(|_| PbrtParseError::RepeatedDirective("Integrator".to_string()))?;
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
                context.current_transform =
                    Transform::from(transform_directive) * context.current_transform;
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
    options: &Options,
    ignore_unrecognized_directives: bool,
) -> Result<World, PbrtParseError> {
    let mut world = World::default();
    let mut context = ParseContext {
        color_space: Some(options.color_space),
        ..Default::default()
    };

    while let Ok(directive) = directive(input) {
        match directive {
            Directive::Entity(entity) => match entity.identifier {
                "Shape" => {
                    world.shapes.push(Shape::from_entity(entity, &context)?);
                }
                "Light" => {
                    world.lights.push(Light::from_entity(entity, &context)?);
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
                context.current_transform =
                    Transform::from(transform_directive) * context.current_transform;
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

    fn file_must_parse_ok(input: &mut impl Read, print_ok_result: bool) -> SceneDescription {
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

    #[test]
    fn test_strip_comments() {
        assert_eq!(
            strip_comments(
                r#"LookAt 3 4 1.5  # eye
.5 .5 0  # look at point LookAt
0 0 1    # up vector
Camera "perspective" "float fov" 45
# more comments"#
                    .to_string()
            ),
            r#"LookAt 3 4 1.5
.5 .5 0
0 0 1
Camera "perspective" "float fov" 45
"#
        );
    }
}
