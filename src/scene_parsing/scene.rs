use std::{cell::OnceCell, collections::HashMap, io::Read};

use uuid::Uuid;

use crate::{
    core::Transform,
    scene_parsing::directives::{AreaLightDesc, MaterialDesc},
};

use super::{
    common::{directive, Directive, FromEntity, GraphicsState, PbrtParseError},
    directives::{
        Accelerator, Camera, ColorSpace, Film, Filter, FromTextureDirective as _, Integrator,
        LightDesc, Sampler, ShapeDesc, TextureDesc,
    },
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
    pub accelerator: Accelerator,
}

#[derive(Debug, Default)]
struct OptionsBuilder {
    camera: OnceCell<Camera>,
    sampler: OnceCell<Sampler>,
    color_space: OnceCell<ColorSpace>,
    film: OnceCell<Film>,
    filter: OnceCell<Filter>,
    integrator: OnceCell<Integrator>,
    accelerator: OnceCell<Accelerator>,
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
        let accelerator = self.accelerator.take().unwrap_or_default();

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
            accelerator,
        })
    }
}

#[derive(Debug, Default)]
pub struct World {
    pub shapes: Vec<ShapeDesc>,
    pub lights: Vec<LightDesc>,
    pub area_light_shapes: Vec<(AreaLightDesc, Vec<ShapeDesc>)>,
    pub textures: HashMap<String, TextureDesc>,
    pub materials: HashMap<String, MaterialDesc>,
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
    let world = parse_world_section(&mut input, ignore_unrecognized_directives)?;

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
    let mut state = GraphicsState::default();

    loop {
        let directive = directive(input).map_err(|e| e.into_inner().unwrap())?;
        match directive {
            Directive::Entity(entity) => match entity.identifier {
                "Camera" => {
                    options_builder
                        .camera
                        .set(Camera::from_entity(entity, &state)?)
                        .map_err(|_| PbrtParseError::RepeatedDirective("Camera".to_string()))?;
                }
                "Sampler" => {
                    options_builder
                        .sampler
                        .set(Sampler::from_entity(entity, &state)?)
                        .map_err(|_| PbrtParseError::RepeatedDirective("Sampler".to_string()))?;
                }
                "ColorSpace" => {
                    options_builder
                        .color_space
                        .set(ColorSpace::from_entity(entity, &state)?)
                        .map_err(|_| PbrtParseError::RepeatedDirective("ColorSpace".to_string()))?;
                }
                "Film" => {
                    options_builder
                        .film
                        .set(Film::from_entity(entity, &state)?)
                        .map_err(|_| PbrtParseError::RepeatedDirective("Film".to_string()))?;
                }
                "Filter" => {
                    options_builder
                        .filter
                        .set(Filter::from_entity(entity, &state)?)
                        .map_err(|_| PbrtParseError::RepeatedDirective("Filter".to_string()))?;
                }
                "Integrator" => {
                    options_builder
                        .integrator
                        .set(Integrator::from_entity(entity, &state)?)
                        .map_err(|_| PbrtParseError::RepeatedDirective("Integrator".to_string()))?;
                }
                "Accelerator" => {
                    options_builder
                        .accelerator
                        .set(Accelerator::from_entity(entity, &state)?)
                        .map_err(|_| {
                            PbrtParseError::RepeatedDirective("Accelerator".to_string())
                        })?;
                }
                invalid_name => {
                    if !ignore_unrecognized_directives {
                        return Err(PbrtParseError::UnrecognizedDirective(
                            invalid_name.to_owned(),
                        ));
                    }
                }
            },
            Directive::Transform(transform_directive) => {
                state.current_transform =
                    Transform::from(transform_directive) * state.current_transform;
            }
            Directive::Texture(_) => {
                return Err(PbrtParseError::IllegalForSection("Texture".to_string()));
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
            Directive::ReverseOrientation => {
                return Err(PbrtParseError::IllegalForSection(
                    "ReverseOrientation".to_string(),
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
    let mut world = World::default();
    let mut state = GraphicsState::default();
    let mut stored_states_stack = Vec::new();

    loop {
        if input.is_empty() {
            break;
        }
        let directive = directive(input).map_err(|e| e.into_inner().unwrap())?;
        match directive {
            Directive::Entity(entity) => match entity.identifier {
                "Shape" => {
                    let shape = ShapeDesc::from_entity(entity, &state)?;
                    if let Some(light_i) = state.current_area_light_index {
                        world.area_light_shapes[light_i].1.push(shape);
                    } else {
                        world.shapes.push(shape);
                    }
                }
                "LightSource" => {
                    world.lights.push(LightDesc::from_entity(entity, &state)?);
                }
                "AreaLightSource" => {
                    // Create area light desc
                    let area_light_desc = AreaLightDesc::from_entity(entity, &state)?;
                    // Create entry for it
                    world.area_light_shapes.push((area_light_desc, Vec::new()));
                    // Set its index as current, so later shapes in scope
                    // can be added to the entry
                    state.current_area_light_index = Some(world.area_light_shapes.len() - 1);
                }
                "Material" => {
                    // Convert to material description
                    let material_desc = MaterialDesc::from_entity(entity, &state)?;
                    // Use random UUID as identifier for unnamed material
                    let random_name = Uuid::new_v4();
                    // Insert into map of all materials
                    world
                        .materials
                        .insert(random_name.to_string(), material_desc);
                    // Set as current name in graphics state
                    state.current_material_name = Some(random_name.to_string());
                }
                "MakeNamedMaterial" => {
                    // Get name
                    let name = entity.identifier.to_owned();
                    // Convert to material description
                    let material_desc = MaterialDesc::from_entity(entity, &state)?;
                    // Insert into map of all materials
                    let existing = world.materials.insert(name.clone(), material_desc);
                    // If material name already existed, this is redefining, error
                    if existing.is_some() {
                        return Err(PbrtParseError::RedefinedName(name));
                    }
                }
                "NamedMaterial" => {
                    // Just set the current material name in the state
                    state.current_material_name = Some(entity.identifier.to_owned());
                    // Not checking if it exists now allows for defining the material afterwards
                }
                invalid_name => {
                    if !ignore_unrecognized_directives {
                        return Err(PbrtParseError::UnrecognizedDirective(
                            invalid_name.to_owned(),
                        ));
                    }
                }
            },
            Directive::Transform(transform_directive) => {
                state.current_transform =
                    Transform::from(transform_directive) * state.current_transform;
            }
            Directive::Texture(texture_directive) => {
                let name = texture_directive.name.to_owned();
                let texture = TextureDesc::from_directive(texture_directive, &state)?;
                if world.textures.insert(name.clone(), texture).is_some() {
                    return Err(PbrtParseError::RedefinedName(name));
                }
            }
            Directive::WorldBegin => {
                return Err(PbrtParseError::IllegalForSection("WorldBegin".to_string()));
            }
            Directive::AttributeBegin => {
                stored_states_stack.push(state.clone());
            }
            Directive::AttributeEnd => {
                if stored_states_stack.is_empty() {
                    return Err(PbrtParseError::IllegalForSection(
                        "AttributeEnd".to_string(),
                    ));
                }
                state = stored_states_stack.pop().unwrap();
            }
            Directive::ReverseOrientation => {
                state.reverse_orientation = !state.reverse_orientation;
            }
        }
    }

    if !stored_states_stack.is_empty() {
        return Err(PbrtParseError::UnclosedAttributeScope);
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
                        ThisIsNotARealDirective "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4
                        WorldBegin"#,
                false
            ).is_err()
        );
    }

    #[test]
    fn test_ignore_invalid_directive() {
        assert!(
            parse_options_section(
                &mut r#"Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4
                        ThisIsNotARealDirective "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4
                        WorldBegin"#,
                true
            ).is_ok()
        );
    }

    #[test]
    fn test_global_options_ok() {
        assert!(parse_options_section(
            &mut r#"Camera "orthographic" "float shutteropen" 1.2 "float shutterclose" 2.4
                    WorldBegin"#,
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
                Material "dielectric"
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
