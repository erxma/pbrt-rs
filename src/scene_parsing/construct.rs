use std::{
    borrow::Cow,
    collections::{hash_map, HashMap},
    io::Read,
    path::PathBuf,
    sync::Arc,
};

use log::info;
use thiserror::Error;

use crate::{
    camera::*,
    color::{RGBColorSpace, SRGB},
    core::{common::*, constants::PI},
    imaging::{BoxFilter, FilterEnum, GaussianFilter, Image, TriangleFilter},
    integrators::*,
    lights::*,
    materials::*,
    media::MediumInterface,
    primitives::*,
    sampling::{
        spectrum::{self, *},
        IndependentSampler, SamplerEnum, StratifiedSampler,
    },
    shapes::*,
    util::error::BuilderError,
};

use super::{
    common::{PbrtParseError, SpectrumDesc},
    directives::*,
    scene::parse_pbrt_file,
};

pub fn create_scene_integrator(
    scene_file: impl Read,
    out_file: Option<PathBuf>,
    ignore_unrecognized_directives: bool,
) -> Result<IntegratorEnum, ReadSceneError> {
    info!("Scene file parsing begin");
    let description = parse_pbrt_file(scene_file, ignore_unrecognized_directives)?;
    info!("Scene file parsing complete");

    info!("Scene construction begin");
    let filter = create_filter(description.options.filter);
    let color_space = get_color_space(description.options.color_space);
    let film = create_film(
        description.options.film,
        filter,
        color_space,
        description.options.camera.exposure_time(),
        out_file,
    )?;
    let camera = create_camera(description.options.camera, film);
    let sampler = create_sampler(description.options.sampler);

    let mut lights = create_lights(description.world.lights, &camera, color_space);

    let mut textures = Textures {
        uncreated_descs: description.world.textures,
        ..Default::default()
    };

    let mut meshes = Meshes::default();
    let materials = create_materials(description.world.materials, color_space, &mut textures)?;
    let mut primitives = create_primitives_for_shapes(
        description.world.shapes,
        &mut textures,
        &materials,
        &mut meshes,
        &camera,
    )?;

    let mut area_light_products = create_area_lights_and_primitives(
        description.world.area_light_shapes,
        &mut textures,
        &materials,
        &mut meshes,
        &camera,
        color_space,
    )?;
    lights.append(&mut area_light_products.lights);
    primitives.append(&mut area_light_products.primitives);

    BilinearPatchMesh::init_mesh_data(meshes.bilinear_patches);
    TriangleMesh::init_mesh_data(meshes.triangles);

    let aggregate = create_aggregate(description.options.accelerator, primitives);
    info!("Scene bounds: {}", aggregate.bounds());

    let integrator = create_integrator(
        description.options.integrator,
        camera,
        sampler,
        aggregate,
        lights,
    );
    info!("Scene construction complete");

    Ok(integrator)
}

#[derive(Error, Debug)]
pub enum ReadSceneError {
    #[error("failed to parse pbrt scene file: {0}")]
    ParseError(#[from] PbrtParseError),
    #[error("error during construct: {0}")]
    BuilderError(#[from] BuilderError),
    #[error("error when loading images: {0}")]
    ImageError(#[from] image::ImageError),
    #[error("texture `{name}` isn't valid for its usage, which expects {expected}")]
    TextureMismatch { name: String, expected: String },
    #[error("texture `{0}` is not defined")]
    UndefinedTexture(String),
    #[error("material `{0}` is not defined")]
    UndefinedMaterial(String),
    #[error("failed to read PLY file: {0}")]
    PlyError(#[from] FromPlyError),
}

fn create_filter(desc: FilterDesc) -> FilterEnum {
    match desc {
        FilterDesc::Box(desc) => BoxFilter::new(Vec2f::new(desc.x_radius, desc.y_radius)).into(),
        FilterDesc::Gaussian(desc) => {
            GaussianFilter::new(Vec2f::new(desc.x_radius, desc.y_radius), desc.std).into()
        }
        FilterDesc::Triangle(desc) => {
            TriangleFilter::new(Vec2f::new(desc.x_radius, desc.y_radius)).into()
        }
    }
}

fn get_color_space(desc: ColorSpaceDesc) -> &'static RGBColorSpace {
    match desc {
        ColorSpaceDesc::Srgb => &SRGB,
    }
}

fn create_film(
    desc: FilmDesc,
    filter: FilterEnum,
    color_space: &'static RGBColorSpace,
    exposure_time: Float,
    override_filename: Option<PathBuf>,
) -> Result<Film, ReadSceneError> {
    let film = match desc {
        FilmDesc::Rgb(desc) => RGBFilm::new(RGBFilmParams {
            full_resolution: Point2i::new(desc.x_resolution as i32, desc.y_resolution as i32),
            pixel_bounds: Bounds2i::from(desc.pixel_bounds.map(|v| v as i32)),
            filter: Arc::new(filter),
            diagonal: desc.diagonal,
            sensor: Arc::new(create_sensor(
                desc.sensor,
                color_space,
                exposure_time,
                desc.iso,
                desc.white_balance_temp,
            )?),
            filename: override_filename.unwrap_or(desc.filename),
            color_space,
            max_component_value: desc.max_component_value,
        })
        .into(),
    };

    Ok(film)
}

fn create_sensor(
    name: SensorName,
    color_space: &RGBColorSpace,
    exposure_time: Float,
    iso: Float,
    white_balance_temp: Option<Float>,
) -> Result<PixelSensor, ReadSceneError> {
    // Note from the original pbrt:
    // "In the talk we mention using 312.5 for historical reasons. The
    // choice of 100 here just means that the other parameters make nice
    // 'round' numbers like 1 and 100."
    let imaging_ratio = exposure_time * iso / 100.0;

    let sensor = match name {
        SensorName::Cie1931 => PixelSensor::with_xyz_matching(color_space, None, imaging_ratio),
        name => {
            let sensor_illum = spectrum::illum_d(white_balance_temp.unwrap_or(6500.0));

            let r = spectrum::get_named_spectrum(&format!("{name}_r"))
                .unwrap_or_else(|| panic!("RGB matching spectra for {name} should be available"));
            let g = spectrum::get_named_spectrum(&format!("{name}_g")).unwrap();
            let b = spectrum::get_named_spectrum(&format!("{name}_b")).unwrap();

            PixelSensor::with_rgb_matching(color_space, r, g, b, &sensor_illum, imaging_ratio)
        }
    };
    Ok(sensor)
}

fn create_camera(desc: CameraDesc, film: Film) -> CameraEnum {
    match desc {
        CameraDesc::Orthographic(desc) => OrthographicCamera::builder()
            .film(film)
            .focal_distance(desc.focal_distance)
            .lens_radius(desc.lens_radius)
            .screen_window(desc.screen_window.unwrap().into())
            .shutter_period(desc.shutter_open..desc.shutter_close)
            .world_from_camera(desc.camera_from_world.inverse())
            .build()
            .into(),

        CameraDesc::Perspective(desc) => PerspectiveCamera::builder()
            .film(film)
            .focal_distance(desc.focal_distance)
            .fov(desc.fov_degs)
            .lens_radius(desc.lens_radius)
            .screen_window(desc.screen_window.unwrap().into())
            .shutter_period(desc.shutter_open..desc.shutter_close)
            .world_from_camera(desc.camera_from_world.inverse())
            .build()
            .into(),
    }
}

fn create_sampler(desc: SamplerDesc) -> SamplerEnum {
    match desc {
        SamplerDesc::Independent(desc) => {
            IndependentSampler::new(desc.pixel_samples, Some(desc.seed)).into()
        }
        SamplerDesc::Stratified(desc) => {
            StratifiedSampler::new(desc.x_samples, desc.y_samples, desc.jitter, Some(desc.seed))
                .into()
        }
    }
}

fn create_integrator(
    desc: IntegratorDesc,
    camera: CameraEnum,
    sampler: SamplerEnum,
    aggregate: PrimitiveEnum,
    lights: Vec<Arc<LightEnum>>,
) -> IntegratorEnum {
    match desc {
        IntegratorDesc::RandomWalk(desc) => {
            RandomWalkIntegrator::new(desc.max_depth, camera, sampler, aggregate, lights).into()
        }
        IntegratorDesc::SimplePath(desc) => SimplePathIntegrator::new(
            desc.max_depth,
            desc.sample_lights,
            desc.sample_bsdf,
            camera,
            sampler,
            aggregate,
            lights,
        )
        .into(),
        IntegratorDesc::Path(desc) => PathIntegrator::new(
            desc.max_depth,
            desc.regularize,
            camera,
            sampler,
            aggregate,
            lights,
            desc.light_sampler,
        )
        .into(),
    }
}

fn create_lights(
    descs: impl IntoIterator<Item = LightDesc>,
    camera: &impl Camera,
    color_space: &'static RGBColorSpace,
) -> Vec<Arc<LightEnum>> {
    descs
        .into_iter()
        .map(|desc| create_light(desc, camera, color_space))
        .collect()
}

fn create_light(
    desc: LightDesc,
    camera: &impl Camera,
    color_space: &'static RGBColorSpace,
) -> Arc<LightEnum> {
    let light = match desc {
        LightDesc::Distant(desc) => {
            let w = (desc.from - desc.to).normalized();
            let (w, v1, v2) = w.coordinate_system();
            let transform = Transform::from_arr([
                [v1.x(), v2.x(), w.x(), 0.0],
                [v1.y(), v2.y(), w.y(), 0.0],
                [v1.z(), v2.z(), w.z(), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]);
            let final_render_from_light = camera
                .camera_transform()
                .render_from_world(desc.world_from_light * transform);

            let radiance = match desc.radiance {
                Some(spec) => Cow::Owned(
                    create_spectrum(spec, SpectrumType::Illuminant, color_space).unwrap(),
                ),
                None => Cow::Borrowed(&color_space.illuminant),
            };

            let mut scale = desc.scale;
            // Scale the light spectrum to be equivalent to 1 nit
            scale /= radiance.to_photometric();
            // Adjust scale to meet target illuminance value.
            // Like for IBLs we measure illuminance as incident on an upward-facing patch.
            if let Some(illuminance) = desc.illuminance {
                scale *= illuminance;
            }

            DirectionalLight::new(final_render_from_light, &*radiance, scale).into()
        }

        LightDesc::Infinite(desc) => {
            // TODO: Support other infinite lights once implemented
            let radiance = match desc.radiance {
                Some(spec) => Cow::Owned(
                    create_spectrum(spec, SpectrumType::Illuminant, color_space).unwrap(),
                ),
                None => {
                    // Default: color space's std illuminant
                    Cow::Borrowed(&color_space.illuminant)
                }
            };

            let mut scale = desc.scale;
            // Scale the light spectrum to be equivalent to 1 nit
            scale /= radiance.to_photometric();
            if let Some(illuminance) = desc.illuminance {
                // If the scene specifies desired illuminance, first calculate
                // the illuminance from a uniform hemispherical emission
                // of L_v then use this to scale the emission spectrum.
                let k_e = PI;
                scale *= illuminance / k_e;
            }

            UniformInfiniteLight::new(&*radiance, scale).into()
        }
    };

    Arc::new(light)
}

#[derive(Debug, Default)]
struct Textures {
    float_textures: HashMap<String, Arc<FloatTextureEnum>>,
    albedo_spectrum_textures: HashMap<String, Arc<SpectrumTextureEnum>>,
    unbounded_spectrum_textures: HashMap<String, Arc<SpectrumTextureEnum>>,
    illuminant_spectrum_textures: HashMap<String, Arc<SpectrumTextureEnum>>,

    uncreated_descs: HashMap<String, TextureDesc>,
}

impl Textures {
    fn get_named_float_texture(
        &mut self,
        name: impl Into<String>,
    ) -> Result<Arc<FloatTextureEnum>, ReadSceneError> {
        let texture = match self.float_textures.entry(name.into()) {
            hash_map::Entry::Occupied(occupied) => occupied.get().clone(),
            hash_map::Entry::Vacant(vacant) => {
                let name = vacant.key();
                let desc = self
                    .uncreated_descs
                    .remove(name)
                    .ok_or_else(|| ReadSceneError::UndefinedTexture(name.clone()))?
                    .into_float()
                    .map_err(|_| ReadSceneError::TextureMismatch {
                        name: name.to_owned(),
                        expected: "float texture".to_string(),
                    })?;
                let texture = Arc::new(create_float_texture(desc)?);
                vacant.insert(texture.clone());
                texture
            }
        };

        Ok(texture)
    }

    fn get_named_spectrum_texture(
        &mut self,
        name: impl Into<String>,
        spectrum_type: SpectrumType,
        color_space: &'static RGBColorSpace,
    ) -> Result<Arc<SpectrumTextureEnum>, ReadSceneError> {
        let spectra = match spectrum_type {
            SpectrumType::Albedo => &mut self.albedo_spectrum_textures,
            SpectrumType::Unbounded => &mut self.unbounded_spectrum_textures,
            SpectrumType::Illuminant => &mut self.illuminant_spectrum_textures,
        };

        let texture = match spectra.entry(name.into()) {
            hash_map::Entry::Occupied(occupied) => occupied.get().clone(),
            hash_map::Entry::Vacant(vacant) => {
                let name = vacant.key();
                let desc = self
                    .uncreated_descs
                    .remove(name)
                    .ok_or_else(|| ReadSceneError::UndefinedTexture(name.clone()))?
                    .into_spectrum()
                    .map_err(|_| ReadSceneError::TextureMismatch {
                        name: name.to_owned(),
                        expected: "spectrum texture".to_string(),
                    })?;
                let texture = Arc::new(create_spectrum_texture(
                    name,
                    desc,
                    spectrum_type,
                    color_space,
                )?);
                vacant.insert(texture.clone());
                texture
            }
        };

        Ok(texture)
    }
}

fn create_float_texture(desc: FloatTextureDesc) -> Result<FloatTextureEnum, ReadSceneError> {
    let create_subtextures = |descs: Vec<FloatTextureDesc>| {
        descs
            .into_iter()
            .map(create_float_texture)
            .collect::<Result<Vec<_>, _>>()
    };

    let texture = match desc {
        FloatTextureDesc::Constant(desc) => ConstantFloatTexture::new(desc.value).into(),
        FloatTextureDesc::Checkerboard2D(desc) => {
            let subtextures = create_subtextures(vec![*desc.tex1, *desc.tex2])?
                .try_into()
                .unwrap();
            CheckerboardFloatTexture::new_2d(subtextures, desc.mapping).into()
        }
        FloatTextureDesc::Checkerboard3D(desc) => {
            let subtextures = create_subtextures(vec![*desc.tex1, *desc.tex2])?
                .try_into()
                .unwrap();
            CheckerboardFloatTexture::new_3d(subtextures, desc.mapping).into()
        }
        FloatTextureDesc::Named(_) => {
            panic!("create_float_texture shouldn't be used for a texture name reference")
        }
    };

    Ok(texture)
}

fn create_spectrum_texture(
    name: &str,
    desc: SpectrumTextureDesc,
    spectrum_type: SpectrumType,
    color_space: &'static RGBColorSpace,
) -> Result<SpectrumTextureEnum, ReadSceneError> {
    let invalid_albedo_err = |_| ReadSceneError::TextureMismatch {
        name: name.to_owned(),
        expected: "RGB albedo spectrum texture (RGB components must be <= 1)".to_string(),
    };

    let create_subtextures = |descs: Vec<SpectrumTextureDesc>| {
        descs
            .into_iter()
            .map(|desc| create_spectrum_texture("", desc, spectrum_type, color_space))
            .collect::<Result<Vec<_>, _>>()
    };

    let texture = match desc {
        SpectrumTextureDesc::Constant(desc) => ConstantSpectrumTexture::new(
            create_spectrum(desc.value, spectrum_type, color_space).map_err(invalid_albedo_err)?,
        )
        .into(),
        SpectrumTextureDesc::Checkerboard2D(desc) => {
            let subtextures = create_subtextures(vec![*desc.tex1, *desc.tex2])?
                .try_into()
                .unwrap();
            CheckerboardSpectrumTexture::new_2d(subtextures, desc.mapping).into()
        }
        SpectrumTextureDesc::Checkerboard3D(desc) => {
            let subtextures = create_subtextures(vec![*desc.tex1, *desc.tex2])?
                .try_into()
                .unwrap();
            CheckerboardSpectrumTexture::new_3d(subtextures, desc.mapping).into()
        }
        SpectrumTextureDesc::Named(_) => {
            panic!("create_spectrum_texture shouldn't be used for a texture name reference")
        }
    };

    Ok(texture)
}

/// Create all materials from their descriptions,
/// replacing them in the name-to-material map.
///
/// Returns `Ok` if all creations succeeded, `Err` upon any failure with the error for that material.
fn create_materials(
    descs: HashMap<String, MaterialDesc>,
    color_space: &'static RGBColorSpace,
    textures: &mut Textures,
) -> Result<HashMap<String, Arc<MaterialEnum>>, ReadSceneError> {
    descs
        .into_iter()
        .map(|(name, desc)| create_material(desc, color_space, textures).map(|mat| (name, mat)))
        .collect()
}

fn create_material(
    desc: MaterialDesc,
    color_space: &'static RGBColorSpace,
    textures: &mut Textures,
) -> Result<Arc<MaterialEnum>, ReadSceneError> {
    let material = match desc {
        MaterialDesc::Diffuse(desc) => {
            let reflectance = get_spectrum_texture(
                textures,
                desc.reflectance,
                SpectrumType::Albedo,
                color_space,
            )?;
            DiffuseMaterial::new(reflectance).into()
        }
        MaterialDesc::Dielectric(desc) => {
            let u_roughness = get_float_texture(
                textures,
                desc.u_roughness.or(desc.roughness.clone()).unwrap(),
            )?;
            let v_roughness =
                get_float_texture(textures, desc.v_roughness.or(desc.roughness).unwrap())?;
            let eta =
                Arc::new(create_spectrum(desc.eta, SpectrumType::Unbounded, color_space).unwrap());
            DielectricMaterial::new(u_roughness, v_roughness, desc.remap_roughness, eta).into()
        }
    };

    Ok(Arc::new(material))
}

// Helpers for getting constructed textures based on description -
// If referring to a named one, query the Textures collection;
// Otherwise, create it ad hoc
fn get_float_texture(
    textures: &mut Textures,
    desc: FloatTextureDesc,
) -> Result<Arc<FloatTextureEnum>, ReadSceneError> {
    match desc {
        FloatTextureDesc::Named(name) => textures.get_named_float_texture(name),
        _ => create_float_texture(desc).map(Arc::new),
    }
}

fn get_spectrum_texture(
    textures: &mut Textures,
    desc: SpectrumTextureDesc,
    spectrum_type: SpectrumType,
    color_space: &'static RGBColorSpace,
) -> Result<Arc<SpectrumTextureEnum>, ReadSceneError> {
    match desc {
        SpectrumTextureDesc::Named(name) => {
            textures.get_named_spectrum_texture(name, spectrum_type, color_space)
        }
        _ => create_spectrum_texture("", desc, spectrum_type, color_space).map(Arc::new),
    }
}

#[derive(Debug, Default)]
struct Meshes {
    bilinear_patches: Vec<BilinearPatchMesh>,
    triangles: Vec<TriangleMesh>,
}

fn create_shape(
    desc: ShapeDesc,
    textures: &mut Textures,
    all_meshes: &mut Meshes,
    camera: &impl Camera,
) -> Result<Vec<Arc<ShapeEnum>>, ReadSceneError> {
    let mut shapes = Vec::new();

    match desc {
        ShapeDesc::Sphere(desc) => {
            let render_from_object = camera
                .camera_transform()
                .render_from_world(desc.world_from_object);

            let sphere = Sphere::builder()
                .radius(desc.radius)
                .z_min(desc.z_min)
                .z_max(desc.z_max)
                .phi_max(desc.phi_max)
                .reverse_orientation(desc.reverse_orientation)
                .render_from_object(render_from_object)
                .build()?;
            shapes.push(Arc::new(Box::new(sphere).into()));
        }
        ShapeDesc::BilinearMesh(desc) => {
            let render_from_object = camera
                .camera_transform()
                .render_from_world(desc.world_from_object);

            let mesh = BilinearPatchMesh::new(
                &render_from_object,
                desc.reverse_orientation,
                desc.indices,
                desc.positions,
                desc.normals,
                desc.uvs,
            );

            // This will be the index to this mesh once it's moved into vec
            let mesh_idx = all_meshes.bilinear_patches.len();

            // For each patch in mesh, create a shape and push to shapes vec
            for blp_idx in 0..mesh.num_patches() {
                let patch = BilinearPatch::new(&mesh, mesh_idx, blp_idx);
                shapes.push(Arc::new(patch.into()));
            }

            // Finally, move mesh into vec of all
            all_meshes.bilinear_patches.push(mesh);
        }
        ShapeDesc::TriangleMesh(desc) => {
            let render_from_object = camera
                .camera_transform()
                .render_from_world(desc.world_from_object);

            let mesh = TriangleMesh::new(
                &render_from_object,
                desc.reverse_orientation,
                desc.indices,
                desc.positions,
                desc.tangents,
                desc.normals,
                desc.uvs,
            );

            // This will be the index to this mesh once it's moved into vec
            let mesh_idx = all_meshes.triangles.len();

            // For each triangle in mesh, create a shape and push to shapes vec
            for tri_idx in 0..mesh.num_triangles() {
                let triangle = Triangle::new(mesh_idx, tri_idx);
                shapes.push(Arc::new(triangle.into()));
            }

            // Finally, move mesh into vec of all
            all_meshes.triangles.push(mesh);
        }
        ShapeDesc::PlyMesh(desc) => {
            let render_from_object = camera
                .camera_transform()
                .render_from_world(desc.world_from_object);

            // Load the PLY file
            info!("Reading PLY file {}", desc.filename.display());
            let mut mesh = TriQuadMesh::load_file(desc.filename)?;

            // If a displacement texture is specified,
            // apply it to the mesh
            if let Some(displacement_name) = desc.displacement_name {
                let displacement = textures.get_named_float_texture(displacement_name)?;

                // Point distance is determined in render space
                let dist_fn = |mut v0: Point3f, mut v1: Point3f| {
                    v0 = &render_from_object * v0;
                    v1 = &render_from_object * v1;
                    v0.distance(v1)
                };
                // Displace point by adding (texture value * normal) to position
                let displace_fn = |pos, n, uv| {
                    let ctx = TextureEvalContext {
                        p: pos,
                        uv,
                        ..TextureEvalContext::default()
                    };

                    let d = UniversalTextureEvaluator::new().eval(&*displacement, &ctx);
                    pos + Vec3f::from(d * n)
                };

                mesh.displace(&dist_fn, desc.edge_length, displace_fn);
            }

            // If mesh contains triangles, create triangle mesh and triangles
            if !mesh.tri_indices.is_empty() {
                let tri_mesh = TriangleMesh::new(
                    &render_from_object,
                    desc.reverse_orientation,
                    mesh.tri_indices,
                    mesh.positions.clone(),
                    None,
                    mesh.normals.clone(),
                    mesh.uv.clone(),
                );
                // This will be the index to this triangle mesh once it's moved into vec
                let mesh_idx = all_meshes.triangles.len();
                // For each triangle in mesh, create a shape and push to shapes vec
                for tri_idx in 0..tri_mesh.num_triangles() {
                    let triangle = Triangle::new(mesh_idx, tri_idx);
                    shapes.push(Arc::new(triangle.into()));
                }
                // Finally, move mesh into vec of all
                all_meshes.triangles.push(tri_mesh);
            }

            // If mesh contains quads, create bilinear patch mesh and bilinear patches
            if !mesh.quad_indices.is_empty() {
                let quad_mesh = BilinearPatchMesh::new(
                    &render_from_object,
                    desc.reverse_orientation,
                    mesh.quad_indices,
                    mesh.positions,
                    mesh.normals,
                    mesh.uv,
                );
                // This will be the index to this bilinear mesh once it's moved into vec
                let mesh_idx = all_meshes.bilinear_patches.len();

                // For each patch in mesh, create a shape and push to shapes vec
                for blp_idx in 0..quad_mesh.num_patches() {
                    let patch = BilinearPatch::new(&quad_mesh, mesh_idx, blp_idx);
                    shapes.push(Arc::new(patch.into()));
                }

                // Finally, move mesh into vec of all
                all_meshes.bilinear_patches.push(quad_mesh);
            }
        }
    };

    Ok(shapes)
}

fn create_primitives_for_shapes(
    shape_descs: Vec<ShapeDesc>,
    textures: &mut Textures,
    materials: &HashMap<String, Arc<MaterialEnum>>,
    all_meshes: &mut Meshes,
    camera: &impl Camera,
) -> Result<Vec<Arc<PrimitiveEnum>>, ReadSceneError> {
    let mut primitives = Vec::new();

    for shape_desc in shape_descs {
        let material =
            materials
                .get(shape_desc.material_name())
                .ok_or(ReadSceneError::UndefinedMaterial(
                    shape_desc.material_name().to_owned(),
                ))?;

        let shapes = create_shape(shape_desc, textures, all_meshes, camera)?;

        // TODO: Currently not supporting any options that would necessitate a GeometricPrimitive

        primitives.extend(
            shapes
                .into_iter()
                .map(|shape| Arc::new(SimplePrimitive::new(shape, material.clone()).into())),
        );
    }

    Ok(primitives)
}

struct AreaLightProducts {
    lights: Vec<Arc<LightEnum>>,
    primitives: Vec<Arc<PrimitiveEnum>>,
}

fn create_area_lights_and_primitives(
    descs: Vec<(AreaLightDesc, Vec<ShapeDesc>)>,
    textures: &mut Textures,
    materials: &HashMap<String, Arc<MaterialEnum>>,
    all_meshes: &mut Meshes,
    camera: &impl Camera,
    color_space: &'static RGBColorSpace,
) -> Result<AreaLightProducts, ReadSceneError> {
    let mut lights = Vec::new();
    let mut primitives = Vec::new();

    for (light_desc, shape_descs) in descs {
        let emission;
        let emission_spec;
        let mut scale;
        let power;
        let two_sided;
        match &light_desc {
            AreaLightDesc::Diffuse(desc) => {
                match &desc.emission {
                    Some(EmissionDesc::ImageFile(path)) => {
                        emission = AreaLightEmission::Image(Arc::new(Image::read(path, None)?));
                        scale = desc.scale / color_space.illuminant.to_photometric();
                    }
                    Some(EmissionDesc::Spectrum(spec_desc)) => {
                        emission_spec =
                            create_spectrum(spec_desc.clone(), SpectrumType::Illuminant, &SRGB)
                                .unwrap();
                        scale = desc.scale / emission_spec.to_photometric();
                        emission = AreaLightEmission::Uniform(&emission_spec);
                    }
                    None => {
                        emission = AreaLightEmission::Uniform(&color_space.illuminant);
                        scale = desc.scale / color_space.illuminant.to_photometric();
                    }
                }

                power = desc.power;
                two_sided = desc.two_sided;
            }
        };

        for shape_desc in shape_descs {
            let material = materials.get(shape_desc.material_name()).ok_or(
                ReadSceneError::UndefinedMaterial(shape_desc.material_name().to_owned()),
            )?;
            let shapes = create_shape(shape_desc, textures, all_meshes, camera)?;

            for shape_arc in shapes {
                if let Some(phi_v) = power {
                    // k_e is the emissive power of the light as defined
                    // by the spectral distribution and texture,
                    // used to normalize the emitted radiance such that
                    // the user-defined power will be the actual power emitted
                    let mut k_e = 1.0;
                    if let AreaLightEmission::Image(_) = &emission {
                        todo!("use luminance vector from image color space")
                    }

                    if two_sided {
                        k_e *= 2.0;
                    }
                    k_e *= shape_arc.area() * PI;

                    scale *= phi_v / k_e;
                }

                let mi = MediumInterface {
                    inside: None,
                    outside: None,
                };

                let light = match &light_desc {
                    AreaLightDesc::Diffuse(light_desc) => {
                        DiffuseAreaLight::builder()
                            // TODO: Use alpha, medium interface, image_color_space options once supported
                            .shape(shape_arc.clone())
                            .emission(emission.clone())
                            .scale(scale)
                            .two_sided(light_desc.two_sided)
                            .image_color_space(color_space)
                            .medium_interface(mi.clone())
                            .build()?
                    }
                };
                let light_arc: Arc<LightEnum> = Arc::new(light.into());
                lights.push(light_arc.clone());

                let primitive =
                    GeometricPrimitive::new(shape_arc, material.clone(), Some(light_arc), mi);
                primitives.push(Arc::new(primitive.into()));
            }
        }
    }

    Ok(AreaLightProducts { lights, primitives })
}

fn create_aggregate(desc: AcceleratorDesc, primitives: Vec<Arc<PrimitiveEnum>>) -> PrimitiveEnum {
    match desc {
        AcceleratorDesc::Bvh(desc) => {
            BVHAggregate::new(primitives, desc.max_node_prims, desc.split_method).into()
        }
        AcceleratorDesc::KdTree(_) => unimplemented!(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpectrumType {
    Albedo,
    Unbounded,
    Illuminant,
}

#[derive(Debug)]
struct InvalidAlbedoRgb;

fn create_spectrum(
    desc: SpectrumDesc,
    spectrum_type: SpectrumType,
    color_space: &'static RGBColorSpace,
) -> Result<SpectrumEnum, InvalidAlbedoRgb> {
    let spectrum = match desc {
        SpectrumDesc::Constant(val) => ConstantSpectrum::new(val).into(),
        SpectrumDesc::Rgb(rgb) => match spectrum_type {
            SpectrumType::Albedo => {
                if rgb.r > 1.0 || rgb.g > 1.0 || rgb.b > 1.0 {
                    return Err(InvalidAlbedoRgb);
                }
                RgbAlbedoSpectrum::new(color_space, rgb).into()
            }
            SpectrumType::Unbounded => RgbUnboundedSpectrum::new(color_space, rgb).into(),
            SpectrumType::Illuminant => RgbIlluminantSpectrum::new(color_space, rgb).into(),
        },

        SpectrumDesc::BlackbodyTemp(temp) => BlackbodySpectrum::new(temp).into(),
    };

    Ok(spectrum)
}
