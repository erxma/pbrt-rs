use std::{
    borrow::Cow,
    collections::{hash_map, HashMap},
    io::Read,
    sync::Arc,
};

use thiserror::Error;

use crate::{
    camera::{
        Camera, CameraEnum, Film, OrthographicCamera, PerspectiveCamera, PixelSensor, RGBFilm,
        RGBFilmParams,
    },
    color::{RGBColorSpace, SRGB},
    core::{constants::PI, Bounds2i, Float, Point2i, Transform, Vec2f},
    imaging::{BoxFilter, FilterEnum, GaussianFilter, TriangleFilter},
    integrators::{IntegratorEnum, RandomWalkIntegrator, SimplePathIntegrator},
    lights::{DirectionalLight, LightEnum, UniformInfiniteLight},
    materials::{
        ConstantFloatTexture, ConstantSpectrumTexture, FloatTextureEnum, SpectrumTextureEnum,
    },
    primitives::PrimitiveEnum,
    sampling::{
        spectrum::{
            self, BlackbodySpectrum, RgbAlbedoSpectrum, RgbIlluminantSpectrum,
            RgbUnboundedSpectrum, SpectrumEnum,
        },
        IndependentSampler, SamplerEnum,
    },
    scene_parsing::scene::parse_pbrt_file,
};

use super::{
    common::{PbrtParseError, Spectrum as SpectrumDesc},
    directives::{
        Camera as CameraDesc, ColorSpace, Film as FilmDesc, Filter, Integrator, Light as LightDesc,
        Sampler, SensorName, Texture,
    },
};

pub fn create_scene_integrator(
    file: impl Read,
    ignore_unrecognized_directives: bool,
) -> Result<IntegratorEnum, ReadSceneError> {
    let description = parse_pbrt_file(file, ignore_unrecognized_directives)?;

    let filter = create_filter(description.options.filter);
    let color_space = get_color_space(description.options.color_space);
    let exposure_time = get_exposure_time(&description.options.camera);
    let film = create_film(description.options.film, filter, color_space, exposure_time)?;
    let camera = create_camera(description.options.camera, film);
    let sampler = create_sampler(description.options.sampler);

    let lights = create_lights(description.world.lights, &camera, color_space);

    todo!()
}

#[derive(Error, Debug)]
pub enum ReadSceneError {
    #[error("failed to parse pbrt scene file")]
    ParseError(#[from] PbrtParseError),
    #[error("texture `{name}` isn't valid for its usage, which expects {expected}")]
    TextureMismatch { name: String, expected: String },
    #[error("texture `{0}` is not defined")]
    UndefinedTexture(String),
}

fn create_filter(desc: Filter) -> FilterEnum {
    match desc {
        Filter::Box(desc) => BoxFilter::new(Vec2f::new(desc.x_radius, desc.y_radius)).into(),
        Filter::Gaussian(desc) => {
            GaussianFilter::new(Vec2f::new(desc.x_radius, desc.y_radius), desc.std).into()
        }
        Filter::Triangle(desc) => {
            TriangleFilter::new(Vec2f::new(desc.x_radius, desc.y_radius)).into()
        }
    }
}

fn get_color_space(desc: ColorSpace) -> &'static RGBColorSpace {
    match desc {
        ColorSpace::Srgb => &SRGB,
    }
}

fn create_film(
    desc: FilmDesc,
    filter: FilterEnum,
    color_space: &'static RGBColorSpace,
    exposure_time: Float,
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
            filename: desc.filename,
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

            PixelSensor::with_rgb_matching(color_space, &r, &g, &b, &sensor_illum, imaging_ratio)
        }
    };
    Ok(sensor)
}

fn get_exposure_time(desc: &CameraDesc) -> Float {
    match desc {
        CameraDesc::Orthographic(desc) => desc.shutter_close - desc.shutter_open,
        CameraDesc::Perspective(desc) => desc.shutter_close - desc.shutter_open,
    }
}

fn create_camera(desc: CameraDesc, film: Film) -> CameraEnum {
    match desc {
        CameraDesc::Orthographic(desc) => OrthographicCamera::builder()
            .film(film)
            .focal_distance(desc.focal_distance)
            .lens_radius(desc.lens_radius)
            .screen_window(desc.screen_window.unwrap().into())
            .shutter_period(desc.shutter_open..desc.shutter_close)
            .world_from_camera(desc.transform)
            .build()
            .into(),

        CameraDesc::Perspective(desc) => PerspectiveCamera::builder()
            .film(film)
            .focal_distance(desc.focal_distance)
            .fov(desc.fov_degs)
            .lens_radius(desc.lens_radius)
            .screen_window(desc.screen_window.unwrap().into())
            .shutter_period(desc.shutter_open..desc.shutter_close)
            .world_from_camera(desc.transform)
            .build()
            .into(),
    }
}

fn create_sampler(desc: Sampler) -> SamplerEnum {
    match desc {
        Sampler::Independent(desc) => {
            IndependentSampler::new(desc.pixel_samples, Some(desc.seed)).into()
        }
    }
}

fn create_integrator(
    desc: Integrator,
    camera: CameraEnum,
    sampler: SamplerEnum,
    aggregate: PrimitiveEnum,
    lights: Vec<Arc<LightEnum>>,
) -> IntegratorEnum {
    match desc {
        Integrator::RandomWalk(desc) => {
            RandomWalkIntegrator::new(desc.max_depth, camera, sampler, aggregate, lights).into()
        }
        Integrator::SimplePath(desc) => SimplePathIntegrator::new(
            desc.max_depth,
            desc.sample_lights,
            desc.sample_bsdf,
            camera,
            sampler,
            aggregate,
            lights,
        )
        .into(),
    }
}

fn create_lights(
    descs: impl IntoIterator<Item = LightDesc>,
    camera: &impl Camera,
    color_space: &'static RGBColorSpace,
) -> Vec<LightEnum> {
    descs
        .into_iter()
        .map(|desc| create_light(desc, camera, color_space))
        .collect()
}

fn create_light(
    desc: LightDesc,
    camera: &impl Camera,
    color_space: &'static RGBColorSpace,
) -> LightEnum {
    match desc {
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
                .render_from_world(desc.render_from_light * transform);

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
            let radiance;
            match desc.radiance {
                Some(spec) => {
                    radiance = Cow::Owned(
                        create_spectrum(spec, SpectrumType::Illuminant, color_space).unwrap(),
                    );
                }
                None => {
                    // Default: color space's std illuminant
                    radiance = Cow::Borrowed(&color_space.illuminant);
                }
            }

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
    }
}

#[derive(Debug, Default)]
struct Textures {
    float_textures: HashMap<String, Arc<FloatTextureEnum>>,
    albedo_spectrum_textures: HashMap<String, Arc<SpectrumTextureEnum>>,
    unbounded_spectrum_textures: HashMap<String, Arc<SpectrumTextureEnum>>,
    illuminant_spectrum_textures: HashMap<String, Arc<SpectrumTextureEnum>>,

    uncreated_descs: HashMap<String, Texture>,
}

impl Textures {
    fn get_float_texture(
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
                    .ok_or_else(|| ReadSceneError::UndefinedTexture(name.clone()))?;
                let texture = Arc::new(create_float_texture(name, desc)?);
                vacant.insert(texture.clone());
                texture
            }
        };

        Ok(texture)
    }

    fn get_spectrum_texture(
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
                    .ok_or_else(|| ReadSceneError::UndefinedTexture(name.clone()))?;
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

fn create_float_texture(name: &str, desc: Texture) -> Result<FloatTextureEnum, ReadSceneError> {
    let not_float_err = |_| ReadSceneError::TextureMismatch {
        name: name.to_owned(),
        expected: "float texture".to_string(),
    };

    let texture = match desc {
        Texture::Constant(desc) => {
            ConstantFloatTexture::new(desc.value.into_float().map_err(not_float_err)?).into()
        }
    };

    Ok(texture)
}

fn create_spectrum_texture(
    name: &str,
    desc: Texture,
    spectrum_type: SpectrumType,
    color_space: &'static RGBColorSpace,
) -> Result<SpectrumTextureEnum, ReadSceneError> {
    let not_spectrum_err = |_| ReadSceneError::TextureMismatch {
        name: name.to_owned(),
        expected: "spectrum texture".to_string(),
    };
    let invalid_albedo_err = |_| ReadSceneError::TextureMismatch {
        name: name.to_owned(),
        expected: "RGB albedo spectrum texture (RGB components must be <= 1)".to_string(),
    };

    let texture = match desc {
        Texture::Constant(desc) => ConstantSpectrumTexture::new(
            create_spectrum(
                desc.value.into_spectrum().map_err(not_spectrum_err)?,
                spectrum_type,
                color_space,
            )
            .map_err(invalid_albedo_err)?,
        )
        .into(),
    };

    Ok(texture)
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
