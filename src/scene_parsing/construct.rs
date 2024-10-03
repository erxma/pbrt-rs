use std::{io::Read, sync::Arc};

use crate::{
    camera::{
        CameraEnum, Film, OrthographicCamera, PerspectiveCamera, PixelSensor, RGBFilm,
        RGBFilmParams,
    },
    color::{RGBColorSpace, SRGB},
    core::{Bounds2i, Float, Point2i, Vec2f},
    imaging::{BoxFilter, FilterEnum, GaussianFilter, TriangleFilter},
    integrators::{IntegratorEnum, RandomWalkIntegrator, SimplePathIntegrator},
    lights::LightEnum,
    primitives::PrimitiveEnum,
    sampling::{spectrum, IndependentSampler, SamplerEnum},
    scene_parsing::scene::parse_pbrt_file,
};

use super::{
    directives::{Camera, ColorSpace, Film as FilmDesc, Filter, Integrator, Sampler},
    PbrtParseError,
};

pub fn create_scene_integrator(
    file: impl Read,
    ignore_unrecognized_directives: bool,
) -> Result<IntegratorEnum, PbrtParseError> {
    let description = parse_pbrt_file(file, ignore_unrecognized_directives)?;

    let filter = create_filter(description.options.filter);
    let color_space = get_color_space(description.options.color_space);
    let exposure_time = get_exposure_time(&description.options.camera);
    let film = create_film(description.options.film, filter, color_space, exposure_time)?;
    let camera = create_camera(description.options.camera, film);
    let sampler = create_sampler(description.options.sampler);

    todo!()
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
) -> Result<Film, PbrtParseError> {
    let film = match desc {
        FilmDesc::Rgb(desc) => RGBFilm::new(RGBFilmParams {
            full_resolution: Point2i::new(desc.x_resolution as i32, desc.y_resolution as i32),
            pixel_bounds: Bounds2i::from(desc.pixel_bounds.map(|v| v as i32)),
            filter: Arc::new(filter),
            diagonal: desc.diagonal,
            sensor: Arc::new(create_sensor(
                &desc.sensor,
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
    name: &str,
    color_space: &'static RGBColorSpace,
    exposure_time: Float,
    iso: Float,
    white_balance_temp: Option<Float>,
) -> Result<PixelSensor, PbrtParseError> {
    // Note from the original pbrt:
    // "In the talk we mention using 312.5 for historical reasons. The
    // choice of 100 here just means that the other parameters make nice
    // 'round' numbers like 1 and 100."
    let imaging_ratio = exposure_time * iso / 100.0;

    let sensor = match name {
        "cie1931" => PixelSensor::with_xyz_matching(color_space, None, imaging_ratio),
        name => {
            let sensor_illum = spectrum::illum_d(white_balance_temp.unwrap_or(6500.0));

            let r = spectrum::get_named_spectrum(&format!("{name}_r")).ok_or(
                PbrtParseError::InvalidValue {
                    name: "sensor".to_string(),
                    value: name.to_owned(),
                },
            )?;
            // If the R spectrum exists, these should as well
            let g = spectrum::get_named_spectrum(&format!("{name}_g")).unwrap();
            let b = spectrum::get_named_spectrum(&format!("{name}_b")).unwrap();

            PixelSensor::with_rgb_matching(color_space, &r, &g, &b, &sensor_illum, imaging_ratio)
        }
    };
    Ok(sensor)
}

fn get_exposure_time(desc: &Camera) -> Float {
    match desc {
        Camera::Orthographic(desc) => desc.shutter_close - desc.shutter_open,
        Camera::Perspective(desc) => desc.shutter_close - desc.shutter_open,
    }
}

fn create_camera(desc: Camera, film: Film) -> CameraEnum {
    match desc {
        Camera::Orthographic(desc) => OrthographicCamera::builder()
            .film(film)
            .focal_distance(desc.focal_distance)
            .lens_radius(desc.lens_radius)
            .screen_window(desc.screen_window.unwrap().into())
            .shutter_period(desc.shutter_open..desc.shutter_close)
            .world_from_camera(desc.transform)
            .build()
            .into(),

        Camera::Perspective(desc) => PerspectiveCamera::builder()
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
