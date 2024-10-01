use std::{io::Read, sync::Arc};

use crate::{
    camera::{CameraEnum, Film, OrthographicCamera, PerspectiveCamera, RGBFilm, RGBFilmParams},
    color::{RGBColorSpace, SRGB},
    core::{Point2i, Vec2f},
    imaging::{BoxFilter, FilterEnum, GaussianFilter, TriangleFilter},
    integrators::{IntegratorEnum, RandomWalkIntegrator, SimplePathIntegrator},
    lights::LightEnum,
    primitives::PrimitiveEnum,
    sampling::{IndependentSampler, SamplerEnum},
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
    let film = create_film(description.options.film, filter, color_space);
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

fn create_film(desc: FilmDesc, filter: FilterEnum, color_space: &'static RGBColorSpace) -> Film {
    match desc {
        FilmDesc::Rgb(desc) => RGBFilm::new(RGBFilmParams {
            full_resolution: Point2i::new(desc.x_resolution as i32, desc.y_resolution as i32),
            pixel_bounds: todo!(),
            filter: Arc::new(filter),
            diagonal: desc.diagonal,
            sensor: todo!(),
            filename: desc.filename,
            color_space,
            max_component_value: todo!(),
        })
        .into(),
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
