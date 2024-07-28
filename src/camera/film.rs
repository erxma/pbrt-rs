use std::{array, sync::atomic::Ordering};

use derive_builder::Builder;
use enum_dispatch::enum_dispatch;

use crate::{
    color::RGBColorSpace,
    geometry::bounds3::Bounds2i,
    image::Filter,
    math::{array2d::Array2D, point::Point2i, square_matrix::SquareMatrix},
    parallel::AtomicF64,
    Float,
};

use super::sensor::PixelSensor;

#[enum_dispatch(FilmTrait)]
#[derive(Clone, Debug)]
pub enum Film<'a> {
    RGBFilm(RGBFilm<'a>),
    GBufferFilm,
    SpectralFilm,
}

#[enum_dispatch]
trait FilmTrait {}

#[derive(Clone, Debug)]
pub struct RGBFilm<'a> {
    full_resolution: Point2i,
    filter: &'a Filter,
    diagonal: Float,
    sensor: &'a PixelSensor,

    color_space: &'a RGBColorSpace<'a>,
    max_component_value: Float,
    write_fp16: bool,
    filter_integral: Float,
    output_rgb_from_sensor_rgb: SquareMatrix<3>,
    pixels: Array2D<RGBPixel>,
}

impl<'a> RGBFilm<'a> {
    pub fn builder() -> RGBFilmBuilder<'a> {
        Default::default()
    }
}

#[derive(Builder)]
#[builder(
    name = "RGBFilmBuilder",
    public,
    build_fn(private, name = "build_params")
)]
struct RGBFilmParams<'a> {
    full_resolution: Point2i,
    filter: &'a Filter,
    diagonal: Float,
    sensor: &'a PixelSensor,

    color_space: &'a RGBColorSpace<'a>,
    max_component_value: Float,
    write_fp16: bool,
    pixels_bounds: Bounds2i,
}

impl<'a> RGBFilmBuilder<'a> {
    pub fn build(&self) -> Result<RGBFilm<'_>, RGBFilmBuilderError> {
        let params = self.build_params()?;

        let filter_integral = params.filter.integral();

        // Compute output_rgb_from_sensor_rgb
        let output_rgb_from_sensor_rgb =
            params.color_space.rgb_from_xyz.clone() * params.sensor.xyz_from_sensor_rgb.clone();

        Ok(RGBFilm {
            full_resolution: params.full_resolution,
            filter: params.filter,
            diagonal: params.diagonal,
            sensor: params.sensor,
            color_space: params.color_space,
            max_component_value: params.max_component_value,
            write_fp16: params.write_fp16,
            filter_integral,
            output_rgb_from_sensor_rgb,
            pixels: Array2D::fill_default(params.pixels_bounds),
        })
    }
}

#[derive(Debug)]
struct RGBPixel {
    rgb_sum: [f64; 3],
    weight_sum: f64,
    rgb_splat: [AtomicF64; 3],
}

impl Default for RGBPixel {
    fn default() -> Self {
        Self {
            rgb_sum: Default::default(),
            weight_sum: Default::default(),
            rgb_splat: array::from_fn(|_| AtomicF64::new(0.0)),
        }
    }
}

impl Clone for RGBPixel {
    // TODO: Should this be kept?
    fn clone(&self) -> Self {
        Self {
            rgb_sum: self.rgb_sum,
            weight_sum: self.weight_sum,
            rgb_splat: self
                .rgb_splat
                .each_ref()
                .map(|val| AtomicF64::new(val.load(Ordering::Acquire))),
        }
    }
}

impl<'a> FilmTrait for RGBFilm<'a> {}

#[derive(Clone, Debug)]
pub struct GBufferFilm {}

impl FilmTrait for GBufferFilm {}

#[derive(Clone, Debug)]
pub struct SpectralFilm {}

impl FilmTrait for SpectralFilm {}

pub struct VisibleSurface {}
