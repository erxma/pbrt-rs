use std::{array, sync::atomic::Ordering};

use derive_builder::Builder;
use enum_dispatch::enum_dispatch;

use crate::{
    color::RGBColorSpace,
    geometry::bounds::Bounds2i,
    image::Filter,
    math::{Array2D, Point2f, Point2i, SquareMatrix, Tuple, Vec2f, Vec2i},
    parallel::AtomicF64,
    sampling::spectrum::{SampledSpectrum, SampledWavelengths},
    Float,
};

use super::sensor::PixelSensor;

#[enum_dispatch(FilmTrait)]
#[derive(Clone, Debug)]
pub enum Film<'a> {
    RGBFilm(RGBFilm<'a>),
}

#[enum_dispatch]
trait FilmTrait {
    #[allow(non_snake_case)]
    fn add_sample(
        &mut self,
        p_film: Point2i,
        L: &SampledSpectrum,
        lambda: &SampledWavelengths,
        visible_surface: &VisibleSurface,
        weight: Float,
    );

    #[allow(non_snake_case)]
    fn add_splat(&mut self, p: Point2f, L: &SampledSpectrum, lambda: &SampledWavelengths);
}

#[derive(Clone, Debug)]
pub struct RGBFilm<'a> {
    full_resolution: Point2i,
    pixel_bounds: Bounds2i,
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
    pixel_bounds: Bounds2i,
    filter: &'a Filter,
    diagonal: Float,
    sensor: &'a PixelSensor,

    color_space: &'a RGBColorSpace<'a>,
    max_component_value: Float,
    write_fp16: bool,
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
            pixel_bounds: params.pixel_bounds,
            filter: params.filter,
            diagonal: params.diagonal,
            sensor: params.sensor,
            color_space: params.color_space,
            max_component_value: params.max_component_value,
            write_fp16: params.write_fp16,
            filter_integral,
            output_rgb_from_sensor_rgb,
            pixels: Array2D::fill_default(params.pixel_bounds),
        })
    }
}

impl<'a> FilmTrait for RGBFilm<'a> {
    #[allow(non_snake_case)]
    fn add_sample(
        &mut self,
        p_film: Point2i,
        L: &SampledSpectrum,
        lambda: &SampledWavelengths,
        _visible_surface: &VisibleSurface,
        weight: Float,
    ) {
        // Convert sample radiance to PixelSensor RGB
        let mut rgb = self.sensor.to_sensor_rgb(L, lambda);

        // Optionally clamp sensor RGB value
        let m = rgb.max_component();
        if m > self.max_component_value {
            rgb *= self.max_component_value / m;
        }

        // Update pixel values with filtered sample contribution
        let pixel = &mut self.pixels[p_film];
        for i in 0..3 {
            pixel.rgb_sum[i] += (weight * rgb[i]) as f64;
        }
        pixel.weight_sum += weight as f64;
    }

    #[allow(non_snake_case)]
    fn add_splat(&mut self, p: Point2f, L: &SampledSpectrum, lambda: &SampledWavelengths) {
        // Convert sample radiance to PixelSensor RGB
        let mut rgb = self.sensor.to_sensor_rgb(L, lambda);

        // Optionally clamp sensor RGB value
        let m = rgb.max_component();
        if m > self.max_component_value {
            rgb *= self.max_component_value / m;
        }

        // Compute bounds of affected pixels for splat, splat_bounds
        let p_discrete = p + Vec2f::new(0.5, 0.5);
        let radius = self.filter.radius();
        let splat_bounds = Bounds2i::new(
            (p_discrete - radius).floor().as_point2i(),
            (p_discrete + radius).floor().as_point2i() + Vec2i::new(1, 1),
        )
        .intersect(self.pixel_bounds);

        for pi in splat_bounds {
            // Evaluate filter at pi and add splat contribution
            let wt = self
                .filter
                .eval((p - Point2f::from(pi) - Vec2f::new(0.5, 0.5)).into());

            if wt != 0.0 {
                for i in 0..3 {
                    self.pixels[pi].rgb_splat[i].fetch_add((wt * rgb[i]) as f64, Ordering::Relaxed);
                }
            }
        }
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

#[derive(Clone, Debug)]
pub struct VisibleSurface {}
