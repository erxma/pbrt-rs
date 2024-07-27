use std::sync::atomic::Ordering;

use derive_builder::Builder;
use enum_dispatch::enum_dispatch;

use crate::{math::array2d::Array2D, parallel::AtomicF64, Float};

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

#[derive(Clone, Debug, Builder)]
struct FilmBase<'a> {
    sensor: &'a PixelSensor,
}

#[derive(Clone, Debug)]
pub struct RGBFilm<'a> {
    base: FilmBase<'a>,
    max_component_value: Float,
    pixels: Array2D<Pixel>,
}

#[derive(Debug)]
struct Pixel {
    rgb_sum: [f64; 3],
    weight_sum: f64,
    rgb_splat: [AtomicF64; 3],
}

impl Clone for Pixel {
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
