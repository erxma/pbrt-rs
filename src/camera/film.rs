use std::{
    array,
    path::PathBuf,
    sync::{atomic::Ordering, Arc},
};

use delegate::delegate;
use derive_builder::Builder;
use enum_dispatch::enum_dispatch;
use num_traits::{AsPrimitive, NumCast};

use crate::{
    color::{RGBColorSpace, RGB},
    geometry::Bounds2i,
    image::{FilterEnum, Image, ImageMetadata},
    math::{
        Array2D, Point2Isize, Point2Usize, Point2f, Point2i, SquareMatrix, Tuple, Vec2f, Vec2i,
    },
    parallel::AtomicF64,
    sampling::spectrum::{SampledSpectrum, SampledWavelengths},
    Float,
};

use super::sensor::PixelSensor;

#[enum_dispatch]
#[derive(Debug)]
pub enum Film {
    RGBFilm(RGBFilm),
}

impl Film {
    delegate! {
        #[through(FilmTrait)]
        to self {
            /// Add a sample to a point on the film. This can be used concurrently on different points of the film,
            /// as it requires only an immutable receiver `&self`.
            ///
            /// # Safety
            /// Exclusivity on each sample point is not guaranteed to be enforced,
            /// except for splats, so data races may occur if adding to the same point concurrently.
            /// Therefore, the user is responsible for ensuring each point is written to by at most one thread at a time.
            /// Concurrent writes to splats are safe.
            #[allow(non_snake_case)]
            pub unsafe fn add_sample_unchecked(
                &self,
                p_film: Point2i,
                L: &SampledSpectrum,
                lambda: &SampledWavelengths,
                visible_surface: Option<&VisibleSurface>,
                weight: Float,
            );
            pub fn uses_visible_surface(&self) -> bool;
            #[allow(non_snake_case)]
            pub fn add_splat(&mut self, p: Point2f, L: &SampledSpectrum, lambda: &SampledWavelengths);
            pub fn sample_wavelengths(&self, u: Float) -> SampledWavelengths;
            pub fn get_image(&self, metadata: &mut ImageMetadata, splat_scale: Float) -> Image;
            pub fn write_image(&self, metadata: ImageMetadata, splat_scale: Float);
            pub fn get_pixel_rgb(&self, p: Point2i, splat_scale: Float) -> RGB;
            pub fn full_resolution(&self) -> Point2i;
            pub fn pixel_bounds(&self) -> Bounds2i;
            pub fn diagonal(&self) -> Float;
            pub fn filter(&self) -> &FilterEnum;
            pub fn sensor(&self) -> &PixelSensor;
        }
    }
}

#[enum_dispatch(Film)]
trait FilmTrait {
    /// Add a sample to a point on the film. This can be used concurrently on different points of the film,
    /// as it requires only an immutable receiver `&self`.
    ///
    /// # Safety
    /// Exclusivity on each sample point is not guaranteed to be enforced,
    /// except for splats, so data races may occur if adding to the same point concurrently.
    /// Therefore, the user is responsible for ensuring each point is written to by at most one thread at a time.
    /// Concurrent writes to splats are safe.
    #[allow(non_snake_case)]
    unsafe fn add_sample_unchecked(
        &self,
        p_film: Point2i,
        L: &SampledSpectrum,
        lambda: &SampledWavelengths,
        visible_surface: Option<&VisibleSurface>,
        weight: Float,
    );

    fn uses_visible_surface(&self) -> bool;

    #[allow(non_snake_case)]
    fn add_splat(&mut self, p: Point2f, L: &SampledSpectrum, lambda: &SampledWavelengths);

    fn sample_wavelengths(&self, u: Float) -> SampledWavelengths;

    fn get_image(&self, metadata: &mut ImageMetadata, splat_scale: Float) -> Image;

    fn write_image(&self, metadata: ImageMetadata, splat_scale: Float);

    fn get_pixel_rgb(&self, p: Point2i, splat_scale: Float) -> RGB;

    fn full_resolution(&self) -> Point2i;
    fn pixel_bounds(&self) -> Bounds2i;
    fn diagonal(&self) -> Float;
    fn filter(&self) -> &FilterEnum;
    fn sensor(&self) -> &PixelSensor;
}

#[derive(Debug)]
pub struct RGBFilm {
    full_resolution: Point2i,
    pixel_bounds: Bounds2i,
    filter: Arc<FilterEnum>,
    diagonal: Float,
    sensor: Arc<PixelSensor>,
    filename: PathBuf,

    color_space: &'static RGBColorSpace,
    max_component_value: Float,
    filter_integral: Float,
    output_rgb_from_sensor_rgb: SquareMatrix<3>,
    pixels: Array2D<RGBPixel>,
}

impl RGBFilm {
    pub fn builder() -> RGBFilmBuilder {
        Default::default()
    }
}

#[derive(Builder)]
#[builder(
    name = "RGBFilmBuilder",
    public,
    build_fn(private, name = "build_params")
)]
struct RGBFilmParams {
    full_resolution: Point2i,
    pixel_bounds: Bounds2i,
    filter: Arc<FilterEnum>,
    diagonal: Float,
    sensor: Arc<PixelSensor>,
    filename: PathBuf,

    color_space: &'static RGBColorSpace,
    max_component_value: Float,
}

impl RGBFilmBuilder {
    pub fn build(&self) -> Result<RGBFilm, RGBFilmBuilderError> {
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
            filename: params.filename,
            color_space: params.color_space,
            max_component_value: params.max_component_value,
            filter_integral,
            output_rgb_from_sensor_rgb,
            pixels: Array2D::fill_default(params.pixel_bounds),
        })
    }
}

impl FilmTrait for RGBFilm {
    #[allow(non_snake_case)]
    unsafe fn add_sample_unchecked(
        &self,
        p_film: Point2i,
        L: &SampledSpectrum,
        lambda: &SampledWavelengths,
        _visible_surface: Option<&VisibleSurface>,
        weight: Float,
    ) {
        let weight: f64 = weight.as_();

        // Convert sample radiance to PixelSensor RGB
        let mut rgb = self.sensor.to_sensor_rgb(L, lambda);

        // Optionally clamp sensor RGB value
        let m = rgb.max_component();
        if m > self.max_component_value {
            rgb *= self.max_component_value / m;
        }

        // Update pixel values with filtered sample contribution
        unsafe {
            self.pixels.mutate_unchecked(p_film, |pixel| {
                for i in 0..3 {
                    let rgb_val: f64 = rgb[i].as_();
                    pixel.rgb_sum[i] += weight * rgb_val;
                }
                pixel.weight_sum += weight;
            });
        }
    }

    fn uses_visible_surface(&self) -> bool {
        false
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
                    self.pixels[pi].rgb_splat[i]
                        .fetch_add(NumCast::from(wt * rgb[i]).unwrap(), Ordering::Relaxed);
                }
            }
        }
    }

    fn sample_wavelengths(&self, u: Float) -> SampledWavelengths {
        SampledWavelengths::sample_visible(u)
    }

    fn get_image(&self, metadata: &mut ImageMetadata, splat_scale: Float) -> Image {
        // FIXME: Make Bounds2Usize?
        let diagonal = self.pixel_bounds.diagonal();
        let dims = Point2Usize::new(diagonal.x() as usize, diagonal.y() as usize);
        let mut image = Image::new(dims, vec!["r", "g", "b"]);
        // TODO: Parallelize
        for p in self.pixel_bounds {
            let rgb = self.get_pixel_rgb(p, splat_scale);
            image.set_channels(
                Point2Isize::from(p).as_point2usize(),
                &[rgb[0], rgb[1], rgb[2]],
            );
        }

        metadata.color_space = Some(self.color_space);

        image
    }

    fn write_image(&self, mut metadata: ImageMetadata, splat_scale: Float) {
        let img = self.get_image(&mut metadata, splat_scale);
        // FIXME: May be error
        img.write(&self.filename, &metadata).unwrap();
    }

    fn get_pixel_rgb(&self, p: Point2i, splat_scale: Float) -> RGB {
        let pixel = &self.pixels[p];
        let rgb_sum = pixel.rgb_sum.map(|v| v.as_());
        let mut rgb = RGB::from(rgb_sum);

        // Normalize with weight sum
        if pixel.weight_sum != 0.0 {
            rgb /= pixel.weight_sum.as_();
        }

        // Add splat value at pixel
        for c in 0..3 {
            let splat: Float = pixel.rgb_splat[c].load(Ordering::Acquire).as_();
            rgb[c] += splat_scale * splat / self.filter_integral;
        }

        // Convert to output RGB color space
        &self.output_rgb_from_sensor_rgb * rgb
    }

    fn full_resolution(&self) -> Point2i {
        self.full_resolution
    }

    fn pixel_bounds(&self) -> Bounds2i {
        self.pixel_bounds
    }

    fn diagonal(&self) -> Float {
        self.diagonal
    }

    fn filter(&self) -> &FilterEnum {
        &self.filter
    }

    fn sensor(&self) -> &PixelSensor {
        &self.sensor
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
