mod filter;

pub use filter::{BoxFilter, Filter, FilterEnum, FilterSample, GaussianFilter, TriangleFilter};

use crate::{
    color::RGBColorSpace,
    core::{Float, Point2Usize},
};
use exr::prelude::write_rgb_file;
use num_traits::AsPrimitive;
use std::path::Path;

pub struct Image {
    resolution: Point2Usize,
    channel_names: Vec<String>,
    // TODO: Support other formats
    values: Vec<f32>,
}

impl Image {
    pub fn new<S, IntoIter>(resolution: Point2Usize, channel_names: IntoIter) -> Self
    where
        S: Into<String>,
        IntoIter: IntoIterator<Item = S>,
    {
        let channel_names: Vec<_> = channel_names.into_iter().map(Into::into).collect();
        let num_channels = channel_names.len();
        Self {
            resolution,
            channel_names,
            values: vec![0.0; num_channels * resolution.x() * resolution.y()],
        }
    }

    pub fn set_channel(&mut self, p: Point2Usize, channel: usize, value: Float) {
        let idx = self.pixel_offset(p) + channel;
        self.values[idx] = value.as_();
    }

    pub fn set_channels(&mut self, p: Point2Usize, values: &[Float]) {
        assert_eq!(values.len(), self.num_channels());
        for (chan, val) in values.iter().enumerate() {
            self.set_channel(p, chan, *val);
        }
    }

    pub fn write(&self, path: &Path, _metadata: &ImageMetadata) -> exr::error::UnitResult {
        assert_eq!(path.extension().unwrap(), "exr");
        self.write_exr(path)
    }

    pub fn write_exr(&self, path: &Path) -> exr::error::UnitResult {
        write_rgb_file(path, self.resolution.x(), self.resolution.y(), |x, y| {
            let pixel_idx = self.pixel_offset(Point2Usize::new(x, y));
            (
                self.values[pixel_idx],
                self.values[pixel_idx + 1],
                self.values[pixel_idx + 2],
            )
        })
    }

    pub fn num_channels(&self) -> usize {
        self.channel_names.len()
    }

    fn pixel_offset(&self, p: Point2Usize) -> usize {
        self.num_channels() * (p.y() * self.resolution.x() + p.x())
    }
}

#[derive(Default)]
pub struct ImageMetadata<'a> {
    pub color_space: Option<&'a RGBColorSpace>,
}
