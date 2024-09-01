mod filter;

use std::path::Path;

pub use filter::{BoxFilter, Filter, FilterEnum};

use crate::{math::Point2i, Float};
use exr::prelude::write_rgb_file;

pub struct Image {
    resolution: Point2i,
    channel_names: Vec<String>,
    // TODO: Support other formats
    values: Vec<f32>,
}

impl Image {
    pub fn new<S, IntoIter>(resolution: Point2i, channel_names: IntoIter) -> Self
    where
        S: Into<String>,
        IntoIter: IntoIterator<Item = S>,
    {
        let channel_names: Vec<_> = channel_names.into_iter().map(Into::into).collect();
        let num_channels = channel_names.len();
        Self {
            resolution,
            channel_names,
            values: vec![0.0; num_channels * resolution.x() as usize * resolution.y() as usize],
        }
    }

    pub fn set_channel(&mut self, p: Point2i, channel: usize, value: Float) {
        let idx = self.pixel_offset(p) + channel;
        self.values[idx] = value;
    }

    pub fn set_channels(&mut self, p: Point2i, values: &[Float]) {
        assert_eq!(values.len(), self.num_channels());
        for (chan, val) in values.iter().enumerate() {
            self.set_channel(p, chan, *val);
        }
    }

    pub fn write(&self, path: &Path) -> exr::error::UnitResult {
        assert_eq!(path.extension().unwrap(), "exr");
        self.write_exr(path)
    }

    pub fn write_exr(&self, path: &Path) -> exr::error::UnitResult {
        write_rgb_file(
            path,
            self.resolution.x() as usize,
            self.resolution.y() as usize,
            |x, y| {
                let pixel_idx = self.pixel_offset(Point2i::new(x as i32, y as i32));
                (
                    self.values[pixel_idx],
                    self.values[pixel_idx + 1],
                    self.values[pixel_idx + 2],
                )
            },
        )
    }

    pub fn num_channels(&self) -> usize {
        self.channel_names.len()
    }

    fn pixel_offset(&self, p: Point2i) -> usize {
        self.num_channels() * (p.y() + self.resolution.x() + p.x()) as usize
    }
}

pub struct ImageMetadata {}
