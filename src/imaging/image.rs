use crate::{
    color::RGBColorSpace,
    core::{Float, Point2Isize, Point2Usize, Point2f},
};
use exr::prelude::write_rgb_file;
use num_traits::AsPrimitive;
use std::path::Path;
use tinyvec::ArrayVec;

#[derive(Debug)]
pub struct Image {
    resolution: Point2Usize,
    channel_names: Vec<String>,

    values: Vec<f32>,
}

impl Image {
    pub fn new(
        resolution: Point2Usize,
        channel_names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let channel_names: Vec<_> = channel_names.into_iter().map(Into::into).collect();
        let num_channels = channel_names.len();
        Self {
            resolution,
            channel_names,
            values: vec![0.0; num_channels * resolution.x() * resolution.y()],
        }
    }

    pub fn get_channel_desc(
        &self,
        requested_names: &[impl AsRef<str>],
    ) -> Option<ImageChannelDesc> {
        let offsets: Option<_> = requested_names
            .iter()
            .map(|req| {
                self.channel_names
                    .iter()
                    .position(|name| name == req.as_ref())
            })
            .collect();

        Some(ImageChannelDesc { offsets: offsets? })
    }

    pub fn get_channel(&self, p: Point2Isize, channel: usize, wrap_mode: WrapMode2D) -> Float {
        // Remap channel pixel coords before reading channel
        let remapped = remap_pixel_coords(p, self.resolution, wrap_mode);
        if remapped.is_none() {
            return 0.0;
        }
        let remapped = remapped.unwrap();

        *self
            .values
            .get(self.pixel_offset(remapped.as_point2usize()) + channel)
            .expect("remapping should have placed point within bounds")
    }

    pub fn bilerp_channel(&self, p: Point2f, channel: usize, wrap_mode: WrapMode2D) -> Float {
        // Compute discrete pixel coords and offsets for p
        let x = p.x() * self.resolution.x() as Float - 0.5;
        let y = p.y() * self.resolution.y() as Float - 0.5;
        let xi = x.floor() as isize;
        let yi = y.floor() as isize;
        let dx = x - x.floor();
        let dy = y - y.floor();

        // Load pixel channel values and return bilinearly interpolated value
        let v = [
            self.get_channel(Point2Isize::new(xi, yi), channel, wrap_mode),
            self.get_channel(Point2Isize::new(xi + 1, yi), channel, wrap_mode),
            self.get_channel(Point2Isize::new(xi, yi + 1), channel, wrap_mode),
            self.get_channel(Point2Isize::new(xi + 1, yi + 1), channel, wrap_mode),
        ];

        (1.0 - dx) * (1.0 - dy) * v[0]
            + dx * (1.0 - dy) * v[1]
            + (1.0 - dx) * dy * v[2]
            + dx * dy * v[3]
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

    pub fn resolution(&self) -> Point2Usize {
        self.resolution
    }

    pub fn num_channels(&self) -> usize {
        self.channel_names.len()
    }

    fn pixel_offset(&self, p: Point2Usize) -> usize {
        self.num_channels() * (p.y() * self.resolution.x() + p.x())
    }
}

#[inline]
fn remap_pixel_coords(
    mut point: Point2Isize,
    resolution: Point2Usize,
    wrap_mode: WrapMode2D,
) -> Option<Point2Isize> {
    let resolution = resolution.as_point2isize();

    if wrap_mode[0] == WrapMode::OctahedralSphere {
        assert_eq!(wrap_mode[1], WrapMode::OctahedralSphere);

        if point.x() < 0 {
            // Mirror x across u = 0, mirror y across v = 0.5
            point = Point2Isize::new(-point.x(), resolution.y() - 1 - point.y());
        } else if point.y() >= resolution.y() {
            // Mirror x across u = 0.5, mirror y across v = 1
            point = Point2Isize::new(
                resolution.x() - 1 - point.x(),
                2 * resolution.y() - 1 - point.y(),
            );
        }

        if resolution.x() == 1 {
            *point.x_mut() = 0;
        }
        if resolution.y() == 1 {
            *point.y_mut() = 0;
        }
    } else {
        for c in 0..2 {
            if point[c] < 0 || point[c] >= resolution[c] {
                match wrap_mode[c] {
                    WrapMode::Black => return None,
                    WrapMode::Clamp => {
                        point[c] = point[c].clamp(0, resolution[c] - 1);
                    }
                    WrapMode::Repeat => {
                        point[c] = point[c] % resolution[c] - 1;
                    }
                    WrapMode::OctahedralSphere => unreachable!(),
                }
            }
        }
    }

    Some(point)
}

#[derive(Default)]
pub struct ImageMetadata<'a> {
    pub color_space: Option<&'a RGBColorSpace>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrapMode {
    Black,
    Clamp,
    Repeat,
    OctahedralSphere,
}

pub type WrapMode2D = [WrapMode; 2];

pub struct ImageChannelDesc {
    pub offsets: ArrayVec<[usize; 4]>,
}

impl ImageChannelDesc {
    pub fn is_identity(&self) -> bool {
        self.offsets
            .iter()
            .enumerate()
            .all(|(i, &offset)| i == offset)
    }
}
