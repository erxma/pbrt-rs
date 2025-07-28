use crate::{
    color::RGBColorSpace,
    core::{Float, Point2Isize, Point2Usize, Point2f},
};
use delegate::delegate;
use std::path::Path;
use strum::{EnumString, VariantNames};
use tinyvec::ArrayVec;

// Wrapping the inner generic impl so that value type is opaque to outside
// Not exporting the enum directly since variants have to be public
#[derive(Debug)]
pub struct Image(inner::ImageEnum);

impl Image {
    pub fn new(
        resolution: Point2Usize,
        channel_names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self(inner::Image::<f32>::new(resolution, channel_names, None).into())
    }

    delegate! {
        to match &self.0 {
            inner::ImageEnum::U8(img) => img,
            inner::ImageEnum::F16(img) => img,
            inner::ImageEnum::F32(img) => img,
        } {
            pub fn get_channel_desc(&self, requested_names: &[impl AsRef<str>]) -> Option<ImageChannelDesc>;
            pub fn get_channel(&self, p: Point2Isize, channel: usize, wrap_mode: WrapMode2D) -> Float;
            pub fn get_channels(&self, p: Point2Isize, wrap_mode: WrapMode2D) -> ImageChannelValues;
            pub fn bilerp_channel(&self, p: Point2f, channel: usize, wrap_mode: WrapMode2D) -> Float;
            pub fn resolution(&self) -> Point2Usize;
            pub fn num_channels(&self) -> usize;
        }

        to match &mut self.0 {
            inner::ImageEnum::U8(img) => img,
            inner::ImageEnum::F16(img) => img,
            inner::ImageEnum::F32(img) => img,
        } {
            pub fn set_channel(&mut self, p: Point2Usize, channel: usize, value: Float);
            pub fn set_channels(&mut self, p: Point2Usize, values: &[Float]);
        }
    }

    pub fn write(&self, path: &Path, metadata: &ImageMetadata) -> image::error::ImageResult<()> {
        match &self.0 {
            inner::ImageEnum::U8(image) => image.write(path, metadata),
            inner::ImageEnum::F16(_) => todo!(),
            inner::ImageEnum::F32(image) => image.write(path, metadata),
        }
    }

    pub fn select_channels(&self, desc: &ImageChannelDesc) -> Option<Self> {
        let inner = match &self.0 {
            inner::ImageEnum::U8(image) => image.select_channels(desc).map(Into::into),
            inner::ImageEnum::F16(image) => image.select_channels(desc).map(Into::into),
            inner::ImageEnum::F32(image) => image.select_channels(desc).map(Into::into),
        };

        inner.map(Self)
    }
}

mod inner {
    use super::*;

    use crate::{
        core::{Float, Point2Isize, Point2Usize, Point2f},
        imaging::{ColorEncoding, ColorEncodingEnum},
    };
    use image::EncodableLayout;
    use itertools::iproduct;
    use log::warn;
    use num_traits::Num;
    use std::{path::Path, str::FromStr as _};
    use tinyvec::ArrayVec;

    #[derive(Debug)]
    pub(super) struct Image<T> {
        resolution: Point2Usize,
        channel_names: Vec<String>,
        color_encoding: Option<ColorEncodingEnum>,

        values: Vec<T>,
    }

    #[derive(Debug, derive_more::From)]
    pub(super) enum ImageEnum {
        U8(Image<u8>),
        F16(Image<half::f16>),
        F32(Image<f32>),
    }

    impl<T: ImageValue> Image<T> {
        pub fn new(
            resolution: Point2Usize,
            channel_names: impl IntoIterator<Item = impl Into<String>>,
            color_encoding: Option<ColorEncodingEnum>,
        ) -> Self {
            let channel_names: Vec<_> = channel_names.into_iter().map(Into::into).collect();
            let num_channels = channel_names.len();
            Self {
                resolution,
                channel_names,
                color_encoding,
                values: vec![T::zero(); num_channels * resolution.x() * resolution.y()],
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

            self.values
                .get(self.pixel_offset(remapped.as_point2usize()) + channel)
                .expect("remapping should have placed point within bounds")
                .to_linear(self.color_encoding.as_ref())
        }

        pub fn get_channels(&self, p: Point2Isize, wrap_mode: WrapMode2D) -> ImageChannelValues {
            let mut values = ArrayVec::new();

            // Remap channel pixel coords before reading channel
            let remapped = remap_pixel_coords(p, self.resolution, wrap_mode);

            if let Some(remapped) = remapped {
                let pixel_offset = self.pixel_offset(remapped.as_point2usize());

                values.extend(
                    self.values
                        .get(pixel_offset..pixel_offset + self.num_channels())
                        .expect("remapping should have placed point within bounds")
                        .into_iter()
                        .map(|v| v.to_linear(self.color_encoding.as_ref())),
                );
            }

            ImageChannelValues(values)
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
            self.values[idx] = ImageValue::from_linear(value, self.color_encoding.as_ref());
        }

        pub fn set_channels(&mut self, p: Point2Usize, values: &[Float]) {
            assert_eq!(values.len(), self.num_channels());
            for (chan, val) in values.iter().enumerate() {
                self.set_channel(p, chan, *val);
            }
        }

        pub fn write(&self, path: &Path, _metadata: &ImageMetadata) -> image::error::ImageResult<()>
        where
            [T]: EncodableLayout,
        {
            // TODO: Use metadata

            let _ = path
                .extension()
                .and_then(|osstr| osstr.to_str())
                .and_then(|s| ImageExtension::from_str(s).ok())
                .expect("Image write path should have supported file extension");

            // Color format that the image values should be written as, determined below
            let color_type;
            // Ref to value vec to ultimately write out, which may not be from self
            // if it needs to be reordered and/or converted
            let mut vals_to_write = &self.values;
            // Possibly needed to hold on to a new, reordered image.
            let reordered_img;

            match self.num_channels() {
                3 => {
                    let desc = self.get_channel_desc(&["R", "G", "B"]);

                    if let Some(desc) = desc {
                        // Reorder in R, G, B order
                        reordered_img = self.select_channels(&desc).unwrap();
                        vals_to_write = &reordered_img.values;
                    } else {
                        warn!(
                            "Image has 3 channels, but they aren't 'R', 'G', 'B'. \
                        Output file may not be as expected ({})",
                            path.display()
                        );
                    }

                    color_type = image::ColorType::Rgb32F;
                }
                4 => {
                    let desc = self.get_channel_desc(&["R", "G", "B", "A"]);

                    if let Some(desc) = desc {
                        // Reorder in R, G, B, A order
                        reordered_img = self.select_channels(&desc).unwrap();
                        vals_to_write = &reordered_img.values;
                    } else {
                        warn!(
                            "Image has 4 channels, but they aren't 'R', 'G', 'B', 'A'. \
                        Output file may not be as expected ({})",
                            path.display()
                        );
                    }

                    color_type = image::ColorType::Rgba32F;
                }
                _ => todo!(),
            }

            image::save_buffer(
                path,
                vals_to_write.as_bytes(),
                self.resolution.x().try_into().unwrap(),
                self.resolution.y().try_into().unwrap(),
                color_type,
            )
        }

        pub fn select_channels(&self, desc: &ImageChannelDesc) -> Option<Self> {
            let new_channel_names: Option<Vec<_>> = desc
                .offsets
                .iter()
                .map(|i| self.channel_names.get(*i).cloned())
                .collect();

            let mut result = Self::new(
                self.resolution,
                new_channel_names?,
                self.color_encoding.clone(),
            );
            for (x, y) in iproduct!(0..self.resolution.x(), 0..self.resolution.y()) {
                let p = Point2Usize::new(x, y);
                let self_offset = self.pixel_offset(p);
                let result_offset = result.pixel_offset(p);
                for i in 0..desc.offsets.len() {
                    result.values[result_offset + i] = self.values[self_offset + desc.offsets[i]];
                }
            }

            Some(result)
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

    pub(super) trait ImageValue: Num + Copy {
        fn to_linear(self, encoding: Option<&impl ColorEncoding>) -> Float;
        fn from_linear(value: Float, encoding: Option<&impl ColorEncoding>) -> Self;
    }

    impl ImageValue for u8 {
        fn to_linear(self, encoding: Option<&impl ColorEncoding>) -> Float {
            encoding
                .expect("Color encoding must be specified to convert u8 to linear")
                .u8_to_linear(self)
        }

        fn from_linear(value: Float, encoding: Option<&impl ColorEncoding>) -> Self {
            encoding
                .expect("Color encoding must be specified to convert u8 to linear")
                .linear_to_u8(value)
        }
    }

    impl ImageValue for half::f16 {
        fn to_linear(self, _encoding: Option<&impl ColorEncoding>) -> Float {
            self.into()
        }

        fn from_linear(value: Float, _encoding: Option<&impl ColorEncoding>) -> Self {
            #[cfg(feature = "use-f64")]
            return Self::from_f64(value);
            #[cfg(not(feature = "use-f64"))]
            return Self::from_f32(value);
        }
    }

    impl ImageValue for f32 {
        fn to_linear(self, _encoding: Option<&impl ColorEncoding>) -> Float {
            self as Float
        }

        fn from_linear(value: Float, _encoding: Option<&impl ColorEncoding>) -> Self {
            value as f32
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, EnumString, VariantNames)]
#[strum(ascii_case_insensitive, serialize_all = "UPPERCASE")]
pub enum ImageExtension {
    Exr,
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

#[derive(Debug)]
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

pub struct ImageChannelValues(pub ArrayVec<[Float; 4]>);
