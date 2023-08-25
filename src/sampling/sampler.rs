use crate::{
    self as pbrt,
    base::camera::CameraSample,
    geometry::point2::{Point2f, Point2i},
};

pub trait Sampler {
    fn start_pixel(&self, p: Point2i);

    fn get_1d(&self) -> pbrt::Float;
    fn get_2d(&self) -> Point2f;

    fn get_camera_sample(&self, p_raster: Point2i) -> CameraSample;

    fn request_1d_array(&mut self, n: usize);
    fn request_2d_array(&mut self, n: usize);
    fn round_count(&self, n: usize);

    fn get_1d_array(&self, n: usize) -> [pbrt::Float];
    fn get_2d_array(&self, n: usize) -> [Point2f];

    fn set_sample_number(&mut self, sample_num: i64) -> bool;
    fn start_next_sample(&mut self) -> bool;

    fn clone_with_seed(&self, seed: i32) -> Self;
}
