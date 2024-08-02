use crate::{
    math::{Point3f, Vec3f},
    medium::Medium,
    Float,
};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ray<'a> {
    /// Origin of the ray.
    pub o: Point3f,
    /// Direction of the ray. Is multiplied for each step.
    pub dir: Vec3f,
    pub t_max: Float,
    pub time: Float,
    pub medium: Option<&'a Medium>,
}

impl<'a> Ray<'a> {
    pub fn new(
        o: Point3f,
        dir: Vec3f,
        t_max: Float,
        time: Float,
        medium: Option<&'a Medium>,
    ) -> Self {
        Self {
            o,
            dir,
            t_max,
            time,
            medium,
        }
    }
}

impl Ray<'_> {
    pub fn at(&self, t: Float) -> Point3f {
        self.o + self.dir * t
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RayDifferential<'a> {
    pub ray: Ray<'a>,
    pub differentials: Option<Differentials>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Differentials {
    pub rx_origin: Point3f,
    pub ry_origin: Point3f,
    pub rx_dir: Vec3f,
    pub ry_dir: Vec3f,
}

impl<'a> RayDifferential<'a> {
    /// Construct a new ray differential with the given ray,
    /// and no differentials set.
    pub fn new(ray: Ray<'a>) -> Self {
        Self {
            ray,
            differentials: None,
        }
    }

    pub fn set_differentials(
        &mut self,
        rx_origin: Point3f,
        rx_dir: Vec3f,
        ry_origin: Point3f,
        ry_dir: Vec3f,
    ) {
        self.differentials = Some(Differentials {
            rx_origin,
            ry_origin,
            rx_dir,
            ry_dir,
        })
    }

    pub fn scale_differentials(&mut self, s: Float) {
        let diffs = &mut self
            .differentials
            .as_mut()
            .expect("Ray differentials should be set to be able to scale them");

        diffs.rx_origin = self.ray.o + (diffs.rx_origin - self.ray.o) * s;
        diffs.ry_origin = self.ray.o + (diffs.ry_origin - self.ray.o) * s;
        diffs.rx_dir = self.ray.dir + (diffs.rx_dir - self.ray.dir) * s;
        diffs.ry_dir = self.ray.dir + (diffs.ry_dir - self.ray.dir) * s;
    }
}
