use std::sync::Arc;

use super::{next_float_down, next_float_up, Float, Normal3f, Point3f, Point3fi, Tuple, Vec3f};
use crate::media::MediumEnum;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ray {
    /// Origin of the ray.
    pub o: Point3f,
    /// Direction of the ray. Is multiplied for each step.
    pub dir: Vec3f,
    pub time: Float,
    pub medium: Option<Arc<MediumEnum>>,
}

impl Ray {
    pub fn new(o: Point3f, dir: Vec3f, time: Float, medium: Option<Arc<MediumEnum>>) -> Self {
        Self {
            o,
            dir,
            time,
            medium,
        }
    }

    pub fn spawn_with_dir(pi: Point3fi, n: Normal3f, time: Float, dir: Vec3f) -> Self {
        Self::new(Self::offset_ray_origin(pi, n, dir), dir, time, None)
    }

    pub fn spawn_from_to(p_from: Point3fi, n: Normal3f, time: Float, p_to: Point3f) -> Self {
        let dir = p_to - p_from.midpoints();
        Self::spawn_with_dir(p_from, n, time, dir)
    }

    pub fn spawn_from_to_fi(
        p_from: Point3fi,
        n_from: Normal3f,
        time: Float,
        p_to: Point3fi,
        n_to: Normal3f,
    ) -> Self {
        let p_from = Self::offset_ray_origin(p_from, n_from, p_to.midpoints() - p_from.midpoints());
        let p_to = Self::offset_ray_origin(p_to, n_to, p_from - p_to.midpoints());
        Self::new(p_from, p_to - p_from, time, None)
    }

    pub fn offset_ray_origin(pi: Point3fi, n: Normal3f, w: Vec3f) -> Point3f {
        // Find vector offset to corner of error bounds and compute initial po
        let d = n.abs().dot_v(pi.error());
        let mut offset = d * Vec3f::from(n);
        if n.dot_v(w) < 0.0 {
            offset = -offset;
        }
        let mut po = pi.midpoints() + offset;

        // Round offset point po away from pi
        for i in 0..3 {
            if offset[i] > 0.0 {
                po[i] = next_float_up(po[i]);
            } else if offset[i] < 0.0 {
                po[i] = next_float_down(po[i]);
            }
        }

        po
    }
}

impl Ray {
    pub fn at(&self, t: Float) -> Point3f {
        self.o + self.dir * t
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RayDifferential {
    pub ray: Ray,
    pub differentials: Option<Differentials>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Differentials {
    pub rx_origin: Point3f,
    pub ry_origin: Point3f,
    pub rx_dir: Vec3f,
    pub ry_dir: Vec3f,
}

impl RayDifferential {
    /// Construct a new ray differential with the given ray and differentials
    pub fn new(ray: Ray, differentials: Differentials) -> Self {
        Self {
            ray,
            differentials: Some(differentials),
        }
    }

    /// Construct a new ray differential with the given ray,
    /// and no differentials set.
    pub fn new_without_diff(ray: Ray) -> Self {
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
