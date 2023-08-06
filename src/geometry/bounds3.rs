use std::{
    cmp::{max, min},
    ops::{Index, IndexMut, Mul},
};

use num_traits::{Bounded, Float, Num};

use crate::{self as pbrt, geometry::routines::gamma};

use super::{
    point3::{Point3, Point3f},
    ray::Ray,
    routines::lerp,
    vec3::{Vec3, Vec3f},
};

/// A 3D axis-aligned bounding box (AABB).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds3<T> {
    pub p_min: Point3<T>,
    pub p_max: Point3<T>,
}

pub type Bounds3i = Bounds3<i32>;
pub type Bounds3f = Bounds3<pbrt::Float>;

impl<T: Num + Copy> Bounds3<T> {
    /// Construct a new bounding box that consists of a single point.
    pub fn new_with_point(p: Point3<T>) -> Self {
        Self { p_min: p, p_max: p }
    }

    /// Returns the coordinates of one of the eight corners of `self`.
    ///
    /// 0 returns `p_min`, 7 returns `p_max`.
    pub fn corner(&self, corner: usize) -> Point3<T> {
        Point3::new(
            self[corner & 1].x,
            self[if corner & 2 != 0 { 1 } else { 0 }].y,
            self[if corner & 4 != 0 { 1 } else { 0 }].z,
        )
    }

    /// Construct a new bounding box that is `self` but expanded by `delta`
    /// on all axes, in both directions on the axis.
    #[inline]
    pub fn expand(self, delta: T) -> Self {
        Self {
            p_min: self.p_min - Vec3::new(delta, delta, delta),
            p_max: self.p_max + Vec3::new(delta, delta, delta),
        }
    }

    /// Checks for a ray-box intersection and returns the the two parametric `t`
    /// values of the intersection, if any, as `(lower, higher)`.
    #[inline]
    pub fn intersect_p(&self, ray: Ray) -> Option<(pbrt::Float, pbrt::Float)>
    where
        T: Into<pbrt::Float>,
    {
        // Convert to Float
        let (p_min, p_max) = (self.p_min.into_(), self.p_max.into_());

        let (mut t0, mut t1) = (0.0, ray.t_max);
        for i in 0..3 {
            // Update interval for ith bounding box slab:
            let inv_ray_dir = 1.0 / ray.dir[i];

            let t_min_plane = (p_min[i] - ray.o[i]) * inv_ray_dir;
            let t_max_plane = (p_max[i] - ray.o[i]) * inv_ray_dir;

            let t_near = t_min_plane.min(t_max_plane);
            let mut t_far = t_min_plane.max(t_max_plane);

            t_far *= 1.0 + 2.0 * gamma(3);

            t0 = if t_near > t0 { t_near } else { t0 };
            t1 = if t_far < t1 { t_far } else { t1 };

            if t0 > t1 {
                return None;
            }
        }

        Some((t0, t1))
    }

    #[inline]
    pub fn intersect_p_with_inv_dir(&self, ray: Ray, inv_dir: Vec3f, dir_is_neg: Vec3<bool>) -> bool
    where
        T: Into<pbrt::Float>,
    {
        // Convert to Float
        let bounds: Bounds3f = self.to_owned().into_();
        // Convert to
        let dir_is_neg: Vec3<usize> = dir_is_neg.into_();

        // Check for ray intersection against x and y slabs
        let tx_min = (bounds[dir_is_neg.x].x - ray.o.x) * inv_dir.x;
        let ty_min = (bounds[dir_is_neg.y].y - ray.o.y) * inv_dir.y;
        let tz_min = (bounds[dir_is_neg.x].x - ray.o.x) * inv_dir.x;
        let mut tx_max = (bounds[1 - dir_is_neg.x].x - ray.o.x) * inv_dir.x;
        let mut ty_max = (bounds[1 - dir_is_neg.y].y - ray.o.y) * inv_dir.y;
        let mut tz_max = (bounds[1 - dir_is_neg.x].x - ray.o.x) * inv_dir.x;

        tx_max *= 1.0 + 2.0 * gamma(3);
        ty_max *= 1.0 + 2.0 * gamma(3);
        tz_max *= 1.0 + 2.0 * gamma(3);

        if tx_min > ty_max || ty_min > tx_max {
            return false;
        }

        let mut t_min = tx_min.max(ty_min);
        let mut t_max = tx_min.min(ty_min);

        if tx_min > tz_max || tz_min > tx_max {
            return false;
        }

        t_min = t_min.max(tz_min);
        t_max = t_max.min(tz_max);

        t_min < ray.t_max && t_max > 0.0
    }
}

impl<T: Num + PartialOrd + Copy> Bounds3<T> {
    /// Obtain the vector from the min to the max point of `self`
    /// (which is along a diagonal line across the box).
    pub fn diagonal(&self) -> Vec3<T> {
        self.p_max - self.p_min
    }

    /// Compute the surface area of `self`.
    pub fn surface_area(&self) -> T
    where
        T: Mul<i32, Output = T>,
    {
        let d = self.diagonal();

        (d.x * d.y + d.x * d.z + d.y * d.z) * 2
    }

    /// Compute the volume of `self`.
    pub fn volume(&self) -> T {
        let d = self.diagonal();

        d.x * d.y * d.z
    }

    /// Determines the axis that `self` is widest on, and returns its index.
    pub fn max_extent(&self) -> usize {
        let d = self.diagonal();
        if d.x > d.y && d.x > d.z {
            0
        } else if d.y > d.z {
            1
        } else {
            2
        }
    }

    /// Returns `true` if `self` and `other` intersect at any point, inclusive
    /// (touching exactly on a corner counts), `false` otherwise.
    pub fn overlaps(self, other: Self) -> bool {
        let x_overlaps = self.p_max.x >= other.p_min.x && self.p_min.x <= other.p_max.x;
        let y_overlaps = self.p_max.y >= other.p_min.y && self.p_min.y <= other.p_max.y;
        let z_overlaps = self.p_max.z >= other.p_min.z && self.p_min.z <= other.p_max.z;

        x_overlaps && y_overlaps && z_overlaps
    }

    /// Linearly interpolate between the min and max points of `self`, on all axes.
    ///
    /// Extrapolates for components of `t` `<0` or `>1`.
    pub fn lerp(self, t: Point3f) -> Point3<pbrt::Float>
    where
        T: Into<pbrt::Float>,
    {
        let b = self.into_();
        Point3::new(
            lerp(b.p_min.x, b.p_max.x, t.x),
            lerp(b.p_min.x, b.p_max.x, t.y),
            lerp(b.p_min.x, b.p_max.x, t.z),
        )
    }
}

impl<T: Ord + Copy> Bounds3<T> {
    /// Construct a new bounding box with two corner points.
    ///
    /// The min and max points are determined by the component-wise mins and maxes
    /// of the given points.
    pub fn new(p1: Point3<T>, p2: Point3<T>) -> Self {
        let p_min = Point3::new(min(p1.x, p2.x), min(p1.y, p2.y), min(p1.z, p2.z));
        let p_max = Point3::new(max(p1.x, p2.x), max(p1.y, p2.y), max(p1.z, p2.z));

        Self { p_min, p_max }
    }

    /// Construct the union of `self` and `other`.
    /// Specifically, a box using the min and max points of the two.
    ///
    /// Note that this new box doesn't necessarily consist of the exact same space
    /// as the two combined.
    pub fn union(self, other: Self) -> Self {
        let p_min = Point3::new(
            min(self.p_min.x, other.p_min.x),
            min(self.p_min.y, other.p_min.y),
            min(self.p_min.z, other.p_min.z),
        );
        let p_max = Point3::new(
            max(self.p_max.x, other.p_max.x),
            max(self.p_max.y, other.p_max.y),
            max(self.p_max.z, other.p_max.z),
        );

        Self { p_min, p_max }
    }

    /// Construct the minimum bounding box that contains `self` as well as a point `p`.
    ///
    /// i.e., expand `self` by the amount needed to reach `p`
    /// (which may be none if `p` is already inside).
    pub fn union_point(self, p: Point3<T>) -> Self {
        let p_min = Point3::new(
            min(self.p_min.x, p.x),
            min(self.p_min.y, p.y),
            min(self.p_min.z, p.z),
        );
        let p_max = Point3::new(
            max(self.p_max.x, p.x),
            max(self.p_max.y, p.y),
            max(self.p_max.z, p.z),
        );

        Self { p_min, p_max }
    }

    /// Construct a bounding box consisting of the intersection of `self` and `other`.
    pub fn intersect(self, other: Self) -> Self {
        let p_min = Point3::new(
            max(self.p_min.x, other.p_min.x),
            max(self.p_min.y, other.p_min.y),
            max(self.p_min.z, other.p_min.z),
        );
        let p_max = Point3::new(
            min(self.p_max.x, other.p_max.x),
            min(self.p_max.y, other.p_max.y),
            min(self.p_max.z, other.p_max.z),
        );

        Self { p_min, p_max }
    }
}

impl<T: Float + Copy> Bounds3<T> {
    /// Construct a new bounding box with two corner points.
    ///
    /// The min and max points are determined by the component-wise mins and maxes
    /// of the given points.
    pub fn new_f(p1: Point3<T>, p2: Point3<T>) -> Self {
        let p_min = Point3::new(p1.x.min(p2.x), p1.y.min(p2.y), p1.z.min(p2.z));
        let p_max = Point3::new(p1.x.max(p2.x), p1.y.max(p2.y), p1.z.max(p2.z));

        Self { p_min, p_max }
    }

    /// Construct the union of `self` and `other`.
    /// Specifically, a box using the min and max points of the two.
    ///
    /// Note that this new box doesn't necessarily consist of the exact same space
    /// as the two combined.
    pub fn union_f(self, other: Self) -> Self {
        let p_min = Point3::new(
            self.p_min.x.min(other.p_min.x),
            self.p_min.y.min(other.p_min.y),
            self.p_min.z.min(other.p_min.z),
        );
        let p_max = Point3::new(
            self.p_max.x.max(other.p_max.x),
            self.p_max.y.max(other.p_max.y),
            self.p_max.z.max(other.p_max.z),
        );

        Self { p_min, p_max }
    }

    /// Construct the minimum bounding box that contains `self` as well as a point `p`.
    ///
    /// i.e., expand `self` by the amount needed to reach `p`
    /// (which may be none if `p` is already inside).
    pub fn union_point_f(self, p: Point3<T>) -> Self {
        let p_min = Point3::new(
            self.p_min.x.min(p.x),
            self.p_min.y.min(p.y),
            self.p_min.z.min(p.z),
        );
        let p_max = Point3::new(
            self.p_max.x.max(p.x),
            self.p_max.y.max(p.y),
            self.p_max.z.max(p.z),
        );

        Self { p_min, p_max }
    }

    /// Construct a bounding box consisting of the intersection of `self` and `other`.
    pub fn intersect_f(self, other: Self) -> Self {
        let p_min = Point3::new(
            self.p_min.x.max(other.p_min.x),
            self.p_min.y.max(other.p_min.y),
            self.p_min.z.max(other.p_min.z),
        );
        let p_max = Point3::new(
            self.p_max.x.min(other.p_max.x),
            self.p_max.y.min(other.p_max.y),
            self.p_max.z.min(other.p_max.z),
        );

        Self { p_min, p_max }
    }
}

impl<T: Bounded + Copy> Bounds3<T> {
    /// Construct an empty box.
    ///
    /// This is done by setting the extents to an invalid config,
    /// such that any operations with it would yield the expected result.
    pub fn empty() -> Self {
        let min_val = T::min_value();
        let max_val = T::max_value();
        let p_min = Point3::new(max_val, max_val, max_val);
        let p_max = Point3::new(min_val, min_val, min_val);

        Self { p_min, p_max }
    }
}

impl<T> Bounds3<T> {
    /// Convert elements into another type.
    pub fn into_<U>(self) -> Bounds3<U>
    where
        T: Into<U>,
    {
        Bounds3 {
            p_min: self.p_min.into_(),
            p_max: self.p_max.into_(),
        }
    }
}

impl<T> Index<usize> for Bounds3<T> {
    type Output = Point3<T>;

    /// Index `self`'s elements by 0, 1.
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.p_min,
            1 => &self.p_max,
            _ => panic!("Index out of bounds for Point3"),
        }
    }
}

impl<T> IndexMut<usize> for Bounds3<T> {
    /// Index `self`'s elements by 0, 1, mutably.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.p_min,
            1 => &mut self.p_max,
            _ => panic!("Index out of bounds for Point3"),
        }
    }
}
