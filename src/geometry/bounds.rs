use std::ops::{Index, IndexMut, RangeInclusive};

use itertools::iproduct;

use crate::{
    self as pbrt,
    math::{
        gamma, lerp, Point2f, Point2i, Point3f, Point3i, Vec2f, Vec2i, Vec3B, Vec3Usize, Vec3f,
        Vec3i,
    },
};

use super::ray::Ray;

/// A 3D axis-aligned bounding box (AABB) of `i32`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bounds3i {
    pub p_min: Point3i,
    pub p_max: Point3i,
}

impl Bounds3i {
    /// Construct a new bounding box with two corner points.
    ///
    /// The min and max points are determined by the component-wise mins and maxes
    /// of the given points.
    pub fn new(p1: Point3i, p2: Point3i) -> Self {
        let p_min = Point3i::new(p1.x().min(p2.x()), p1.y().min(p2.y()), p1.z().min(p2.z()));
        let p_max = Point3i::new(p1.x().max(p2.x()), p1.y().max(p2.y()), p1.z().max(p2.z()));

        Self { p_min, p_max }
    }

    /// Construct a new bounding box that consists of a single point.
    pub fn new_with_point(p: Point3i) -> Self {
        Self { p_min: p, p_max: p }
    }

    /// Returns the coordinates of one of the eight corners of `self`.
    ///
    /// 0 returns `p_min`, 7 returns `p_max`.
    pub fn corner(&self, corner: usize) -> Point3i {
        Point3i::new(
            self[corner & 1].x(),
            self[if corner & 2 != 0 { 1 } else { 0 }].y(),
            self[if corner & 4 != 0 { 1 } else { 0 }].z(),
        )
    }

    /// Construct a new bounding box that is `self` but expanded by `delta`
    /// on all axes, in both directions on the axis.
    #[inline]
    pub fn expand(self, delta: i32) -> Self {
        Self {
            p_min: self.p_min - Vec3i::new(delta, delta, delta),
            p_max: self.p_max + Vec3i::new(delta, delta, delta),
        }
    }

    /// Checks for a ray-box intersection and returns the the two parametric `t`
    /// values of the intersection, if any, as `(lower, higher)`.
    #[inline]
    pub fn intersect_p(&self, ray: Ray) -> Option<(pbrt::Float, pbrt::Float)> {
        // Convert to Float
        let bounds: Bounds3f = self.to_owned().into();

        let (mut t0, mut t1) = (0.0, ray.t_max);
        for i in 0..3 {
            // Update interval for ith bounding box slab:
            let inv_ray_dir = 1.0 / ray.dir[i];

            let t_min_plane = (bounds.p_min[i] - ray.o[i]) * inv_ray_dir;
            let t_max_plane = (bounds.p_max[i] - ray.o[i]) * inv_ray_dir;

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
    pub fn intersect_p_with_inv_dir(&self, ray: Ray, inv_dir: Vec3i, dir_is_neg: Vec3B) -> bool {
        // Convert to float
        let bounds: Bounds3f = self.to_owned().into();
        let inv_dir: Vec3f = inv_dir.into();
        // Convert to usize for easy multiplication
        let dir_is_neg: Vec3Usize = dir_is_neg.into_();

        // Check for ray intersection against x and y slabs
        let tx_min = (bounds[dir_is_neg.x].x() - ray.o.x()) * inv_dir.x();
        let ty_min = (bounds[dir_is_neg.y].y() - ray.o.y()) * inv_dir.y();
        let tz_min = (bounds[dir_is_neg.x].x() - ray.o.x()) * inv_dir.x();
        let mut tx_max = (bounds[1 - dir_is_neg.x].x() - ray.o.x()) * inv_dir.x();
        let mut ty_max = (bounds[1 - dir_is_neg.y].y() - ray.o.y()) * inv_dir.y();
        let mut tz_max = (bounds[1 - dir_is_neg.x].x() - ray.o.x()) * inv_dir.x();

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

    /// Obtain the vector from the min to the max point of `self`
    /// (which is along a diagonal line across the box).
    pub fn diagonal(&self) -> Vec3i {
        self.p_max - self.p_min
    }

    /// Compute the surface area of `self`.
    pub fn surface_area(&self) -> i32 {
        let d = self.diagonal();

        (d.x() * d.y() + d.x() * d.z() + d.y() * d.z()) * 2
    }

    /// Compute the volume of `self`.
    pub fn volume(&self) -> i32 {
        let d = self.diagonal();
        d.x() * d.y() * d.z()
    }

    /// Determines the axis that `self` is widest on, and returns its index.
    pub fn max_extent(&self) -> usize {
        let d = self.diagonal();
        if d.x() > d.y() && d.x() > d.z() {
            0
        } else if d.y() > d.z() {
            1
        } else {
            2
        }
    }

    /// Returns `true` if `self` and `other` intersect at any point, inclusive
    /// (touching exactly on a corner counts), `false` otherwise.
    pub fn overlaps(self, other: Self) -> bool {
        let x_overlaps = self.p_max.x() >= other.p_min.x() && self.p_min.x() <= other.p_max.x();
        let y_overlaps = self.p_max.y() >= other.p_min.y() && self.p_min.y() <= other.p_max.y();
        let z_overlaps = self.p_max.z() >= other.p_min.z() && self.p_min.z() <= other.p_max.z();

        x_overlaps && y_overlaps && z_overlaps
    }

    pub fn contains(&self, p: Point3i) -> bool {
        p.x() >= self.p_min.x()
            && p.x() <= self.p_max.x()
            && p.y() >= self.p_min.y()
            && p.y() <= self.p_max.y()
            && p.z() >= self.p_min.z()
            && p.z() <= self.p_max.z()
    }

    /// Linearly interpolate between the min and max points of `self`, on all axes.
    ///
    /// Extrapolates for components of `t` `<0` or `>1`.
    pub fn lerp(self, t: Point3i) -> Point3f {
        let bounds: Bounds3f = self.into();
        let t: Point3f = t.into();

        Point3f::new(
            lerp(bounds.p_min.x(), bounds.p_max.x(), t.x()),
            lerp(bounds.p_min.x(), bounds.p_max.x(), t.y()),
            lerp(bounds.p_min.x(), bounds.p_max.x(), t.z()),
        )
    }

    /// Construct the union of `self` and `other`.
    /// Specifically, a box using the min and max points of the two.
    ///
    /// Note that this new box doesn't necessarily consist of the exact same space
    /// as the two combined.
    pub fn union(self, other: Self) -> Self {
        let p_min = Point3i::new(
            self.p_min.x().min(other.p_min.x()),
            self.p_min.y().min(other.p_min.y()),
            self.p_min.z().min(other.p_min.z()),
        );
        let p_max = Point3i::new(
            self.p_max.x().max(other.p_max.x()),
            self.p_max.y().max(other.p_max.y()),
            self.p_max.z().max(other.p_max.z()),
        );

        Self { p_min, p_max }
    }

    /// Construct the minimum bounding box that contains `self` as well as a point `p`.
    ///
    /// i.e., expand `self` by the amount needed to reach `p`
    /// (which may be none if `p` is already inside).
    pub fn union_point(self, p: Point3i) -> Self {
        let p_min = Point3i::new(
            self.p_min.x().min(p.x()),
            self.p_min.y().min(p.y()),
            self.p_min.z().min(p.z()),
        );
        let p_max = Point3i::new(
            self.p_max.x().max(p.x()),
            self.p_max.y().max(p.y()),
            self.p_max.z().max(p.z()),
        );

        Self { p_min, p_max }
    }

    /// Construct a bounding box consisting of the intersection of `self` and `other`.
    pub fn intersect(self, other: Self) -> Self {
        let p_min = Point3i::new(
            self.p_min.x().max(other.p_min.x()),
            self.p_min.y().max(other.p_min.y()),
            self.p_min.z().max(other.p_min.z()),
        );
        let p_max = Point3i::new(
            self.p_max.x().min(other.p_max.x()),
            self.p_max.y().min(other.p_max.y()),
            self.p_max.z().min(other.p_max.z()),
        );

        Self { p_min, p_max }
    }

    /// Construct an empty box.
    ///
    /// This is done by setting the extents to an invalid config,
    /// such that any operations with it would yield the expected result.
    pub fn empty() -> Self {
        let min_val = i32::MIN;
        let max_val = i32::MAX;
        let p_min = Point3i::new(max_val, max_val, max_val);
        let p_max = Point3i::new(min_val, min_val, min_val);

        Self { p_min, p_max }
    }
}

impl Index<usize> for Bounds3i {
    type Output = Point3i;

    /// Index `self`'s elements by 0, 1.
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.p_min,
            1 => &self.p_max,
            _ => panic!("Index out of bounds for Bounds3"),
        }
    }
}

impl IndexMut<usize> for Bounds3i {
    /// Index `self`'s elements by 0, 1, mutably.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.p_min,
            1 => &mut self.p_max,
            _ => panic!("Index out of bounds for Bounds3"),
        }
    }
}

/// A 3D axis-aligned bounding box (AABB) of `f32`,
/// or `f64` if feature `use-f64` is enabled.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bounds3f {
    pub p_min: Point3f,
    pub p_max: Point3f,
}

impl Bounds3f {
    /// Construct a new bounding box with two corner points.
    ///
    /// The min and max points are determined by the component-wise mins and maxes
    /// of the given points.
    pub fn new(p1: Point3f, p2: Point3f) -> Self {
        let p_min = Point3f::new(p1.x().min(p2.x()), p1.y().min(p2.y()), p1.z().min(p2.z()));
        let p_max = Point3f::new(p1.x().max(p2.x()), p1.y().max(p2.y()), p1.z().max(p2.z()));

        Self { p_min, p_max }
    }

    /// Construct a new bounding box that consists of a single point.
    pub fn new_with_point(p: Point3f) -> Self {
        Self { p_min: p, p_max: p }
    }

    /// Returns the coordinates of one of the eight corners of `self`.
    ///
    /// 0 returns `p_min`, 7 returns `p_max`.
    pub fn corner(&self, corner: usize) -> Point3f {
        Point3f::new(
            self[corner & 1].x(),
            self[if corner & 2 != 0 { 1 } else { 0 }].y(),
            self[if corner & 4 != 0 { 1 } else { 0 }].z(),
        )
    }

    /// Construct a new bounding box that is `self` but expanded by `delta`
    /// on all axes, in both directions on the axis.
    #[inline]
    pub fn expand(self, delta: pbrt::Float) -> Self {
        Self {
            p_min: self.p_min - Vec3f::new(delta, delta, delta),
            p_max: self.p_max + Vec3f::new(delta, delta, delta),
        }
    }

    /// Checks for a ray-box intersection and returns the the two parametric `t`
    /// values of the intersection, if any, as `(lower, higher)`.
    #[inline]
    pub fn intersect_p(&self, ray: Ray) -> Option<(pbrt::Float, pbrt::Float)> {
        let (mut t0, mut t1) = (0.0, ray.t_max);
        for i in 0..3 {
            // Update interval for ith bounding box slab:
            let inv_ray_dir = 1.0 / ray.dir[i];

            let t_min_plane = (self.p_min[i] - ray.o[i]) * inv_ray_dir;
            let t_max_plane = (self.p_max[i] - ray.o[i]) * inv_ray_dir;

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
    pub fn intersect_p_with_inv_dir(&self, ray: Ray, inv_dir: Vec3f, dir_is_neg: Vec3B) -> bool {
        let bounds: Bounds3f = self.to_owned();
        // Convert to
        let dir_is_neg: Vec3Usize = dir_is_neg.into_();

        // Check for ray intersection against x and y slabs
        let tx_min = (bounds[dir_is_neg.x].x() - ray.o.x()) * inv_dir.x();
        let ty_min = (bounds[dir_is_neg.y].y() - ray.o.y()) * inv_dir.y();
        let tz_min = (bounds[dir_is_neg.x].x() - ray.o.x()) * inv_dir.x();
        let mut tx_max = (bounds[1 - dir_is_neg.x].x() - ray.o.x()) * inv_dir.x();
        let mut ty_max = (bounds[1 - dir_is_neg.y].y() - ray.o.y()) * inv_dir.y();
        let mut tz_max = (bounds[1 - dir_is_neg.x].x() - ray.o.x()) * inv_dir.x();

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

    /// Obtain the vector from the min to the max point of `self`
    /// (which is along a diagonal line across the box).
    pub fn diagonal(&self) -> Vec3f {
        self.p_max - self.p_min
    }

    /// Compute the surface area of `self`.
    pub fn surface_area(&self) -> pbrt::Float {
        let d = self.diagonal();

        (d.x() * d.y() + d.x() * d.z() + d.y() * d.z()) * 2.0
    }

    /// Compute the volume of `self`.
    pub fn volume(&self) -> pbrt::Float {
        let d = self.diagonal();
        d.x() * d.y() * d.z()
    }

    /// Determines the axis that `self` is widest on, and returns its index.
    pub fn max_extent(&self) -> usize {
        let d = self.diagonal();
        if d.x() > d.y() && d.x() > d.z() {
            0
        } else if d.y() > d.z() {
            1
        } else {
            2
        }
    }

    /// Returns `true` if `self` and `other` intersect at any point, inclusive
    /// (touching exactly on a corner counts), `false` otherwise.
    pub fn overlaps(self, other: Self) -> bool {
        let x_overlaps = self.p_max.x() >= other.p_min.x() && self.p_min.x() <= other.p_max.x();
        let y_overlaps = self.p_max.y() >= other.p_min.y() && self.p_min.y() <= other.p_max.y();
        let z_overlaps = self.p_max.z() >= other.p_min.z() && self.p_min.z() <= other.p_max.z();

        x_overlaps && y_overlaps && z_overlaps
    }

    pub fn contains(&self, p: Point3f) -> bool {
        p.x() >= self.p_min.x()
            && p.x() <= self.p_max.x()
            && p.y() >= self.p_min.y()
            && p.y() <= self.p_max.y()
            && p.z() >= self.p_min.z()
            && p.z() <= self.p_max.z()
    }

    /// Linearly interpolate between the min and max points of `self`, on all axes.
    ///
    /// Extrapolates for components of `t` `<0` or `>1`.
    pub fn lerp(self, t: Point3f) -> Point3f {
        Point3f::new(
            lerp(self.p_min.x(), self.p_max.x(), t.x()),
            lerp(self.p_min.x(), self.p_max.x(), t.y()),
            lerp(self.p_min.x(), self.p_max.x(), t.z()),
        )
    }

    /// Construct the union of `self` and `other`.
    /// Specifically, a box using the min and max points of the two.
    ///
    /// Note that this new box doesn't necessarily consist of the exact same space
    /// as the two combined.
    pub fn union(self, other: Self) -> Self {
        let p_min = Point3f::new(
            self.p_min.x().min(other.p_min.x()),
            self.p_min.y().min(other.p_min.y()),
            self.p_min.z().min(other.p_min.z()),
        );
        let p_max = Point3f::new(
            self.p_max.x().max(other.p_max.x()),
            self.p_max.y().max(other.p_max.y()),
            self.p_max.z().max(other.p_max.z()),
        );

        Self { p_min, p_max }
    }

    /// Construct the minimum bounding box that contains `self` as well as a point `p`.
    ///
    /// i.e., expand `self` by the amount needed to reach `p`
    /// (which may be none if `p` is already inside).
    pub fn union_point(self, p: Point3f) -> Self {
        let p_min = Point3f::new(
            self.p_min.x().min(p.x()),
            self.p_min.y().min(p.y()),
            self.p_min.z().min(p.z()),
        );
        let p_max = Point3f::new(
            self.p_max.x().max(p.x()),
            self.p_max.y().max(p.y()),
            self.p_max.z().max(p.z()),
        );

        Self { p_min, p_max }
    }

    /// Construct a bounding box consisting of the intersection of `self` and `other`.
    pub fn intersect(self, other: Self) -> Self {
        let p_min = Point3f::new(
            self.p_min.x().max(other.p_min.x()),
            self.p_min.y().max(other.p_min.y()),
            self.p_min.z().max(other.p_min.z()),
        );
        let p_max = Point3f::new(
            self.p_max.x().min(other.p_max.x()),
            self.p_max.y().min(other.p_max.y()),
            self.p_max.z().min(other.p_max.z()),
        );

        Self { p_min, p_max }
    }

    /// Construct an empty box.
    ///
    /// This is done by setting the extents to an invalid config,
    /// such that any operations with it would yield the expected result.
    pub fn empty() -> Self {
        let min_val = pbrt::Float::MIN;
        let max_val = pbrt::Float::MAX;
        let p_min = Point3f::new(max_val, max_val, max_val);
        let p_max = Point3f::new(min_val, min_val, min_val);

        Self { p_min, p_max }
    }

    // TODO: Does this have to for 3f only?
    /// Return the boudning sphere of `self`, as the (center, radius) of the sphere.
    pub fn bounding_sphere(&self) -> (Point3f, pbrt::Float) {
        let center = (self.p_min + self.p_max) / 2.0;
        let radius = if self.contains(center) {
            center.distance(self.p_max)
        } else {
            0.0
        };

        (center, radius)
    }
}

impl From<Bounds3i> for Bounds3f {
    fn from(v: Bounds3i) -> Self {
        Self::new(v.p_min.into(), v.p_max.into())
    }
}

impl Index<usize> for Bounds3f {
    type Output = Point3f;

    /// Index `self`'s elements by 0, 1.
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.p_min,
            1 => &self.p_max,
            _ => panic!("Index out of bounds for Bounds3"),
        }
    }
}

impl IndexMut<usize> for Bounds3f {
    /// Index `self`'s elements by 0, 1, mutably.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.p_min,
            1 => &mut self.p_max,
            _ => panic!("Index out of bounds for Bounds3"),
        }
    }
}

/// A 2D axis-aligned bounding box (AABB) of `i32`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bounds2i {
    pub p_min: Point2i,
    pub p_max: Point2i,
}

impl Bounds2i {
    /// Construct a new bounding box with two corner points.
    ///
    /// The min and max points are determined by the component-wise mins and maxes
    /// of the given points.
    pub fn new(p1: Point2i, p2: Point2i) -> Self {
        let p_min = Point2i::new(p1.x().min(p2.x()), p1.y().min(p2.y()));
        let p_max = Point2i::new(p1.x().max(p2.x()), p1.y().max(p2.y()));

        Self { p_min, p_max }
    }

    /// Construct a new bounding box that consists of a single point.
    pub fn new_with_point(p: Point2i) -> Self {
        Self { p_min: p, p_max: p }
    }

    /// Returns the coordinates of one of the four corners of `self`.
    ///
    /// 0 returns `p_min`, 3 returns `p_max`.
    pub fn corner(&self, corner: usize) -> Point2i {
        Point2i::new(
            self[corner & 1].x(),
            self[if corner & 2 != 0 { 1 } else { 0 }].y(),
        )
    }

    /// Construct a new bounding box that is `self` but expanded by `delta`
    /// on all axes, in both directions on the axis.
    #[inline]
    pub fn expand(self, delta: i32) -> Self {
        Self {
            p_min: self.p_min - Vec2i::new(delta, delta),
            p_max: self.p_max + Vec2i::new(delta, delta),
        }
    }

    /// Obtain the vector from the min to the max point of `self`
    /// (which is along a diagonal line across the box).
    pub fn diagonal(&self) -> Vec2i {
        self.p_max - self.p_min
    }

    /// Compute the area of `self`.
    pub fn area(&self) -> i32 {
        let d = self.diagonal();
        d.x() * d.y()
    }

    /// Determines the axis that `self` is widest on, and returns its index.
    pub fn max_extent(&self) -> usize {
        let d = self.diagonal();
        if d.x() > d.y() {
            0
        } else {
            1
        }
    }

    /// Returns `true` if `self` and `other` intersect at any point, inclusive
    /// (touching exactly on a corner counts), `false` otherwise.
    pub fn overlaps(self, other: Self) -> bool {
        let x_overlaps = self.p_max.x() >= other.p_min.x() && self.p_min.x() <= other.p_max.x();
        let y_overlaps = self.p_max.y() >= other.p_min.y() && self.p_min.y() <= other.p_max.y();

        x_overlaps && y_overlaps
    }

    pub fn contains(&self, p: Point2i) -> bool {
        p.x() >= self.p_min.x()
            && p.x() <= self.p_max.x()
            && p.y() >= self.p_min.y()
            && p.y() <= self.p_max.y()
    }

    /// Linearly interpolate between the min and max points of `self`, on all axes.
    ///
    /// Extrapolates for components of `t` `<0` or `>1`.
    pub fn lerp(self, t: Point2i) -> Point2f {
        let bounds: Bounds2f = self.into();
        let t: Point2f = t.into();

        Point2f::new(
            lerp(bounds.p_min.x(), bounds.p_max.x(), t.x()),
            lerp(bounds.p_min.x(), bounds.p_max.x(), t.y()),
        )
    }

    /// Construct the union of `self` and `other`.
    /// Specifically, a box using the min and max points of the two.
    ///
    /// Note that this new box doesn't necessarily consist of the exact same space
    /// as the two combined.
    pub fn union(self, other: Self) -> Self {
        let p_min = Point2i::new(
            self.p_min.x().min(other.p_min.x()),
            self.p_min.y().min(other.p_min.y()),
        );
        let p_max = Point2i::new(
            self.p_max.x().max(other.p_max.x()),
            self.p_max.y().max(other.p_max.y()),
        );

        Self { p_min, p_max }
    }

    /// Construct the minimum bounding box that contains `self` as well as a point `p`.
    ///
    /// i.e., expand `self` by the amount needed to reach `p`
    /// (which may be none if `p` is already inside).
    pub fn union_point(self, p: Point2i) -> Self {
        let p_min = Point2i::new(self.p_min.x().min(p.x()), self.p_min.y().min(p.y()));
        let p_max = Point2i::new(self.p_max.x().max(p.x()), self.p_max.y().max(p.y()));

        Self { p_min, p_max }
    }

    /// Construct a bounding box consisting of the intersection of `self` and `other`.
    pub fn intersect(self, other: Self) -> Self {
        let p_min = Point2i::new(
            self.p_min.x().max(other.p_min.x()),
            self.p_min.y().max(other.p_min.y()),
        );
        let p_max = Point2i::new(
            self.p_max.x().min(other.p_max.x()),
            self.p_max.y().min(other.p_max.y()),
        );

        Self { p_min, p_max }
    }

    /// Construct an empty box.
    ///
    /// This is done by setting the extents to an invalid config,
    /// such that any operations with it would yield the expected result.
    pub fn empty() -> Self {
        let min_val = i32::MIN;
        let max_val = i32::MAX;
        let p_min = Point2i::new(max_val, max_val);
        let p_max = Point2i::new(min_val, min_val);

        Self { p_min, p_max }
    }
}

impl Index<usize> for Bounds2i {
    type Output = Point2i;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.p_min,
            1 => &self.p_max,
            _ => panic!("Index out of bounds for Bounds3"),
        }
    }
}

impl IndexMut<usize> for Bounds2i {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.p_min,
            1 => &mut self.p_max,
            _ => panic!("Index out of bounds for Bounds3"),
        }
    }
}

/// A 2D axis-aligned bounding box (AABB) of `f32`,
/// or `f64` if feature `use-f64` is enabled.
impl IntoIterator for Bounds2i {
    type Item = Point2i;
    type IntoIter = Bounds2iIterator;

    fn into_iter(self) -> Self::IntoIter {
        Bounds2iIterator::new(self)
    }
}

pub struct Bounds2iIterator(itertools::Product<RangeInclusive<i32>, RangeInclusive<i32>>);

impl Bounds2iIterator {
    pub fn new(bounds: Bounds2i) -> Self {
        Self(iproduct!(
            bounds.p_min.x()..=bounds.p_max.x(),
            bounds.p_min.y()..=bounds.p_max.y()
        ))
    }
}

impl Iterator for Bounds2iIterator {
    type Item = Point2i;

    fn next(&mut self) -> Option<Self::Item> {
        let (x, y) = self.0.next()?;
        Some(Point2i::new(x, y))
    }
}

/// A 2D axis-aligned bounding box (AABB) of `i32`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bounds2f {
    pub p_min: Point2f,
    pub p_max: Point2f,
}

impl Bounds2f {
    /// Construct a new bounding box with two corner points.
    ///
    /// The min and max points are determined by the component-wise mins and maxes
    /// of the given points.
    pub fn new(p1: Point2f, p2: Point2f) -> Self {
        let p_min = Point2f::new(p1.x().min(p2.x()), p1.y().min(p2.y()));
        let p_max = Point2f::new(p1.x().max(p2.x()), p1.y().max(p2.y()));

        Self { p_min, p_max }
    }

    /// Construct a new bounding box that consists of a single point.
    pub fn new_with_point(p: Point2f) -> Self {
        Self { p_min: p, p_max: p }
    }

    /// Returns the coordinates of one of the four corners of `self`.
    ///
    /// 0 returns `p_min`, 3 returns `p_max`.
    pub fn corner(&self, corner: usize) -> Point2f {
        Point2f::new(
            self[corner & 1].x(),
            self[if corner & 2 != 0 { 1 } else { 0 }].y(),
        )
    }

    /// Construct a new bounding box that is `self` but expanded by `delta`
    /// on all axes, in both directions on the axis.
    #[inline]
    pub fn expand(self, delta: pbrt::Float) -> Self {
        Self {
            p_min: self.p_min - Vec2f::new(delta, delta),
            p_max: self.p_max + Vec2f::new(delta, delta),
        }
    }

    /// Obtain the vector from the min to the max point of `self`
    /// (which is along a diagonal line across the box).
    pub fn diagonal(&self) -> Vec2f {
        self.p_max - self.p_min
    }

    /// Compute the area of `self`.
    pub fn area(&self) -> pbrt::Float {
        let d = self.diagonal();
        d.x() * d.y()
    }

    /// Determines the axis that `self` is widest on, and returns its index.
    pub fn max_extent(&self) -> usize {
        let d = self.diagonal();
        if d.x() > d.y() {
            0
        } else {
            1
        }
    }

    /// Returns `true` if `self` and `other` intersect at any point, inclusive
    /// (touching exactly on a corner counts), `false` otherwise.
    pub fn overlaps(self, other: Self) -> bool {
        let x_overlaps = self.p_max.x() >= other.p_min.x() && self.p_min.x() <= other.p_max.x();
        let y_overlaps = self.p_max.y() >= other.p_min.y() && self.p_min.y() <= other.p_max.y();

        x_overlaps && y_overlaps
    }

    pub fn contains(&self, p: Point2f) -> bool {
        p.x() >= self.p_min.x()
            && p.x() <= self.p_max.x()
            && p.y() >= self.p_min.y()
            && p.y() <= self.p_max.y()
    }

    /// Linearly interpolate between the min and max points of `self`, on all axes.
    ///
    /// Extrapolates for components of `t` `<0` or `>1`.
    pub fn lerp(self, t: Point3f) -> Point2f {
        Point2f::new(
            lerp(self.p_min.x(), self.p_max.x(), t.x()),
            lerp(self.p_min.x(), self.p_max.x(), t.y()),
        )
    }

    /// Construct the union of `self` and `other`.
    /// Specifically, a box using the min and max points of the two.
    ///
    /// Note that this new box doesn't necessarily consist of the exact same space
    /// as the two combined.
    pub fn union(self, other: Self) -> Self {
        let p_min = Point2f::new(
            self.p_min.x().min(other.p_min.x()),
            self.p_min.y().min(other.p_min.y()),
        );
        let p_max = Point2f::new(
            self.p_max.x().max(other.p_max.x()),
            self.p_max.y().max(other.p_max.y()),
        );

        Self { p_min, p_max }
    }

    /// Construct the minimum bounding box that contains `self` as well as a point `p`.
    ///
    /// i.e., expand `self` by the amount needed to reach `p`
    /// (which may be none if `p` is already inside).
    pub fn union_point(self, p: Point2f) -> Self {
        let p_min = Point2f::new(self.p_min.x().min(p.x()), self.p_min.y().min(p.y()));
        let p_max = Point2f::new(self.p_max.x().max(p.x()), self.p_max.y().max(p.y()));

        Self { p_min, p_max }
    }

    /// Construct a bounding box consisting of the intersection of `self` and `other`.
    pub fn intersect(self, other: Self) -> Self {
        let p_min = Point2f::new(
            self.p_min.x().max(other.p_min.x()),
            self.p_min.y().max(other.p_min.y()),
        );
        let p_max = Point2f::new(
            self.p_max.x().min(other.p_max.x()),
            self.p_max.y().min(other.p_max.y()),
        );

        Self { p_min, p_max }
    }

    /// Construct an empty box.
    ///
    /// This is done by setting the extents to an invalid config,
    /// such that any operations with it would yield the expected result.
    pub fn empty() -> Self {
        let min_val = pbrt::Float::MIN;
        let max_val = pbrt::Float::MAX;
        let p_min = Point2f::new(max_val, max_val);
        let p_max = Point2f::new(min_val, min_val);

        Self { p_min, p_max }
    }

    /// Return the bounding sphere of `self`, as the (center, radius) of the sphere.
    pub fn bounding_sphere(&self) -> (Point2f, pbrt::Float) {
        let center = (self.p_min + self.p_max) / 2.0;
        let radius = if self.contains(center) {
            center.distance(self.p_max)
        } else {
            0.0
        };

        (center, radius)
    }
}

impl From<Bounds2i> for Bounds2f {
    fn from(v: Bounds2i) -> Self {
        Self::new(v.p_min.into(), v.p_max.into())
    }
}

impl Index<usize> for Bounds2f {
    type Output = Point2f;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.p_min,
            1 => &self.p_max,
            _ => panic!("Index out of bounds for Bounds2"),
        }
    }
}

impl IndexMut<usize> for Bounds2f {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.p_min,
            1 => &mut self.p_max,
            _ => panic!("Index out of bounds for Bounds2"),
        }
    }
}
