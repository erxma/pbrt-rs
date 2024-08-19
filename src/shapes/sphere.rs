use std::mem;

use crate::{
    float::PI,
    geometry::{
        Bounds3f, DirectionCone, Frame, Ray, SampleInteraction, SurfaceInteraction, Transform,
    },
    math::{
        difference_of_products, gamma, safe_acos, safe_sqrt, spherical_direction, Interval,
        Normal3f, Point2f, Point3f, Point3fi, Tuple, Vec3f, Vec3fi,
    },
    util::sampling::sample_uniform_sphere,
    Float,
};

use super::{QuadricIntersection, Shape, ShapeIntersection, ShapeSample, ShapeSampleContext};

#[derive(Clone, Debug)]
pub struct Sphere {
    radius: Float,
    z_min: Float,
    z_max: Float,
    theta_z_min: Float,
    theta_z_max: Float,
    phi_max: Float,
    render_from_object: Transform,
    object_from_render: Transform,
    reverse_orientation: bool,
    transform_swaps_handedness: bool,
}

impl Shape for Sphere {
    fn bounds(&self) -> Bounds3f {
        &self.render_from_object
            * Bounds3f::new(
                Point3f::new(-self.radius, -self.radius, self.z_min),
                Point3f::new(self.radius, self.radius, self.z_max),
            )
    }

    fn normal_bounds(&self) -> DirectionCone {
        DirectionCone::ENTIRE_SPHERE
    }

    fn intersect(&self, ray: &Ray, t_max: Option<Float>) -> Option<ShapeIntersection> {
        let t_max = t_max.unwrap_or(Float::INFINITY);

        let isect = self.basic_intersect(ray, t_max);

        isect.map(|isect| {
            let intr = self.interaction_from_intersection(&isect, -ray.dir, ray.time);
            ShapeIntersection {
                intr,
                t_hit: isect.t_hit,
            }
        })
    }

    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool {
        let t_max = t_max.unwrap_or(Float::INFINITY);
        self.basic_intersect(ray, t_max).is_some()
    }

    fn area(&self) -> Float {
        self.phi_max * self.radius * (self.z_max - self.z_min)
    }

    fn sample(&self, u: Point2f) -> Option<ShapeSample> {
        // Sample from uniform sphere, then scale by self's radius
        let mut p_obj = Point3f::ZERO + self.radius * sample_uniform_sphere(u);

        // Reproject p_obj to sphere surface and compute error
        p_obj *= self.radius / p_obj.distance(Point3f::ZERO);
        let p_obj_err = gamma(5) * Vec3f::from(p_obj).abs();

        // Compute surface normal for sample and return ShapeSample
        let n_obj = Normal3f::from(p_obj);
        let mut n = (&self.render_from_object * n_obj).normalized();
        if self.reverse_orientation {
            n *= -1.0;
        }
        // Compute uv coords for sphere sample
        let theta = safe_acos(p_obj.z() / self.radius);
        let mut phi = p_obj.y().atan2(p_obj.x());
        if phi < 0.0 {
            phi += 2.0 * PI;
        }
        let uv = Point2f::new(
            phi / self.phi_max,
            (theta - self.theta_z_min) / (self.theta_z_max - self.theta_z_min),
        );

        let pi = &self.render_from_object * Point3fi::new_fi(p_obj, p_obj_err);
        let intr = SampleInteraction::new(pi, None, n, uv);
        let pdf = self.pdf(&intr);
        Some(ShapeSample { intr, pdf })
    }

    fn sample_with_context(&self, ctx: &ShapeSampleContext, u: Point2f) -> Option<ShapeSample> {
        let p_center = &self.render_from_object * Point3f::ZERO;
        let p_origin = ctx.offset_ray_origin(p_center.into());
        let ctx_p = ctx.pi.midpoints_only();
        // If p is inside sphere, sample uniformly
        if p_origin.distance_squared(p_center) <= self.radius * self.radius {
            // Sample shape by area, compute incident dir wi
            let mut shape_sample = self.sample(u).unwrap();
            let sample_intr = &mut shape_sample.intr;
            sample_intr.time = ctx.time;
            let mut wi = sample_intr.pi.midpoints_only() - ctx_p;
            if wi.length_squared() == 0.0 {
                return None;
            }
            wi = wi.normalized();
            // Compute area sampling PDF is ss to solid angle measure
            shape_sample.pdf /= Vec3f::from(sample_intr.n).absdot(-wi)
                / ctx_p.distance_squared(sample_intr.pi.midpoints_only());
            if shape_sample.pdf.is_infinite() {
                return None;
            }

            Some(shape_sample)
        } else {
            // Point outside of sphere, sample uniformly within the subtended cone.

            // Compute quantities related to cone theta_max
            let sin_theta_max = self.radius / ctx_p.distance(p_center);
            let sin2_theta_max = sin_theta_max * sin_theta_max;
            let cos_theta_max = safe_sqrt(1.0 - sin2_theta_max);
            let mut one_minus_cos_theta_max = 1.0 - cos_theta_max;

            // Compute theta, phi for sample in cone
            let mut cos_theta = (cos_theta_max - 1.0) * u.x() + 1.0;
            let mut sin2_theta = 1.0 - cos_theta * cos_theta;
            // For small angles, compute cone sample via Taylor series expansion
            if sin2_theta_max < 0.00068523
            /* < sin^2(1.5deg) */
            {
                sin2_theta = sin2_theta_max * u.x();
                cos_theta = (1.0 - sin2_theta).sqrt();
                one_minus_cos_theta_max = sin2_theta_max / 2.0;
            }

            // Compute angle alpha from center of sphere to sample point on surface
            let cos_alpha = sin2_theta / sin_theta_max
                + cos_theta * safe_sqrt(1.0 - sin2_theta / sin_theta_max * sin2_theta_max);
            let sin_alpha = safe_sqrt(1.0 - cos_alpha * cos_alpha);

            // Compute surface normal and sample point on sphere
            let phi = u.y() * 2.0 * PI;
            let w = spherical_direction(sin_alpha, cos_alpha, phi);
            let sampling_frame = Frame::from_z((p_center - ctx_p).normalized());
            let mut n = Normal3f::from(sampling_frame.from_local(-w));
            let p = p_center + self.radius * Vec3f::from(n);
            if self.reverse_orientation {
                n *= -1.0;
            }

            // Compute error bounds for sample point
            let p_err = gamma(5) * Vec3f::from(p).abs();
            // Compute uv coords for sample point on sphere
            let theta = safe_acos(p.z() / self.radius);
            let mut sphere_phi = p.y().atan2(p.x());
            if sphere_phi < 0.0 {
                sphere_phi += 2.0 * PI;
            }
            let uv = Point2f::new(
                sphere_phi / self.phi_max,
                (theta - self.theta_z_min) / (self.theta_z_max - self.theta_z_min),
            );

            // Return sample info
            let intr = SampleInteraction::new(Point3fi::new_fi(p, p_err), Some(ctx.time), n, uv);
            let pdf = 1.0 / (2.0 * PI * one_minus_cos_theta_max);

            Some(ShapeSample { intr, pdf })
        }
    }

    fn pdf(&self, _interaction: &SampleInteraction) -> Float {
        1.0 / self.area()
    }

    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vec3f) -> Float {
        let p_center = &self.render_from_object * Point3f::ZERO;
        let p_origin = ctx.offset_ray_origin(p_center.into());
        let ctx_p = ctx.pi.midpoints_only();
        // Similarly to sample_with_context...if p is inside sphere, sample uniformly
        if p_origin.distance_squared(p_center) <= self.radius * self.radius {
            // Return solid angle PDF for point inside sphere:

            // Intersect sample ray with shape geometry
            let ray = ctx.spawn_ray(wi);
            let isect = self.intersect(&ray, None);
            match isect {
                Some(isect) => {
                    // Compute PDF in solid angle measure from intersection point
                    let pdf = (1.0 / self.area())
                        / (isect.intr.n.absdot_v(-wi)
                            / ctx_p.distance_squared(isect.intr.pi.midpoints_only()));

                    if pdf.is_finite() {
                        pdf
                    } else {
                        0.0
                    }
                }
                None => 0.0,
            }
        } else {
            // Outside of sphere.
            // Compute general solid angle sphere PDF
            let sin2_theta_max = self.radius * self.radius / ctx_p.distance_squared(p_center);
            let cos_theta_max = safe_sqrt(1.0 - sin2_theta_max);
            let mut one_minus_cos_theta_max = 1.0 - cos_theta_max;
            // For small angles, compute more accurate 1-cos via Taylor series expansion
            if sin2_theta_max < 0.00068523
            /* < sin^2(1.5deg) */
            {
                one_minus_cos_theta_max = sin2_theta_max / 2.0;
            }

            1.0 / (2.0 * PI * one_minus_cos_theta_max)
        }
    }
}

impl Sphere {
    /// Perform a basic ray-sphere intersection test, and return basic info about the point if found.
    pub fn basic_intersect(&self, ray: &Ray, t_max: Float) -> Option<QuadricIntersection> {
        // Transform ray's origin, direction to object space
        let o_obj = &self.object_from_render * Point3fi::from(ray.o);
        let dir_obj = &self.object_from_render * Vec3fi::from(ray.dir);

        // Solve quadratic equation to compute sphere t0 and t1:
        // Compute sphere quadratic coefficients
        // a = d.x^2 + d.y^2 + d.z^2
        let a = dir_obj.x().squared() + dir_obj.y().squared() + dir_obj.z().squared();
        // b = 2(d.x*o.x + d.y*o.y + )
        let b = 2.0 * (dir_obj.x() * o_obj.x() + dir_obj.y() * o_obj.y() + dir_obj.z() * o_obj.z());
        // c = o.x^2 + o.y^2 + o.z^2 - r^2
        let c = o_obj.x().squared() + o_obj.y().squared() + o_obj.z().squared()
            - Interval::new_exact(self.radius).squared();
        // Compute sphere quadratic discriminant, discrim
        let v = Vec3fi::from(o_obj - b / (2.0 * a) * dir_obj);
        let len = v.length();
        let discrim =
            4.0 * a * (Interval::new_exact(self.radius) + len) * Interval::new_exact(self.radius)
                - len;
        if discrim.lower_bound() < 0.0 {
            return None;
        }
        // Compute quadratic t values with variant of quadratic equation
        let discrim_sqrt = discrim.sqrt();
        let q = if b.midpoint() < 0.0 {
            -0.5 * (b - discrim_sqrt)
        } else {
            -0.5 * (b + discrim_sqrt)
        };
        let mut t0 = q / a;
        let mut t1 = c / q;
        // Swap t0, t1 so that t0's lower bound is lesser
        // Because they are intervals, which is lesser is ambiguous,
        // lower is used to avoid returning a hit that is potentially
        // farther away than an actual closer hit.
        if t0.lower_bound() > t1.lower_bound() {
            mem::swap(&mut t0, &mut t1);
        }

        // Check quadric shape t0 and t1 for nearest intersection
        // Also ambiguous; err on side of return no intersection rather than invalid one
        if t0.upper_bound() > t_max || t1.lower_bound() <= 0.0 {
            return None;
        }
        let mut t_shape_hit;
        if t0.lower_bound() > 0.0 {
            t_shape_hit = t0;
        } else if t1.upper_bound() <= t_max {
            t_shape_hit = t1;
        } else {
            return None;
        }

        let compute_point_and_phi = |t_shape_hit: Interval| {
            // Compute sphere hit position and phi
            let mut p_hit =
                o_obj.midpoints_only() + t_shape_hit.midpoint() * dir_obj.midpoints_only();
            // Refine sphere intersection point
            p_hit *= self.radius / p_hit.distance(Point3f::ZERO);
            if p_hit.x() == 0.0 && p_hit.y() == 0.0 {
                *p_hit.x_mut() = 1e-5 * self.radius;
            }
            let mut phi = p_hit.y().atan2(p_hit.x());
            if phi <= 0.0 {
                phi += 2.0 * PI;
            }
            (p_hit, phi)
        };

        let (mut p_hit, mut phi) = compute_point_and_phi(t_shape_hit);

        // Test sphere intersection against clipping params
        // Skip z test if the range includes the entire sphere (the computed p_hit.z may be slightly out of range due to floating point)
        if (self.z_min > -self.radius && p_hit.z() < self.z_min)
            || (self.z_max < self.radius && p_hit.z() > self.z_max)
            || phi > self.phi_max
        {
            // Intersection invalid
            // Try again with t1, if possible
            if t_shape_hit == t1 || t1.upper_bound() > t_max {
                return None;
            }

            t_shape_hit = t1;
            (p_hit, phi) = compute_point_and_phi(t_shape_hit);

            if (self.z_min > -self.radius && p_hit.z() < self.z_min)
                || (self.z_max < self.radius && p_hit.z() > self.z_max)
                || phi > self.phi_max
            {
                // Also invalid
                return None;
            }
        }

        // Ray hit the sphere, return info
        Some(QuadricIntersection {
            // Computed in object space, but also correct t in render space.
            t_hit: t_shape_hit.midpoint(),
            p_obj: p_hit,
            phi,
        })
    }

    /// Upgrade QuadricIntersection to a full SurfaceInteraction.
    pub fn interaction_from_intersection(
        &self,
        isect: &QuadricIntersection,
        wo: Vec3f,
        time: Float,
    ) -> SurfaceInteraction {
        #![allow(non_snake_case)]
        let p_hit = isect.p_obj;
        let phi = isect.phi;

        // Find parametric representation of sphere hit
        let u = phi / self.phi_max;
        let cos_theta = p_hit.z() / self.radius;
        let theta = safe_acos(cos_theta);
        let v = (theta - self.theta_z_min) / (self.theta_z_max - self.theta_z_min);

        // Compute error bounds for sphere intersection
        let p_error = gamma(5) * Vec3f::from(p_hit).abs();

        // Compute sphere dp/du and dp/dv
        let z_radius = (p_hit.x().powi(2) + p_hit.y().powi(2)).sqrt();
        let cos_phi = p_hit.x() / z_radius;
        let sin_phi = p_hit.y() / z_radius;
        let dpdu = Vec3f::new(-self.phi_max * p_hit.y(), self.phi_max * p_hit.x(), 0.0);
        let sin_theta = safe_sqrt(1.0 - cos_theta.powi(2));
        let dpdv = (self.theta_z_max - self.theta_z_min)
            * Vec3f::new(
                p_hit.z() * cos_phi,
                p_hit.z() * sin_phi,
                -self.radius * sin_theta,
            );

        // Compute sphere dn/du and dn/dv
        let d2p_duu = -self.phi_max.powi(2) * Vec3f::new(p_hit.x(), p_hit.y(), 0.0);
        let d2p_duv = (self.theta_z_max - self.theta_z_min)
            * p_hit.z()
            * self.phi_max
            * Vec3f::new(-sin_phi, cos_phi, 0.0);
        let d2p_dvv = -(self.theta_z_max - self.theta_z_min).powi(2) * Vec3f::from(p_hit);
        // Compute coefficients for fundamental forms
        let E = dpdu.dot(dpdu);
        let F = dpdu.dot(dpdv);
        let G = dpdv.dot(dpdv);
        let n = dpdu.cross(dpdv).normalized();
        let e = n.dot(d2p_duu);
        let f = n.dot(d2p_duv);
        let g = n.dot(d2p_dvv);
        // Compute dn/du and dn/dv from fundamental form
        let EGF2 = difference_of_products(E, G, F, F);
        let inv_EGF2 = if EGF2 == 0.0 { 0.0 } else { 1.0 / EGF2 };
        let dndu =
            Normal3f::from((f * F - e * G) * inv_EGF2 * dpdu + (e * F - f * E) * inv_EGF2 * dpdv);
        let dndv =
            Normal3f::from((g * F - f * G) * inv_EGF2 * dpdu + (f * F - g * E) * inv_EGF2 * dpdv);

        let flip_normal = self.reverse_orientation ^ self.transform_swaps_handedness;
        let wo_object = &self.object_from_render * wo;
        SurfaceInteraction::builder()
            .pi(Point3fi::new_fi(p_hit, p_error))
            .uv(Point2f::new(u, v))
            .wo(wo_object)
            .dpdu(dpdu)
            .dpdv(dpdv)
            .dndu(dndu)
            .dndv(dndv)
            .time(time)
            .flip_normal(flip_normal)
            .build()
            .unwrap()
    }
}
