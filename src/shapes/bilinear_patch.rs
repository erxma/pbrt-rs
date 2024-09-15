use std::sync::{Arc, OnceLock};

use itertools::iproduct;

use crate::{
    core::{
        difference_of_products, gamma, lerp, solve_quadratic, spherical_quad_area, Bounds3f,
        DirectionCone, Float, Normal3f, Point2f, Point3f, Point3fi, Ray, SampleInteraction,
        SquareMatrix, SurfaceInteraction, SurfaceInteractionParams, Transform, Tuple, Vec3f,
    },
    memory::{
        NORMAL3F_BUFFER_CACHE, POINT2F_BUFFER_CACHE, POINT3F_BUFFER_CACHE, USIZE_BUFFER_CACHE,
    },
    sampling::routines::{
        bilinear_pdf, invert_bilinear, invert_spherical_rectangle_sample, sample_bilinear,
        sample_spherical_rectangle, PiecewiseConstant2D,
    },
};

use super::{Shape, ShapeIntersection, ShapeSample, ShapeSampleContext};

#[derive(Clone, Debug)]
pub struct BilinearPatch {
    mesh_idx: usize,
    blp_idx: usize,
    area: Float,
}

impl BilinearPatch {
    const MIN_SPHERICAL_SAMPLE_AREA: Float = 1e-4;

    pub fn new(mesh: &BilinearPatchMesh, mesh_idx: usize, blp_idx: usize) -> Self {
        // Determine area of bilinear patch

        // Get bilinear patch vertices
        let (p00, p10, p01, p11) = mesh.positions(blp_idx);

        let area;
        if mesh.patch_is_rectangle(blp_idx) {
            area = p00.distance(p01) * p00.distance(p10);
        } else {
            // Compute approx area of patch using Riemann sum evaled at NA x NA points
            const NA: usize = 3;

            let mut p = [[Point3f::ZERO; NA + 1]; NA + 1];
            for (i, j) in iproduct!(0..=NA, 0..=NA) {
                let u = i as Float / NA as Float;
                let v = j as Float / NA as Float;
                p[i][j] = lerp(lerp(p00, p01, v), lerp(p10, p11, v), u);
            }

            let mut riemann_sum = 0.0;
            for (i, j) in iproduct!(0..NA, 0..NA) {
                riemann_sum += 0.5
                    * (p[i + 1][j + 1] - p[i][j])
                        .cross(p[i + 1][j] - p[i][j + 1])
                        .length();
            }
            area = riemann_sum;
        }

        Self {
            mesh_idx,
            blp_idx,
            area,
        }
    }

    fn mesh(&self) -> &BilinearPatchMesh {
        BilinearPatchMesh::get(self.mesh_idx).unwrap()
    }

    pub fn mesh_positions(&self) -> (Point3f, Point3f, Point3f, Point3f) {
        self.mesh().positions(self.blp_idx)
    }

    pub fn mesh_vertex_normals(&self) -> Option<(Normal3f, Normal3f, Normal3f, Normal3f)> {
        self.mesh().vertex_normals(self.blp_idx)
    }

    pub fn mesh_uvs(&self) -> Option<(Point2f, Point2f, Point2f, Point2f)> {
        self.mesh().uvs(self.blp_idx)
    }

    pub fn interaction_from_intersection(
        mesh: &BilinearPatchMesh,
        blp_idx: usize,
        uv: Point2f,
        time: Float,
        outgoing: Vec3f,
    ) -> SurfaceInteraction {
        #![allow(non_snake_case)]

        // Get bilinear patch vertices
        let (p00, p10, p01, p11) = mesh.positions(blp_idx);

        // Compute patch point p, dp/du, dp/dv for (u, v)
        let p = lerp(lerp(p00, p01, uv[1]), lerp(p10, p11, uv[1]), uv[0]);
        let dpdu: Vec3f = lerp(p10, p11, uv[1]) - lerp(p00, p01, uv[1]);
        let dpdv: Vec3f = lerp(p01, p11, uv[0]) - lerp(p00, p10, uv[0]);

        // TODO: Skipping this part for now
        // Compute (s, t) texcoords at patch (u, v)
        /*
        let st = uv;
        let duds = 1.0;
        let dudt = 0.0;
        let dvds = 0.0;
        let dvdt = 1.0;
        // if Some mesh.uv...
        */

        // Find partial derivatives dndu and dndv for patch
        let d2p_duu = Vec3f::ZERO;
        let d2p_dvv = Vec3f::ZERO;
        let d2p_duv = (p00 - p01) + (p11 - p10);
        // Compute coeffs for fundamental forms
        let E = dpdu.dot(dpdu);
        let F = dpdu.dot(dpdv);
        let G = dpdv.dot(dpdv);
        let n = dpdu.cross(dpdu).normalized();
        let e = n.dot(d2p_duu);
        let f = n.dot(d2p_duv);
        let g = n.dot(d2p_dvv);
        // Compute dn/du and dn/dv from coeffs
        let EGF2 = difference_of_products(E, G, F, F);
        let inv_EGF2 = if EGF2 != 0.0 { 1.0 / EGF2 } else { 0.0 };
        let dndu: Normal3f =
            ((f * F - e * G) * inv_EGF2 * dpdu + (e * F - f * E) * inv_EGF2 * dpdv).into();
        let dndv: Normal3f =
            ((g * F - f * G) * inv_EGF2 * dpdu + (f * F - g * E) * inv_EGF2 * dpdv).into();
        // TODO: Update dn/du and dn/dv to account for (s, t) parameterization

        // Initialize intersection point error
        let p_abs_sum = p00.abs() + p10.abs() + p01.abs() + p11.abs();
        let p_err = gamma(6) * Vec3f::from(p_abs_sum);

        let flip_normal = mesh.reverse_orientation ^ mesh.transform_swaps_handedness;
        let mut isect = SurfaceInteraction::new(SurfaceInteractionParams {
            pi: Point3fi::new_fi(p, p_err),
            wo: outgoing,
            uv,
            dpdu,
            dpdv,
            dndu,
            dndv,
            time,
            flip_normal,
        });

        // Compute patch shading normal if necessary
        if let Some((n00, n10, n01, n11)) = mesh.vertex_normals(blp_idx) {
            let mut ns = lerp(lerp(n00, n01, uv[1]), lerp(n10, n11, uv[1]), uv[0]);
            if ns.length_squared() > 0.0 {
                ns = ns.normalized();
                // Set shading geometry for patch intersection
                let dndu = lerp(n10, n11, uv[1]) - lerp(n00, n01, uv[1]);
                let dndv = lerp(n01, n11, uv[0]) - lerp(n00, n10, uv[0]);
                // TODO: Update dn/du and dn/dv to account for (s, t) parameterization
                let rotation = Transform::rotate_from_to(isect.n.normalized().into(), ns.into());
                isect.set_shading_geometry(ns, &rotation * dpdu, rotation * dpdv, dndu, dndv, true);
            }
        }

        isect
    }

    /// Returns `true` if `self`'s vertices form a rectangle.
    fn is_rectangle(&self) -> bool {
        self.mesh().patch_is_rectangle(self.blp_idx)
    }
}

impl Shape for BilinearPatch {
    fn bounds(&self) -> Bounds3f {
        // Get patch vertices
        let (p00, p10, p01, p11) = self.mesh_positions();
        // Bounds is bounding box of the four corners
        Bounds3f::new(p00, p01).union(Bounds3f::new(p10, p11))
    }

    fn normal_bounds(&self) -> DirectionCone {
        // Get patch vertices
        let (p00, p10, p01, p11) = self.mesh_positions();
        // If patch is a triangle, return bounds for single surface normal
        if p00 == p10 || p10 == p11 || p01 == p11 || p00 == p11 {
            let dpdu = lerp(p10, p11, 0.5) - lerp(p00, p01, 0.5);
            let dpdv = lerp(p01, p11, 0.5) - lerp(p00, p10, 0.5);
            let mut n: Normal3f = dpdu.cross(dpdv).normalized().into();

            if let Some((vn00, vn10, vn01, vn11)) = self.mesh_vertex_normals() {
                let ns = (vn00 + vn01 + vn10 + vn11) / 4.0;
                n = n.face_forward(ns.into());
            } else if self.mesh().reverse_orientation ^ self.mesh().transform_swaps_handedness {
                n = -n;
            }
            return DirectionCone::from_dir(n.into());
        }

        // Compute patch normal at (0, 0)
        let mut n00: Normal3f = (p10 - p00).cross(p01 - p00).normalized().into();
        let mut n10: Normal3f = (p11 - p10).cross(p00 - p10).normalized().into();
        let mut n01: Normal3f = (p00 - p01).cross(p11 - p01).normalized().into();
        let mut n11: Normal3f = (p01 - p11).cross(p10 - p11).normalized().into();
        if let Some((vn00, vn10, vn01, vn11)) = self.mesh_vertex_normals() {
            n00 = n00.face_forward(vn00.into());
            n10 = n10.face_forward(vn10.into());
            n01 = n01.face_forward(vn01.into());
            n11 = n11.face_forward(vn11.into());
        } else if self.mesh().reverse_orientation ^ self.mesh().transform_swaps_handedness {
            n00 = -n00;
            n10 = -n10;
            n01 = -n01;
            n11 = -n11;
        }

        // Compute average normal and return bounds
        let n = (n00 + n10 + n01 + n11).normalized();
        // Cos of max angle any normal makes with average
        let cos_theta = [n.dot(n00), n.dot(n01), n.dot(n10), n.dot(n11)]
            .into_iter()
            .reduce(Float::min)
            .unwrap();

        DirectionCone::new(n.into(), cos_theta.clamp(-1.0, 1.0))
    }

    fn intersect(&self, ray: &Ray, t_max: Option<Float>) -> Option<ShapeIntersection> {
        let t_max = t_max.unwrap_or(Float::INFINITY);
        // Get positions
        let (p00, p10, p01, p11) = self.mesh_positions();

        let blp_isect = intersect_bilinear_patch(ray, t_max, p00, p10, p01, p11)?;

        let intr = Self::interaction_from_intersection(
            self.mesh(),
            self.blp_idx,
            blp_isect.uv,
            ray.time,
            -ray.dir,
        );
        Some(ShapeIntersection {
            intr,
            t_hit: blp_isect.t,
        })
    }

    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool {
        let t_max = t_max.unwrap_or(Float::INFINITY);

        let (p00, p10, p01, p11) = self.mesh_positions();
        intersect_bilinear_patch(ray, t_max, p00, p10, p01, p11).is_some()
    }

    fn area(&self) -> Float {
        self.area
    }

    fn sample(&self, u: Point2f) -> Option<ShapeSample> {
        // Get positions
        let (p00, p10, p01, p11) = self.mesh_positions();

        let (uv, mut pdf) = if let Some(distrib) = &self.mesh().image_distribution {
            todo!()
        } else if !self.is_rectangle() {
            // Sample patch (u, v) with approx uniform area sampling

            // Init w array with differential area at patch corners
            let w = [
                (p10 - p00).cross(p01 - p00).length(),
                (p10 - p00).cross(p11 - p10).length(),
                (p01 - p00).cross(p11 - p01).length(),
                (p11 - p10).cross(p11 - p01).length(),
            ];

            let uv = sample_bilinear(u, &w);
            let pdf = bilinear_pdf(uv, &w);
            (uv, pdf)
        } else {
            (u, 1.0)
        };

        // Compute patch geometric quantities at sampled (u, v)

        // Compute p, dp/du, dp/dv
        let pu0 = lerp(p00, p01, uv[1]);
        let pu1 = lerp(p10, p11, uv[1]);
        let p = lerp(pu0, pu1, uv[0]);
        let dpdu = pu1 - pu0;
        let dpdv = lerp(p01, p11, uv[0]) - lerp(p00, p10, uv[0]);
        if dpdu.length_squared() == 0.0 || dpdv.length_squared() == 0.0 {
            return None;
        }

        let st = uv;
        if let Some(uv) = self.mesh_uvs() {
            todo!()
        }

        // Compute surface normal for sampled (u, v)
        let mut n: Normal3f = dpdu.cross(dpdv).normalized().into();
        // Flip normal if necessary
        if let Some((n00, n10, n01, n11)) = self.mesh_vertex_normals() {
            let ns = lerp(lerp(n00, n01, uv[1]), lerp(n10, n11, uv[1]), uv[0]);
            n = n.face_forward(ns.into());
        } else if self.mesh().reverse_orientation ^ self.mesh().transform_swaps_handedness {
            n = -n;
        }

        // Compute error for (u, v)
        let p_abs_sum = p00.abs() + p01.abs() + p10.abs() + p11.abs();
        let p_err = gamma(6) * Vec3f::from(p_abs_sum);

        // Return sample
        let intr = SampleInteraction::new(Point3fi::new_fi(p, p_err), None, n, st);
        pdf /= dpdu.cross(dpdv).length();
        Some(ShapeSample { intr, pdf })
    }

    fn sample_with_context(&self, ctx: &ShapeSampleContext, mut u: Point2f) -> Option<ShapeSample> {
        // Get positions
        let (p00, p10, p01, p11) = self.mesh_positions();

        // Sample patch with respect to solid angle from reference point
        let v00 = (p00 - ctx.pi.midpoints()).normalized();
        let v10 = (p10 - ctx.pi.midpoints()).normalized();
        let v01 = (p01 - ctx.pi.midpoints()).normalized();
        let v11 = (p11 - ctx.pi.midpoints()).normalized();
        if !self.is_rectangle()
            || self.mesh().image_distribution.is_some()
            || spherical_quad_area(v00, v10, v11, v01) <= Self::MIN_SPHERICAL_SAMPLE_AREA
        {
            let mut ss = self.sample(u)?;
            ss.intr.time = ctx.time;
            let mut wi = ss.intr.pi.midpoints() - ctx.pi.midpoints();
            if wi.length_squared() == 0.0 {
                return None;
            }
            wi = wi.normalized();

            // Convert area sampling PDF in ss to solid angle measure
            ss.pdf /= ss.intr.n.absdot(Normal3f::from(-wi))
                / ctx.pi.midpoints().distance_squared(ss.intr.pi.midpoints());
            if ss.pdf.is_infinite() {
                return None;
            }

            Some(ss)
        } else {
            // Sample dir to rectangular patch
            let mut pdf = 1.0;

            // Warp uniform sample u to account for incident cos theta factor
            if let Some(ns) = ctx.ns {
                let ns = ns.into();
                // Compute cos theta weights for rectangle seen from reference point
                let w = [
                    v00.absdot(ns).max(0.01),
                    v10.absdot(ns).max(0.01),
                    v01.absdot(ns).max(0.01),
                    v11.absdot(ns).max(0.01),
                ];

                u = sample_bilinear(u, &w);
                pdf *= bilinear_pdf(u, &w);
            }

            // Sample spherical rectangle at reference point
            let eu = p10 - p00;
            let ev = p01 - p00;
            let (p, quad_pdf) = sample_spherical_rectangle(ctx.pi.midpoints(), p00, eu, ev, u);
            pdf *= quad_pdf;

            // Compute (u, v) and surface normal for sampled point
            let uv = Point2f::new(
                (p - p00).dot(eu) / p10.distance_squared(p00),
                (p - p00).dot(ev) / p01.distance_squared(p00),
            );
            let mut n: Normal3f = eu.cross(ev).normalized().into();
            // Flip normal if necessary
            if let Some((n00, n10, n01, n11)) = self.mesh_vertex_normals() {
                let ns = lerp(lerp(n00, n01, uv[1]), lerp(n10, n11, uv[1]), uv[0]);
                n = n.face_forward(ns.into());
            } else if self.mesh().reverse_orientation ^ self.mesh().transform_swaps_handedness {
                n = -n;
            }

            // Compute st texcoords for (u, v)
            let st = uv;
            if let Some(uv) = self.mesh_uvs() {
                todo!();
            }

            let intr = SampleInteraction::new(p.into(), Some(ctx.time), n, st);
            Some(ShapeSample { intr, pdf })
        }
    }

    fn pdf(&self, interaction: &SampleInteraction) -> Float {
        // Get positons
        let (p00, p10, p01, p11) = self.mesh_positions();

        // Compute parametric (u, v) of point on patch
        let mut uv = interaction.uv;
        if let Some((uv00, uv10, uv01, uv11)) = self.mesh_uvs() {
            uv = invert_bilinear(uv, &[uv00, uv10, uv01, uv11]);
        }

        // Compute PDF for sampling (u, v)
        let pdf = if let Some(distrib) = &self.mesh().image_distribution {
            todo!()
        } else if !self.is_rectangle() {
            // Init w array with differential area at patch corners
            let w = [
                (p10 - p00).cross(p01 - p00).length(),
                (p10 - p00).cross(p11 - p10).length(),
                (p01 - p00).cross(p11 - p01).length(),
                (p11 - p10).cross(p11 - p01).length(),
            ];

            bilinear_pdf(uv, &w)
        } else {
            1.0
        };

        // FInd dp/du and dp/dv at patch (u, v)
        let pu0 = lerp(p00, p01, uv[1]);
        let pu1 = lerp(p10, p11, uv[1]);
        let dpdu = pu1 - pu0;
        let dpdv = lerp(p01, p11, uv[0]) - lerp(p00, p10, uv[0]);

        // Scale to get PDF with respect to surface area
        pdf / dpdu.cross(dpdv).length()
    }

    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vec3f) -> Float {
        // Get positons
        let (p00, p10, p01, p11) = self.mesh_positions();

        // Compute solid angle PDF for patch from reference point
        // Intersect sample ray with shape geometry
        let ray = ctx.spawn_ray_with_dir(wi);
        let isect = match self.intersect(&ray, None) {
            Some(isect) => isect,
            None => {
                return 0.0;
            }
        };

        let v00 = (p00 - ctx.pi.midpoints()).normalized();
        let v10 = (p10 - ctx.pi.midpoints()).normalized();
        let v01 = (p01 - ctx.pi.midpoints()).normalized();
        let v11 = (p11 - ctx.pi.midpoints()).normalized();
        if !self.is_rectangle()
            || self.mesh().image_distribution.is_some()
            || spherical_quad_area(v00, v10, v11, v01) <= Self::MIN_SPHERICAL_SAMPLE_AREA
        {
            let intr = SampleInteraction::new(
                isect.intr.pi,
                Some(isect.intr.time),
                isect.intr.n,
                isect.intr.uv,
            );
            // Return solid angle PDF for area-sampled patch
            let pdf = self.pdf(&intr)
                * (ctx
                    .pi
                    .midpoints()
                    .distance_squared(isect.intr.pi.midpoints())
                    / isect.intr.n.absdot(Normal3f::from(wi)));

            if pdf.is_finite() {
                pdf
            } else {
                0.0
            }
        } else {
            // Return PDF for sample in spherical rectangle
            let pdf = 1.0 / spherical_quad_area(v00, v10, v11, v01);
            if let Some(ns) = ctx.ns {
                let ns = ns.into();
                // Compute cos theta weights for rectangle seen from ref point
                let w = [
                    v00.absdot(ns).max(0.01),
                    v10.absdot(ns).max(0.01),
                    v01.absdot(ns).max(0.01),
                    v11.absdot(ns).max(0.01),
                ];

                let u = invert_spherical_rectangle_sample(
                    ctx.pi.midpoints(),
                    p00,
                    p10 - p00,
                    p01 - p00,
                    isect.intr.pi.midpoints(),
                );
                pdf * bilinear_pdf(u, &w)
            } else {
                pdf
            }
        }
    }
}

pub fn intersect_bilinear_patch(
    ray: &Ray,
    t_max: Float,
    p00: Point3f,
    p10: Point3f,
    p01: Point3f,
    p11: Point3f,
) -> Option<BilinearIntersection> {
    // Find quadratic coeffs for distance from ray to u iso-lines
    let a = (p10 - p00).cross(p01 - p11).dot(ray.dir);
    let c = (p00 - ray.o).cross(ray.dir).dot(p01 - p00);
    let b = (p10 - ray.o).cross(ray.dir).dot(p11 - p10) - (a + c);

    // Solve quadratic for patch u intersection
    let (u1, u2) = solve_quadratic(a, b, c)?;

    // Find epsilon to ensure that candidate is greater than zero
    let epsilon = gamma(10)
        * (ray.o.abs().max_component()
            + ray.dir.abs().max_component()
            + p00.abs().max_component()
            + p10.abs().max_component()
            + p01.abs().max_component()
            + p11.abs().max_component());

    // Compute v and t for first u intersection
    // Only a valid intersection if between 0 and 1
    let mut t = t_max;
    let mut u = None;
    let mut v = None;
    if (0.0..=1.0).contains(&u1) {
        // Precompute some common terms
        let u_o = lerp(p00, p10, u1);
        let u_dir = lerp(p01, p11, u1) - u_o;
        let delta_o = u_o - ray.o;
        let perp = ray.dir.cross(u_dir);
        let p2 = perp.length_squared();

        // Compute matrix determinants for v and t numerators
        let v1 = SquareMatrix::new([
            [delta_o.x(), ray.dir.x(), perp.x()],
            [delta_o.y(), ray.dir.y(), perp.y()],
            [delta_o.z(), ray.dir.z(), perp.z()],
        ])
        .determinant();
        let t1 = SquareMatrix::new([
            [delta_o.x(), u_dir.x(), perp.x()],
            [delta_o.y(), u_dir.y(), perp.y()],
            [delta_o.z(), u_dir.z(), perp.z()],
        ])
        .determinant();

        // Set u, v, t if intersection is valid
        // (may not be despite positive t due to float error)
        if t1 > p2 * epsilon && (0.0..=p2).contains(&v1) {
            u = Some(u1);
            v = Some(v1 / p2);
            t = t1 / p2;
        }
    }

    // Compute v and t for second u intersection
    // Only a valid intersection if between 0 and 1,
    // No need to repeat if it's the same intersection
    if (0.0..=1.0).contains(&u2) && u2 != u1 {
        // Precompute some common terms
        let u_o = lerp(p00, p10, u2);
        let u_dir = lerp(p01, p11, u2) - u_o;
        let delta_o = u_o - ray.o;
        let perp = ray.dir.cross(u_dir);
        let p2 = perp.length_squared();

        // Compute matrix determinants for v and t numerators
        let v2 = SquareMatrix::new([
            [delta_o.x(), ray.dir.x(), perp.x()],
            [delta_o.y(), ray.dir.y(), perp.y()],
            [delta_o.z(), ray.dir.z(), perp.z()],
        ])
        .determinant();
        let t2 = SquareMatrix::new([
            [delta_o.x(), u_dir.x(), perp.x()],
            [delta_o.y(), u_dir.y(), perp.y()],
            [delta_o.z(), u_dir.z(), perp.z()],
        ])
        .determinant()
            / p2;

        // Set u, v, t if intersection is valid, and is closer than t1
        // (may not be despite positive t due to float error)
        if t2 > p2 * epsilon && (0.0..=p2).contains(&v2) && t2 < t {
            u = Some(u2);
            v = Some(v2 / p2);
            t = t2;
        }
    }

    // Check intersection against t_max and possibly return intersection
    if t < t_max {
        // u and v should have been set
        Some(BilinearIntersection {
            uv: Point2f::new(u.unwrap(), v.unwrap()),
            t,
        })
    } else {
        None
    }
}

#[derive(Debug)]
pub struct BilinearPatchMesh {
    pub indices: Arc<Vec<usize>>,
    /// Vertex positions in render space.
    pub positions: Arc<Vec<Point3f>>,
    /// Per-vertex normals in render space, if any.
    pub normals: Option<Arc<Vec<Normal3f>>>,
    /// Vertex UVs, if any.
    pub uv: Option<Arc<Vec<Point2f>>>,
    pub reverse_orientation: bool,
    pub transform_swaps_handedness: bool,
    pub image_distribution: Option<PiecewiseConstant2D>,
}

static MESHES: OnceLock<Vec<BilinearPatchMesh>> = OnceLock::new();

impl BilinearPatchMesh {
    pub fn new(
        render_from_obj: &Transform,
        reverse_orientation: bool,
        indices: Vec<usize>,
        mut positions: Vec<Point3f>,
        normals: Option<Vec<Normal3f>>,
        uv: Option<Vec<Point2f>>,
    ) -> Self {
        assert_eq!(
            indices.len() % 4,
            0,
            "Number of vertex indices for a bilinear patch mesh must be multiple of 4, but got {}",
            indices.len()
        );

        // Lookup indices in cache
        let indices = USIZE_BUFFER_CACHE.lookup_or_add(indices);

        let positions = {
            // Transform positions to render space
            for p in positions.iter_mut() {
                *p = render_from_obj * *p;
            }
            POINT3F_BUFFER_CACHE.lookup_or_add(positions)
        };

        let normals = normals.map(|mut vec| {
            // Num must match num of indices
            assert_eq!(vec.len(), indices.len());
            // Transform normals to render space
            for n in vec.iter_mut() {
                *n = render_from_obj * *n;
                if reverse_orientation {
                    *n = -*n;
                }
            }
            NORMAL3F_BUFFER_CACHE.lookup_or_add(vec)
        });

        let uv = uv.map(|vec| {
            // Num must match num of indices
            assert_eq!(vec.len(), indices.len());
            POINT2F_BUFFER_CACHE.lookup_or_add(vec)
        });

        Self {
            indices,
            positions,
            normals,
            uv,
            reverse_orientation,
            transform_swaps_handedness: render_from_obj.swaps_handedness(),
            image_distribution: None,
        }
    }

    pub fn get(idx: usize) -> Option<&'static Self> {
        MESHES
            .get()
            .expect("Should not try to get() a mesh from storage before storage's been initialized")
            .get(idx)
    }

    pub fn init_mesh_data(all_meshes: Vec<BilinearPatchMesh>) {
        MESHES
            .set(all_meshes)
            .expect("Mesh storage shouldn't be set more than once")
    }

    pub fn positions(&self, blp_idx: usize) -> (Point3f, Point3f, Point3f, Point3f) {
        let verts = &self.indices[4 * blp_idx..4 * blp_idx + 4];
        let p00 = self.positions[verts[0]];
        let p10 = self.positions[verts[1]];
        let p01 = self.positions[verts[2]];
        let p11 = self.positions[verts[3]];
        (p00, p10, p01, p11)
    }

    pub fn vertex_normals(
        &self,
        blp_idx: usize,
    ) -> Option<(Normal3f, Normal3f, Normal3f, Normal3f)> {
        if let Some(vert_n) = &self.normals {
            let verts = &self.indices[4 * blp_idx..4 * blp_idx + 4];
            let n00 = vert_n[verts[0]];
            let n10 = vert_n[verts[1]];
            let n01 = vert_n[verts[2]];
            let n11 = vert_n[verts[3]];
            Some((n00, n10, n01, n11))
        } else {
            None
        }
    }

    pub fn uvs(&self, blp_idx: usize) -> Option<(Point2f, Point2f, Point2f, Point2f)> {
        if let Some(uv) = &self.uv {
            let verts = &self.indices[4 * blp_idx..4 * blp_idx + 4];
            let uv00 = uv[verts[0]];
            let uv10 = uv[verts[1]];
            let uv01 = uv[verts[2]];
            let uv11 = uv[verts[3]];
            Some((uv00, uv10, uv01, uv11))
        } else {
            None
        }
    }

    /// Returns `true` if the vertices of the patch at `blp_idx` form a rectangle.
    pub fn patch_is_rectangle(&self, blp_idx: usize) -> bool {
        let (p00, p10, p01, p11) = self.positions(blp_idx);
        // Test for coincident vertices (handled early to prevent invalid ops below)
        if p00 == p01 || p01 == p11 || p11 == p10 || p10 == p00 {
            return false;
        }

        // Check if verts are coplanar,
        // Compute the surface normal formed by three of the verts,
        // and test if the vector from one of them to the fourth is nearly
        // perpendicular to it
        let normal = (p10 - p00).cross(p01 - p00).normalized();
        if (p11 - p00).normalized().absdot(normal) > 1e-5 {
            return false;
        }

        // Check if verts form a rectangle.
        // If any vert's dist squared to mean position has relative error > 1e-4
        // to that of p00 consider as not a rectangle
        let p_center = (p00 + p10 + p01 + p11) / 4.0;
        let dist_sq = [
            p00.distance_squared(p_center),
            p10.distance_squared(p_center),
            p01.distance_squared(p_center),
            p11.distance_squared(p_center),
        ];
        if dist_sq
            .iter()
            .any(|d2| (d2 - dist_sq[0]).abs() / dist_sq[0] > 1e-4)
        {
            return false;
        }

        true
    }
}

pub struct BilinearIntersection {
    pub uv: Point2f,
    pub t: Float,
}
